import os.path as osp
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE  # noqa: E402
from tool.generate_image_sunburst import generate_image_sunburst  # noqa: E402
from tool.tool import ToolCallError  # noqa: E402
from utils import put  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@nvidiak"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "poster_generation_lite_v0.md").read_text().strip()


class poster_generation_lite(Agent):
    schema = {
        "type": "function",
        "function": {
            "name": "poster_generation_lite",
            "description": (
                "Poster art director. Given a brief, it plans the poster and makes exactly one image-model call that "
                "renders the finished poster, then stops. Returns the tool result with the absolute path of the poster image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The brief: subject, audience, tone, and any required text"},
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Target aspect ratio of the poster"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional reference images: absolute paths, file://, http(s), or data: URLs",
                    },
                },
                "required": ["prompt", "aspect_ratio"],
            },
        },
    }

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(model_name, system_prompt, tool_registry=[generate_image_sunburst], **kwargs)

    def run_core(self, args: dict[str, Any]) -> str:
        aspect_ratio = args.get("aspect_ratio")
        if aspect_ratio not in ASPECT_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {aspect_ratio!r}")
        self.add_user_message(f"{args['prompt']}\n\nTarget aspect ratio: {aspect_ratio}", image_urls=args.get("image_urls") or None)
        finish_reason = self.step()
        if finish_reason != "tool_calls":
            raise RuntimeError(f"{self.model_name} ended with finish_reason={finish_reason!r} instead of calling {generate_image_sunburst.schema['function']['name']}")
        self.execute_tool_calls()
        return self.messages[-1].get("content") or ""

    def save_history_core(self, root: str) -> None:
        self.save_derived(osp.join(root, "derived"))
        super().save_history_core(root)

    def save_derived(self, derived: str) -> None:
        for instance in self.tool_instances.values():
            for record in instance.calls:
                prompt = record["arguments"].get("prompt")
                if prompt is not None:
                    put(prompt.encode("utf-8"), osp.join(derived, "prompt.txt"))
                for path in record["files"]:
                    if Path(path).is_file():
                        put(Path(path).read_bytes(), osp.join(derived, "poster.png"))
