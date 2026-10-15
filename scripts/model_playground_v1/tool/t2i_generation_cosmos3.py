import json
import os.path as osp
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image_cosmos3_slurm_v1 import generate_image_cosmos3_slurm_v1  # noqa: E402
from utils import put  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@k3slurm"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "t2i_generation_cosmos3_v0.md").read_text().strip()
MAX_STEPS = 4
NUDGE_MESSAGE = "You replied with text. Call generate_image now through a function call; do not write the call as text."


class t2i_generation_cosmos3(Agent):
    """Kimi turns one taxonomy topic into a Cosmos3 structured JSON prompt (official T2I template) and calls
    generate_image (Cosmos3-Super-Text2Image) once; the run stops as soon as the image exists. Every request uses
    tool_choice="required"; a template violation is returned to Kimi as a tool error and it may call again.
    Saves the usual trace (setting.json, chat.json, tool/generate_image/call/<id>/) plus <root>/derived/image.png and
    <root>/derived/meta.json = {taxonomy, prompt, upsampled_prompt}: Kimi's short request and the structured JSON the
    image was rendered from (Kimi plays the official upsampler, prompt/t2i_generation_cosmos3_v0.md)."""

    tool_registry_classes = [generate_image_cosmos3_slurm_v1]

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(model_name, system_prompt, tool_registry=list(type(self).tool_registry_classes), **kwargs)
        self.taxonomy = None

    def image_tool(self) -> generate_image_cosmos3_slurm_v1:
        return self.tool_instances[generate_image_cosmos3_slurm_v1.schema["function"]["name"]]

    def build_request(self) -> dict[str, Any]:
        return {**super().build_request(), "tool_choice": "required"}

    def run_core(self, args: dict[str, Any]) -> str:
        self.taxonomy = args.get("taxonomy")
        self.add_user_message(args["prompt"])
        for _ in range(MAX_STEPS):
            finish_reason = self.step()
            if finish_reason == "tool_calls":
                self.execute_tool_calls()
            else:
                self.add_user_message(NUDGE_MESSAGE)
            if self.image_tool().output is not None:
                return self.messages[-1].get("content") or ""
        raise RuntimeError(f"{self.model_name} made no image within {MAX_STEPS} steps")

    def save_history_core(self, root: str) -> None:
        super().save_history_core(root)
        image = self.image_tool().output
        meta = {"taxonomy": self.taxonomy, "prompt": image["request"], "upsampled_prompt": image["prompt"]}
        put(image["png"], osp.join(root, "derived", "image.png"))
        put(json.dumps(meta, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "derived", "meta.json"))
