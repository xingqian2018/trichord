import os.path as osp
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image_qwen_image import generate_image_qwen_image  # noqa: E402
from utils import put  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@k3slurm"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "t2i_generation_v0.md").read_text().strip()


class t2i_generation(Agent):
    """Kimi writes a base prompt for one taxonomy topic and makes exactly one generate_image call, then stops.
    Saves only <root>/image.webp and <root>/prompt.txt (the rewritten prompt the image was rendered from)."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(model_name, system_prompt, tool_registry=[generate_image_qwen_image], **kwargs)

    def image_tool(self) -> generate_image_qwen_image:
        return self.tool_instances[generate_image_qwen_image.schema["function"]["name"]]

    def run_core(self, args: dict[str, Any]) -> str:
        self.add_user_message(args["prompt"])
        finish_reason = self.step()
        if finish_reason != "tool_calls":
            raise RuntimeError(f"{self.model_name} ended with finish_reason={finish_reason!r} instead of calling generate_image")
        self.execute_tool_calls()
        if self.image_tool().output is None:
            raise RuntimeError(f"generate_image produced no image: {self.messages[-1].get('content')}")
        return self.messages[-1].get("content") or ""

    def save_history_core(self, root: str) -> None:
        output = self.image_tool().output
        put(output["webp"], osp.join(root, "image.webp"))
        put(output["prompt"].encode("utf-8"), osp.join(root, "prompt.txt"))
