import json
import os.path as osp
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image_qwen_image_v1 import generate_image_qwen_image_v1  # noqa: E402
from tool.upsample_prompt_qwen_image import upsample_prompt_qwen_image  # noqa: E402
from utils import put  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@k3slurm"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "t2i_generation_v1.md").read_text().strip()
MAX_STEPS = 6
NUDGE_MESSAGE = "You replied with text. Continue through a function call (upsample_prompt, then generate_image); do not write calls as text."


class t2i_generation_v1(Agent):
    """Kimi writes a base prompt for one taxonomy topic, calls upsample_prompt (Qwen-Image-2.1-PE-T2I), may edit the
    expanded description, and calls generate_image (Qwen-Image-2.1) once; the run stops as soon as the image exists,
    there is no review or second render. Every request uses tool_choice="required".
    Saves the usual trace (setting.json, chat.json, tool/<fn>/call/<id>/) plus <root>/derived/image.png and
    <root>/derived/meta.json = {taxonomy, prompt, upsampled_prompt, render_prompt}: Kimi's base prompt, the rewriter's
    expansion, and the description the image was actually rendered from (equal to upsampled_prompt unless Kimi edited it)."""

    tool_registry_classes = [upsample_prompt_qwen_image, generate_image_qwen_image_v1]

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(model_name, system_prompt, tool_registry=list(type(self).tool_registry_classes), **kwargs)
        self.taxonomy = None
        self.image_tool().upsampler = self.upsample_tool()

    def upsample_tool(self) -> upsample_prompt_qwen_image:
        return self.tool_instances[upsample_prompt_qwen_image.schema["function"]["name"]]

    def image_tool(self) -> generate_image_qwen_image_v1:
        return self.tool_instances[generate_image_qwen_image_v1.schema["function"]["name"]]

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
        upsampled, image = self.upsample_tool().output, self.image_tool().output
        meta = {"taxonomy": self.taxonomy, "prompt": upsampled["prompt"], "upsampled_prompt": upsampled["upsampled_prompt"],
                "render_prompt": image["render_prompt"]}
        put(image["png"], osp.join(root, "derived", "image.png"))
        put(json.dumps(meta, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "derived", "meta.json"))
