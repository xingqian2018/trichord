import io
import json
import os.path as osp
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, ToolCallError, migrate_files  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayImageGenerator, image_conversion, put, resolve_model_string, sniff_image_fmt  # noqa: E402

DEFAULT_MODEL_NAME = "nano-banana-2.0@nvidiak"

ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "4:3": "1184x880",
    "3:4": "880x1184",
    "16:9": "1360x768",
    "9:16": "768x1360",
}

ASPECT_RATIO_TO_GEMINI_PHRASE = {
    "1:1": "1:1 square aspect ratio, roughly resolution at 1024x1024",
    "4:3": "4:3 aspect ratio, landscape orientation, roughly resolution at 1184x880",
    "3:4": "3:4 aspect ratio, portrait orientation, roughly resolution at 880x1184",
    "16:9": "16:9 widescreen aspect ratio, landscape orientation, roughly resolution at 1360x768",
    "9:16": "9:16 vertical aspect ratio, portrait orientation, tall image, roughly resolution at 768x1360",
}


class generate_image(Tool):
    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt and save it to disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"]},
                    "output_path": {"type": "string", "description": "Absolute path to save the generated PNG as"},
                },
                "required": ["prompt", "aspect_ratio", "output_path"],
            },
        },
    }

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, scratch_root: Optional[str] = None):
        super().__init__()
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.image_path_history: list[str] = []
        self.model_name = model_name
        self.gateway = UnifiedGatewayImageGenerator(GATEWAY_CONFIG, num_concurrency=1, num_max_retry=2, timeout=300)

    def generate(self, prompt: str, aspect_ratio: str, output_path: str) -> str:
        if aspect_ratio not in ASPECT_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {aspect_ratio!r}")
        size = ASPECT_RATIO_TO_SIZE[aspect_ratio]
        if "gemini" in resolve_model_string(self.model_name).lower():
            prompt = f"{prompt}\n\n{ASPECT_RATIO_TO_GEMINI_PHRASE[aspect_ratio]}."
        out = Path(output_path)
        if out.suffix.lower() != ".png":
            raise ToolCallError(f"output_path must end with .png, got {out.name!r}")
        request = self.gateway.build_request(self.model_name, prompt, size=size, output_format="png")
        result = self.gateway.query([request])[0]
        if result is None:
            raise RuntimeError(f"Image generation failed for model '{self.model_name}'")
        data = result["images"][0]
        if sniff_image_fmt(data) != "png":
            buf = io.BytesIO()
            image_conversion(data, dst_fmt="pil").save(buf, format="PNG")
            data = buf.getvalue()
        put(data, str(out))
        return str(out)

    def run_core(self, args: dict) -> str:
        out = Path(args["output_path"])
        if self.scratch_root is not None and (not out.is_absolute() or not out.resolve().is_relative_to(self.scratch_root)):
            raise ToolCallError(f"the target save path {out} is not available; use an absolute path under {self.scratch_root}/")
        if out.exists():
            raise ToolCallError(f"the target save path {out} already exists; overwriting is not allowed, choose a new filename")
        path = self.generate(args["prompt"], args["aspect_ratio"], str(out))
        self.image_path_history.append(path)
        return f"generate_image saved to {path}"

    def save_history_core(self, path: str) -> None:
        setting = {
            "tool": type(self).__name__,
            "model_name": self.model_name,
            "model_string": resolve_model_string(self.model_name),
            "scratch_root": str(self.scratch_root) if self.scratch_root else None,
            "aspect_ratio_to_size": ASPECT_RATIO_TO_SIZE,
        }
        put(json.dumps(setting, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(path, "setting.json"))
        if self.result_text is not None:
            put(self.result_text.encode("utf-8"), osp.join(path, "result.txt"))
        migrate_files(path, self.image_path_history)
