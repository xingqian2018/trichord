import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, migrate_files  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayImageGenerator, resolve_model_string  # noqa: E402

DEFAULT_MODEL_NAME = "nano-banana-2.0"

ASPECT_RATIO_TO_SIZE = {
    "1:1": "960x960",
    "4:3": "1104x832",
    "3:4": "832x1104",
    "16:9": "1280x720",
    "9:16": "720x1280",
}

ASPECT_RATIO_TO_PHRASE = {
    "1:1": "1:1 square aspect ratio",
    "4:3": "4:3 aspect ratio, landscape orientation",
    "3:4": "3:4 aspect ratio, portrait orientation",
    "16:9": "16:9 widescreen aspect ratio, landscape orientation",
    "9:16": "9:16 vertical aspect ratio, portrait orientation, tall image",
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
        self.images: list[str] = []
        self.model_name = model_name
        self.gateway = UnifiedGatewayImageGenerator(GATEWAY_CONFIG, num_concurrency=1, num_max_retry=2, timeout=300)

    def generate(self, prompt: str, aspect_ratio: str, output_path: str) -> str:
        size = ASPECT_RATIO_TO_SIZE.get(aspect_ratio, "1024x1024")
        if "gemini" in resolve_model_string(self.model_name).lower():
            phrase = ASPECT_RATIO_TO_PHRASE.get(aspect_ratio, f"{aspect_ratio} aspect ratio")
            prompt = f"{prompt}\n\n{phrase}."
        request = self.gateway.build_request(self.model_name, prompt, size=size)
        result = self.gateway.query([request])[0]
        if result is None:
            raise RuntimeError(f"Image generation failed for model '{self.model_name}'")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(result["images"][0])
        return str(out)

    def run(self, args: dict) -> str:
        out = Path(args["output_path"])
        if self.scratch_root is not None and (not out.is_absolute() or not out.resolve().is_relative_to(self.scratch_root)):
            raise ValueError(f"the target save path {out} is not available; use an absolute path under {self.scratch_root}/")
        path = self.generate(args["prompt"], args.get("aspect_ratio", "1:1"), str(out))
        self.images.append(path)
        return f"[generate_image saved to {path}]"

    def produced_files(self) -> list[str]:
        return list(self.images)

    def save_history(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.result_text is not None:
            (target / "result.txt").write_text(self.result_text)
        moved = migrate_files(target, self.images)
        self.images[:] = [moved.get(x, x) for x in self.images]
