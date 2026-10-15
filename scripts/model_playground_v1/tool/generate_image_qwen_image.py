import base64
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slurm_pool import SlurmPool  # noqa: E402
from tool.tool import Tool, ToolCallError  # noqa: E402
from utils import encode_image  # noqa: E402

QWEN_IMAGE_MODEL = "Qwen/Qwen-Image-2.1"
WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 1.0


class generate_image_qwen_image(Tool):
    """Render exactly the given description at the given aspect ratio on our Qwen-Image-2.1 replicas (no rewriting
    inside). The agent links `upsampler` (the upsample_prompt tool); rendering is refused until a description has been
    upsampled."""

    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Render one image from a detailed description at the given aspect ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The detailed description to render"},
                    "aspect_ratio": {"type": "string", "enum": list(WH_RATIO_TO_SIZE), "description": "Aspect ratio of the image"},
                },
                "required": ["prompt", "aspect_ratio"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self, image_format: str = "webp", image_quality: int = 85, webp_method: int = 5):
        super().__init__()
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.webp_method = webp_method
        self.renderer = SlurmPool("QI", "SLURM_QWEN_IMAGE_API_KEY", timeout=1800)
        self.upsampler: Optional[Any] = None
        self.output: Optional[dict[str, Any]] = None

    def render(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        def query(client: Any) -> bytes:
            response = client.images.generate(
                model=QWEN_IMAGE_MODEL,
                prompt=prompt,
                size=f"{width}x{height}",
                n=1,
                extra_body={"num_inference_steps": NUM_INFERENCE_STEPS, "true_cfg_scale": TRUE_CFG_SCALE, "seed": seed},
            )
            return base64.b64decode(response.data[0].b64_json)

        return self.renderer.call(query)

    def run_core(self, args: dict[str, Any]) -> str:
        if self.upsampler is not None and self.upsampler.output is None:
            raise ToolCallError("call upsample_prompt first, then call generate_image with the description it returns")
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise ToolCallError("missing required argument 'prompt'; call generate_image again with the description")
        ratio = args.get("aspect_ratio")
        if ratio not in WH_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(WH_RATIO_TO_SIZE)}, got {ratio!r}")
        width, height = WH_RATIO_TO_SIZE[ratio]
        seed = random.randrange(2**31)
        image = encode_image(self.render(prompt, width, height, seed), self.image_format, self.image_quality, self.webp_method)
        fd, path = tempfile.mkstemp(prefix="qwen_image_", suffix="." + self.image_format)
        with os.fdopen(fd, "wb") as f:
            f.write(image)
        self.add_file(path)
        self.output = {"render_prompt": prompt, "aspect_ratio": ratio, "width": width, "height": height, "seed": seed, "format": self.image_format, "image": image}
        return f"generate_image finished: {width}x{height} image"

    def setting(self) -> dict[str, Any]:
        return {
            **super().setting(),
            "image_model": QWEN_IMAGE_MODEL,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "image_format": self.image_format,
            "image_quality": self.image_quality,
            "webp_method": self.webp_method,
        }
