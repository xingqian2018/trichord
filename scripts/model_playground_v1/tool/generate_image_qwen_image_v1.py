import io
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from tool.generate_image_qwen_image import WH_RATIO_TO_SIZE, generate_image_qwen_image  # noqa: E402
from tool.tool import ToolCallError  # noqa: E402


class generate_image_qwen_image_v1(generate_image_qwen_image):
    """Render exactly the given description at the given aspect ratio (no rewriting inside). The agent links
    `upsampler` (the upsample_prompt tool); rendering is refused until a description has been upsampled."""

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

    def __init__(self):
        super().__init__()
        self.upsampler: Optional[Any] = None

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
        png = self.render(prompt, width, height, seed)
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            buf = io.BytesIO()
            Image.open(io.BytesIO(png)).save(buf, format="PNG")
            png = buf.getvalue()
        fd, path = tempfile.mkstemp(prefix="qwen_image_", suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png)
        self.add_file(path)
        self.output = {"render_prompt": prompt, "aspect_ratio": ratio, "width": width, "height": height, "seed": seed, "png": png}
        return f"generate_image finished: {width}x{height} image"

    def setting(self) -> dict[str, Any]:
        base = super().setting()
        base.pop("rewriter_model", None)
        return base
