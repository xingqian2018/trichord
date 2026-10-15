import base64
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from slurm_pool import SlurmPool  # noqa: E402
from tool.tool import Tool, ToolCallError  # noqa: E402

QWEN_IMAGE_MODEL = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE_PE_MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_SYSTEM_PROMPT = (Path(__file__).resolve().parents[1] / "prompt" / "qwen_image_2p1_pe_t2i_system.txt").read_text().strip()

# Official Qwen-Image-2.1 presets for the rewriter's wh_ratio (Qwen-Image-2.1-PE-T2I model card)
WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
DEFAULT_WH_RATIO = "1:1"
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 1.0
PE_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192, "extra_body": {"top_k": 20}}
PE_MAX_ATTEMPTS = 3


def parse_rewrite(text: str) -> dict[str, str]:
    answer = text.rpartition("</think>")[2].strip()
    result = json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
    if not isinstance(result.get("rewritten_prompt"), str) or not result["rewritten_prompt"].strip():
        raise ValueError("rewriter returned no rewritten_prompt")
    return result


class generate_image_qwen_image(Tool):
    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate one image from a text description. The description is expanded and rendered automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What the image should show"},
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self):
        super().__init__()
        self.rewriter = SlurmPool("QPE", "SLURM_QWEN_IMAGE_PE_API_KEY", timeout=600)
        self.renderer = SlurmPool("QI", "SLURM_QWEN_IMAGE_API_KEY", timeout=1800)
        self.output: Optional[dict[str, Any]] = None

    def rewrite(self, prompt: str) -> dict[str, str]:
        def query(client: Any) -> str:
            response = client.chat.completions.create(
                model=QWEN_IMAGE_PE_MODEL,
                messages=[{"role": "system", "content": PE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                **PE_SAMPLING,
            )
            return response.choices[0].message.content or ""

        error: Optional[Exception] = None
        for _ in range(PE_MAX_ATTEMPTS):
            try:
                return parse_rewrite(self.rewriter.call(query))
            except (json.JSONDecodeError, ValueError) as e:
                error = e
        raise RuntimeError(f"prompt rewriter gave no valid JSON after {PE_MAX_ATTEMPTS} attempts: {error}")

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
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise ToolCallError("missing required argument 'prompt'; call generate_image again with a description")
        rewrite = self.rewrite(prompt)
        wh_ratio = rewrite.get("wh_ratio") if rewrite.get("wh_ratio") in WH_RATIO_TO_SIZE else DEFAULT_WH_RATIO
        width, height = WH_RATIO_TO_SIZE[wh_ratio]
        seed = random.randrange(2**31)
        png = self.render(rewrite["rewritten_prompt"], width, height, seed)
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            buf = io.BytesIO()
            Image.open(io.BytesIO(png)).save(buf, format="PNG")
            png = buf.getvalue()
        fd, path = tempfile.mkstemp(prefix="qwen_image_", suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png)
        self.add_file(path)
        self.output = {"prompt": prompt, "upsampled_prompt": rewrite["rewritten_prompt"], "wh_ratio": wh_ratio,
                       "width": width, "height": height, "seed": seed, "png": png}
        return f"generate_image finished: {width}x{height} image"

    def setting(self) -> dict[str, Any]:
        return {
            **super().setting(),
            "image_model": QWEN_IMAGE_MODEL,
            "rewriter_model": QWEN_IMAGE_PE_MODEL,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "true_cfg_scale": TRUE_CFG_SCALE,
        }
