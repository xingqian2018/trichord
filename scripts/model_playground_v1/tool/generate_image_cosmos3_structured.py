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

COSMOS3_MODEL = "LeptonTR/Cosmos3-Super-Text2Image"
# Official Cosmos3-Super-Text2Image agentic_upsampling defaults (constants.py / prompt_upsampler.py, "768" tier)
ASPECT_RATIO_TO_SIZE = {
    "1:1": (1024, 1024),
    "4:3": (1184, 880),
    "3:4": (880, 1184),
    "16:9": (1360, 768),
    "9:16": (768, 1360),
}
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
FLOW_SHIFT = 3.0
EXTRA_ARGS = {"guardrails": False, "use_resolution_template": False}
COUNT_KEYS = ["number_of_subjects", "number_of_arms", "number_of_legs", "number_of_hands", "number_of_fingers"]
OBJECT_KEYS = ["subject_details", "lighting", "aesthetics", "cinematography", "quadrant_scan"]
TEMPLATE_KEYS = [
    "subjects", "subject_details", "background_setting", "lighting", "aesthetics", "cinematography", "style_medium",
    "artistic_style", "context", "text_and_signage_elements", "quadrant_scan", "comprehensive_t2i_caption",
]


class generate_image_cosmos3_structured(Tool):
    """Render a Cosmos3 structured JSON prompt (the official T2I template) on our Cosmos3 replicas.
    The tool fills `resolution` and `aspect_ratio` in the JSON from the chosen aspect ratio, as the official upsampler does."""

    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Render one image from a structured JSON description at the given aspect ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "The short natural-language image request the description was expanded from"},
                    "prompt": {"type": "object", "description": "The structured image description, with every top-level key of the template"},
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Aspect ratio of the image"},
                },
                "required": ["request", "prompt", "aspect_ratio"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self):
        super().__init__()
        self.renderer = SlurmPool("C3", "SLURM_COSMOS3_API_KEY", timeout=1800)
        self.output: Optional[dict[str, Any]] = None

    def render(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        def query(client: Any) -> bytes:
            response = client.images.generate(
                model=COSMOS3_MODEL,
                prompt=prompt,
                size=f"{width}x{height}",
                n=1,
                extra_body={"num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE, "flow_shift": FLOW_SHIFT,
                            "negative_prompt": "", "seed": seed, "extra_args": dict(EXTRA_ARGS)},
            )
            return base64.b64decode(response.data[0].b64_json)

        return self.renderer.call(query)

    @staticmethod
    def problems(prompt: dict[str, Any]) -> list[str]:
        found = []
        missing = [k for k in TEMPLATE_KEYS if k not in prompt]
        if missing:
            found.append(f"missing top-level keys {missing}")
        extra = [k for k in prompt if k not in TEMPLATE_KEYS + ["resolution", "aspect_ratio"]]
        if extra:
            found.append(f"keys not in the template {extra}")
        if "subjects" in prompt and not isinstance(prompt["subjects"], list):
            found.append("subjects must be a list")
        if "text_and_signage_elements" in prompt and not isinstance(prompt["text_and_signage_elements"], list):
            found.append("text_and_signage_elements must be a list")
        for key in OBJECT_KEYS:
            if key in prompt and not isinstance(prompt[key], dict):
                found.append(f"{key} must be an object")
        caption = prompt.get("comprehensive_t2i_caption")
        if "comprehensive_t2i_caption" in prompt and (not isinstance(caption, str) or not caption.strip()):
            found.append("comprehensive_t2i_caption must be a non-empty paragraph")
        for i, subject in enumerate(prompt.get("subjects") or []):
            if not isinstance(subject, dict):
                found.append(f"subjects[{i}] must be an object")
                continue
            bad = [k for k in COUNT_KEYS if k in subject and (isinstance(subject[k], bool) or not isinstance(subject[k], int))]
            if bad:
                found.append(f"subjects[{i}] {bad} must be JSON integers")
        return found

    def run_core(self, args: dict[str, Any]) -> str:
        request = (args.get("request") or "").strip()
        if not request:
            raise ToolCallError("missing required argument 'request'; pass the short image request the description expands")
        prompt = args.get("prompt")
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                raise ToolCallError("prompt must be a JSON object following the template, not free text")
        if not isinstance(prompt, dict):
            raise ToolCallError("prompt must be a JSON object following the template")
        found = self.problems(prompt)
        if found:
            raise ToolCallError("the structured prompt does not follow the template: " + "; ".join(found) + ". Fix it and call generate_image again.")
        ratio = args.get("aspect_ratio")
        if ratio not in ASPECT_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {ratio!r}")
        width, height = ASPECT_RATIO_TO_SIZE[ratio]
        structured = {**{k: prompt[k] for k in TEMPLATE_KEYS}, "resolution": {"H": height, "W": width}, "aspect_ratio": ratio.replace(":", ",")}
        seed = random.randrange(2**31)
        png = self.render(json.dumps(structured), width, height, seed)
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            buf = io.BytesIO()
            Image.open(io.BytesIO(png)).save(buf, format="PNG")
            png = buf.getvalue()
        fd, path = tempfile.mkstemp(prefix="cosmos3_", suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png)
        self.add_file(path)
        self.output = {"request": request, "prompt": structured, "aspect_ratio": ratio, "width": width, "height": height, "seed": seed, "png": png}
        return f"generate_image finished: {width}x{height} image"

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), "image_model": COSMOS3_MODEL, "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE, "flow_shift": FLOW_SHIFT, "extra_args": EXTRA_ARGS}
