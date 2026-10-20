import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image_cosmos3_with_strict_prompt_format_slurm import ASPECT_RATIO_TO_SIZE, generate_image_cosmos3_with_strict_prompt_format_slurm  # noqa: E402
from tool.poster_generation_v1p3 import poster_generation_v1p3, poster_visual_critic_v1p3, render_poster_v1p3, require_args  # noqa: E402
from tool.tool import ToolCallError  # noqa: E402
from tool.upsample_prompt_cosmos3 import upsample_prompt_cosmos3  # noqa: E402
from utils import put  # noqa: E402


class upsample_prompt_cosmos3_v2(upsample_prompt_cosmos3):
    def run_core(self, args: dict) -> str:
        require_args(self.schema, args)
        return super().run_core(args)


class generate_image_cosmos3_v2(generate_image_cosmos3_with_strict_prompt_format_slurm):
    """generate_image for the v2 poster agent: generate_image_cosmos3_with_strict_prompt_format_slurm (template check, official settings, our
    Cosmos3 replicas) with the poster pipeline's output_path, so the HTML can reference the saved PNG. The agent passes
    the structured JSON returned by upsample_prompt; rendering is refused until upsample_prompt has been called."""

    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Render one image from the structured JSON returned by upsample_prompt and save it as a PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "object", "description": "The structured JSON object returned by upsample_prompt, fixed where needed"},
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Aspect ratio of the image"},
                    "output_path": {"type": "string", "description": "Absolute path to save the generated PNG as"},
                },
                "required": ["prompt", "aspect_ratio", "output_path"],
            },
        },
    }

    def __init__(self, scratch_root: Optional[str] = None):
        super().__init__(image_format="png")
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.upsampler: Optional[upsample_prompt_cosmos3] = None

    def run_core(self, args: dict[str, Any]) -> str:
        require_args(self.schema, args)
        if self.upsampler is not None and self.upsampler.output is None:
            raise ToolCallError("call upsample_prompt first, then call generate_image with the structured JSON it returns")
        out = Path(args["output_path"])
        if out.suffix.lower() != ".png":
            raise ToolCallError(f"output_path must end with .png, got {out.name!r}")
        if self.scratch_root is not None and (not out.is_absolute() or not out.resolve().is_relative_to(self.scratch_root)):
            raise ToolCallError(f"the target save path {out} is not available; use an absolute path under {self.scratch_root}/")
        if out.exists():
            raise ToolCallError(f"the target save path {out} already exists; overwriting is not allowed, choose a new filename")
        request = self.upsampler.output["prompt"] if self.upsampler is not None else "poster image"
        result = super().run_core({"request": request, "prompt": args["prompt"], "aspect_ratio": args["aspect_ratio"]})
        if self.output is None:
            return result
        put(self.output["image"], str(out))
        return f"generate_image saved to {out}"

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), "scratch_root": str(self.scratch_root) if self.scratch_root else None}


class poster_generation_v2(poster_generation_v1p3):
    """v2 = v1p5 with Kimi-K3 prompt upsampling in front of Cosmos3, as two tools like the Qwen-Image-2.1 t2i pipeline:
    for each visual the agent calls upsample_prompt (a separate Kimi-K3 conversation with the official Cosmos3 upsampling
    rules, tool/upsample_prompt_cosmos3.py) and then generate_image with the returned structured JSON, rendered with the
    official settings. System prompt: prompt/poster_generation_agentic_layout_t2i_required_v2.md."""

    tool_registry_classes = [upsample_prompt_cosmos3_v2, generate_image_cosmos3_v2, render_poster_v1p3, poster_visual_critic_v1p3]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.image_tool().upsampler = self.tool_instances[upsample_prompt_cosmos3_v2.schema["function"]["name"]]

    def image_tool(self) -> generate_image_cosmos3_v2:
        return self.tool_instances[generate_image_cosmos3_v2.schema["function"]["name"]]
