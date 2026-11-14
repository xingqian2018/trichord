import copy
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.poster_chart_generation_v1 import poster_chart_generation_v1, poster_visual_critic_chart_v1  # noqa: E402
from tool.poster_generation_v1p3 import render_poster_v1p3  # noqa: E402
from tool.generate_image_cosmos3_with_strict_prompt_format_slurm import COSMOS3_MODEL  # noqa: E402
from tool.poster_generation_v2 import generate_image_cosmos3_v2, upsample_prompt_cosmos3_v2  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.aspect_ratio import ASPECT_RATIOS, SIZE_1K, SIZE_2K, size_text  # noqa: E402

RENDER_ASPECT_RATIO_TO_SIZE_V2 = {r: size_text(SIZE_2K[r]) for r in ASPECT_RATIOS}
IMAGE_ASPECT_RATIO_TO_SIZE_V2 = {r: SIZE_1K[r] for r in ASPECT_RATIOS}


def with_aspect_ratios(schema: dict[str, Any], ratios: Iterable[str]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    schema["function"]["parameters"]["properties"]["aspect_ratio"]["enum"] = list(ratios)
    return schema


class upsample_prompt_cosmos3_chart_v2(upsample_prompt_cosmos3_v2):
    schema = with_aspect_ratios(upsample_prompt_cosmos3_v2.schema, ASPECT_RATIOS)
    aspect_ratio_to_size = IMAGE_ASPECT_RATIO_TO_SIZE_V2


class generate_image_cosmos3_chart_v2(generate_image_cosmos3_v2):
    """generate_image on the original Cosmos3-Super-Text2Image (c3_serve), pinned so C3_SERVED_NAME cannot switch it to x9c."""

    schema = with_aspect_ratios(generate_image_cosmos3_v2.schema, ASPECT_RATIOS)
    aspect_ratio_to_size = IMAGE_ASPECT_RATIO_TO_SIZE_V2

    def __init__(self, scratch_root: Optional[str] = None):
        super().__init__(scratch_root=scratch_root)
        self.served_model_name = COSMOS3_MODEL


class render_poster_chart_v2(render_poster_v1p3):
    """render_poster at 2K (1:1 = 2048x2048, every ratio at about the same pixel area), seven ratios; v1 rendered at the
    1 MP Cosmos3 sizes."""

    schema = with_aspect_ratios(render_poster_v1p3.schema, ASPECT_RATIOS)
    aspect_ratio_to_size = RENDER_ASPECT_RATIO_TO_SIZE_V2


class poster_chart_generation_v2(poster_chart_generation_v1):
    """Chart posters v1 with 2K renders and seven aspect ratios (16:9 3:2 4:3 1:1 3:4 2:3 9:16) for the poster, its
    generated images and the upsampler, images from the original Cosmos3-Super-Text2Image instead of x9c; same prompts,
    critic and checks as v1."""

    tool_registry_classes = [upsample_prompt_cosmos3_chart_v2, generate_image_cosmos3_chart_v2, render_poster_chart_v2, poster_visual_critic_chart_v1]
    aspect_ratio_to_size = RENDER_ASPECT_RATIO_TO_SIZE_V2
    schema = with_aspect_ratios(poster_chart_generation_v1.schema, ASPECT_RATIOS)
    schema["function"]["parameters"]["properties"]["prompt"]["description"] = (
        "The brief: subject, audience, tone, required text, and the target aspect ratio (" + ", ".join(ASPECT_RATIOS) + ")"
    )
