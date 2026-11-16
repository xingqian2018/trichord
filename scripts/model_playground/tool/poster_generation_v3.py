import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.poster_chart_generation_v1 import IMAGE_REQUIRED_WARNING, RENDER_REQUIRED_WARNING, SAVED_RE  # noqa: E402
from tool.poster_chart_generation_v2 import (  # noqa: E402
    RENDER_ASPECT_RATIO_TO_SIZE_V2,
    generate_image_cosmos3_chart_v2,
    render_poster_chart_v2,
    upsample_prompt_cosmos3_chart_v2,
    with_aspect_ratios,
)
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_generation_v1p2 import K3SLURM_MODEL_NAME  # noqa: E402
from tool.poster_generation_v1p3 import CRITIC_NUM_MAX_RETRY, CRITIC_TIMEOUT, poster_visual_critic_v1p3  # noqa: E402
from tool.poster_generation_v2 import poster_generation_v2  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.aspect_ratio import ASPECT_RATIOS  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
CRITIC_SYSTEM_PROMPT_V3 = (PROMPT_DIR / "poster_visual_critic_v3.md").read_text().strip()
WEB_FONT_WAIT_MS = 10000
POSTER_MAX_TOKENS = 32768
POSTER_TIMEOUT = 1200


class render_poster_v3(render_poster_chart_v2):
    """render_poster at 2K with seven ratios that lets Google Fonts load before the screenshot: without a virtual time
    budget headless Chrome sometimes captures the page before the web fonts arrive and renders the fallback font."""

    chrome_args = (f"--virtual-time-budget={WEB_FONT_WAIT_MS}",)


class poster_visual_critic_v3(poster_visual_critic_v1p3):
    """Same tool (schema name poster_visual_critic) on Slurm-hosted Kimi-K3, with a review prompt that makes small text a
    must-fix and checks the display type (prompt/poster_visual_critic_v3.md)."""

    def __init__(self, model_name: str = K3SLURM_MODEL_NAME, **kwargs: Any):
        Agent.__init__(
            self,
            model_name,
            CRITIC_SYSTEM_PROMPT_V3,
            temperature=0.0,
            num_max_retry=CRITIC_NUM_MAX_RETRY,
            timeout=CRITIC_TIMEOUT,
            **kwargs,
        )


class poster_generation_v3(poster_generation_v2):
    """Posters v2 specialized for expressive Google Fonts typography and no small text (prompt/poster_generation_v3.md and
    the v3 critic), on the chart v2 setup: 2K renders, seven aspect ratios, images from the original
    Cosmos3-Super-Text2Image; a reply without a rendered version or without a generated image gets one more turn."""

    tool_registry_classes = [upsample_prompt_cosmos3_chart_v2, generate_image_cosmos3_chart_v2, render_poster_v3, poster_visual_critic_v3]
    aspect_ratio_to_size = RENDER_ASPECT_RATIO_TO_SIZE_V2
    schema = with_aspect_ratios(poster_generation_v2.schema, ASPECT_RATIOS)
    schema["function"]["parameters"]["properties"]["prompt"]["description"] = (
        "The brief: subject, audience, tone, required text, and the target aspect ratio (" + ", ".join(ASPECT_RATIOS) + ")"
    )

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("max_tokens", POSTER_MAX_TOKENS)
        kwargs.setdefault("timeout", POSTER_TIMEOUT)
        super().__init__(*args, **kwargs)

    def generated_paths(self) -> list[str]:
        return [path for m in self.messages if m.get("role") == "tool" for path in SAVED_RE.findall(m.get("content") or "")]

    def uses_generated_image(self) -> bool:
        return self.cached_html is not None and any(path in self.cached_html for path in self.generated_paths())

    def run_core(self, args: dict[str, Any]) -> str:
        result = super().run_core(args)
        if self.cached_html is None:
            self.add_user_message(RENDER_REQUIRED_WARNING)
            result = poster_generation.run_core(self, {})
        if not self.uses_generated_image():
            self.add_user_message(IMAGE_REQUIRED_WARNING)
            result = poster_generation.run_core(self, {})
        return result
