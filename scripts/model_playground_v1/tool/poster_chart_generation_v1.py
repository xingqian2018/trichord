import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_generation_v1p3 import CRITIC_NUM_MAX_RETRY, CRITIC_TIMEOUT, poster_visual_critic_v1p3, render_poster_v1p3  # noqa: E402
from tool.poster_generation_v1p2 import K3SLURM_MODEL_NAME  # noqa: E402
from tool.poster_generation_v2 import generate_image_cosmos3_v2, poster_generation_v2, upsample_prompt_cosmos3_v2  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
CHART_CRITIC_SYSTEM_PROMPT = (PROMPT_DIR / "poster_visual_critic_chart_v1.md").read_text().strip()
RENDER_REQUIRED_WARNING = (
    "You delivered HTML without rendering and reviewing it. Call render_poster with that HTML, then poster_visual_critic "
    "on the returned path, fix the must-fix points, and deliver only the version that was last rendered and reviewed."
)
IMAGE_REQUIRED_WARNING = (
    "Your poster has no generated image, but at least one is required. Add one in a planned slot: call upsample_prompt, "
    "then generate_image with the returned JSON and a new output_path under /tmp/poster_agent/, use the returned path as "
    "the <img> src, then render_poster and poster_visual_critic again, and deliver only the last rendered and reviewed version."
)
SAVED_RE = re.compile(r"generate_image saved to (\S+)")


class poster_visual_critic_chart_v1(poster_visual_critic_v1p3):
    """Same tool (schema name poster_visual_critic) on Slurm-hosted Kimi-K3, with a strict data-poster review prompt:
    edges, every chart's box, geometry against labeled numbers (pie/donut shares, bar lengths, time spacing), legends."""

    def __init__(self, model_name: str = K3SLURM_MODEL_NAME, **kwargs: Any):
        Agent.__init__(
            self,
            model_name,
            CHART_CRITIC_SYSTEM_PROMPT,
            temperature=0.0,
            num_max_retry=CRITIC_NUM_MAX_RETRY,
            timeout=CRITIC_TIMEOUT,
            **kwargs,
        )


class poster_chart_generation_v1(poster_generation_v2):
    """Data posters with inline-SVG charts (prompt/poster_chart_generation_v1.md) on the v2 tools, with the strict chart
    critic; a reply that delivers HTML without any render_poster call gets one more turn asking it to render and review."""

    tool_registry_classes = [upsample_prompt_cosmos3_v2, generate_image_cosmos3_v2, render_poster_v1p3, poster_visual_critic_chart_v1]

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
