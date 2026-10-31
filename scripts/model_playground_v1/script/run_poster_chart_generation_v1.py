r"""
Chart posters v1: data posters whose focal point is inline-SVG charts (tool/poster_chart_generation_v1.py), on the v2
tools (optional images via Kimi-K3 upsample_prompt + Cosmos3 generate_image, render_poster) with a strict chart critic
(prompt/poster_visual_critic_chart_v1.md). A reply that skips render_poster gets one more turn asking it to render and review.
System prompt: prompt/poster_chart_generation_v1.md. Same services and launch as run_poster_generation_v2.py, with
    python3 script/run_poster_chart_generation_v1.py --model kimi-k3@k3slurm --prompt_preset poster_chart_generation_v1 \
        --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_chart_generation_v1_kimik3_cosmos3x9cupsampled_kimik3 \
        --aspect_ratio random --reasoning_effort random --max_samples_per_process 625
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()

from run_poster_generation import main  # noqa: E402
from tool.poster_chart_generation_v1 import poster_chart_generation_v1  # noqa: E402

if __name__ == "__main__":
    main(poster_chart_generation_v1)
