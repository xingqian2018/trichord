r"""
Chart posters v2 = chart v1 (tool/poster_chart_generation_v1.py) rendered at 2K (render_poster at about 2048 px, v1 rendered at
the 1 MP Cosmos3 sizes) with seven aspect ratios, 16:9 3:2 4:3 1:1 3:4 2:3 9:16, for the poster, the generated images and the
upsampler, and images from the original Cosmos3-Super-Text2Image (c3_serve, serving/serve.sh cosmos3_super_t2i) instead of x9c
(tool/poster_chart_generation_v2.py). Same system prompt, critic and Kimi-K3 service as v1; launch with
C3_ENDPOINTS=$(cat $RUN/c3_serve/broker_url)/v1 and no C3_SERVED_NAME, then
    python3 script/run_poster_chart_generation_v2.py --model kimi-k3@k3slurm --prompt_preset poster_chart_generation_v1 \
        --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_chart_generation_v2_kimik3_cosmos3x9cupsampled_kimik3 \
        --aspect_ratio random --reasoning_effort random --max_samples_per_process 625
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()

from run_poster_generation import main  # noqa: E402
from tool.poster_chart_generation_v2 import poster_chart_generation_v2  # noqa: E402

if __name__ == "__main__":
    main(poster_chart_generation_v2)
