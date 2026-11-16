r"""
Posters v3 = posters v2 specialized for expressive typography and no small text (tool/poster_generation_v3.py):
- system prompt prompt/poster_generation_v3.md: display fonts with character loaded from Google Fonts (system font names
  are not installed in the render container and fall back to DejaVu/Liberation), at most three families, and minimum
  text sizes relative to the shorter canvas side (headline 7%, body 3%, anything 2.2%), cutting copy instead of shrinking it;
- critic prompt prompt/poster_visual_critic_v3.md: small text is a must-fix, the display type is checked;
- render_poster waits for web fonts (--virtual-time-budget), 2K renders, seven aspect ratios 16:9 3:2 4:3 1:1 3:4 2:3 9:16,
  images from the original Cosmos3-Super-Text2Image (c3_serve), as chart v2.
The render node needs internet access to fonts.googleapis.com / fonts.gstatic.com.
Launch with C3_ENDPOINTS=$(cat $RUN/c3_serve/broker_url)/v1 and no C3_SERVED_NAME, then
    python3 script/run_poster_generation_v3.py --model kimi-k3@k3slurm --prompt_preset poster_generation_v3 \
        --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v3_kimik3_cosmos3upsampled_kimik3 \
        --aspect_ratio random --reasoning_effort random --max_samples_per_process 625
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v3 import poster_generation_v3  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v3)
