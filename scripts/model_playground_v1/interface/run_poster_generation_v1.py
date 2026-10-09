# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground_v1/interface/run_poster_generation_v1.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v1 import poster_generation_v1  # noqa: E402

main(poster_generation_v1)
