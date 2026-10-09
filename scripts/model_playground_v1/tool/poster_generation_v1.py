import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image_cosmos3 import generate_image_cosmos3  # noqa: E402
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_visual_critic import poster_visual_critic  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402


class poster_generation_v1(poster_generation):
    tool_registry_classes = [generate_image_cosmos3, render_poster, poster_visual_critic]
