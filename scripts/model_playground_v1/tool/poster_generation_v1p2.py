import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image_cosmos3 import generate_image_cosmos3  # noqa: E402
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_visual_critic import poster_visual_critic  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402

K3SLURM_MODEL_NAME = "kimi-k3@k3slurm"


class poster_visual_critic_k3slurm(poster_visual_critic):
    """Same tool (schema name poster_visual_critic), but the critic itself runs on Slurm-hosted Kimi-K3."""

    def __init__(self, model_name: str = K3SLURM_MODEL_NAME, **kwargs):
        super().__init__(model_name, **kwargs)


class poster_generation_v1p2(poster_generation):
    tool_registry_classes = [generate_image_cosmos3, render_poster, poster_visual_critic_k3slurm]
