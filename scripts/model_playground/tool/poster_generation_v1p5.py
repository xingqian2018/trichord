import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image_cosmos3_slurm import generate_image_cosmos3_slurm  # noqa: E402
from tool.poster_generation_v1p3 import poster_generation_v1p3, poster_visual_critic_v1p3, render_poster_v1p3, require_args  # noqa: E402


class generate_image_cosmos3_slurm_v1p5(generate_image_cosmos3_slurm):
    def run_core(self, args: dict) -> str:
        require_args(self.schema, args)
        return super().run_core(args)


class poster_generation_v1p5(poster_generation_v1p3):
    tool_registry_classes = [generate_image_cosmos3_slurm_v1p5, render_poster_v1p3, poster_visual_critic_v1p3]
