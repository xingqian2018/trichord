import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image_cosmos3 import generate_image_cosmos3  # noqa: E402
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_visual_critic import poster_visual_critic  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402

K3SLURM_MODEL_NAME = "kimi-k3@k3slurm"
COSMOS3_EXTRA_ARGS = {"guardrails": False}


class generate_image_cosmos3_noguard(generate_image_cosmos3):
    """Same tool (schema name generate_image); every Cosmos3 request carries extra_args guardrails=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        build_request = self.gateway.build_request

        def build_request_noguard(*a, **kw) -> dict[str, Any]:
            request = build_request(*a, **kw)
            extra_body = request.setdefault("extra_body", {})
            extra_body["extra_args"] = {**extra_body.get("extra_args", {}), **COSMOS3_EXTRA_ARGS}
            return request

        self.gateway.build_request = build_request_noguard

    def setting(self) -> dict:
        return {**super().setting(), "extra_args": COSMOS3_EXTRA_ARGS}


class poster_visual_critic_k3slurm(poster_visual_critic):
    """Same tool (schema name poster_visual_critic), but the critic itself runs on Slurm-hosted Kimi-K3."""

    def __init__(self, model_name: str = K3SLURM_MODEL_NAME, **kwargs):
        super().__init__(model_name, **kwargs)


class poster_generation_v1p2(poster_generation):
    tool_registry_classes = [generate_image_cosmos3_noguard, render_poster, poster_visual_critic_k3slurm]
