import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image_cosmos3 import COSMOS3_ASPECT_RATIO_TO_SIZE, generate_image_cosmos3  # noqa: E402
from tool.poster_generation import poster_generation  # noqa: E402
from tool.poster_generation_v1p2 import K3SLURM_MODEL_NAME, poster_visual_critic_k3slurm  # noqa: E402
from tool.poster_visual_critic import SYSTEM_PROMPT as CRITIC_SYSTEM_PROMPT  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402
from tool.tool import ToolCallError  # noqa: E402

CRITIC_TIMEOUT = 600
CRITIC_NUM_MAX_RETRY = 2


def require_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    missing = [name for name in schema["function"]["parameters"].get("required", []) if args.get(name) in (None, "")]
    if missing:
        raise ToolCallError(f"missing required argument(s): {missing}; call {schema['function']['name']} again with all of them")


def require_local_image(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https", "data"):
        return
    path = Path(parsed.path if parsed.scheme == "file" else url)
    if not path.is_file():
        raise ToolCallError(f"image_url {url} does not exist; pass the exact path returned by render_poster or generate_image")


def fallback_setting(agent: Agent) -> dict[str, Any]:
    return {
        "fallback_model_name": getattr(agent.gateway, "fallback_model_name", None),
        "fallback_requests": getattr(agent.gateway, "fallback_requests", 0),
    }


class generate_image_cosmos3_v1p3(generate_image_cosmos3):
    def run_core(self, args: dict) -> str:
        require_args(self.schema, args)
        return super().run_core(args)


class render_poster_v1p3(render_poster):
    aspect_ratio_to_size = COSMOS3_ASPECT_RATIO_TO_SIZE

    def run_core(self, args: dict) -> str:
        require_args(self.schema, args)
        return super().run_core(args)


class poster_visual_critic_v1p3(poster_visual_critic_k3slurm):
    def __init__(self, model_name: str = K3SLURM_MODEL_NAME, **kwargs: Any):
        Agent.__init__(
            self,
            model_name,
            CRITIC_SYSTEM_PROMPT,
            temperature=0.0,
            num_max_retry=CRITIC_NUM_MAX_RETRY,
            timeout=CRITIC_TIMEOUT,
            **kwargs,
        )

    def run_core(self, args: dict) -> str:
        require_args(self.schema, args)
        require_local_image(args["image_url"])
        return super().run_core(args)

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), **fallback_setting(self)}


class poster_generation_v1p3(poster_generation):
    tool_registry_classes = [generate_image_cosmos3_v1p3, render_poster_v1p3, poster_visual_critic_v1p3]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        for instance in self.tool_instances.values():
            if isinstance(instance, Agent):
                instance.max_tokens = self.max_tokens

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), **fallback_setting(self)}
