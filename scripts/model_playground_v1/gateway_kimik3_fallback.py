import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from gateway_kimik3 import K3_GATEWAY_KEY, RoutingGatewayVLM  # noqa: E402
from utils import UnifiedGatewayVLM, resolve_model_string  # noqa: E402

FALLBACK_MODEL_NAME = "kimi-k3@nvidiak"
CONTEXT_OVERFLOW_MARKER = "maximum context length"


def is_context_overflow(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 400 and CONTEXT_OVERFLOW_MARKER in str(error)


class FallbackRoutingGatewayVLM(RoutingGatewayVLM):
    fallback_model_name = FALLBACK_MODEL_NAME

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.use_fallback = False
        self.fallback_requests = 0

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        if client != K3_GATEWAY_KEY:
            return await super().query_core(client, request)
        if not self.use_fallback:
            try:
                return await self.k3().query_core(None, request)
            except Exception as e:
                if not is_context_overflow(e):
                    raise
                self.use_fallback = True
                logger.warning(f"{self.label} context over the k3slurm limit, switching this agent to {self.fallback_model_name}")
        self.fallback_requests += 1
        fallback_client, fallback_request = UnifiedGatewayVLM.resolve_client(self, {**request, "model": resolve_model_string(self.fallback_model_name)})
        return await UnifiedGatewayVLM.query_core(self, fallback_client, fallback_request)


def install() -> None:
    import tool.agent
    tool.agent.UnifiedGatewayVLM = FallbackRoutingGatewayVLM
