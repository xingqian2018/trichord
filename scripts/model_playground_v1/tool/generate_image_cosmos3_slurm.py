import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from tool.generate_image_cosmos3 import generate_image_cosmos3  # noqa: E402
from utils import UnifiedGatewayImageGenerator, load_credential_secret  # noqa: E402

COSMOS3_MODEL = "Slurm/Cosmos3-Super-Text2Image"
POOL_KEY = "c3slurm"
DEFAULT_MODEL_NAME = f"{COSMOS3_MODEL}@{POOL_KEY}"
API_KEY_NAME = "SLURM_COSMOS3_API_KEY"
WAIT_SECONDS = 30

# Self-hosted replicas only (scripts/serving/serve_cosmos3_super_t2i.sbatch); Lepton is never called.
# Endpoints are found on the batch host, which has squeue:
#   python3 k3_endpoints.py --out $C3_ENDPOINTS_FILE --user xingqianx --job-prefix c3-xqx \
#       --model Slurm/Cosmos3-Super-Text2Image --api-key-name SLURM_COSMOS3_API_KEY
# and read here from C3_ENDPOINTS (comma list) or C3_ENDPOINTS_FILE (json list).
# The bearer key is SLURM_COSMOS3_API_KEY in credentials/gateway.json.
BAD_UNTIL: dict[str, float] = {}


def load_c3_endpoints() -> list[str]:
    if os.environ.get("C3_ENDPOINTS"):
        return [u.strip() for u in os.environ["C3_ENDPOINTS"].split(",") if u.strip()]
    path = os.environ.get("C3_ENDPOINTS_FILE")
    if path and Path(path).is_file():
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            return []
    return []


class UnifiedGatewayCosmos3T2I(UnifiedGatewayImageGenerator):
    label = "Cosmos3T2I"

    def __init__(self, num_concurrency=1, num_max_retry=2, timeout=1800, cooldown_seconds: float = 120, refresh_seconds: float = 300):
        super().__init__({}, num_concurrency=num_concurrency, num_max_retry=num_max_retry, timeout=timeout)
        from openai import AsyncOpenAI
        self.AsyncOpenAI = AsyncOpenAI
        self.api_key = load_credential_secret(API_KEY_NAME)
        if not self.api_key:
            raise RuntimeError(f"{API_KEY_NAME} missing from credentials/gateway.json")
        self.cooldown_seconds = cooldown_seconds
        self.refresh_seconds = refresh_seconds
        self.clients: dict[str, Any] = {}
        self.refreshed_at = 0.0

    def refresh(self, force: bool = False) -> None:
        if not force and self.clients and time.time() - self.refreshed_at < self.refresh_seconds:
            return
        self.refreshed_at = time.time()
        urls = load_c3_endpoints()
        self.clients = {u: self.clients.get(u) or self.AsyncOpenAI(base_url=u, api_key=self.api_key, max_retries=0) for u in urls}

    def resolve_client(self, request: dict) -> tuple[Any, dict]:
        return POOL_KEY, {**request, "model": COSMOS3_MODEL}

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        import openai
        self.refresh()
        while True:
            now = time.time()
            names = [u for u in self.clients if BAD_UNTIL.get(u, 0) <= now]
            random.shuffle(names)
            for name in names:
                try:
                    return await super().query_core(self.clients[name], request)
                except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                    BAD_UNTIL[name] = time.time() + self.cooldown_seconds
                    logger.warning(f"{self.label} endpoint {name} failed, cooling down: {type(e).__name__}: {e}")
            logger.warning(f"{self.label}: no endpoint available ({len(self.clients)} known), waiting {WAIT_SECONDS}s")
            await asyncio.sleep(WAIT_SECONDS)
            self.refresh(force=True)


class generate_image_cosmos3_slurm(generate_image_cosmos3):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, scratch_root: Optional[str] = None):
        super().__init__(model_name=model_name, scratch_root=scratch_root)
        build_request = self.gateway.build_request
        self.gateway.close()
        self.gateway = UnifiedGatewayCosmos3T2I(timeout=self.gateway_timeout)
        self.gateway.build_request = build_request
