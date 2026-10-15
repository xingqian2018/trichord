"""Synchronous OpenAI clients over self-hosted Slurm replicas (scripts/serving/serve.sh).

The batch host refreshes a json list of base URLs (k3_endpoints.py --every 300) into <PREFIX>_ENDPOINTS_FILE;
<PREFIX>_ENDPOINTS (comma list) overrides it. The bearer key is an entry of credentials/gateway.json.
A failing replica cools down; with no replica available the call waits and re-reads the list until its deadline.
"""

import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from utils import load_credential_secret  # noqa: E402

WAIT_SECONDS = 30
BAD_UNTIL: dict[str, float] = {}
BAD_LOCK = threading.Lock()


class SlurmPool:
    def __init__(self, env_prefix: str, api_key_name: str, timeout: float = 900, cooldown_seconds: float = 120, refresh_seconds: float = 300):
        from openai import OpenAI
        self.OpenAI = OpenAI
        self.env_prefix = env_prefix
        self.api_key = load_credential_secret(api_key_name)
        if not self.api_key:
            raise RuntimeError(f"{api_key_name} missing from credentials/gateway.json")
        self.timeout = timeout
        self.cooldown_seconds = cooldown_seconds
        self.refresh_seconds = refresh_seconds
        self.clients: dict[str, Any] = {}
        self.refreshed_at = 0.0

    def load_urls(self) -> list[str]:
        if os.environ.get(f"{self.env_prefix}_ENDPOINTS"):
            return [u.strip() for u in os.environ[f"{self.env_prefix}_ENDPOINTS"].split(",") if u.strip()]
        path = os.environ.get(f"{self.env_prefix}_ENDPOINTS_FILE")
        if path and Path(path).is_file():
            try:
                return json.loads(Path(path).read_text())
            except json.JSONDecodeError:
                return []
        return []

    def refresh(self, force: bool = False) -> None:
        if not force and self.clients and time.time() - self.refreshed_at < self.refresh_seconds:
            return
        self.refreshed_at = time.time()
        self.clients = {u: self.clients.get(u) or self.OpenAI(base_url=u, api_key=self.api_key, timeout=self.timeout, max_retries=0) for u in self.load_urls()}

    def call(self, fn: Callable[[Any], Any], deadline_seconds: float = 1800) -> Any:
        import openai
        deadline = time.time() + deadline_seconds
        self.refresh()
        while True:
            now = time.time()
            with BAD_LOCK:
                names = [u for u in self.clients if BAD_UNTIL.get(u, 0) <= now]
            random.shuffle(names)
            for name in names:
                try:
                    return fn(self.clients[name])
                except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                    with BAD_LOCK:
                        BAD_UNTIL[name] = time.time() + self.cooldown_seconds
                    logger.warning(f"{self.env_prefix} endpoint {name} failed, cooling down: {type(e).__name__}: {e}")
            if time.time() + WAIT_SECONDS > deadline:
                raise RuntimeError(f"{self.env_prefix}: no endpoint answered within {deadline_seconds}s ({len(self.clients)} known)")
            logger.warning(f"{self.env_prefix}: no endpoint available ({len(self.clients)} known), waiting {WAIT_SECONDS}s")
            time.sleep(WAIT_SECONDS)
            self.refresh(force=True)
