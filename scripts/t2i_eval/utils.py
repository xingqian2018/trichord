# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

CREDENTIAL_FILE = Path(__file__).resolve().parents[2] / "credentials" / "gateway.json"


def load_credential_secret(key: str) -> str:
    try:
        with open(CREDENTIAL_FILE, "r") as f:
            return json.load(f).get(key, "").strip()
    except FileNotFoundError:
        return ""


GATEWAY_CONFIG = {
    "nvidia": {
        "url": "https://inference-api.nvidia.com/v1",
        "api": load_credential_secret("NVIDIA_GATEWAY"),
        "modelstr": [
            "aws/anthropic/bedrock-claude-opus-5-5",
            "openai/openai/gpt-6-astra",
            "nvidia/moonshotai/kimi-k3",
        ],
    },
}

MODEL_CHOICE = {
    "opus-5.5": "aws/anthropic/bedrock-claude-opus-5-5",
    "gpt-6-astra": "openai/openai/gpt-6-astra",
    "kimi-k3": "nvidia/moonshotai/kimi-k3",
}


def resolve_model_string(model_name: str) -> str:
    return MODEL_CHOICE.get(model_name, model_name)


class UnifiedGatewayLLM:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency: int = 4, num_max_retry: int = 1, timeout: int = 100):
        from openai import AsyncOpenAI

        self.key_to_gateway: dict[str, AsyncOpenAI] = {}
        self.modelstr_to_key: dict[str, str] = {}
        for keyname, cfg in gateway_configs.items():
            if not cfg.get("api"):
                logger.warning(f"Skipping gateway '{keyname}': missing API key in {CREDENTIAL_FILE}.")
                continue
            self.key_to_gateway[keyname] = AsyncOpenAI(api_key=cfg["api"], base_url=cfg["url"])
            for modelstr in cfg["modelstr"]:
                self.modelstr_to_key.setdefault(modelstr, keyname)
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def build_request(
        self,
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> dict[str, Any]:
        return {
            "model": resolve_model_string(model_name),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

    async def query_core(self, request: dict) -> str:
        model = request["model"]
        client = self.key_to_gateway[self.modelstr_to_key[model]]
        for attempt in range(self.num_max_retry):
            try:

                async def _stream() -> str:
                    stream = await client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result

                return await asyncio.wait_for(_stream(), timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{model} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s")
            except Exception as e:
                logger.warning(f"{model} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}")
        return ""

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=5)

        async def gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(one(r) for r in request_list))

        result = asyncio.run_coroutine_threadsafe(gather(), self._loop).result()
        pbar.close()
        return result

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
