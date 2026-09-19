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
import base64
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
            "azure/openai/gpt-5.1",
            "openai/openai/gpt-5.4",
            "openai/openai/gpt-5.5",
            "openai/openai/gpt-6-astra",
            "gcp/google/gemini-3.1-pro-preview",
            "gcp/google/gemini-3-pro",
            "gcp/google/gemini-3-flash-preview",
            "gcp/google/gemini-3.8-flash",
            "gcp/google/gemini-2.5-pro",
            "us/gcp/google/gemini-2.5-flash",
            "nvidia/qwen/qwen-235b",
            "nvidia/qwen/qwen3-5-397b-a17b",
            "aws/anthropic/bedrock-claude-opus-4-7",
        ],
    },
    "nvidiak": {
        "url": "https://inference-api.nvidia.com/v1",
        "api": load_credential_secret("NVIDIA_GATEWAY_K"),
        "modelstr": [
            "openai/openai/gpt-6-astra",
            "gcp/google/gemini-3.1-pro-preview",
            "gcp/google/gemini-3.8-flash",
            "aws/anthropic/bedrock-claude-opus-4-7",
        ],
    },
    "nvidiams": {
        "url": "https://inference-api.nvidia.com/v1",
        "api": load_credential_secret("NVIDIA_GATEWAY_MS"),
        "modelstr": [
            "gcp/google/gemini-3.1-pro-preview",
            "aws/anthropic/bedrock-claude-opus-4-7",
        ],
    },
    "hpsv3": {
        "url": "https://b5k2m9x7-scorer-hpsv3.xenon.lepton.run/v1",
        "api": load_credential_secret("LEPTON_REWARD"),
        "modelstr": ["hpsv3"],
    },
    "pickscore": {
        "url": "https://b5k2m9x7-scorer-pickscore.xenon.lepton.run/v1",
        "api": load_credential_secret("LEPTON_REWARD"),
        "modelstr": ["pickscore"],
    },
    "paddleocrv5": {
        "url": "https://b5k2m9x7-scorer-paddleocrv5.xenon.lepton.run/v1",
        "api": load_credential_secret("LEPTON_REWARD"),
        "modelstr": ["paddleocrv5", "paddleocrv5_strict", "paddleocrv5_pned_gned"],
    },
}


MODEL_CHOICE = {
    "gpt-5.1":                   "azure/openai/gpt-5.1",
    "gpt-5.4":                   "openai/openai/gpt-5.4",
    "gpt-5.5":                   "openai/openai/gpt-5.5",
    "gpt-6@nvidia":              "openai/openai/gpt-6-astra",
    "gpt-6@nvidiak":             "openai/openai/gpt-6-astra",
    "gemini-2.5-flash":          "us/gcp/google/gemini-2.5-flash",
    "gemini-2.5-pro":            "gcp/google/gemini-2.5-pro",
    "gemini-3-flash":            "gcp/google/gemini-3-flash-preview",
    "gemini-3.8-flash@nvidia":   "gcp/google/gemini-3.8-flash",
    "gemini-3.8-flash@nvidiak":  "gcp/google/gemini-3.8-flash",
    "gemini-3.1-pro@nvidia":     "gcp/google/gemini-3.1-pro-preview",
    "gemini-3.1-pro@nvidiak":    "gcp/google/gemini-3.1-pro-preview",
    "qwen-235b":                 "nvidia/qwen/qwen-235b",
    "opus-4.7@nvidia":           "aws/anthropic/bedrock-claude-opus-4-7",
    "opus-4.7@nvidiak":          "aws/anthropic/bedrock-claude-opus-4-7",
    "opus-4.7@nvidiams":         "aws/anthropic/bedrock-claude-opus-4-7",
    "hpsv3":                     "hpsv3",
    "pickscore":                 "pickscore",
    "paddleocrv5":               "paddleocrv5",
    "paddleocrv5_strict":        "paddleocrv5_strict",
    "paddleocrv5_pned_gned":     "paddleocrv5_pned_gned",
}


def image_bytes_to_data_url(image_bytes: bytes, image_fmt: str = "webp") -> str:
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime_type = mime_types.get(image_fmt.lower(), f"image/{image_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def video_bytes_to_data_url(video_bytes: bytes, video_fmt: str = "mp4") -> str:
    mime_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "avi": "video/avi",
    }
    mime_type = mime_types.get(video_fmt.lower(), f"video/{video_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(video_bytes).decode('utf-8')}"


def resolve_model_string(model_name: str) -> str:
    if model_name.count("@") > 1:
        raise ValueError(f"model_name '{model_name}' contains more than one '@'")
    modelstr = MODEL_CHOICE.get(model_name, model_name)
    if modelstr.count("@") > 1:
        raise ValueError(f"resolved modelstr '{modelstr}' contains more than one '@'")
    if "@" in model_name and "@" not in modelstr:
        gateway = model_name.split("@", 1)[1]
        return f"{modelstr}@{gateway}"
    return modelstr


class UnifiedGatewayVLM:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=100):
        from openai import AsyncOpenAI
        self.key_to_gateway: dict[str, AsyncOpenAI] = dict()
        self.modelstr_to_key: dict[str, str] = dict()
        self.modelstr_ambiguous: set[str] = set()
        for keyname, cfg in gateway_configs.items():
            if not cfg.get("api"):
                logger.warning(f"Skipping gateway '{keyname}': missing API key.")
                continue
            self.key_to_gateway[keyname] = AsyncOpenAI(api_key=cfg["api"], base_url=cfg["url"])
            for modelstr in cfg["modelstr"]:
                if modelstr in self.modelstr_ambiguous:
                    pass
                elif modelstr in self.modelstr_to_key:
                    del self.modelstr_to_key[modelstr]
                    self.modelstr_ambiguous.add(modelstr)
                else:
                    self.modelstr_to_key[modelstr] = keyname
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request: dict) -> str:
        model_field = request["model"]
        if "@" in model_field:
            actual_model, gateway_key = model_field.rsplit("@", 1)
            client = self.key_to_gateway[gateway_key]
        else:
            actual_model = model_field
            client = self.key_to_gateway[self.modelstr_to_key[model_field]]
        request = {**request, "model": actual_model}
        for attempt in range(self.num_max_retry):
            try:
                async def _stream():
                    stream = await client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result
                return await asyncio.wait_for(_stream(), timeout=self.timeout)
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"VLM API for {actual_model} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s")
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"VLM API for {actual_model} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}")
        logger.warning("VLM query failed after max retries")
        return ""

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def build_request(
        self,
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        image_bytes: Optional[bytes] = None,
        image_fmt: str = "webp",
        video_bytes: Optional[bytes] = None,
        video_fmt: str = "mp4",
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ) -> dict[str, Any]:
        model_str = resolve_model_string(model_name)
        request = {
            "model": model_str,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": content_prompt}]},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        extra = []
        if image_bytes is not None:
            extra.append({"type": "image_url", "image_url": {"url": image_bytes_to_data_url(image_bytes, image_fmt)}})
        if video_bytes is not None:
            extra.append({"type": "image_url", "image_url": {"url": video_bytes_to_data_url(video_bytes, video_fmt)}})
        if extra:
            request["messages"][-1]["content"].extend(extra)
        if output_fmt in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()


class UnifiedGatewayLLM:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=100):
        from openai import AsyncOpenAI
        self.key_to_gateway: dict[str, AsyncOpenAI] = dict()
        self.modelstr_to_key: dict[str, str] = dict()
        self.modelstr_ambiguous: set[str] = set()
        for keyname, cfg in gateway_configs.items():
            if not cfg.get("api"):
                logger.warning(f"Skipping gateway '{keyname}': missing API key.")
                continue
            self.key_to_gateway[keyname] = AsyncOpenAI(api_key=cfg["api"], base_url=cfg["url"])
            for modelstr in cfg["modelstr"]:
                if modelstr in self.modelstr_ambiguous:
                    pass
                elif modelstr in self.modelstr_to_key:
                    del self.modelstr_to_key[modelstr]
                    self.modelstr_ambiguous.add(modelstr)
                else:
                    self.modelstr_to_key[modelstr] = keyname
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request: dict) -> str:
        model_field = request["model"]
        if "@" in model_field:
            actual_model, gateway_key = model_field.rsplit("@", 1)
            client = self.key_to_gateway[gateway_key]
        else:
            actual_model = model_field
            client = self.key_to_gateway[self.modelstr_to_key[model_field]]
        request = {**request, "model": actual_model}
        for attempt in range(self.num_max_retry):
            try:
                async def _stream():
                    stream = await client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result
                return await asyncio.wait_for(_stream(), timeout=self.timeout)
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"LLM API for {actual_model} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s")
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"LLM API for {actual_model} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}")
        logger.warning("LLM query failed after max retries")
        return ""

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def build_request(
        self,
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        model_str = resolve_model_string(model_name)
        request = {
            "model": model_str,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if output_fmt in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()
