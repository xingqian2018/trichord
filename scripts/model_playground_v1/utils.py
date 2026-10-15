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
import hashlib
import io
import json
import random
import threading
from pathlib import Path
from typing import Any, Optional

from urllib.parse import urlparse

import fsspec
from loguru import logger
from PIL import Image
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
            "azure/openai/gpt-6-astra",
            "gcp/google/gemini-3.1-pro-preview",
            "gcp/google/gemini-3-pro",
            "gcp/google/gemini-3-flash-preview",
            "gcp/google/gemini-3.8-flash",
            "gcp/google/gemini-2.5-pro",
            "us/gcp/google/gemini-2.5-flash",
            "gcp/google/gemini-3.1-flash-image",
            "openai/openai/gpt-image-2",
            "azure/openai/gpt-image-2",
            "nvidia/qwen/qwen-235b",
            "nvidia/qwen/qwen3-5-397b-a17b",
            "nvidia/moonshotai/kimi-k3",
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
            "nvidia/moonshotai/kimi-k3",
            "gcp/google/gemini-3.1-flash-image",
            "openai/openai/gpt-image-2",
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
    "lepton_cosmos3_t2i": {
        "url": "https://b5k2m9x7-qwen3-super-t2i-for-datagen.xenon.lepton.run/v1",
        "api": load_credential_secret("LEPTON_COSMOS3_SUPER_T2I"),
        "modelstr": ["LeptonTR/Cosmos3-Super-Text2Image"],
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
    "gpt-6-astra@nvidia":        "openai/openai/gpt-6-astra",
    "gpt-6-astra-azure@nvidia":  "azure/openai/gpt-6-astra",
    "gpt-6-astra@nvidiak":       "openai/openai/gpt-6-astra",
    "gemini-2.5-flash":          "us/gcp/google/gemini-2.5-flash",
    "gemini-2.5-pro":            "gcp/google/gemini-2.5-pro",
    "gemini-3-flash":            "gcp/google/gemini-3-flash-preview",
    "gemini-3.8-flash@nvidia":   "gcp/google/gemini-3.8-flash",
    "gemini-3.8-flash@nvidiak":  "gcp/google/gemini-3.8-flash",
    "gemini-3.1-pro@nvidia":     "gcp/google/gemini-3.1-pro-preview",
    "gemini-3.1-pro@nvidiak":    "gcp/google/gemini-3.1-pro-preview",
    "qwen-235b":                 "nvidia/qwen/qwen-235b",
    "kimi-k3@nvidia":            "nvidia/moonshotai/kimi-k3",
    "kimi-k3@nvidiak":           "nvidia/moonshotai/kimi-k3",
    "nano-banana-2.0@nvidia":    "gcp/google/gemini-3.1-flash-image",
    "nano-banana-2.0@nvidiak":   "gcp/google/gemini-3.1-flash-image",
    "gpt-image-2.0@nvidia":      "openai/openai/gpt-image-2",
    "gpt-image-2.0@nvidiak":     "openai/openai/gpt-image-2",
    "gpt-image-2.0-azure":       "azure/openai/gpt-image-2",
    "opus-4.7@nvidia":           "aws/anthropic/bedrock-claude-opus-4-7",
    "opus-4.7@nvidiak":          "aws/anthropic/bedrock-claude-opus-4-7",
    "opus-4.7@nvidiams":         "aws/anthropic/bedrock-claude-opus-4-7",
    "hpsv3":                     "hpsv3",
    "pickscore":                 "pickscore",
    "paddleocrv5":               "paddleocrv5",
    "paddleocrv5_strict":        "paddleocrv5_strict",
    "paddleocrv5_pned_gned":     "paddleocrv5_pned_gned",
}


def video_bytes_to_data_url(video_bytes: bytes, video_fmt: str = "mp4") -> str:
    mime_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "avi": "video/avi",
    }
    mime_type = mime_types.get(video_fmt.lower(), f"video/{video_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(video_bytes).decode('utf-8')}"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


S3_CREDENTIAL_FILE = Path(__file__).resolve().parents[2] / "credentials" / "gcs.secret"


def put(data: bytes, path: str | Path) -> str:
    url = str(path)
    scheme = urlparse(url).scheme
    if scheme in ("", "file"):
        target = Path(url[len("file://"):] if scheme == "file" else url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return url
    assert scheme == "s3", f"unsupported destination {url!r}; use a local path or s3://bucket/key"
    import boto3
    from botocore.config import Config

    cred = json.loads(S3_CREDENTIAL_FILE.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=cred["endpoint_url"],
        region_name=cred["region_name"],
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        config=Config(s3={"addressing_style": "path"}, request_checksum_calculation="when_required"),
    )
    parsed = urlparse(url)
    client.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=data)
    return url


def sniff_image_fmt(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


def read_url_bytes(url: str) -> bytes:
    scheme = urlparse(url).scheme
    if scheme == "data":
        return base64.b64decode(url.split(",", 1)[1])
    if scheme in ("", "file", "http", "https", "s3", "gs"):
        with fsspec.open(url, "rb") as f:
            return f.read()
    raise ValueError(f"unsupported url scheme {scheme!r} in {url}")


def image_conversion(value: Any, src_fmt: str = "", dst_fmt: str = "bytes") -> Any:
    if not src_fmt:
        if isinstance(value, (bytes, bytearray)):
            src_fmt = "bytes"
        elif isinstance(value, Image.Image):
            src_fmt = "pil"
        else:
            src_fmt = "data_url" if urlparse(str(value)).scheme == "data" else "url"
    if src_fmt == "data_url" and dst_fmt == "data_url":
        return value

    if src_fmt == "bytes":
        data = bytes(value)
    elif src_fmt == "pil":
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        data = buf.getvalue()
    elif src_fmt in ("data_url", "url"):
        data = read_url_bytes(str(value))
    else:
        raise ValueError(f"unknown image src_fmt {src_fmt!r}")

    if dst_fmt == "bytes":
        return data
    if dst_fmt == "pil":
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    if dst_fmt == "data_url":
        mime_type = f"image/{sniff_image_fmt(data)}"
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    raise ValueError(f"unknown image dst_fmt {dst_fmt!r}")


REASONING_EFFORT_LEVELS = ["low", "medium", "high", "max"]
REASONING_EFFORT_DEFAULTS = {"kimi": "max", "gpt": "high"}


REASONING_EFFORT_SUPPORTED = {"kimi": ["low", "high", "max"], "gpt": ["low", "medium", "high"]}


def default_reasoning_effort(model_name: str) -> Optional[str]:
    model_string = resolve_model_string(model_name).lower()
    for family, effort in REASONING_EFFORT_DEFAULTS.items():
        if family in model_string:
            return effort
    return None


def random_reasoning_effort(model_name: str, rng: Optional[random.Random] = None) -> Optional[str]:
    model_string = resolve_model_string(model_name).lower()
    for family, levels in REASONING_EFFORT_SUPPORTED.items():
        if family in model_string:
            return (rng or random).choice(levels)
    return None


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


class UnifiedGateway:
    label = "Gateway"

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

    def resolve_client(self, request: dict) -> tuple[Any, dict]:
        model_field = request["model"]
        if "@" in model_field:
            actual_model, gateway_key = model_field.rsplit("@", 1)
            client = self.key_to_gateway[gateway_key]
        else:
            actual_model = model_field
            client = self.key_to_gateway[self.modelstr_to_key[model_field]]
        return client, {**request, "model": actual_model}

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        stream = await client.chat.completions.create(**request)
        content = None
        reasoning = None
        reasoning_key = None
        finish_reason = None
        tool_calls: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content = (content or "") + delta.content
            for key in ("reasoning_content", "reasoning"):
                trace = getattr(delta, key, None)
                if trace:
                    reasoning_key = reasoning_key or key
                    reasoning = (reasoning or "") + trace
                    break
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = tool_calls.setdefault(tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason
        if finish_reason is None:
            raise RuntimeError("stream ended without a finish_reason (possible dropped connection)")
        response: dict[str, Any] = {}
        if reasoning is not None:
            response[reasoning_key] = reasoning
        if content is not None:
            response["content"] = content
        if tool_calls:
            response["tool_calls"] = list(tool_calls.values())
        response["finish_reason"] = finish_reason
        return response

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[Optional[dict[str, Any]]]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[Optional[dict[str, Any]]]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> Optional[dict[str, Any]]:
                async with sem:
                    client, req = self.resolve_client(req)
                    actual_model = req["model"]
                    result = None
                    for attempt in range(self.num_max_retry):
                        try:
                            result = await asyncio.wait_for(self.query_core(client, req), timeout=self.timeout)
                            break
                        except KeyboardInterrupt:
                            raise
                        except asyncio.TimeoutError:
                            if attempt == 0 or attempt == self.num_max_retry - 1:
                                logger.warning(f"{self.label} API for {actual_model} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s")
                        except Exception as e:
                            if attempt == 0 or attempt == self.num_max_retry - 1:
                                logger.warning(f"{self.label} API for {actual_model} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}")
                    else:
                        logger.warning(f"{self.label} query failed after max retries")
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()


class UnifiedGatewayLLM(UnifiedGateway):
    label = "LLM"

    def build_request(
        self,
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        request = {
            "model": resolve_model_string(model_name),
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
        if tools:
            request["tools"] = tools
        return request


class UnifiedGatewayVLM(UnifiedGateway):
    label = "VLM"

    def build_request(
        self,
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        image_bytes: Optional[bytes] = None,
        video_bytes: Optional[bytes] = None,
        video_fmt: str = "mp4",
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        request = {
            "model": resolve_model_string(model_name),
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
            extra.append({"type": "image_url", "image_url": {"url": image_conversion(image_bytes, dst_fmt="data_url")}})
        if video_bytes is not None:
            extra.append({"type": "image_url", "image_url": {"url": video_bytes_to_data_url(video_bytes, video_fmt)}})
        if extra:
            request["messages"][-1]["content"].extend(extra)
        if output_fmt in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        if tools:
            request["tools"] = tools
        return request


class UnifiedGatewayImageGenerator(UnifiedGateway):
    label = "ImageGenerator"

    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=300):
        super().__init__(gateway_configs, num_concurrency=num_concurrency, num_max_retry=num_max_retry, timeout=timeout)

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        response = await client.images.generate(**request)
        images = [base64.b64decode(item.b64_json) for item in response.data if item.b64_json]
        return {"images": images} if images else None

    def build_request(self, model_name: str, prompt: str, size: Optional[str] = None, n: int = 1, output_format: Optional[str] = None) -> dict[str, Any]:
        request = {"model": resolve_model_string(model_name), "prompt": prompt, "n": n}
        if size:
            request["size"] = size
        if output_format:
            request["output_format"] = output_format
        return request
