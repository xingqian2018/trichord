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
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
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
    "kimi-k3":                   "nvidia/moonshotai/kimi-k3",
    "nano-banana-2.0":           "gcp/google/gemini-3.1-flash-image",
    "gpt-image-2.0":             "openai/openai/gpt-image-2",
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


SCRATCH_ROOT = Path("/tmp/model_playground")
ALLOWED_OUTPUT_ROOT = Path("/tmp")


def resolve_output_path(requested: Optional[str], default_dir: Path, suffix: str) -> Path:
    req = Path(requested) if requested else None
    if req is not None and req.is_absolute() and str(req.resolve()).startswith(str(ALLOWED_OUTPUT_ROOT.resolve()) + "/"):
        out = req.resolve()
    else:
        name = req.name if req is not None and req.name else f"{uuid.uuid4().hex}{suffix}"
        out = Path(default_dir) / name
    if out.suffix.lower() != suffix:
        out = out.with_suffix(suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


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

    async def query_core(self, request: dict) -> dict[str, str]:
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
                    content = None
                    reasoning = None
                    finish_reason = None
                    tool_calls: dict[int, dict[str, str]] = {}
                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            content = (content or "") + delta.content
                        trace = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        if trace:
                            reasoning = (reasoning or "") + trace
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                entry = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                                if tc.id:
                                    entry["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        entry["name"] += tc.function.name
                                    if tc.function.arguments:
                                        entry["arguments"] += tc.function.arguments
                        if chunk.choices[0].finish_reason is not None:
                            finish_reason = chunk.choices[0].finish_reason
                    if finish_reason is None:
                        logger.warning(f"VLM API for {actual_model}: stream ended without a finish_reason (possible dropped connection)")
                    tool_calls_str = json.dumps([tool_calls[i] for i in sorted(tool_calls)], ensure_ascii=False) if tool_calls else None
                    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls_str, "finish_reason": finish_reason}
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
        return {"content": None, "reasoning": None, "tool_calls": None, "finish_reason": None}

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[dict[str, str]]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[dict[str, str]]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> dict[str, str]:
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
        tools: Optional[list[dict[str, Any]]] = None,
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
        if tools:
            request["tools"] = tools
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

    async def query_core(self, request: dict) -> dict[str, str]:
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
                    content = None
                    reasoning = None
                    finish_reason = None
                    tool_calls: dict[int, dict[str, str]] = {}
                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            content = (content or "") + delta.content
                        trace = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        if trace:
                            reasoning = (reasoning or "") + trace
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                entry = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                                if tc.id:
                                    entry["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        entry["name"] += tc.function.name
                                    if tc.function.arguments:
                                        entry["arguments"] += tc.function.arguments
                        if chunk.choices[0].finish_reason is not None:
                            finish_reason = chunk.choices[0].finish_reason
                    if finish_reason is None:
                        logger.warning(f"LLM API for {actual_model}: stream ended without a finish_reason (possible dropped connection)")
                    tool_calls_str = json.dumps([tool_calls[i] for i in sorted(tool_calls)], ensure_ascii=False) if tool_calls else None
                    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls_str, "finish_reason": finish_reason}
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
        return {"content": None, "reasoning": None, "tool_calls": None, "finish_reason": None}

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[dict[str, str]]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[dict[str, str]]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> dict[str, str]:
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
        tools: Optional[list[dict[str, Any]]] = None,
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
        if tools:
            request["tools"] = tools
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()


class UnifiedGatewayImageGenerator:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=300):
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

    async def query_core(self, request: dict) -> dict[str, Any]:
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
                async def _generate():
                    response = await client.images.generate(**request)
                    images = [base64.b64decode(item.b64_json) for item in response.data if item.b64_json]
                    return {"images": images or None}
                return await asyncio.wait_for(_generate(), timeout=self.timeout)
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"Image API for {actual_model} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s")
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(f"Image API for {actual_model} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}")
        logger.warning("Image query failed after max retries")
        return {"images": None}

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[dict[str, Any]]:
        mininterval = 0.1 if len(request_list) < 500 else 30
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None, mininterval=mininterval)

        async def coroutine_gather() -> list[dict[str, Any]]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def build_request(self, model_name: str, prompt: str, size: Optional[str] = None, n: int = 1) -> dict[str, Any]:
        request = {"model": resolve_model_string(model_name), "prompt": prompt, "n": n}
        if size:
            request["size"] = size
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()


CSS_DIM_RE = re.compile(r"(width|height):\s*(\d+)px", re.IGNORECASE)
LOCAL_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
FALLBACK_CANVAS_SIZE = (1080, 1350)
RENDER_TARGET_AREA = 2048 * 2048
CHROME_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]


def infer_canvas_size(html_code: str) -> tuple[int, int]:
    best = None
    for rule in html_code.split("}"):
        dims = dict((prop.lower(), int(px)) for prop, px in CSS_DIM_RE.findall(rule))
        if "width" in dims and "height" in dims:
            area = dims["width"] * dims["height"]
            if best is None or area > best[0]:
                best = (area, dims["width"], dims["height"])
    if best is None:
        return FALLBACK_CANVAS_SIZE
    return best[1], best[2]


def inject_reset_css(html_code: str, width: int, height: int) -> str:
    reset = (
        "<style>"
        "html,body{margin:0!important;padding:0!important;"
        f"width:{width}px!important;height:{height}px!important;overflow:hidden!important;}}"
        "</style>"
    )
    match = re.search(r"<head[^>]*>", html_code, re.IGNORECASE)
    if match:
        return html_code[: match.end()] + reset + html_code[match.end() :]
    match = re.search(r"<html[^>]*>", html_code, re.IGNORECASE)
    if match:
        return html_code[: match.end()] + f"<head>{reset}</head>" + html_code[match.end() :]
    return reset + html_code


def inline_local_images(html_code: str) -> str:
    def replace(match):
        src = match.group(2)
        path = Path(src[7:] if src.startswith("file://") else src)
        if not path.is_absolute() or not path.is_file():
            return match.group(0)
        data_url = image_bytes_to_data_url(path.read_bytes(), path.suffix.lstrip(".") or "png")
        return match.group(1) + data_url + match.group(3)

    return LOCAL_IMG_SRC_RE.sub(replace, html_code)


def find_chrome_binary() -> str | None:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def render_html_to_png(html_code: str, png_path: Path, target_area: int = RENDER_TARGET_AREA) -> Path:
    chrome_bin = find_chrome_binary()
    if chrome_bin is None:
        raise RuntimeError(f"No headless Chrome/Chromium binary found (tried {CHROME_CANDIDATES}).")
    width, height = infer_canvas_size(html_code)
    scale = (target_area / (width * height)) ** 0.5
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "poster.html"
        html_path.write_text(inject_reset_css(inline_local_images(html_code), width, height))
        cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={scale}",
            f"--screenshot={png_path}",
            f"file://{html_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not png_path.exists():
            raise RuntimeError(f"Chrome render failed: {result.stderr.decode(errors='replace')}")
    return png_path
