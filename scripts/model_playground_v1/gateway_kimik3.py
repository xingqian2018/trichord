"""Gateway for Kimi-K3 served by plain vLLM on Slurm (no --enable-auto-tool-choice / --tool-call-parser),
e.g. the k3-scale2048 endpoints on oci-jhb.

Why a dedicated gateway (verified 2026-10-09 against vLLM 0.30 K3 serving):
  * top-level `tools` + tool_choice auto -> HTTP 400, the server has no tool parser
  * tool_choice="none" -> the K3 chat template injects "MUST NOT call any tools"
  * tools passed as chat_template_kwargs={"tools": ...} render the normal tool-declare block
  * the server's kimi_k3 reasoning parser drops the tools section from `content`, so the raw output is
    recovered with return_token_ids + /detokenize and parsed here into OpenAI-style tool_calls

Endpoints come from k3_endpoints.load_endpoints() (K3_ENDPOINTS / K3_ENDPOINTS_FILE / squeue).

Use everywhere a model name is accepted:
    import gateway_kimik3; gateway_kimik3.install()      # before any Agent is built
    ... --model kimi-k3@k3slurm
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from k3_endpoints import K3_MODEL, http_json, load_endpoints  # noqa: E402
from utils import UnifiedGatewayVLM  # noqa: E402

K3_GATEWAY_KEY = "k3slurm"

# K3 output channels (the prompt already ends with "<|open|>think<|sep|>"):
#   <think><|close|>think<|sep|><|open|>response<|sep|><text><|close|>response<|sep|>
#   <|open|>tools<|sep|><|open|>call tool="NAME" index="1"<|sep|>
#     <|open|>argument key="K" type="string"<|sep|>V<|close|>argument<|sep|>...<|close|>call<|sep|>
#   <|close|>tools<|sep|><|close|>message<|sep|>
THINK_END = "<|close|>think<|sep|>"
RESPONSE_RE = re.compile(r"<\|open\|>response<\|sep\|>(.*?)<\|close\|>response<\|sep\|>", re.DOTALL)
CALL_RE = re.compile(r'<\|open\|>call tool="(?P<name>[^"]+)"[^<]*?<\|sep\|>(?P<body>.*?)<\|close\|>call<\|sep\|>', re.DOTALL)
ARG_RE = re.compile(r'<\|open\|>argument key="(?P<key>[^"]+)" type="(?P<type>[^"]+)"<\|sep\|>(?P<val>.*?)<\|close\|>argument<\|sep\|>', re.DOTALL)
SPECIAL_RE = re.compile(r"<\|[^|<>]+\|>")

_ENDPOINT_CACHE: dict[str, Any] = {"urls": [], "at": 0.0}
_ENDPOINT_LOCK = threading.Lock()
_BAD_UNTIL: dict[str, float] = {}  # shared by all gateway instances in the process


def cached_endpoints(max_age: float) -> list[str]:
    with _ENDPOINT_LOCK:
        if not _ENDPOINT_CACHE["urls"] or time.time() - _ENDPOINT_CACHE["at"] >= max_age:
            urls = load_endpoints()
            if urls or not _ENDPOINT_CACHE["urls"]:
                _ENDPOINT_CACHE["urls"] = urls
            _ENDPOINT_CACHE["at"] = time.time()
        return list(_ENDPOINT_CACHE["urls"])


def pre_format(request: dict[str, Any]) -> dict[str, Any]:
    request = {**request, "model": K3_MODEL, "stream": True}
    extra = dict(request.pop("extra_body", None) or {})
    tools = request.pop("tools", None)
    request.pop("tool_choice", None)
    if tools:
        extra["chat_template_kwargs"] = {**extra.get("chat_template_kwargs", {}), "tools": tools}
        extra.update(return_token_ids=True, skip_special_tokens=False, spaces_between_special_tokens=False)
    if extra:
        request["extra_body"] = extra
    return request


def _arg_value(typ: str, val: str) -> Any:
    if typ == "string":
        return val
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val


def parse_k3_output(raw: str) -> dict[str, Any]:
    think, _, rest = raw.partition(THINK_END) if THINK_END in raw else ("", "", raw)
    m = RESPONSE_RE.search(rest)
    content = m.group(1).strip() if m else None
    tool_calls = []
    for c in CALL_RE.finditer(rest):
        args = {a.group("key"): _arg_value(a.group("type"), a.group("val")) for a in ARG_RE.finditer(c.group("body"))}
        tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                           "function": {"name": c.group("name"), "arguments": json.dumps(args, ensure_ascii=False)}})
    if content is None and not tool_calls:
        content = SPECIAL_RE.sub("", rest).strip()
    return {"reasoning": SPECIAL_RE.sub("", think).strip() or None, "content": content or None, "tool_calls": tool_calls}


class EndpointError(Exception):
    """Transport-level failure: try another endpoint."""


class UnifiedGatewayKimiK3(UnifiedGatewayVLM):
    label = "KimiK3"

    def __init__(self, endpoints: Optional[list[str]] = None, num_concurrency=4, num_max_retry=2, timeout=600,
                 cooldown_seconds: float = 120, refresh_seconds: float = 300, own_loop: bool = True):
        if own_loop:
            super().__init__({}, num_concurrency=num_concurrency, num_max_retry=num_max_retry, timeout=timeout)
        else:  # delegate: only query_core is used, on the caller's event loop
            self.num_concurrency, self.num_max_retry, self.timeout = num_concurrency, num_max_retry, timeout
        from openai import AsyncOpenAI
        self._AsyncOpenAI = AsyncOpenAI
        self._fixed_endpoints = endpoints
        self.cooldown_seconds = cooldown_seconds
        self.refresh_seconds = refresh_seconds
        self._clients: dict[str, Any] = {}
        self._refreshed_at = 0.0
        self._refresh()

    def _refresh(self) -> None:
        if self._clients and time.time() - self._refreshed_at < self.refresh_seconds:
            return
        self._refreshed_at = time.time()
        urls = self._fixed_endpoints or cached_endpoints(self.refresh_seconds)
        if not urls:
            raise RuntimeError("no Kimi-K3 endpoint found (set K3_ENDPOINTS / K3_ENDPOINTS_FILE, see k3_endpoints.py)")
        if set(urls) != set(self._clients):
            urls = list(urls)
            random.shuffle(urls)  # spread processes over endpoints
            self._clients = {u: self._clients.get(u) or self._AsyncOpenAI(base_url=u, api_key="EMPTY", max_retries=0) for u in urls}
            self._cycle = itertools.cycle(list(self._clients))

    def _pick(self, tried: set[str]) -> Optional[str]:
        self._refresh()
        now = time.time()
        for _ in range(len(self._clients)):
            url = next(self._cycle)
            if url not in tried and _BAD_UNTIL.get(url, 0) <= now:
                return url
        return None

    def resolve_client(self, request: dict) -> tuple[Any, dict]:
        return None, {**request, "model": K3_MODEL}

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        tried: set[str] = set()
        while (url := self._pick(tried)) is not None:
            tried.add(url)
            try:
                return await self._query_endpoint(url, request)
            except EndpointError as e:
                _BAD_UNTIL[url] = time.time() + self.cooldown_seconds
                logger.warning(f"{self.label} endpoint {url} failed, cooling down: {e}")
        raise RuntimeError(f"{self.label}: all endpoints unavailable")

    async def _query_endpoint(self, url: str, request: dict) -> dict[str, Any]:
        import openai
        client = self._clients[url]
        req = pre_format(request)
        client_tools = "chat_template_kwargs" in req.get("extra_body", {})
        content, reasoning, reasoning_key, finish_reason = None, None, None, None
        token_ids: list[int] = []
        try:
            stream = await client.chat.completions.create(**req)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    content = (content or "") + delta.content
                for key in ("reasoning_content", "reasoning"):
                    trace = getattr(delta, key, None)
                    if trace:
                        reasoning_key = reasoning_key or key
                        reasoning = (reasoning or "") + trace
                        break
                ids = getattr(choice, "token_ids", None) or (choice.model_extra or {}).get("token_ids")
                if ids:
                    token_ids.extend(ids)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            raise EndpointError(f"{type(e).__name__}: {e}") from e
        if finish_reason is None:
            raise EndpointError("stream ended without a finish_reason (possible dropped connection)")

        tool_calls: list[dict[str, Any]] = []
        if client_tools:
            root = url.rstrip("/").removesuffix("/v1")
            try:
                raw = (await asyncio.to_thread(http_json, f"{root}/detokenize", {"model": K3_MODEL, "tokens": token_ids}, 60))["prompt"]
            except Exception as e:
                raise EndpointError(f"detokenize failed: {e}") from e
            parsed = parse_k3_output(raw)
            reasoning = parsed["reasoning"] or reasoning
            content = parsed["content"]
            tool_calls = parsed["tool_calls"]
            if tool_calls and finish_reason == "stop":
                finish_reason = "tool_calls"

        response: dict[str, Any] = {}
        if reasoning is not None:
            response[reasoning_key or "reasoning_content"] = reasoning
        if content is not None:
            response["content"] = content
        if tool_calls:
            response["tool_calls"] = tool_calls
        response["finish_reason"] = finish_reason
        return response


class RoutingGatewayVLM(UnifiedGatewayVLM):
    """UnifiedGatewayVLM that sends `*@k3slurm` models to UnifiedGatewayKimiK3 and everything else to the base."""

    _k3: Optional[UnifiedGatewayKimiK3] = None

    def k3(self) -> UnifiedGatewayKimiK3:
        # one delegate per instance: AsyncOpenAI clients must stay on this gateway's event loop
        if self._k3 is None:
            self._k3 = UnifiedGatewayKimiK3(timeout=self.timeout, own_loop=False)
        return self._k3

    def resolve_client(self, request: dict) -> tuple[Any, dict]:
        if request["model"].endswith(f"@{K3_GATEWAY_KEY}"):
            return K3_GATEWAY_KEY, {**request, "model": K3_MODEL}
        return super().resolve_client(request)

    async def query_core(self, client: Any, request: dict) -> Optional[dict[str, Any]]:
        if client == K3_GATEWAY_KEY:
            return await self.k3().query_core(None, request)
        return await super().query_core(client, request)


def install() -> None:
    """Route `<anything>@k3slurm` through UnifiedGatewayKimiK3 for every Agent built afterwards."""
    import tool.agent
    tool.agent.UnifiedGatewayVLM = RoutingGatewayVLM
