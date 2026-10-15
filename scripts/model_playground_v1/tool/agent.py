from __future__ import annotations

import json
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.tool import Tool, externalize_media  # noqa: E402
from utils import put  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayVLM, default_reasoning_effort, image_conversion, resolve_model_string  # noqa: E402


class Agent(Tool):
    schema = {
        "type": "function",
        "function": {
            "name": "agent",
            "description": "A helpful agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "input chat message"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional images to look at: absolute paths, file://, http(s), or data: URLs",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tool_registry: Optional[list[type[Tool]]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        gateway_configs: dict[str, dict] = GATEWAY_CONFIG,
        num_max_retry: int = 2,
        timeout: int = 600,
        max_worker_thread_for_tool: int = 8,
    ):
        super().__init__()
        registry = {cls.schema["function"]["name"]: cls for cls in tool_registry or []}
        assert len(registry) == len(tool_registry or []), "duplicate tool function names in tool_registry"
        self.tool_instances: dict[str, Tool] = {function_name: cls() for function_name, cls in registry.items()}
        self.tools = [cls.schema for cls in registry.values()]

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else default_reasoning_effort(model_name)
        self.max_worker_thread_for_tool = max_worker_thread_for_tool
        self.gateway = UnifiedGatewayVLM(gateway_configs, num_concurrency=1, num_max_retry=num_max_retry, timeout=timeout)

    def build_request(self) -> dict[str, Any]:
        request = {
            "model": resolve_model_string(self.model_name),
            "messages": [{"role": "system", "content": self.system_prompt}, *self.messages],
            "max_tokens": int(self.max_tokens),
            "stream": True,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.tools:
            request["tools"] = self.tools
        return request

    def add_user_message(self, text: str, image_urls: Optional[list[str]] = None) -> None:
        if image_urls:
            content: Any = [{"type": "text", "text": text}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": image_conversion(url, dst_fmt="data_url")}})
        else:
            content = text
        self.append({"role": "user", "content": content})

    def step(self) -> str:
        result = self.gateway.query([self.build_request()])[0]
        if result is None:
            raise RuntimeError(f"{self.model_name} query failed after retries")
        finish_reason = result.pop("finish_reason")
        self.append({"role": "assistant", **result})
        return finish_reason

    def latest_tool_calls(self) -> list[dict[str, Any]]:
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return []
        return self.messages[-1].get("tool_calls") or []

    def execute_one_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        function_name = call["function"]["name"]
        instance = self.tool_instances.get(function_name)
        if instance is None:
            text = f"Unknown tool. Available tools: {list(self.tool_instances)}"
        else:
            raw_arguments = call["function"]["arguments"]
            text = instance.run(json.loads(raw_arguments) if raw_arguments else {}, tool_call_id=call["id"])
        return {"role": "tool", "tool_call_id": call["id"], "content": text}

    def execute_tool_calls(self) -> None:
        calls = self.latest_tool_calls()
        if not calls:
            return
        with ThreadPoolExecutor(max_workers=min(len(calls), self.max_worker_thread_for_tool)) as pool:
            results = list(pool.map(self.execute_one_tool_call, calls))
        for msg in results:
            self.append(msg)

    def run_core(self, args: dict[str, Any]) -> str:
        if args.get("prompt") is not None:
            self.add_user_message(args["prompt"], image_urls=args.get("image_urls") or None)
        while True:
            finish_reason = self.step()
            if finish_reason == "stop":
                return self.messages[-1].get("content") or ""
            if finish_reason == "tool_calls":
                self.execute_tool_calls()
            else:
                raise RuntimeError(f"{self.model_name} stopped with finish_reason={finish_reason!r}")

    def setting(self) -> dict[str, Any]:
        return {
            **super().setting(),
            "model_name": self.model_name,
            "model_string": resolve_model_string(self.model_name),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "tools": list(self.tool_instances),
        }

    def save_history_core(self, root: str) -> None:
        put(json.dumps(self.setting(), ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "setting.json"))
        request = self.build_request()
        chat = {k: v for k, v in request.items() if k not in ("model", "max_tokens", "stream", "messages")}
        chat["messages"] = externalize_media(request["messages"], root)
        put(json.dumps(chat, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "chat.json"))
        self.save_calls(root)
        for function_name, instance in self.tool_instances.items():
            if instance.calls:
                instance.save_history(osp.join(root, "tool", function_name))
