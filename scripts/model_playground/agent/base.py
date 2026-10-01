from __future__ import annotations

import base64
import copy
import io
import json
import os.path as osp
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, ToolCallError  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayVLM, default_reasoning_effort, hash_bytes, image_conversion, put, resolve_model_string  # noqa: E402

FIRST_NAMES = ["Ada", "Iris", "Leo", "Mira", "Noah", "Uma", "Theo", "Zara", "Kai", "Nina", "Ravi", "Sage", "Elio", "Vera", "Omar", "Lina", "Xingqian"]
LAST_NAMES = ["Chen", "Okafor", "Silva", "Novak", "Haddad", "Ivers", "Moreau", "Tanaka", "Quinn", "Bauer", "Rossi", "Mehta", "Larsen", "Diaz", "Kowal", "Nakamura", "Xu"]


def random_agent_name(exclude: set[str] = set(), max_attempts: int = len(FIRST_NAMES) * len(LAST_NAMES)) -> str:
    for _ in range(max_attempts):
        name = f"{random.choice(FIRST_NAMES)}_{random.choice(LAST_NAMES)}"
        if name not in exclude:
            return name
    raise RuntimeError(f"Could not draw a unique agent name after {max_attempts} attempts ({len(exclude)} names taken)")


def convert_agent_schema_to_tool_schema(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for schema in agents:
        function = {k: v for k, v in schema["agent"].items() if k not in ("name", "duty")}
        converted.append({"type": "function", "function": {"name": schema["agent"]["duty"], **function}})
    return converted


def externalize_media(messages: list[dict[str, Any]], root: str, saved: set[str]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(messages)
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            url = block.get("image_url", {}).get("url") if isinstance(block, dict) else None
            if not url or not url.startswith("data:"):
                continue
            header, payload = url.split(",", 1)
            fmt = header[len("data:"):].split(";")[0].split("/")[-1]
            data = base64.b64decode(payload)
            filename = f"{hash_bytes(data)}.{fmt}"
            if filename not in saved:
                put(data, osp.join(root, "media", filename))
                saved.add(filename)
            block["image_url"]["url"] = f"media/{filename}"
    return messages


class UnknownToolError(ToolCallError):
    def __init__(self, message: str, message_extended: str):
        super().__init__(message)
        self.message = message
        self.message_extended = message_extended


def get_agent_id(agent_duty: str, agent_name: str):
    return f"{agent_duty}_agent_({agent_name})"


class Agent(Tool):
    schema = {
        "type": "agent",
        "agent": {
            "name": "<placeholder>",
            "duty": "agent",
            "description": "A helpful agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "input chat message",
                    },
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

    def __init__(
        self,
        agent_name: str,
        model_name: str,
        system_prompt: str,
        tool_registry: Optional[list[type[Tool]]] = None,
        agent_registry: Optional[list[type[Agent]]] = None,
        allow_different_agent_with_same_functionality: bool = False,
        agent_list: Optional[list[str]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        gateway_configs: dict[str, dict] = GATEWAY_CONFIG,
        num_max_retry: int = 2,
        timeout: int = 600,
        max_worker_thread_for_agent_and_tool: int = 8,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.agent_id = get_agent_id(self.schema["agent"]["duty"], agent_name)
        self.lock = threading.Lock()
        self.max_worker_thread_for_agent_and_tool = max_worker_thread_for_agent_and_tool
        self.schema = {**type(self).schema, "agent": {**type(self).schema["agent"], "type": type(self).__name__.lower(), "name": agent_name}}

        tool_registry = {cls.schema["function"]["name"].lower(): cls for cls in (tool_registry or [])}
        agent_registry = {cls.schema["agent"]["duty"].lower(): cls for cls in (agent_registry or [])}
        assert not (set(tool_registry) & set(agent_registry)), "disallow same names for tool and agent"

        # Sort of tools #
        self.tool_registry = tool_registry
        self.tool_instances: dict[str, Tool] = {}
        self.tools = [cls.schema for _, cls in tool_registry.items()]

        # Sort of sub agents #
        if not allow_different_agent_with_same_functionality:
            assert agent_list is None, "agent_list shouldn't be set unless allow_different_agent_with_same_functionality=True"
            agent_list = list(agent_registry)
        agent_list = [name.lower() for name in (agent_list or [])]
        assert set(agent_list) <= set(agent_registry), f"agent_list refers to unregistered agent types: {agent_list}"

        self.agent_instances: dict[str, Agent] = {}
        self.agents = []
        taken = {self.agent_name}
        for duty in agent_list:
            sub_name = random_agent_name(exclude=taken)
            taken.add(sub_name)
            instance = agent_registry[duty](sub_name)
            instance_schema = copy.deepcopy(instance.schema)
            instance_schema["agent"].update({"name": sub_name, "duty": duty})
            self.agents.append(instance_schema)
            self.agent_instances[get_agent_id(duty, sub_name)] = instance

        names = [t["function"]["name"] for t in self.tools] + [a["agent"]["duty"] for a in self.agents]
        assert len(set(names)) == len(names), f"duplicate tool/agent declarations: {names}"

        # Other model request things #
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else default_reasoning_effort(model_name)
        self.max_tokens = max_tokens
        self.gateway = UnifiedGatewayVLM(gateway_configs, num_concurrency=1, num_max_retry=num_max_retry, timeout=timeout)

        # Message history #
        self.messages: list[dict[str, Any]] = []
        self.messages_extended: list[dict[str, Any]] = []
        self.call_counter = 0

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
        if self.tools or self.agents:
            request["tools"] = [*self.tools, *convert_agent_schema_to_tool_schema(self.agents)]
        return request

    def next_tool_call_system_id(self, tool_name: str) -> str:
        call_id = f"{tool_name}_tool_call_{self.call_counter}"
        self.call_counter += 1
        return call_id

    def next_agent_call_system_id(self, agent_duty: str, agent_name: str) -> str:
        call_id = f"{agent_duty}_agent_call_{self.call_counter}({agent_name})"
        self.call_counter += 1
        return call_id

    def append(self, msg: dict[str, Any], msg_ext: dict[str, Any]) -> None:
        self.messages.append(msg)
        self.messages_extended.append(msg_ext)

    def add_user_message(self, text: str, image_urls: Optional[list[str]] = None) -> None:
        if image_urls:
            content: Any = [{"type": "text", "text": text}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": image_conversion(url, dst_fmt="data_url")}})
        else:
            content = text
        msg = {"role": "user", "content": content}
        self.append(msg, copy.deepcopy(msg))

    def sloppy_find_agent_name(self, role: str) -> Optional[str]:
        exact = [a["agent"]["name"] for a in self.agents if a["agent"]["duty"] == role]
        if len(exact) >= 1:
            return exact[0]
        return None

    def modify_tool_query_msg_to_extended_format(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ext_tool_calls = []
        for call in tool_calls:
            function_name = call["function"]["name"]
            agent_name = self.sloppy_find_agent_name(function_name)
            if agent_name is not None:
                function = copy.deepcopy(call["function"])
                agent_duty = function.pop("name")
                agent_call_id = self.next_agent_call_system_id(agent_duty, agent_name)
                agent = {
                    "name": agent_name,
                    "duty": agent_duty,
                    **function
                }
                agent_call = {"id": agent_call_id, "type": "agent", "agent": agent}
                ext_tool_calls.append(agent_call)
            else:
                call = copy.deepcopy(call)
                call["id"] = self.next_tool_call_system_id(function_name)
                ext_tool_calls.append(call)
        return ext_tool_calls

    def step(self) -> str:
        result = self.gateway.query([self.build_request()])[0]
        if result is None:
            raise RuntimeError(f"{self.model_name} query failed after retries")

        finish_reason = result.pop("finish_reason")
        msg: dict[str, Any] = {"role": "assistant", **result}

        msg_ext = copy.deepcopy(msg)
        if "tool_calls" in msg_ext:
            msg_ext["tool_calls"] = self.modify_tool_query_msg_to_extended_format(msg["tool_calls"])
        self.append(msg, msg_ext)
        return finish_reason

    def latest_tool_call_exts(self) -> list[dict[str, Any]]:
        if not self.messages_extended or self.messages_extended[-1]["role"] != "assistant":
            return []
        return self.messages_extended[-1].get("tool_calls") or []

    def latest_tool_call_ids(self) -> list[str]:
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return []
        return [call["id"] for call in self.messages[-1].get("tool_calls") or []]


    def get_tool_or_agent_instance(self, call: dict[str, Any]) -> Tool:
        agent_dutys = list(set(a["agent"]["duty"] for a in self.agents))
        if call["type"] == "function":
            tool_name = call["function"]["name"]
            if tool_name in self.tool_registry:
                self.tool_instances[call["id"]] = self.tool_registry[tool_name]()
                return self.tool_instances[call["id"]]
            else:
                raise UnknownToolError(
                    f"Unknown tool. Available tools: {[*self.tool_registry, *agent_dutys]}",
                    f"Unknown tool. Available tools: {[*self.tool_registry]}",
                )
        elif call["type"] == "agent":
            agent_id = get_agent_id(call["agent"]["duty"], call["agent"]["name"])
            if agent_id not in self.agent_instances:
                raise UnknownToolError(
                    f"Unknown tool. Available tools: {[*self.tool_registry, *agent_dutys]}",
                    f"Unknown agent. Available agents: {[{'duty': a['agent']['duty'], 'name': a['agent']['name']} for a in self.agents]}",
                )
            else:
                return self.agent_instances[agent_id]
        else:
            raise UnknownToolError(
                f"Unknown tool type! Should be function!",
                f"Unknown tool or agent type! Should be either function or agent!",
            )

    def execute_one_tool_or_agent_call(self, call_id: str, call_ext: dict[str, Any]) -> Callable[[str], None]:
        try:
            instance = self.get_tool_or_agent_instance(call_ext)
            raw_arguments = call_ext["agent"]["arguments"] if call_ext["type"] == "agent" else call_ext["function"]["arguments"]
            args = json.loads(raw_arguments) if raw_arguments else {}
            if call_ext["type"] == "agent":
                with instance.lock:
                    text = instance.run(args)
            else:
                text = instance.run(args)
            text_ext = text
        except UnknownToolError as e:
            text, text_ext = e.message, e.message_extended

        msg = {"role": "tool", "tool_call_id": call_id, "content": text}
        if call_ext["type"] == "agent":
            msg_ext = {"role": "agent", "agent_call_id": call_ext["id"], "content": text_ext}
        else:
            msg_ext = {"role": "tool", "tool_call_id": call_ext["id"], "content": text_ext}
        return partial(self.append, msg, msg_ext)

    def execute_tool_or_agent_calls(self) -> None:
        call_ids = self.latest_tool_call_ids()
        call_exts = self.latest_tool_call_exts()
        if len(call_ids) != len(call_exts):
            assert False, "fatal error, inconsistent message history"
        if not call_ids:
            return
        with ThreadPoolExecutor(max_workers=min(len(call_exts), self.max_worker_thread_for_agent_and_tool)) as pool:
            appends_callers = list(pool.map(self.execute_one_tool_or_agent_call, call_ids, call_exts))
        for append_caller in appends_callers:
            append_caller()

    def run_core(self, args: dict[str, Any]) -> str:
        if args.get("prompt") is not None:
            self.add_user_message(args["prompt"], image_urls=args.get("image_urls") or None)
        while True:
            finish_reason = self.step()
            if finish_reason == "stop":
                return self.messages[-1]["content"]
            if finish_reason == "tool_calls":
                self.execute_tool_or_agent_calls()
            else:
                raise RuntimeError(f"{self.model_name} stopped with finish_reason={finish_reason!r}")

    def save_chat_header(self, isext: bool) -> list[dict[str, Any]]:
        chat_message_header: list[dict[str, Any]] = []
        if self.tools or self.agents:
            schemas = [*self.tools, *self.agents] if isext else [*self.tools, *convert_agent_schema_to_tool_schema(self.agents)]
            chat_message_header.append({"role": "system", "type": "tool-declare", "content": json.dumps(schemas, ensure_ascii=False)})
        if self.reasoning_effort is not None:
            chat_message_header.append({"role": "system", "type": "reasoning-effort", "content": self.reasoning_effort})
        chat_message_header.append({"role": "system", "content": self.system_prompt})
        return chat_message_header

    def save_history_core(self, path: str) -> None:
        root = osp.join(path, self.agent_id.replace("(", "").replace(")", ""))
        setting = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_duty": self.schema["agent"]["duty"],
            "model_name": self.model_name,
            "model_string": resolve_model_string(self.model_name),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "tools": list(self.tool_registry),
            "agents": [a["agent"]["duty"] for a in self.agents],
        }
        put(json.dumps(setting, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "setting.json"))

        saved_media: set[str] = set()
        chat = {
            "messages": [*self.save_chat_header(isext=False), *externalize_media(self.messages, root, saved_media)],
            "messages_extended": [*self.save_chat_header(isext=True), *externalize_media(self.messages_extended, root, saved_media)],
        }
        put(json.dumps(chat, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "chat.json"))

        for msg in self.messages_extended:
            if msg["role"] != "tool":
                continue
            tool_call_id = msg["tool_call_id"]
            out_dir = osp.join(root, "tool", tool_call_id)
            if tool_call_id in self.tool_instances:
                self.tool_instances[tool_call_id].save_history(out_dir)
            else:
                put(msg["content"].encode("utf-8"), osp.join(out_dir, "result.txt"))

        for instance in self.agent_instances.values():
            if instance.messages:
                instance.save_history(osp.join(root, "agent"))

        if self.result_text is not None:
            put(self.result_text.encode("utf-8"), osp.join(root, "result.txt"))
