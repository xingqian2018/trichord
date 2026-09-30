from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayVLM, image_bytes_to_data_url, resolve_model_string  # noqa: E402

FIRST_NAMES = ["Ada", "Iris", "Leo", "Mira", "Noah", "Uma", "Theo", "Zara", "Kai", "Nina", "Ravi", "Sage", "Elio", "Vera", "Omar", "Lina", "Xingqian"]
LAST_NAMES = ["Chen", "Okafor", "Silva", "Novak", "Haddad", "Ivers", "Moreau", "Tanaka", "Quinn", "Bauer", "Rossi", "Mehta", "Larsen", "Diaz", "Kowal", "Nakamura", "Xu"]


def random_agent_name(exclude: set[str] = set(), max_attempts: int = len(FIRST_NAMES) * len(LAST_NAMES)) -> str:
    for _ in range(max_attempts):
        name = f"{random.choice(FIRST_NAMES)}, {random.choice(LAST_NAMES)}"
        if name not in exclude:
            return name
    raise RuntimeError(f"Could not draw a unique agent name after {max_attempts} attempts ({len(exclude)} names taken)")


def convert_agent_schema_to_tool_schema(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for schema in agents:
        function = {k: v for k, v in schema["agent"].items() if k not in ("type", "name")}
        converted.append({"type": "function", "function": {"name": schema["agent"]["type"], **function}})
    return converted


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


def to_image_bytes(image: Any) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class UnknownToolError(ValueError):
    def __init__(self, message: str, extended_message: str):
        super().__init__(message)
        self.extended_message = extended_message


class Agent(Tool):
    schema = {
        "type": "agent",
        "agent": {
            "type": "agent",
            "name": "<placeholder>",
            "description": "A helpful agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The brief: subject, audience, tone, required text, and the target aspect ratio (1:1, 4:3, 3:4, 16:9, 9:16)",
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
        tool_registry: Optional[dict[str, type[Tool]]] = None,
        agent_registry: Optional[dict[str, type[Agent]]] = None,
        allow_different_agent_with_same_functionality: bool = False,
        agent_list: Optional[list[str]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        gateway_configs: dict[str, dict] = GATEWAY_CONFIG,
        num_max_retry: int = 2,
        timeout: int = 600,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.schema = {**type(self).schema, "agent": {**type(self).schema["agent"], "type": type(self).__name__.lower(), "name": agent_name}}

        tool_registry = {k.lower(): v for k, v in (tool_registry or {}).items()}
        agent_registry = {k.lower(): v for k, v in (agent_registry or {}).items()}
        assert not (set(tool_registry) & set(agent_registry)), "disallow same names for tool and agent"

        # Sort of tools #
        self.tool_registry = tool_registry
        self.tool_instances: dict[str, Tool] = {}
        self.tools = [cls.schema for _, cls in tool_registry.items()]
        self.tool_call_counter = 0

        # Sort of sub agents #
        if not allow_different_agent_with_same_functionality:
            assert agent_list is None, "agent_list shouldn't be set unless allow_different_agent_with_same_functionality=True"
            agent_list = list(agent_registry)
        agent_list = [name.lower() for name in (agent_list or [])]
        assert set(agent_list) <= set(agent_registry), f"agent_list refers to unregistered agent types: {agent_list}"

        self.agent_instances: dict[str, Agent] = {}
        self.agents = []
        taken = {self.agent_name}
        for type_name in agent_list:
            sub_name = random_agent_name(exclude=taken)
            taken.add(sub_name)
            instance = agent_registry[type_name](sub_name)
            function_name = type_name if agent_list.count(type_name) == 1 else f"{type_name}__{sub_name.replace(', ', '_')}"
            instance.schema = {**instance.schema, "agent": {**instance.schema["agent"], "type": function_name, "name": sub_name}}
            self.agents.append(instance.schema)
            self.agent_instances[sub_name] = instance
        names = [t["function"]["name"] for t in self.tools] + [a["agent"]["type"] for a in self.agents]
        assert len(set(names)) == len(names), f"duplicate tool/agent declarations: {names}"

        # Other model request things #
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.gateway = UnifiedGatewayVLM(gateway_configs, num_concurrency=1, num_max_retry=num_max_retry, timeout=timeout)

        # Message history #
        self.messages: list[dict[str, Any]] = []
        self.messages_extended: list[dict[str, Any]] = []

    def build_request(self) -> dict[str, Any]:
        request = {
            "model": resolve_model_string(self.model_name),
            "messages": [{"role": "system", "content": self.system_prompt}, *self.messages],
            "max_tokens": int(self.max_tokens),
            "stream": True,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if self.tools or self.agents:
            request["tools"] = [*self.tools, *convert_agent_schema_to_tool_schema(self.agents)]
        return request

    def next_tool_call_system_id(self, tool_name: str) -> str:
        self.tool_call_counter += 1
        return f"{tool_name}_tool_call_{self.tool_call_counter}"

    def append(self, msg: dict[str, Any], extended: Optional[dict[str, Any]] = None) -> None:
        self.messages.append(msg)
        self.messages_extended.append(extended if extended is not None else dict(msg))

    def add_user_message(self, text: str, images: Optional[list[Any]] = None) -> None:
        if images:
            content: Any = [{"type": "text", "text": text}]
            for image in images:
                data = to_image_bytes(image)
                content.append({"type": "image_url", "image_url": {"url": image_bytes_to_data_url(data, sniff_image_fmt(data))}})
        else:
            content = text
        self.append({"role": "user", "content": content})

    def sloppy_find_agent_instance(self, function_name: str) -> Optional[Agent]:
        exact = [self.agent_instances[a["agent"]["name"]] for a in self.agents if a["agent"]["type"] == function_name]
        if len(exact) == 1:
            return exact[0]
        return None

    def tag_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call["function"]["name"]
        system_id = self.next_tool_call_system_id(name)
        agent = self.sloppy_find_agent_instance(name)
        if agent is not None:
            return {"id": system_id, "type": "agent", "agent_name": agent.agent_name, "function": call["function"]}
        return {**call, "id": system_id}

    def step(self) -> None:
        result = self.gateway.query([self.build_request()])[0]
        tool_calls = json.loads(result["tool_calls"]) if result["tool_calls"] else None

        msg: dict[str, Any] = {"role": "assistant", "content": result["content"] or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in tool_calls
            ]
        if result["reasoning"]:
            msg["reasoning_content"] = result["reasoning"]

        extended: dict[str, Any] = {
            "role": "assistant",
            "reasoning": result["reasoning"],
            "content": msg["content"],
            "tool_calls": [self.tag_tool_call(c) for c in msg["tool_calls"]] if tool_calls else None,
        }
        self.append(msg, extended)

    def pending_tool_calls(self) -> list[dict[str, str]]:
        answered: set[str] = set()
        pos = len(self.messages) - 1
        while pos >= 0 and self.messages[pos]["role"] == "tool":
            answered.add(self.messages[pos]["tool_call_id"])
            pos -= 1
        if pos < 0 or self.messages[pos]["role"] != "assistant":
            return []
        wire_calls = self.messages[pos].get("tool_calls") or []
        extended_calls = self.messages_extended[pos].get("tool_calls") or []
        return [
            {"id": w["id"], "system_id": x["id"], "name": w["function"]["name"], "arguments": w["function"]["arguments"]}
            for w, x in zip(wire_calls, extended_calls)
            if w["id"] not in answered
        ]

    def add_tool_result(self, call_id: str, system_id: str, text: str, extended_text: Optional[str] = None) -> None:
        tool = self.tool_instances.get(system_id)
        if tool is not None:
            tool.result_text = text
        extended: dict[str, Any] = {"role": "tool", "tool_call_id": system_id, "content": extended_text or text}
        if isinstance(tool, Agent):
            extended = {"role": "tool", "type": "agent", "agent_name": tool.agent_name, "tool_call_id": system_id, "content": extended_text or text}
        self.append({"role": "tool", "tool_call_id": call_id, "content": text}, extended)

    def new_tool_instance(self, name: str) -> Tool:
        if name in self.tool_registry:
            return self.tool_registry[name]()
        agent = self.sloppy_find_agent_instance(name)
        if agent is not None:
            return agent
        agent_functions = sorted(a["agent"]["type"] for a in self.agents)
        raise UnknownToolError(
            f"Unknown tool '{name}'. Available tools: {sorted([*self.tool_registry, *agent_functions])}",
            f"Unknown tool or agent '{name}'. Available tools: {sorted(self.tool_registry)}. Available agents: {agent_functions}",
        )

    def run(self, args: dict[str, Any]) -> str:
        try:
            for call in self.pending_tool_calls():
                self.add_tool_result(call["id"], call["system_id"], f"Tool call '{call['name']}' was interrupted before a result was recorded.")
            if args.get("prompt") is not None:
                self.add_user_message(args["prompt"], images=args.get("images") or None)
            while True:
                self.step()
                pending = self.pending_tool_calls()
                if not pending:
                    self.result_text = self.messages[-1].get("content") or ""
                    return self.result_text
                for call in pending:
                    extended_text = None
                    try:
                        tool = self.new_tool_instance(call["name"])
                        self.tool_instances[call["system_id"]] = tool
                        text = tool.run(json.loads(call["arguments"]) if call["arguments"] else {})
                    except UnknownToolError as e:
                        text = f"Tool call '{call['name']}' failed: {e}"
                        extended_text = f"Tool call '{call['name']}' failed: {e.extended_message}"
                    except Exception as e:
                        text = f"Tool call '{call['name']}' failed: {type(e).__name__}: {e}"
                    self.add_tool_result(call["id"], call["system_id"], text, extended_text)
        except Exception as e:
            for call in self.pending_tool_calls():
                self.add_tool_result(call["id"], call["system_id"], f"Run aborted: {type(e).__name__}: {e}")
            self.result_text = f"Run failed: {type(e).__name__}: {e}"
            self.append({"role": "assistant", "content": self.result_text})
            return self.result_text

    def call_folder(self, system_id: str) -> str:
        handler = self.tool_instances.get(system_id)
        if isinstance(handler, Agent):
            return f"agent/{handler.agent_name}"
        return f"tool/{system_id}"

    def header_records(self, extended: bool) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self.tools or self.agents:
            schemas = [*self.tools, *self.agents] if extended else [*self.tools, *convert_agent_schema_to_tool_schema(self.agents)]
            records.append({"role": "system", "type": "tool-declare", "content": json.dumps(schemas, ensure_ascii=False)})
        if self.reasoning_effort:
            records.append({"role": "system", "type": "reasoning-effort", "content": self.reasoning_effort})
        records.append({"role": "system", "content": self.system_prompt})
        return records

    def save_history(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        chat = {
            "agent_name": self.agent_name,
            "messages": [*self.header_records(extended=False), *self.messages],
            "messages_extended": [*self.header_records(extended=True), *self.messages_extended],
        }
        (root / "chat.json").write_text(json.dumps(chat, ensure_ascii=False, indent=2))

        saved: set[str] = set()
        for msg in self.messages_extended:
            if msg["role"] != "tool":
                continue
            system_id = msg["tool_call_id"]
            folder = self.call_folder(system_id)
            if folder in saved:
                continue
            saved.add(folder)
            out_dir = root / folder
            if system_id in self.tool_instances:
                self.tool_instances[system_id].save_history(out_dir)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "result.txt").write_text(msg["content"])

        if self.result_text is not None:
            (root / "result.txt").write_text(self.result_text)
