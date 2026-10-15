# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground/main2.py

import base64
import inspect
import json
import os
import os.path as osp
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from agent.base import convert_agent_schema_to_tool_schema, random_agent_name  # noqa: E402
from agent.poster_generation import poster_generation  # noqa: E402
from agent.poster_visual_critic import poster_visual_critic_agent  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE, generate_image  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402
from utils import GATEWAY_CONFIG, REASONING_EFFORT_SUPPORTED, default_reasoning_effort, hash_bytes, image_conversion, put, resolve_model_string  # noqa: E402

TOOL_REGISTRY = [generate_image, render_poster]
AGENT_REGISTRY = [poster_visual_critic_agent]
MCP_SERVER_FLAG = "--poster-mcp-server"
MCP_SERVER_NAME = "poster"
RECORD_DIR_ENV = "POSTER_RECORD_DIR"
CALLS_FILE = "calls.jsonl"
FATAL_FILE = "fatal.json"

DSH_ROOT = Path.home() / "Software" / "dsh"
DSH_BIN = DSH_ROOT / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
DSH_HOME = DSH_ROOT / "home"
NODE_BIN_DIR = Path.home() / "Software" / "node" / "bin"
DSH_PROVIDER = "poster-gateway"
DSH_API_KEY_ENV = "POSTER_GATEWAY_API_KEY"
DSH_DISABLED_PLUGINS = [
    "session-log-deepseek",
    "session-title-llm",
    "mcp-resources",
    "plan-mode",
    "tool-bash",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "tool-skill",
    "tool-goal",
    "tool-todo",
    "tool-web",
    "tool-workflow",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-subagent-control",
    "tool-subagent-list-agents",
]
DSH_MODEL_CHOICES = ["kimi-k3@nvidia", "kimi-k3@nvidiak", "gpt-6-astra@nvidia", "gpt-6-astra@nvidiak"]
DSH_CONTEXT_WINDOW = 262144
DSH_TOOL_CALL_TIMEOUT_MS = 900000
RUN_ROOT = Path("/tmp/model_playground/dsh_runs")
RESULT_ROOT = Path.home() / "agentic_result"
PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_DEFAULT_EFFORT = "model default"
TAXONOMY_NONE = "(none)"


def poster_tool_registry() -> dict[str, Any]:
    return {cls.schema["function"]["name"]: cls for cls in TOOL_REGISTRY}


def poster_agent_registry() -> dict[str, Any]:
    return {cls.schema["agent"]["duty"]: cls for cls in AGENT_REGISTRY}


def poster_function_schemas() -> list[dict[str, Any]]:
    schemas = [cls.schema for cls in TOOL_REGISTRY]
    schemas += convert_agent_schema_to_tool_schema([cls.schema for cls in AGENT_REGISTRY])
    return [schema["function"] for schema in schemas]


def harness_tool_name(raw_name: str) -> str:
    return f"mcp__{MCP_SERVER_NAME}__{raw_name}"


def raw_tool_name(harness_name: str) -> str:
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    return harness_name[len(prefix):] if harness_name.startswith(prefix) else harness_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class PosterToolRecorder:
    def __init__(self, record_dir: str):
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.record_dir / CALLS_FILE
        self.lock = threading.Lock()
        previous = read_jsonl(self.calls_path)
        self.call_counter = len(previous)
        self.taken_agent_names = {record["agent_name"] for record in previous if record.get("agent_name")}
        self.tool_registry = poster_tool_registry()
        self.agent_registry = poster_agent_registry()

    def create_instance(self, name: str) -> tuple[str, Any, Optional[str]]:
        with self.lock:
            counter = self.call_counter
            self.call_counter += 1
            if name in self.tool_registry:
                return f"{name}_tool_call_{counter}", self.tool_registry[name](), None
            sub_name = random_agent_name(exclude=self.taken_agent_names)
            self.taken_agent_names.add(sub_name)
            return f"{name}_agent_call_{counter}({sub_name})", self.agent_registry[name](sub_name), sub_name

    def append_record(self, record: dict[str, Any]) -> None:
        with self.lock:
            with self.calls_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, name: str, args: dict[str, Any]) -> str:
        system_id, instance, agent_name = self.create_instance(name)
        record = {"system_id": system_id, "name": name, "agent_name": agent_name, "arguments": args, "started_at": time.time()}
        try:
            text = instance.run(args)
        except Exception as e:
            record.update({"ended_at": time.time(), "fatal_error": f"{type(e).__name__}: {e}"})
            self.append_record(record)
            put(json.dumps(record, ensure_ascii=False, indent=4).encode("utf-8"), str(self.record_dir / FATAL_FILE))
            raise
        record.update({"ended_at": time.time(), "result_text": text, "image_path_history": list(getattr(instance, "image_path_history", None) or [])})
        if agent_name is not None:
            instance.save_history(str(self.record_dir / "agent"))
            record["agent_id"] = instance.agent_id
        self.append_record(record)
        return text


def run_poster_mcp_server() -> None:
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    recorder = PosterToolRecorder(os.environ[RECORD_DIR_ENV])
    tools = [types.Tool.model_validate({"name": s["name"], "description": s["description"], "inputSchema": s["parameters"]}) for s in poster_function_schemas()]
    known = {tool.name for tool in tools}

    async def run_tool(name: str, arguments: Optional[dict[str, Any]]):
        if name not in known:
            return types.CallToolResult.model_validate({"content": [{"type": "text", "text": f"Unknown tool. Available tools: {sorted(known)}"}], "isError": True})
        text = await anyio.to_thread.run_sync(recorder.run, name, dict(arguments or {}))
        return types.CallToolResult.model_validate({"content": [{"type": "text", "text": text}]})

    if "on_list_tools" in inspect.signature(Server.__init__).parameters:
        async def list_tools(ctx, params):
            return types.ListToolsResult(tools=tools)

        async def call_tool(ctx, params):
            return await run_tool(params.name, params.arguments)

        server = Server(MCP_SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)
    else:
        server = Server(MCP_SERVER_NAME)

        @server.list_tools()
        async def list_tools():
            return tools

        @server.call_tool()
        async def call_tool(name, arguments):
            return await run_tool(name, arguments)

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)


class MediaTable:
    def __init__(self):
        self.urls: dict[str, str] = {}
        self.payloads: list[bytes] = []

    def add(self, data: bytes, saved_path: str) -> str:
        name = f"image_{len(self.urls)}{Path(saved_path).suffix}"
        self.urls[name] = saved_path
        self.payloads.append(data)
        return name

    def key(self) -> Optional[str]:
        return hash_bytes(b"".join(self.payloads)) if self.payloads else None


def parse_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    return json.loads(arguments) if arguments else {}


def conversation_record(conversations: list[list[dict[str, Any]]], media: MediaTable) -> dict[str, Any]:
    return {"conversations": conversations, "media_urls": media.urls, "media_key": media.key()}


def harness_block(block: dict[str, Any], root: str, media: MediaTable) -> dict[str, Any]:
    if block.get("type") == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block.get("type") == "image" and block.get("data"):
        data = base64.b64decode(block["data"])
        fmt = (block.get("mimeType") or "image/png").split("/")[-1]
        saved_path = osp.join(root, "media", f"{hash_bytes(data)}.{fmt}")
        put(data, saved_path)
        return {"type": "image", "image": media.add(data, saved_path)}
    return block


def harness_log_to_conversation(rows: list[dict[str, Any]], root: str, media: MediaTable) -> list[dict[str, Any]]:
    system_texts: list[str] = []
    tools: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for row in rows:
        kind = row.get("type")
        data = row.get("data") or {}
        if kind == "request/header" and not tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}} for t in data["header"].get("tools") or []]
        elif kind == "system/message":
            text = "".join(b.get("text", "") for b in data["message"]["content"] if b.get("type") == "text")
            if not system_texts:
                system_texts.append(text)
            elif text not in system_texts:
                system_texts.append(text)
                messages.append({"role": "system", "content": [{"type": "text", "text": text}]})
        elif kind == "user/message":
            messages.append({"role": "user", "content": [harness_block(b, root, media) for b in data.get("content") or []]})
        elif kind == "assistant/message":
            reasoning, content, tool_calls = [], [], []
            for block in data["message"].get("content") or []:
                if block.get("type") == "reasoning":
                    reasoning.append(block.get("text", ""))
                elif block.get("type") == "tool-call":
                    tool_calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": parse_arguments(block.get("arguments"))}})
                else:
                    content.append(harness_block(block, root, media))
            message: dict[str, Any] = {"role": "assistant"}
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            message["content"] = content
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        elif kind == "tool/result":
            result = data["message"]
            messages.append({"role": "user", "tool_call_id": result["toolCallId"], "content": [harness_block(b, root, media) for b in result.get("content") or []]})
    system = {"role": "system", "content": [{"type": "text", "text": system_texts[0] if system_texts else ""}], "tools": tools}
    return [system, *messages]


def agent_chat_to_conversation(chat: dict[str, Any], local_dir: Path, saved_dir: str, media: MediaTable) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    system_text = ""
    messages: list[dict[str, Any]] = []
    for msg in chat["messages"]:
        if msg["role"] == "system" and msg.get("type") == "tool-declare":
            tools = json.loads(msg["content"])
            continue
        if msg["role"] == "system" and msg.get("type"):
            continue
        if msg["role"] == "system" and not system_text:
            system_text = msg["content"]
            continue
        raw = msg.get("content")
        blocks = [{"type": "text", "text": raw}] if isinstance(raw, str) else []
        for block in raw if isinstance(raw, list) else []:
            if block.get("type") == "image_url":
                url = block["image_url"]["url"]
                blocks.append({"type": "image", "image": media.add((local_dir / url).read_bytes(), osp.join(saved_dir, url))})
            else:
                blocks.append(block)
        message: dict[str, Any] = {"role": "user" if msg["role"] == "tool" else msg["role"]}
        if msg.get("tool_call_id"):
            message["tool_call_id"] = msg["tool_call_id"]
        if msg.get("reasoning_content"):
            message["reasoning_content"] = msg["reasoning_content"]
        message["content"] = blocks
        if msg.get("tool_calls"):
            message["tool_calls"] = [{**call, "function": {**call["function"], "arguments": parse_arguments(call["function"].get("arguments"))}} for call in msg["tool_calls"]]
        messages.append(message)
    return [{"role": "system", "content": [{"type": "text", "text": system_text}], "tools": tools}, *messages]


def supported_efforts(model_string: str) -> list[str]:
    for family, levels in REASONING_EFFORT_SUPPORTED.items():
        if family in model_string:
            return levels
    return []


class DshPosterSession:
    def __init__(self, agent_name: str, model_name: str, instructions: str, reasoning_effort: Optional[str]):
        self.agent_name = agent_name
        self.model_name = model_name
        self.model_string = resolve_model_string(model_name)
        self.model_id, self.gateway_key = self.model_string.rsplit("@", 1)
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else default_reasoning_effort(model_name)
        self.run_dir = RUN_ROOT / agent_name
        self.workspace = self.run_dir / "workspace"
        self.record_dir = self.run_dir / "records"
        self.sessions_dir = self.run_dir / "sessions"
        self.patch_path = self.run_dir / "patch.yml"
        self.dsh_session_id: Optional[str] = None
        self.events: list[dict[str, Any]] = []
        self.error: Optional[str] = None
        self.turn_count = 0
        self.save_history_lock = threading.Lock()
        self.history_saved = False

    def build_patch(self) -> list[dict[str, Any]]:
        efforts = supported_efforts(self.model_string)
        route: dict[str, Any] = {
            "displayName": f"{self.gateway_key} gateway",
            "apiKeyEnv": DSH_API_KEY_ENV,
            "api": "openai-completions",
            "baseURL": GATEWAY_CONFIG[self.gateway_key]["url"],
            "models": [{"id": self.model_id, "name": self.model_name, "contextWindow": DSH_CONTEXT_WINDOW, "reasoningEfforts": {e: e for e in efforts} if efforts else False}],
        }
        if "kimi" in self.model_string:
            route["compat"] = {"thinkingFormat": "deepseek"}
        if self.reasoning_effort is not None:
            route["reasoning"] = self.reasoning_effort
        server_env = {RECORD_DIR_ENV: str(self.record_dir), "HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")}
        mcp_entry = {
            "id": f"mcp-{MCP_SERVER_NAME}",
            "name": "@deepseek-ai/dsh-mcp-client",
            "config": {
                "serverName": MCP_SERVER_NAME,
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(Path(__file__).resolve()), MCP_SERVER_FLAG],
                "env": server_env,
                "toolCallTimeoutMs": DSH_TOOL_CALL_TIMEOUT_MS,
                "failOnStartupError": True,
            },
        }
        return [
            {"id": "agent-default-model", "config": {"provider": DSH_PROVIDER, "model": self.model_id}},
            {"id": "llm-pi-ai", "config": {"providers": {DSH_PROVIDER: route}}},
            {"id": "session-persistence-jsonl", "config": {"root": str(self.sessions_dir), "compression": "none"}},
            {"id": "system-prompt", "config": {"personaPrefix": self.instructions}},
            *[{"id": plugin_id, "disabled": True} for plugin_id in DSH_DISABLED_PLUGINS],
            {"insert": [mcp_entry]},
        ]

    def dsh_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PATH": f"{NODE_BIN_DIR}:{os.environ.get('PATH', '')}",
            "DSH_HOME": str(DSH_HOME),
            "DSH_TELEMETRY_MODE": "DISABLED",
            "DSH_PERMISSION_MODE": "danger-full-access",
            DSH_API_KEY_ENV: GATEWAY_CONFIG[self.gateway_key]["api"],
        }

    def stage_images(self, images: list[bytes]) -> list[str]:
        paths = []
        for data in images:
            path = self.workspace / "inputs" / f"{hash_bytes(data)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_conversion(image_conversion(data, dst_fmt="pil"), dst_fmt="bytes"))
            paths.append(str(path))
        return paths

    def run_turn(self, text: str, images: list[bytes], aspect_ratio: str) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        if self.dsh_session_id is None:
            put(yaml.safe_dump(self.build_patch(), allow_unicode=True, sort_keys=False).encode("utf-8"), str(self.patch_path))
        message = f"{text}\n\nTarget aspect ratio: {aspect_ratio}"
        image_paths = self.stage_images(images)
        if image_paths:
            message += "\n\nReference images:\n" + "\n".join(image_paths)
        self.events.append({"type": "user", "text": message, "turn": self.turn_count})
        self.turn_count += 1

        cmd = ["node", str(DSH_BIN), "--profile", "headless", "--patch", str(self.patch_path), "--json"]
        if self.dsh_session_id is not None:
            cmd += ["--session-id", self.dsh_session_id]
        cmd.append("-")
        stderr_path = self.run_dir / f"stderr_turn{self.turn_count - 1}.log"
        with stderr_path.open("w") as stderr_file:
            process = subprocess.Popen(cmd, cwd=self.workspace, env=self.dsh_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file, text=True)
            process.stdin.write(message)
            process.stdin.close()
            for line in process.stdout:
                if not line.strip():
                    continue
                event = json.loads(line)
                self.events.append(event)
                if event.get("type") == "session":
                    self.dsh_session_id = event["sessionId"]
                if (self.record_dir / FATAL_FILE).is_file() or self.mcp_failed(stderr_path):
                    process.kill()
                    break
            returncode = process.wait()
        if self.mcp_failed(stderr_path):
            raise RuntimeError(f"MCP server '{MCP_SERVER_NAME}' failed to start: {stderr_path.read_text()[-2000:]}")
        fatal = self.record_dir / FATAL_FILE
        if fatal.is_file():
            raise RuntimeError(f"Fatal tool error: {json.loads(fatal.read_text()).get('fatal_error')}")
        if returncode != 0:
            raise RuntimeError(f"dsh exited with {returncode}: {stderr_path.read_text()[-2000:]}")

    def mcp_failed(self, stderr_path: Path) -> bool:
        return stderr_path.is_file() and f"mcp-{MCP_SERVER_NAME} (@deepseek-ai/dsh-mcp-client): Error" in stderr_path.read_text()

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.record_dir / CALLS_FILE)

    def call_id_to_record(self) -> dict[str, dict[str, Any]]:
        pending = list(self.records())
        mapping = {}
        for event in self.events:
            if event.get("type") != "tool_call":
                continue
            name = raw_tool_name(event["tool"])
            for idx, record in enumerate(pending):
                if record["name"] == name and record["arguments"] == event.get("input"):
                    mapping[event["callId"]] = pending.pop(idx)
                    break
        return mapping

    def final_text(self) -> Optional[str]:
        finals = [event["text"] for event in self.events if event.get("type") == "final"]
        return finals[-1] if finals else None

    def save_derived(self, derived: str, records: list[dict[str, Any]]) -> None:
        renders = [r for r in records if r["name"] == render_poster.schema["function"]["name"]]
        critiques = [r for r in records if r["name"] == poster_visual_critic_agent.schema["agent"]["duty"] and r["arguments"].get("image_url")]
        self.cached_html = renders[-1]["arguments"].get("html") if renders else None
        self.cached_image = None
        if critiques:
            try:
                self.cached_image = image_conversion(critiques[-1]["arguments"]["image_url"], dst_fmt="pil")
            except Exception:
                pass
        poster_generation.save_derived(self, derived)

    def save_history(self, path: str) -> str:
        with self.save_history_lock:
            if self.history_saved:
                raise RuntimeError("DshPosterSession.save_history may only be called once per session")
            self.history_saved = True
        root = osp.join(path, self.agent_name)
        records = self.records()
        self.save_derived(osp.join(root, "derived"), records)

        setting = {
            "agent_name": self.agent_name,
            "harness": "deepseek-harness",
            "harness_version": subprocess.run(["node", str(DSH_BIN), "--version"], capture_output=True, text=True, env=self.dsh_env()).stdout.strip(),
            "dsh_session_id": self.dsh_session_id,
            "model_name": self.model_name,
            "model_string": self.model_string,
            "reasoning_effort": self.reasoning_effort,
            "disabled_harness_plugins": DSH_DISABLED_PLUGINS,
            "tools": [harness_tool_name(name) for name in poster_tool_registry()],
            "agents": [harness_tool_name(name) for name in poster_agent_registry()],
        }
        put(json.dumps(setting, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "setting.json"))
        put(self.patch_path.read_bytes(), osp.join(root, "harness", "patch.yml"))

        for log in sorted(self.sessions_dir.rglob("*.jsonl")):
            put(log.read_bytes(), osp.join(root, "harness", "session", str(log.relative_to(self.sessions_dir))))
        mapping = {call_id: record["system_id"] for call_id, record in self.call_id_to_record().items()}
        events = {"dsh_session_id": self.dsh_session_id, "events": self.events, "call_id_to_system_id": mapping}
        put(json.dumps(events, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "harness", "events.json"))

        main_logs = sorted(p for p in self.sessions_dir.rglob("session.v*.jsonl") if self.dsh_session_id and self.dsh_session_id in str(p))
        media = MediaTable()
        conversation = harness_log_to_conversation(read_jsonl(main_logs[-1]), root, media) if main_logs else []
        put(json.dumps(conversation_record([conversation], media), ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "chat.json"))

        tool_registry = poster_tool_registry()
        for record in records:
            if record["name"] not in tool_registry or "result_text" not in record:
                continue
            instance = tool_registry[record["name"]]()
            instance.result_text = record["result_text"]
            instance.image_path_history = record.get("image_path_history") or []
            instance.save_history(osp.join(root, "tool", record["system_id"]))
        agent_dir = self.record_dir / "agent"
        for file in sorted(agent_dir.rglob("*")) if agent_dir.is_dir() else []:
            if not file.is_file():
                continue
            relative = str(file.relative_to(agent_dir))
            if file.name != "chat.json":
                put(file.read_bytes(), osp.join(root, "agent", relative))
                continue
            saved_dir = osp.join(root, "agent", str(file.parent.relative_to(agent_dir)))
            agent_media = MediaTable()
            agent_conversation = agent_chat_to_conversation(json.loads(file.read_text()), file.parent, saved_dir, agent_media)
            put(json.dumps(conversation_record([agent_conversation], agent_media), ensure_ascii=False, indent=4).encode("utf-8"), osp.join(saved_dir, "chat.json"))
            put(file.read_bytes(), osp.join(saved_dir, "chat_original.json"))

        final = self.final_text()
        if final is not None:
            put(final.encode("utf-8"), osp.join(root, "result.txt"))
        return root


if MCP_SERVER_FLAG in sys.argv:
    run_poster_mcp_server()
    sys.exit(0)


import streamlit as st  # noqa: E402

from taxonomy.helper import TOPIC_MESSAGE_TEMPLATE, Taxonomy  # noqa: E402


def list_prompt_presets() -> dict[str, Path]:
    if not PROMPT_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(PROMPT_DIR.glob("*.md")) if p.stem != "poster_visual_critic"}


@st.cache_resource
def get_taxonomy() -> Taxonomy:
    return Taxonomy()


def run_turn(session: DshPosterSession, user_input: str, images: list, aspect_ratio: str) -> None:
    try:
        session.run_turn(user_input, images, aspect_ratio)
    except Exception as e:
        session.error = f"{type(e).__name__}: {e}"


st.set_page_config(page_title="DeepSeek Harness Playground", layout="wide")
st.title("DeepSeek Harness Playground")

with st.sidebar:
    model_name = st.selectbox("Model", DSH_MODEL_CHOICES, index=0)
    effort_choice = st.selectbox(
        f"Reasoning effort (model default: {default_reasoning_effort(model_name) or 'none'})",
        [MODEL_DEFAULT_EFFORT, *supported_efforts(resolve_model_string(model_name))],
        index=0,
    )
    reasoning_effort = None if effort_choice == MODEL_DEFAULT_EFFORT else effort_choice
    aspect_ratio = st.selectbox("Poster aspect ratio", list(ASPECT_RATIO_TO_SIZE), index=0)
    if not DSH_BIN.is_file():
        st.error(f"DeepSeek Harness not found at {DSH_BIN}")

    prompt_presets = list_prompt_presets()

    def load_preset():
        selected = st.session_state.preset_select
        if selected == "Custom":
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            st.session_state.system_prompt = prompt_presets[selected].read_text()

    st.radio("Instructions preset", ["Custom", *prompt_presets.keys()], key="preset_select", on_change=load_preset)
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    instructions = st.text_area("Instructions (harness persona prefix)", key="system_prompt", height=150)

    def load_topic():
        selected = st.session_state.taxonomy_select
        if selected == TAXONOMY_NONE:
            st.session_state.show_draft = False
            return
        st.session_state.show_draft = True
        st.session_state.draft_editor = TOPIC_MESSAGE_TEMPLATE.format(topic=selected)

    st.selectbox("Taxonomy topic", [TAXONOMY_NONE, *get_taxonomy().list_topics()], key="taxonomy_select", on_change=load_topic)

    if st.button("Clear chat"):
        st.session_state.pop("dsh_session", None)
        st.session_state.pop("dsh_worker", None)
        st.rerun()


def new_session() -> DshPosterSession:
    agent_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return DshPosterSession(agent_name, model_name, instructions, reasoning_effort)


def session_matches_sidebar(candidate: DshPosterSession) -> bool:
    expected_effort = reasoning_effort if reasoning_effort is not None else default_reasoning_effort(model_name)
    return candidate.model_name == model_name and candidate.instructions == instructions and candidate.reasoning_effort == expected_effort


if "dsh_session" not in st.session_state:
    st.session_state.dsh_session = new_session()
elif not st.session_state.dsh_session.events and not session_matches_sidebar(st.session_state.dsh_session):
    st.session_state.dsh_session = new_session()

session: DshPosterSession = st.session_state.dsh_session
if session.events and not session_matches_sidebar(session):
    st.sidebar.caption("Sidebar changes apply after Clear chat.")

with st.sidebar:
    st.markdown("**Tools exposed to the harness (MCP)**")
    for schema in poster_function_schemas():
        st.caption(f"• {harness_tool_name(schema['name'])}")
    with st.expander("Tool definitions"):
        st.code(json.dumps(poster_function_schemas(), indent=4, ensure_ascii=False), language="json")

st.caption(f"agent `{session.agent_name}` · model `{session.model_string}` · dsh session `{session.dsh_session_id or 'not started'}`")


def render_event(event: dict[str, Any], records_by_call: dict[str, dict[str, Any]]):
    event_type = event.get("type")
    if event_type == "user":
        with st.chat_message("user"):
            st.markdown(event["text"])
    elif event_type == "thinking":
        with st.chat_message("assistant"):
            with st.expander("🧠 Thinking"):
                st.markdown(event.get("text") or "")
    elif event_type == "text":
        with st.chat_message("assistant"):
            st.markdown(event.get("text") or "")
    elif event_type == "tool_call":
        record = records_by_call.get(event["callId"])
        label = record["system_id"] if record else event["callId"]
        with st.chat_message("assistant"):
            with st.expander(f"🔧 {event['tool']} · {label}"):
                st.code(json.dumps(event.get("input"), indent=4, ensure_ascii=False), language="json")
    elif event_type == "tool_result":
        record = records_by_call.get(event["callId"])
        with st.chat_message("tool", avatar="🛠️"):
            result = event.get("result")
            st.markdown(result if isinstance(result, str) else f"```json\n{json.dumps(result, indent=4, ensure_ascii=False)}\n```")
            for path in (record or {}).get("image_path_history") or []:
                if Path(path).is_file():
                    st.image(path)
            st.caption(f"{record['system_id'] if record else event['callId']} · {event.get('status')}")
    elif event_type == "status" and event.get("phase") == "turn_end":
        reason = (event.get("reason") or {}).get("kind")
        if reason != "completed":
            st.warning(f"Turn ended: {reason}")


def is_running() -> bool:
    worker = st.session_state.get("dsh_worker")
    return worker is not None and worker.is_alive()


running = is_running()
render_baseline = len(session.events)
records_by_call = session.call_id_to_record()
for event in session.events[:render_baseline]:
    render_event(event, records_by_call)


@st.fragment(run_every=1.0)
def live_tail():
    if not is_running():
        if st.session_state.get("dsh_was_running"):
            st.session_state.dsh_was_running = False
            st.rerun(scope="app")
        return
    st.session_state.dsh_was_running = True
    live_records = session.call_id_to_record()
    for event in session.events[render_baseline:]:
        render_event(event, live_records)
    st.status("harness running...", state="running", expanded=False)


live_tail()

if session.error and not running:
    st.error(session.error)

if not running and session.dsh_session_id is not None:
    if "dsh_save_dir" not in st.session_state:
        st.session_state.dsh_save_dir = str(RESULT_ROOT)
    path_col, btn_col = st.columns([5, 1])
    with path_col:
        st.text_input("Save root (a folder named by the agent name is created inside)", key="dsh_save_dir", label_visibility="collapsed")
    with btn_col:
        if st.button("Save result", use_container_width=True):
            save_root = st.session_state.dsh_save_dir
            if not save_root.startswith("s3://"):
                save_root = str(Path(save_root).expanduser())
            try:
                st.success(f"Saved to {session.save_history(save_root)}")
            except Exception as e:
                st.error(f"Save failed: {type(e).__name__}: {e}")

if st.session_state.get("show_draft"):
    st.text_area("Draft message (from taxonomy topic)", key="draft_editor", height=100)
    send_col, discard_col = st.columns([1, 5])
    with send_col:
        if st.button("Send draft", disabled=running):
            st.session_state.submit_draft = st.session_state.draft_editor
            st.session_state.show_draft = False
            st.rerun()
    with discard_col:
        if st.button("Discard draft"):
            st.session_state.show_draft = False
            st.rerun()

chat_value = st.chat_input("Describe the poster...", accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp", "gif"], disabled=running)
draft = st.session_state.pop("submit_draft", None)
user_input, attached = None, []
if chat_value is not None and (chat_value.text or chat_value.files):
    user_input = chat_value.text or ""
    attached = [f.getvalue() for f in chat_value.files]
elif draft:
    user_input = draft

if user_input is not None and not running:
    session.error = None
    worker = threading.Thread(target=run_turn, args=(session, user_input, attached, aspect_ratio), daemon=True)
    st.session_state.dsh_worker = worker
    worker.start()
    st.rerun()
