import filecmp
import io
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import GATEWAY_CONFIG, UnifiedGatewayVLM, image_bytes_to_data_url, resolve_model_string  # noqa: E402

EMPTY_REPLY = "*(empty response — check the terminal log for API errors)*"


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


def tool_placeholder(call_id: str) -> str:
    return f"<<tool_result:{call_id}>>"


class Agent:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        gateway_configs: dict[str, dict] = GATEWAY_CONFIG,
        num_max_retry: int = 2,
        timeout: int = 600,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.gateway = UnifiedGatewayVLM(gateway_configs, num_concurrency=1, num_max_retry=num_max_retry, timeout=timeout)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.messages: list[dict[str, Any]] = []
        self.meta: list[dict[str, Any]] = []
        self.tool_results: dict[str, dict[str, Any]] = {}
        self.tool_call_counter = 0
        self.last_rendered_html_path: Optional[str] = None

    def set_model(self, model_name: str):
        self.model_name = model_name

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def set_tools(self, tools: Optional[list[dict[str, Any]]]):
        self.tools = tools or []

    def reset(self):
        self.messages.clear()
        self.meta.clear()
        self.tool_results.clear()
        self.tool_call_counter = 0
        self.last_rendered_html_path = None

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
        if self.tools:
            request["tools"] = self.tools
        return request

    def next_tool_call_id(self, tool_name: str) -> str:
        self.tool_call_counter += 1
        return f"{tool_name}_{self.tool_call_counter:09d}_{uuid.uuid4().hex[:8]}"

    def append(self, msg: dict[str, Any], meta: Optional[dict[str, Any]] = None) -> int:
        self.messages.append(msg)
        self.meta.append(meta or {})
        return len(self.messages) - 1

    def add_user_message(self, text: str, images: Optional[list[Any]] = None) -> int:
        if images:
            content: Any = [{"type": "text", "text": text}]
            for image in images:
                data = to_image_bytes(image)
                content.append({"type": "image_url", "image_url": {"url": image_bytes_to_data_url(data, sniff_image_fmt(data))}})
        else:
            content = text
        return self.append({"role": "user", "content": content})

    def step(self) -> int:
        result = self.gateway.query([self.build_request()])[0]
        meta: dict[str, Any] = {"reasoning": result["reasoning"], "finish_reason": result["finish_reason"]}
        has_tool_calls = result["finish_reason"] == "tool_calls" and result["tool_calls"]
        tool_calls = json.loads(result["tool_calls"]) if has_tool_calls else []
        if not tool_calls:
            meta["tool_calls"] = result["tool_calls"]
            meta["rendered_html_path"] = self.last_rendered_html_path
            return self.append({"role": "assistant", "content": result["content"] or EMPTY_REPLY}, meta)
        for call in tool_calls:
            call["id"] = self.next_tool_call_id(call["name"])
        meta["tool_calls"] = json.dumps(tool_calls, ensure_ascii=False)
        return self.append(
            {
                "role": "assistant",
                "content": result["content"] or "",
                "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in tool_calls
                ],
            },
            meta,
        )

    def pending_tool_calls(self) -> list[dict[str, str]]:
        answered: set[str] = set()
        pos = len(self.messages) - 1
        while pos >= 0 and self.messages[pos]["role"] == "tool":
            answered.add(self.messages[pos]["tool_call_id"])
            pos -= 1
        if pos < 0 or self.messages[pos]["role"] != "assistant":
            return []
        return [
            {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in self.messages[pos].get("tool_calls", [])
            if c["id"] not in answered
        ]

    def add_tool_result(self, call_id: str, tool_result: dict[str, Any]) -> int:
        if tool_result.get("html_path"):
            self.last_rendered_html_path = tool_result["html_path"]
        self.tool_results[call_id] = tool_result
        meta = {"images": tool_result.get("images", []), "html_path": tool_result.get("html_path")}
        return self.append({"role": "tool", "tool_call_id": call_id, "content": tool_result["text"]}, meta)

    def save(self, root_path: str | Path) -> Path:
        root = Path(root_path)
        (root / "tool").mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for msg, meta in zip(self.messages, self.meta):
            record: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
            if msg["role"] == "tool":
                record["content"] = tool_placeholder(msg["tool_call_id"])
            if meta.get("reasoning"):
                record["thinking"] = meta["reasoning"]
            for key in ("tool_calls", "tool_call_id"):
                if key in msg:
                    record[key] = msg[key]
            if "finish_reason" in meta:
                record["finish_reason"] = meta["finish_reason"]
            records.append(record)
        (root / "chat.json").write_text(json.dumps({"session_id": self.session_id, "message": records}, ensure_ascii=False, indent=2))

        mapping_index: dict[str, str] = {}
        for call_id, tool_result in self.tool_results.items():
            tool_dir = root / "tool" / call_id
            tool_dir.mkdir(parents=True, exist_ok=True)
            (tool_dir / "result.txt").write_text(tool_result["text"])
            produced = [*(tool_result.get("images") or []), *([tool_result["html_path"]] if tool_result.get("html_path") else [])]
            mapping: dict[str, str] = {}
            for src in produced:
                src_path = Path(src)
                if not src_path.is_file():
                    continue
                dst = tool_dir / src_path.name
                if dst.exists() and not filecmp.cmp(src_path, dst, shallow=False):
                    dst = tool_dir / f"{src_path.stem}_{uuid.uuid4().hex[:8]}{src_path.suffix}"
                if not dst.exists():
                    shutil.copy2(src_path, dst)
                mapping[str(src_path)] = dst.name
            if mapping:
                (tool_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
            if tool_result.get("chat"):
                (tool_dir / "chat.json").write_text(json.dumps(tool_result["chat"], ensure_ascii=False, indent=2))
            mapping_index[call_id] = f"tool/{call_id}"
        (root / "tool_call_mapping.json").write_text(json.dumps(mapping_index, ensure_ascii=False, indent=2))
        return root
