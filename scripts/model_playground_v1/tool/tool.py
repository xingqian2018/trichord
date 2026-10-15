import base64
import copy
import json
import os.path as osp
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import hash_bytes, put  # noqa: E402


class ToolCallError(Exception):
    pass


def migrate_files(target_dir: str, paths: list[str]) -> None:
    mapping: dict[str, list[str]] = {}
    for src in paths:
        src_path = Path(src)
        if not src_path.is_file():
            continue
        data = src_path.read_bytes()
        name = f"{hash_bytes(data)}{src_path.suffix.lower()}"
        put(data, osp.join(target_dir, name))
        src_path.unlink()
        mapping.setdefault(name, []).append(str(src_path))
    if mapping:
        put(json.dumps(mapping, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(target_dir, "file_mapping.json"))


def externalize_media(messages: list[dict[str, Any]], root: str) -> list[dict[str, Any]]:
    messages = copy.deepcopy(messages)
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            url = block.get("image_url", {}).get("url") if block.get("type") == "image_url" else None
            if not url or not url.startswith("data:"):
                continue
            header, payload = url.split(",", 1)
            fmt = header[len("data:"):].split(";")[0].split("/")[-1]
            data = base64.b64decode(payload)
            filename = f"{hash_bytes(data)}.{fmt}"
            put(data, osp.join(root, "media", filename))
            block["image_url"]["url"] = f"media/{filename}"
    return messages


class Tool:
    schema = {
        "type": "function",
        "function": {
            "name": "tool",
            "description": "An example of how to make a tool schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "input argument"},
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = True

    def __init__(self):
        self.function_name = type(self).schema["function"]["name"]
        self.system_prompt = ""
        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.current_call = threading.local()
        self.result_text: Optional[str] = None
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.save_history_lock = threading.Lock()
        self.history_saved = False

    def append(self, msg: dict[str, Any]) -> None:
        with self.lock:
            self.messages.append(msg)

    def add_file(self, path: str) -> None:
        self.current_call.record["files"].append(path)

    def run(self, args: dict[str, Any], tool_call_id: Optional[str] = None) -> str:
        with nullcontext() if self.allow_parallel_calls else self.run_lock:
            with self.lock:
                record = {"tool_call_id": tool_call_id, "arguments": args, "result": None, "files": []}
                self.calls.append(record)
            self.current_call.record = record
            try:
                text = self.run_core(args)
            except ToolCallError as e:
                text = str(e)
            finally:
                self.current_call.record = None
            record["result"] = text
            self.result_text = text
            return text

    def run_core(self, args: dict[str, Any]) -> str:
        raise NotImplementedError

    def setting(self) -> dict[str, Any]:
        return {"function_name": self.function_name, "class": type(self).__name__}

    def save_history(self, path: str | Path) -> str:
        with self.save_history_lock:
            if self.history_saved:
                raise RuntimeError(f"{type(self).__name__}.save_history may only be called once per instance")
            self.history_saved = True
        self.save_history_core(str(path))
        return str(path)

    def save_calls(self, root: str) -> None:
        for record in self.calls:
            if record["tool_call_id"] is None:
                continue
            call_dir = osp.join(root, "call", record["tool_call_id"])
            call = {"tool_call_id": record["tool_call_id"], "arguments": record["arguments"], "result": record["result"]}
            put(json.dumps(call, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(call_dir, "call.json"))
            migrate_files(call_dir, record["files"])

    def save_history_core(self, root: str) -> None:
        put(json.dumps(self.setting(), ensure_ascii=False, indent=4).encode("utf-8"), osp.join(root, "setting.json"))
        self.save_calls(root)
