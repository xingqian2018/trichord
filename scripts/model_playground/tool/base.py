import json
import os.path as osp
import threading
from pathlib import Path
from typing import Any, Optional

from utils import hash_bytes, put  # noqa: E402


class ToolCallError(Exception):
    pass


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

    def __init__(self):
        self.result_text: Optional[str] = None
        self.save_history_lock = threading.Lock()
        self.history_saved = False

    def run(self, args: dict[str, Any]) -> str:
        try:
            self.result_text = self.run_core(args)
        except ToolCallError as e:
            self.result_text = str(e)
        return self.result_text

    def run_core(self, args: dict[str, Any]) -> str:
        raise NotImplementedError

    def save_history(self, path: str | Path) -> None:
        with self.save_history_lock:
            if self.history_saved:
                raise RuntimeError(f"{type(self).__name__}.save_history may only be called once per instance")
            self.history_saved = True
        self.save_history_core(str(path))

    def save_history_core(self, path: str) -> None:
        raise NotImplementedError


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
