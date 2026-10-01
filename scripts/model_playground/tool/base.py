import filecmp
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional


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

    def run(self, args: dict[str, Any]) -> str:
        raise NotImplementedError

    def save_history(self, path: str | Path) -> None:
        raise NotImplementedError


def migrate_files(target_dir: Path, paths: list[str]) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for src in paths:
        src_path = Path(src)
        if not src_path.is_file():
            continue
        if src_path.resolve().parent == target_dir.resolve():
            mapping[str(src_path)] = src_path.name
            continue
        dst = target_dir / src_path.name
        if dst.exists() and not filecmp.cmp(src_path, dst, shallow=False):
            dst = target_dir / f"{src_path.stem}_{uuid.uuid4().hex[:8]}{src_path.suffix}"
        if not dst.exists():
            shutil.copy2(src_path, dst)
        mapping[str(src_path)] = dst.name
        moved[str(src_path)] = str(dst)
    if mapping:
        mapping_file = target_dir / "file_mapping.json"
        existing = json.loads(mapping_file.read_text()) if mapping_file.exists() else {}
        existing.update(mapping)
        mapping_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    return moved
