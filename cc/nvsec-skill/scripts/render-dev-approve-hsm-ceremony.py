#!/usr/bin/env python3
"""Render a deliberately failing DEV_QUORUM ceremony placeholder template."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


SAFE_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")
DEFAULT_CU = "cu_test_ceremony_02"


def _safe_name(value: str, field: str) -> str:
    if not SAFE_NAME.fullmatch(value) or value.casefold() == "_all":
        raise ValueError(f"{field} must be a narrowly scoped name")
    return value


def _render(cu: str, label: str) -> bytes:
    return (
        "actions:\n"
        "  - action: unlock\n"
        f"    username: {cu}\n"
        "    token: __NVSEC_HSM_APPROVAL_DATA__\n"
        "\n"
        "  - action: find\n"
        f"    key: {label}\n"
        "    type: secret\n"
        "\n"
        "  - action: lockout\n"
        f"    username: {cu}\n"
    ).encode("utf-8")


def _write_private(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(body)
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cu", default=DEFAULT_CU)
    args = parser.parse_args()

    label = _safe_name(args.label, "label")
    cu = _safe_name(args.cu, "cu")
    _write_private(
        args.output,
        _render(cu, label),
    )
    print(f"Rendered Dev ceremony placeholder template: {args.output}")
    print(f"Deliberately missing key label: {label}")


if __name__ == "__main__":
    main()
