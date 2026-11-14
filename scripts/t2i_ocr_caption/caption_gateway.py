# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------
"""Caption images through the NVIDIA inference gateway with the SILA caption v2 prompts.

Usage:
    python scripts/t2i_ocr_caption/caption_gateway.py \
        --input /path/to/images_dir_or_list.jsonl \
        --output /path/to/captions.jsonl \
        --model gemini-3.1-pro@nvidiak \
        --prompt dense

--input is a directory (scanned recursively for images), a .txt with one image path/url per line,
or a .jsonl whose rows carry "image" (path/url) and optionally "id". Rows already present in
--output with a caption are skipped, so re-running resumes.
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROMPTS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gateway import GATEWAY_CONFIG, UnifiedGatewayVLM, resolve_model_string  # noqa: E402
from utils.image_io import read_url_bytes  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
USER_TEXT = "Caption this image."

TAGGED_RESPONSE = re.compile(
    r"\A\s*<image_breakdown>(?P<breakdown>.*?)</image_breakdown>\s*<caption>(?P<caption>.*?)</caption>\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
XML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def load_items(input_path: str) -> list[dict[str, str]]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        return [{"id": str(p.relative_to(path)), "image": str(p)} for p in files]
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                row = json.loads(line)
                items.append({"id": str(row.get("id", row["image"])), "image": row["image"]})
            else:
                items.append({"id": line, "image": line})
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids in {input_path}")
    return items


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("caption"):
                    done.add(row["id"])
    return done


def prepare_image(image_bytes: bytes, max_long_side: int) -> bytes:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    if max_long_side <= 0 or max(image.size) <= max_long_side:
        return image_bytes
    image = ImageOps.exif_transpose(image)
    scale = max_long_side / max(image.size)
    resized = image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def parse_response(content: Optional[str], prompt_name: str) -> tuple[Optional[str], Optional[str]]:
    if not content:
        return None, None
    text = THINK_BLOCK.sub("", content).strip()
    if prompt_name == "dense":
        match = TAGGED_RESPONSE.match(text)
        if match is None:
            return None, None
        breakdown, caption = match.group("breakdown").strip(), match.group("caption").strip()
    else:
        breakdown, caption = None, text
    caption = " ".join(caption.split())
    if not caption or XML_TAG.search(caption):
        return None, None
    return caption, breakdown


def build_requests(
    gateway: UnifiedGatewayVLM, items: list[dict[str, str]], args: argparse.Namespace, system_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requests, kept, failed = [], [], []
    for item in items:
        try:
            image_bytes = prepare_image(read_url_bytes(item["image"]), args.max_long_side)
        except Exception as exc:
            failed.append({**item, "error": f"image load failed: {type(exc).__name__}: {exc}"})
            continue
        request = gateway.build_request(
            model_name=args.model,
            system_prompt=system_prompt,
            content_prompt=USER_TEXT,
            image_bytes=image_bytes,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.reasoning_effort:
            request["reasoning_effort"] = args.reasoning_effort
        requests.append(request)
        kept.append(item)
    return requests, kept, failed


def run_chunk(
    gateway: UnifiedGatewayVLM, items: list[dict[str, str]], args: argparse.Namespace, system_prompt: str
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    pending = items
    for attempt in range(1, args.parse_attempts + 1):
        requests, kept, failed = build_requests(gateway, pending, args, system_prompt)
        for row in failed:
            records[row["id"]] = {**row, "caption": None, "breakdown": None, "finish_reason": None}
        responses = gateway.query(requests, pbar_desc=f"caption attempt {attempt}") if requests else []
        retry = []
        for item, response in zip(kept, responses):
            content = response.get("content") if response else None
            finish_reason = response.get("finish_reason") if response else None
            caption, breakdown = parse_response(content, args.prompt)
            error = None
            if caption is None:
                error = "gateway returned nothing" if response is None else f"unparseable response (finish_reason={finish_reason})"
                retry.append(item)
            records[item["id"]] = {**item, "caption": caption, "breakdown": breakdown, "finish_reason": finish_reason, "error": error}
        if not retry:
            break
        pending = retry
    model_string = resolve_model_string(args.model)
    return [{**records[item["id"]], "model": model_string, "prompt": args.prompt} for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-3.1-pro@nvidiak")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="dense")
    parser.add_argument("--max_long_side", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high", "max"], default=None)
    parser.add_argument("--num_concurrency", type=int, default=64)
    parser.add_argument("--num_max_retry", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--parse_attempts", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items = load_items(args.input)
    if args.limit > 0:
        items = items[: args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output_path)
    todo = [item for item in items if item["id"] not in done]
    logger.info(f"{len(items)} images, {len(done)} already captioned, {len(todo)} to caption with {args.model} / {args.prompt}")
    if not todo:
        return

    system_prompt = PROMPTS[args.prompt]()
    gateway = UnifiedGatewayVLM(GATEWAY_CONFIG, num_concurrency=args.num_concurrency, num_max_retry=args.num_max_retry, timeout=args.timeout)
    num_ok = 0
    try:
        for start in range(0, len(todo), args.chunk_size):
            rows = run_chunk(gateway, todo[start : start + args.chunk_size], args, system_prompt)
            with open(output_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_ok += sum(row["caption"] is not None for row in rows)
            logger.info(f"{min(start + args.chunk_size, len(todo))}/{len(todo)} processed, {num_ok} captioned")
    finally:
        gateway.close()
    logger.info(f"done: {num_ok}/{len(todo)} captioned -> {output_path}")


if __name__ == "__main__":
    main()
