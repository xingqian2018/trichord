"""Prepare and smoke-test a copied Qwen3.5 dense safetensors checkpoint.

The training export remains immutable. ``prepare`` is valid only for a copied
checkpoint with the native Cosmos training names. It rewrites safetensors
headers without touching tensor payloads and normalizes tokenizer metadata for
vLLM/Transformers compatibility.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

_NATIVE_A_LOG = ".linear_attn._fp32_params.A_log"
_VLLM_A_LOG = ".linear_attn.A_log"
_NATIVE_TOKENIZER = "TokenizersBackend"
_VLLM_TOKENIZER = "Qwen2Tokenizer"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_safetensors_header(path: pathlib.Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"invalid safetensors length prefix: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise TypeError(f"invalid safetensors header: {path}")
    return header_size, header


def _rewrite_safetensors_header(
    path: pathlib.Path,
    renames: list[tuple[str, str]],
) -> None:
    header_size, header = _read_safetensors_header(path)
    for old_key, new_key in renames:
        if old_key not in header:
            raise KeyError(f"missing indexed tensor {old_key!r} in {path.name}")
        if new_key in header:
            raise ValueError(f"tensor-key collision for {new_key!r} in {path.name}")
        header[new_key] = header.pop(old_key)

    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > header_size:
        raise ValueError(f"rewritten header grew from {header_size} to {len(encoded)} bytes: {path}")
    with path.open("r+b") as handle:
        handle.seek(8)
        handle.write(encoded + b" " * (header_size - len(encoded)))
        handle.flush()
        os.fsync(handle.fileno())


def inspect_checkpoint(model_dir: pathlib.Path, expected: str) -> dict[str, Any]:
    required = (
        ".complete",
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete checkpoint {model_dir}: missing={missing}")

    config = _read_json(model_dir / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"expected model_type=qwen3_5, got {config.get('model_type')!r}")
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError(f"unexpected architecture: {config.get('architectures')!r}")
    text_config = config.get("text_config") or {}
    if text_config.get("num_experts") not in (None, 0):
        raise ValueError(f"expert checkpoint is forbidden: num_experts={text_config.get('num_experts')}")

    marker = _read_json(model_dir / ".complete")
    if marker.get("status") != "complete":
        raise ValueError(f"checkpoint marker is not complete: {marker}")

    index = _read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight_map")
    expert_keys = [key for key in weight_map if ".experts." in key or ".shared_expert." in key]
    if expert_keys:
        raise ValueError(f"expert weights are forbidden: {expert_keys[:3]}")

    native_keys = [key for key in weight_map if _NATIVE_A_LOG in key]
    vllm_keys = [key for key in weight_map if _VLLM_A_LOG in key]
    tokenizer_class = _read_json(model_dir / "tokenizer_config.json").get("tokenizer_class")
    expected_counts = {
        "native": (48, 0, _NATIVE_TOKENIZER),
        "vllm": (0, 48, _VLLM_TOKENIZER),
    }
    expected_native, expected_vllm, expected_tokenizer = expected_counts[expected]
    actual = (len(native_keys), len(vllm_keys), tokenizer_class)
    wanted = (expected_native, expected_vllm, expected_tokenizer)
    if actual != wanted:
        raise ValueError(f"checkpoint state mismatch for {expected}: actual={actual} expected={wanted}")

    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        keys_by_shard[str(shard_name)].append(key)
    a_log_dtypes: set[str] = set()
    for shard_name, indexed_keys in sorted(keys_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"index references missing shard: {shard_path}")
        _, header = _read_safetensors_header(shard_path)
        absent = [key for key in indexed_keys if key not in header]
        if absent:
            raise ValueError(f"shard header is missing indexed keys in {shard_name}: {absent[:3]}")
        for key in indexed_keys:
            if _NATIVE_A_LOG in key or _VLLM_A_LOG in key:
                a_log_dtypes.add(str(header[key].get("dtype")))
    if a_log_dtypes != {"BF16"}:
        raise ValueError(f"compat safetensors require BF16 A_log, got {sorted(a_log_dtypes)}")

    return {
        "a_log_dtypes": sorted(a_log_dtypes),
        "architectures": config.get("architectures"),
        "checkpoint_status": marker.get("status"),
        "expected_state": expected,
        "indexed_weight_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "model_type": config.get("model_type"),
        "native_a_log_keys": len(native_keys),
        "normalized_a_log_keys": len(vllm_keys),
        "shard_count": len(keys_by_shard),
        "tokenizer_class": tokenizer_class,
        "weight_count": len(weight_map),
    }


def prepare_checkpoint(model_dir: pathlib.Path, summary_path: pathlib.Path) -> None:
    before = inspect_checkpoint(model_dir, "native")
    index_path = model_dir / "model.safetensors.index.json"
    index = _read_json(index_path)
    weight_map = index["weight_map"]

    renames_by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    updated_weight_map: dict[str, str] = {}
    for old_key, shard_name in weight_map.items():
        new_key = old_key.replace(_NATIVE_A_LOG, _VLLM_A_LOG)
        if new_key in updated_weight_map:
            raise ValueError(f"key collision after normalization: {new_key}")
        updated_weight_map[new_key] = shard_name
        if new_key != old_key:
            renames_by_shard[str(shard_name)].append((old_key, new_key))
    rename_count = sum(len(items) for items in renames_by_shard.values())
    if rename_count != 48:
        raise ValueError(f"expected exactly 48 A_log renames, got {rename_count}")

    rewritten_shards: list[str] = []
    for shard_name in sorted(renames_by_shard):
        _rewrite_safetensors_header(model_dir / shard_name, renames_by_shard[shard_name])
        rewritten_shards.append(shard_name)
    index["weight_map"] = updated_weight_map
    _write_json_atomic(index_path, index)

    tokenizer_path = model_dir / "tokenizer_config.json"
    tokenizer_config = _read_json(tokenizer_path)
    if tokenizer_config.get("tokenizer_class") != _NATIVE_TOKENIZER:
        raise ValueError(f"unexpected tokenizer_class={tokenizer_config.get('tokenizer_class')!r}")
    tokenizer_config["tokenizer_class"] = _VLLM_TOKENIZER
    _write_json_atomic(tokenizer_path, tokenizer_config)

    after = inspect_checkpoint(model_dir, "vllm")
    summary = {
        "before": before,
        "after": after,
        "model_dir": str(model_dir),
        "renamed_a_log_keys": rename_count,
        "rewritten_shards": rewritten_shards,
    }
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": "Bearer not-needed", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def _create_image(path: pathlib.Path) -> str:
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (320, 240), (24, 48, 96))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 50, 240, 190), fill=(245, 240, 220), outline=(0, 0, 0), width=4)
        draw.ellipse((130, 90, 190, 150), fill=(220, 30, 45), outline=(0, 0, 0), width=3)
        image.save(path, format="PNG")
        return "pillow"
    except Exception as pillow_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"cannot generate image with Pillow: {pillow_error}") from pillow_error
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=1",
                "-frames:v",
                "1",
                "-y",
                str(path),
            ],
            check=True,
        )
        return "ffmpeg"


def _create_video(path: pathlib.Path) -> str:
    try:
        import av
        import numpy as np

        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for frame_index in range(8):
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            image[:, :, 0] = frame_index * 30
            image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
            image[:, :, 2] = np.linspace(255, 0, 240, dtype=np.uint8)[:, None]
            left = 20 + frame_index * 32
            image[80:160, left : left + 48] = (255, 255, 255)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return "pyav"
    except Exception as pyav_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"cannot generate video with PyAV: {pyav_error}") from pyav_error
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=2",
                "-t",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(path),
            ],
            check=True,
        )
        return "ffmpeg"


def _response_summary(response: dict[str, Any], latency: float) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"endpoint returned empty content: {response}")
    if prompt_tokens <= 0 or completion_tokens <= 0 or latency <= 0 or not math.isfinite(latency):
        raise RuntimeError(f"invalid response accounting: latency={latency} usage={usage}")
    return {
        "content": content.strip(),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "latency_seconds": latency,
        "usage": usage,
    }


def probe_endpoint(base_url: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _http_json("GET", f"{base_url}/v1/models", None, timeout=30)
    model_id = models["data"][0]["id"]
    common = {
        "model": model_id,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    image_path = output_dir / "synthetic_320x240.png"
    video_path = output_dir / "synthetic_4s_2fps.mp4"
    image_generator = _create_image(image_path)
    video_generator = _create_video(video_path)
    image_uri = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    video_uri = "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode("ascii")
    payloads = {
        "text": {
            **common,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply briefly: deployment test."}]}],
        },
        "image": {
            **common,
            "max_tokens": 96,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": "Describe the central shape and its color in one sentence."},
                    ],
                }
            ],
        },
        "video": {
            **common,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_uri}},
                        {
                            "type": "text",
                            "text": "Describe the moving object and dominant visual pattern in one sentence.",
                        },
                    ],
                }
            ],
            "mm_processor_kwargs": {
                "fps": 2,
                "min_frames": 4,
                "max_frames": 256,
                "min_pixels": 4096,
                "max_pixels": 25165824,
            },
        },
    }

    results: dict[str, Any] = {"model_id": model_id, "models_response": models}
    for modality in ("text", "image", "video"):
        started = time.perf_counter()
        response = _http_json("POST", f"{base_url}/v1/chat/completions", payloads[modality], timeout=600)
        result = _response_summary(response, time.perf_counter() - started)
        result["modality"] = modality
        if modality == "image":
            result.update(generator=image_generator, size_bytes=image_path.stat().st_size)
        elif modality == "video":
            result.update(generator=video_generator, size_bytes=video_path.stat().st_size)
        _write_json_atomic(output_dir / f"probe_{modality}.json", result)
        results[modality] = result

    text_tokens = int(results["text"]["usage"]["prompt_tokens"])
    for modality in ("image", "video"):
        visual_tokens = int(results[modality]["usage"]["prompt_tokens"])
        if visual_tokens <= text_tokens:
            raise RuntimeError(f"{modality} did not add visual tokens: text={text_tokens} visual={visual_tokens}")
    _write_json_atomic(output_dir / "probe_results.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    inspect_parser.add_argument("--expected", choices=("native", "vllm"), required=True)
    inspect_parser.add_argument("--output", type=pathlib.Path)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--summary-path", type=pathlib.Path, required=True)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--base-url", required=True)
    probe_parser.add_argument("--output-dir", type=pathlib.Path, required=True)

    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_checkpoint(args.model_dir, args.expected)
        if args.output:
            _write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "prepare":
        prepare_checkpoint(args.model_dir, args.summary_path)
    else:
        probe_endpoint(args.base_url.rstrip("/"), args.output_dir)


if __name__ == "__main__":
    main()
