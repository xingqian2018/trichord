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

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gateway import GATEWAY_CONFIG, UnifiedGatewayLLM  # noqa: E402

DEFAULT_SYSTEM_PROMPT = (
    Path.home()
    / "Project/imaginaire4_next/projects/cosmos3/agentic/taskflows/open_loop_image_prompt_upsampling/k19_clean_faithful.txt"
)

ORIENTATION_BY_ASPECT = {
    "1:1": "square",
    "16:9": "landscape-oriented",
    "4:3": "landscape-oriented",
    "3:2": "landscape-oriented",
    "9:16": "vertical",
    "3:4": "portrait-oriented",
    "2:3": "portrait-oriented",
}

MIN_WORDS = 390
MAX_WORDS = 490

_CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)


def build_system_prompt(template: str, aspect: str) -> str:
    return template % {"aspect": aspect, "orientation": ORIENTATION_BY_ASPECT[aspect]}


def parse_caption(response: str, orientation: str) -> tuple[Optional[str], str]:
    matches = _CAPTION_RE.findall(response or "")
    if not matches:
        return None, "no_caption_tag"
    raw = matches[-1].strip()
    if re.search(r"\n\s*\n", raw):
        return None, "multi_paragraph"
    caption = " ".join(raw.split())
    if not caption.startswith(f"A {orientation} "):
        return None, "bad_opening"
    num_words = len(caption.split())
    if not MIN_WORDS <= num_words <= MAX_WORDS:
        return None, f"words_{num_words}"
    return caption, "ok"


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"blend": path.stem, "benchmark": data}
    return data


def load_state(path: Path) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def upsample_blend(
    llm: UnifiedGatewayLLM,
    manifest: dict,
    state: dict[str, dict],
    state_path: Path,
    system_prompt: str,
    orientation: str,
    model: str,
    max_rounds: int,
    limit: Optional[int],
) -> None:
    entries = manifest["benchmark"][:limit] if limit else manifest["benchmark"]
    for round_idx in range(max_rounds):
        pending = [e for e in entries if str(e["id"]) not in state or state[str(e["id"])].get("caption") is None]
        if not pending:
            return
        requests = [
            llm.build_request(model_name=model, system_prompt=system_prompt, content_prompt=e["prompt"], temperature=1.0, max_tokens=16384)
            for e in pending
        ]
        responses = llm.query(requests, pbar_desc=f"{manifest['blend']} round {round_idx}")
        reasons: dict[str, int] = {}
        for entry, response in zip(pending, responses):
            caption, reason = parse_caption((response or {}).get("content") or "", orientation)
            key = str(entry["id"])
            record = state.get(key, {"caption": None, "tries": []})
            record["tries"].append(reason)
            record["caption"] = caption
            state[key] = record
            reasons[reason] = reasons.get(reason, 0) + 1
        save_json(state, state_path)
        logger.info(f"[{manifest['blend']}] round {round_idx}: {len(pending)} calls, {reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsample QImageBench prompts for Cosmos3.5 with the k19 dense caption prompt")
    parser.add_argument("--blend_dir", type=str, default=str(Path(__file__).resolve().parent / "blend"))
    parser.add_argument("--blends", type=str, nargs="+", default=["arena1000", "cosmos500"])
    parser.add_argument("--aspect", type=str, default="1:1", choices=list(ORIENTATION_BY_ASPECT))
    parser.add_argument("--model", type=str, default="kimi-k3")
    parser.add_argument("--name", type=str, default="c3p5dense_k19_sq")
    parser.add_argument("--system_prompt", type=str, default=str(DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--num_concurrency", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    blend_dir = Path(args.blend_dir)
    orientation = ORIENTATION_BY_ASPECT[args.aspect]
    system_prompt = build_system_prompt(Path(args.system_prompt).read_text(encoding="utf-8"), args.aspect)
    suffix = args.name if args.aspect == "1:1" else f"{args.name}_{args.aspect.replace(':', 'x')}"
    llm = UnifiedGatewayLLM(
        gateway_configs={"nvidia": GATEWAY_CONFIG["nvidia"]}, num_concurrency=args.num_concurrency, timeout=args.timeout
    )

    try:
        for blend in args.blends:
            manifest = load_manifest(blend_dir / f"{blend}.json")
            state_path = blend_dir / f"{blend}_{suffix}_state.json"
            state = load_state(state_path)
            upsample_blend(
                llm, manifest, state, state_path, system_prompt, orientation, args.model, args.max_rounds, args.limit
            )
            entries = manifest["benchmark"][: args.limit] if args.limit else manifest["benchmark"]
            missing = [str(e["id"]) for e in entries if state.get(str(e["id"]), {}).get("caption") is None]
            if missing:
                logger.error(f"[{blend}] {len(missing)} prompts still without a valid caption: {missing[:20]}")
                continue
            if args.limit:
                logger.info(f"[{blend}] --limit {args.limit} done, captions are in {state_path}")
                continue
            out = dict(manifest)
            out["benchmark"] = [
                {
                    **e,
                    "prompt_upsampled": state[str(e["id"])]["caption"],
                    "upsampler_model": args.model,
                    "upsample_aspect": args.aspect,
                }
                for e in manifest["benchmark"]
            ]
            out_path = blend_dir / f"{blend}_{suffix}.json"
            save_json(out, out_path)
            logger.info(f"[{blend}] wrote {len(out['benchmark'])} prompts to {out_path}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
