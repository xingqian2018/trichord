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

# Instruction please see QIMAGEBENCH.md


import argparse
import json
from pathlib import Path

HF_DATASET_REPO = "Qwen/Qwen-Image-Bench"
HF_DATASET_FILE = "qwen_image_bench_hf_v0518.jsonl"


def load_hf_rows(local_jsonl: str | None) -> list[dict]:
    if local_jsonl is None:
        from huggingface_hub import hf_hub_download

        local_jsonl = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_DATASET_FILE, repo_type="dataset")
    with open(local_jsonl, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_manifest(rows: list[dict], generation_language: str) -> dict:
    prompt_key = f"prompt_{generation_language}"
    return {
        "source": f"hf://datasets/{HF_DATASET_REPO}/{HF_DATASET_FILE}",
        "generation_language": generation_language,
        "benchmark": [
            {
                "id": f"{int(row['ID']):06d}",
                "prompt": row[prompt_key],
                "prompt_cn": row["prompt_cn"],
                "prompt_en": row["prompt_en"],
                "dims_en": row["dims_en"],
            }
            for row in sorted(rows, key=lambda r: int(r["ID"]))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen-Image-Bench generation/scoring manifests")
    parser.add_argument("--local_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="qib1000")
    args = parser.parse_args()

    rows = load_hf_rows(args.local_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for language in ("cn", "en"):
        manifest = build_manifest(rows, language)
        output_path = output_dir / f"{args.name}_{language}.json"
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(manifest['benchmark'])} rows to {output_path}")


if __name__ == "__main__":
    main()
