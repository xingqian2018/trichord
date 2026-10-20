from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config

# cd /home/xingqianx/Project/trichord && python reports/score_report_20261020_dataset_ablation/gather_ugb_scores.py

ROOT = "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5"
HAMID = "v2_1170L_gpt5p5_cosmos3p5_dense_hamid"
RUNS = [
    ("pre-SFT noneg", f"{HAMID}/c3p5_super_v2p5_iter741260_noneg"),
    ("000 neg", f"{HAMID}/c3p5_exp003_000_pretrain_iter25000"),
    ("001 neg", f"{HAMID}/c3p5_exp003_001_pretrain_midtrain_iter25000"),
    ("000 noneg", f"{HAMID}/c3p5_exp003_000_pretrain_iter25000_noneg"),
    ("001 noneg", f"{HAMID}/c3p5_exp003_001_pretrain_midtrain_iter25000_noneg"),
    ("002 noneg", f"{HAMID}/c3p5_exp003_002_pretrain_getty_iter25000_noneg"),
    ("pre-SFT cfg6 noneg", f"{HAMID}/c3p5_super_v2p5_iter741260_cfg6_noneg"),
    ("pre-SFT recipe shift6", f"{HAMID}/c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0"),
    ("002 recipe shift6", f"{HAMID}/c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seed0_noneg"),
    ("ruofan g4", f"{HAMID}/c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault"),
    ("ruofan g6", f"{HAMID}/c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault"),
    ("Qwen-2.1 no PE", "v2_1170L/qwen_image_2p1_nope"),
    ("Qwen-2.1 PE", "v2_1170L/qwen_image_2p1_pe"),
]
DIMENSIONS = [
    "Action", "Attribute", "Compound", "Entity Layout", "Grammar", "Logical Reasoning",
    "Relationship", "Style", "Text Generation", "World Knowledge",
]
CREDENTIAL_PATH = os.path.expanduser("~/Project/trichord/credentials/gcs.secret")


def s3_client():
    with open(CREDENTIAL_PATH) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        endpoint_url=cred["endpoint_url"],
        region_name=cred.get("region_name", "us-east-1"),
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        config=Config(s3={"addressing_style": "path"}),
    )


def load_result(client, run: str) -> dict | None:
    bucket, _, prefix = ROOT.removeprefix("s3://").partition("/")
    try:
        body = client.get_object(Bucket=bucket, Key=f"{prefix}/{run}/unigenbench_result.json")["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(body)


def main() -> None:
    client = s3_client()
    columns = []
    for name, run in RUNS:
        result = load_result(client, run)
        if result is None:
            print(f"skip {name}: no unigenbench_result.json under {run}")
            continue
        stats = result["stats"]
        column = {subset: 100 * stats[subset]["overall_accuracy"] for subset in ("all", "orig", "phi")}
        for dimension in DIMENSIONS:
            entry = stats["all"]["big_class_stats"].get(dimension)
            column[dimension] = 100 * entry["accuracy"] if entry else float("nan")
        column["success"] = f"{len(result.get('breakdown', {}))}/1170"
        column["judge"] = result.get("judge_model", "?")
        columns.append((name, column))

    print("| Metric | " + " | ".join(name for name, _ in columns) + " |")
    print("|" + "---|" * (len(columns) + 1))
    for row in ["all", "orig", "phi"] + DIMENSIONS:
        print(f"| {row} | " + " | ".join(f"{column[row]:.2f}" for _, column in columns) + " |")
    for row in ["success", "judge"]:
        print(f"| {row} | " + " | ".join(str(column[row]) for _, column in columns) + " |")


if __name__ == "__main__":
    main()
