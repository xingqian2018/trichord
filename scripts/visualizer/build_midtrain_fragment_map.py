from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import yaml

MAIN_TABLE = "gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance"
REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES_PATH = REPO_ROOT / "reports/image_data_stats/dataset_categories.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "midtrain_fragment_map.json"
CANDIDATE_RANGES = [(37045, 37045), (37159, 37159), (37168, 37171), (40561, 10 ** 9)]
GCS_CONFIG_PATH = os.path.expanduser("~/.config/dir/config.yaml")
GCS_CONFIG_PROFILE = "team-gcs-cosmos"


def setup_google_auth() -> None:
    with open(GCS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sa_key = json.loads(cfg["gcp"]["service_account_profiles"][GCS_CONFIG_PROFILE]["json_string"])
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(sa_key, tf)
    tf.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name


def in_candidates(fid: int) -> bool:
    return any(lo <= fid <= hi for lo, hi in CANDIDATE_RANGES)


def count_sources(dataset: "lance.LanceDataset", fid: int) -> tuple[int, dict[str, int]]:
    table = dataset.get_fragment(fid).to_table(columns=["source_dataset"])
    return fid, dict(Counter(table.column("source_dataset").to_pylist()))


def main() -> None:
    setup_google_auth()
    midtrain = json.load(open(CATEGORIES_PATH))["Midtrain"]
    dataset = lance.dataset(MAIN_TABLE)
    fids = sorted(f.fragment_id for f in dataset.get_fragments() if in_candidates(f.fragment_id))
    print(f"scanning {len(fids)} fragments, table version {dataset.version}", flush=True)
    per_dataset: dict[str, dict[int, int]] = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, (fid, counts) in enumerate(pool.map(lambda f: count_sources(dataset, f), fids)):
            for name, n in counts.items():
                if name in midtrain:
                    per_dataset[name][fid] = n
            if i % 50 == 0:
                print(f"  {i}/{len(fids)}", flush=True)
    result = {
        "main_table": MAIN_TABLE,
        "table_version": dataset.version,
        "datasets": {
            name: {
                "rows": sum(per_dataset[name].values()),
                "fragments": {str(fid): n for fid, n in sorted(per_dataset[name].items())},
            }
            for name in midtrain
            if name in per_dataset
        },
    }
    missing = [name for name in midtrain if name not in per_dataset]
    OUTPUT_PATH.write_text(json.dumps(result, indent=1))
    for name, v in result["datasets"].items():
        ids = list(v["fragments"])
        print(f"{name:32s} rows={v['rows']:>12,} fragments={len(ids):4d} {ids[0]}-{ids[-1]}")
    if missing:
        print("missing from candidate ranges:", missing, file=sys.stderr)
    print("written", OUTPUT_PATH)


if __name__ == "__main__":
    main()
