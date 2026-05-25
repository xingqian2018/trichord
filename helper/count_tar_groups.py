#!/usr/bin/env python3
"""
count_tar_groups.py — count WebDS tars per 100-index group for a scene-text SGD dataset.

Usage:
    python count_tar_groups.py <dataset_name>
    python count_tar_groups.py <full_gcs_path>

Examples:
    python count_tar_groups.py synthetic_scene_text_v1
    python count_tar_groups.py gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_v1
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/Project/bashrc"))
from s3_omni import parse_s3input, make_async_s3client, list_keys_with_size, _ENDPOINT_URL, _REGION

WEBDS_ROOT = "gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text"


def resolve(arg):
    if ":" in arg:
        return arg.rstrip("/")
    return f"{WEBDS_ROOT}/{arg}"


def count_groups(path):
    profile, bucket, prefix = parse_s3input(path + "/")
    client = make_async_s3client(profile, _ENDPOINT_URL[profile], _REGION[profile])
    items = list_keys_with_size(client, bucket, prefix)

    counts = defaultdict(int)
    for key, _, _ in items:
        if not key.endswith(".tar"):
            continue
        if "/images/" not in key:
            continue
        stem = key.rsplit("/", 1)[-1][:-4]
        if not stem.isdigit() or len(stem) != 9:
            continue
        group = (int(stem) // 100) * 100
        counts[group] += 1

    return dict(counts)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = resolve(sys.argv[1])
    print(f"Scanning: {path}")

    counts = count_groups(path)
    if not counts:
        print("No tars found.")
        sys.exit(0)

    print(f"\n{'Group':<12} {'Tars':>6}")
    print("-" * 20)
    for g in sorted(counts):
        mark = "  ✅" if counts[g] == 100 else ""
        print(f"{g}-{g+100:<8}  {counts[g]:>4}{mark}")


if __name__ == "__main__":
    main()
