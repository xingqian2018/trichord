#!/usr/bin/env python3
"""
wdkeys — count tars per key inside a webdataset.

Usage:
    wdkeys <path-or-shortname>

Argument forms:
    1. Full s3-omni path:  gcs:nv-00-10206-vfm/webdataset_image_text_related/zennodo10k
                           (with or without trailing /)
    2. Bare dataset name:  zennodo10k
                           — searched under the two working roots:
                             gcs:nv-00-10206-vfm/webdataset_image_text_related/
                             gcs:nv-00-10206-vfm/webdataset_synthetic/

What it does:
    Walks the dataset, finds every wdinfo.json. The directory containing each
    wdinfo.json is the "bucket leaf" (e.g. resolution_*/aspect_ratio_*/). Its
    immediate subdirectories are the *keys* (images/, metas/, ...). For each key,
    every .tar file beneath it is counted. Output is a per-key total summed
    across all bucket leaves, plus a per-leaf breakdown.

Output:
    <key>             <total_tars>
    ...
    (and a per-leaf table if more than one leaf exists)
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/Project/bashrc"))
from s3_omni import (
    parse_s3input, make_async_s3client, list_keys_with_size,
    _ENDPOINT_URL, _REGION,
)

WORKING_ROOTS = [
    "gcs:nv-00-10206-vfm/webdataset_image_text_related",
    "gcs:nv-00-10206-vfm/webdataset_synthetic",
]


def looks_like_path(arg):
    return ":" in arg and "/" in arg


def resolve(arg):
    if looks_like_path(arg):
        return [arg.rstrip("/")]
    candidates = []
    for root in WORKING_ROOTS:
        full = f"{root}/{arg}"
        profile, bucket, prefix = parse_s3input(full + "/")
        client = make_async_s3client(profile, _ENDPOINT_URL[profile], _REGION[profile])
        items = list_keys_with_size(client, bucket, prefix)
        if items:
            candidates.append(full)
    return candidates


def scan(path):
    profile, bucket, prefix = parse_s3input(path + "/")
    client = make_async_s3client(profile, _ENDPOINT_URL[profile], _REGION[profile])
    items = list_keys_with_size(client, bucket, prefix)
    if not items:
        return None

    leaves = set()
    for k, _r, _s in items:
        if k.endswith("/wdinfo.json") or k.endswith("wdinfo.json"):
            tail = k[len(prefix):] if k.startswith(prefix) else k
            leaf = tail.rsplit("/wdinfo.json", 1)[0] if "/wdinfo.json" in tail else ""
            leaves.add(leaf)

    if not leaves:
        leaves = {""}

    per_leaf = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for k, _r, _s in items:
        if not k.endswith(".tar"):
            continue
        tail = k[len(prefix):] if k.startswith(prefix) else k
        matched_leaf = None
        for leaf in leaves:
            test = leaf + "/" if leaf else ""
            if tail.startswith(test):
                if matched_leaf is None or len(leaf) > len(matched_leaf):
                    matched_leaf = leaf
        if matched_leaf is None:
            continue
        rest = tail[len(matched_leaf) + (1 if matched_leaf else 0):]
        parts = rest.split("/", 1)
        if not parts or not parts[0]:
            continue
        key = parts[0]
        per_leaf[matched_leaf][key] += 1
        totals[key] += 1

    return {
        "path": path,
        "leaves": sorted(leaves),
        "per_leaf": {l: dict(d) for l, d in per_leaf.items()},
        "totals": dict(totals),
    }


def render(res):
    print(f"\n=== {res['path']} ===")
    if not res["totals"]:
        print("(no tars found under any wdinfo-rooted leaf)")
        return
    keys = sorted(res["totals"], key=lambda k: -res["totals"][k])
    width = max(len(k) for k in keys)

    print("\nTotals:")
    for k in keys:
        print(f"  {k:<{width}}  {res['totals'][k]:>8}")

    nonempty_leaves = [l for l in res["leaves"]
                       if any(res["per_leaf"].get(l, {}).get(k, 0) for k in keys)]
    if len(nonempty_leaves) > 1:
        print("\nPer-leaf:")
        leaf_w = max(len(l) for l in nonempty_leaves)
        header = f"  {'leaf':<{leaf_w}}  " + "  ".join(f"{k:>{max(len(k),6)}}" for k in keys)
        print(header)
        for leaf in nonempty_leaves:
            row = res["per_leaf"].get(leaf, {})
            cells = "  ".join(f"{row.get(k, 0):>{max(len(k),6)}}" for k in keys)
            print(f"  {leaf:<{leaf_w}}  {cells}")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        sys.exit(1)
    arg = argv[1]
    paths = resolve(arg)
    if not paths:
        print(f"No dataset found for '{arg}'.", file=sys.stderr)
        print(f"Tried under:", file=sys.stderr)
        for r in WORKING_ROOTS:
            print(f"  - {r}/", file=sys.stderr)
        sys.exit(2)
    if len(paths) > 1:
        print(f"Found '{arg}' in multiple roots:")
        for p in paths:
            print(f"  - {p}")
        print()
    for p in paths:
        res = scan(p)
        if res:
            render(res)


if __name__ == "__main__":
    main(sys.argv)
