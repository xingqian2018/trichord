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
"""Write captions from a caption_gateway.py jsonl back into a Lance table as a new column.

Usage:
    python scripts/t2i_ocr_caption/write_caption_column.py \
        --table gs://.../poster_..._slice_from_maintable_YYYYmmdd.lance \
        --captions /path/to/captions.jsonl \
        --column caption_sp_for_html_poster_v0

The jsonl "id" is matched against --id_column (default uuid); rows without a caption get null.
An existing column of the same name is only replaced with --overwrite.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LANCE_INITIAL_UPLOAD_SIZE", str(1 << 30))

import lance
import pyarrow as pa
from loguru import logger


def open_table(uri: str) -> lance.LanceDataset:
    storage_options = None
    if uri.startswith(("s3://", "gs://")):
        cred = json.loads((Path(__file__).resolve().parents[2] / "credentials" / "gcs.secret").read_text())
        storage_options = {
            "aws_access_key_id": cred["aws_access_key_id"],
            "aws_secret_access_key": cred["aws_secret_access_key"],
            "aws_endpoint": cred["endpoint_url"],
            "aws_region": cred["region_name"],
        }
        uri = "s3://" + uri[len("gs://"):] if uri.startswith("gs://") else uri
    return lance.dataset(uri, storage_options=storage_options)


def load_captions(paths: list[str]) -> dict[str, str]:
    captions = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("caption"):
                        captions[row["id"]] = row["caption"]
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True)
    parser.add_argument("--captions", required=True, nargs="+")
    parser.add_argument("--column", required=True)
    parser.add_argument("--id_column", default="uuid")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = open_table(args.table)
    captions = load_captions(args.captions)
    ids = dataset.to_table(columns=[args.id_column]).column(args.id_column).to_pylist()
    matched = sum(i in captions for i in ids)
    unknown = len(set(captions) - set(ids))
    logger.info(f"{args.table}: {len(ids)} rows, {len(captions)} captions, {matched} matched, {unknown} caption ids not in table")
    if unknown:
        raise ValueError(f"{unknown} caption ids are not in the table")

    if args.column in dataset.schema.names:
        if not args.overwrite:
            raise ValueError(f"column {args.column} already exists, pass --overwrite to replace it")
        dataset.drop_columns([args.column])
        dataset = open_table(args.table)

    def fill(batch: pa.RecordBatch) -> pa.RecordBatch:
        values = [captions.get(i) for i in batch.column(args.id_column).to_pylist()]
        return pa.record_batch([pa.array(values, pa.string())], names=[args.column])

    dataset.add_columns(fill, read_columns=[args.id_column], batch_size=1024)
    dataset = open_table(args.table)
    written = dataset.to_table(columns=[args.id_column, args.column])
    pairs = zip(written.column(args.id_column).to_pylist(), written.column(args.column).to_pylist())
    mismatched = sum(captions.get(i) != c for i, c in pairs)
    filled = written.num_rows - written.column(args.column).null_count
    logger.info(f"wrote {args.column}: {filled}/{written.num_rows} rows non-null, {mismatched} mismatched, version {dataset.version}")
    if mismatched:
        raise ValueError(f"{mismatched} rows do not match the captions file")


if __name__ == "__main__":
    main()
