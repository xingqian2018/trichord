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
"""Ingest a caption column from slice tables into specific fragments of a main Lance table.

Follows the SILA first-publish pattern:
    1. add the column to the schema as an all-null nullable field (metadata only)
    2. write one data file per target fragment holding only that column, aligned to physical rows
    3. publish all files in one DataReplacement commit (touches only the target fragments)

Usage (run with the same pylance major version as the table's writers):
    python ingest_caption_column.py \
        --table gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
        --sources gs://.../poster_..._slice_from_maintable_20261114.lance ... \
        --column caption_sp_for_html_poster_v0 \
        --fragment_id 40992-40996 \
        --dry_run
"""

import argparse
import os
import time
import uuid as uuidlib

import lance
import pyarrow as pa
from lance.file import LanceFileWriter
from lance.fragment import DataFile
from loguru import logger


def parse_fragment_ids(spec: str) -> list[int]:
    ids = set()
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def storage_options() -> dict[str, str]:
    try:
        from cosmos_data.gcs_utils import setup_gcs_auth
        from cosmos_sila.core.utils import LANCE_STORAGE_OPTIONS
    except ImportError:
        return {}
    setup_gcs_auth()
    return dict(LANCE_STORAGE_OPTIONS)


def load_captions(sources: list[str], column: str, id_column: str, options: dict[str, str]) -> dict[str, str]:
    captions = {}
    for source in sources:
        table = lance.dataset(source, storage_options=options).to_table(columns=[id_column, column])
        ids = table.column(id_column).to_pylist()
        values = table.column(column).to_pylist()
        if any(v is None for v in values):
            raise ValueError(f"{source} has null {column}")
        overlap = set(ids) & set(captions)
        if overlap or len(set(ids)) != len(ids):
            raise ValueError(f"{source} has duplicate ids")
        captions.update(zip(ids, values))
        logger.info(f"{source}: {len(ids)} captions")
    return captions


def align_fragment(fragment: lance.LanceFragment, captions: dict[str, str], id_column: str) -> tuple[list, int]:
    values = [None] * fragment.physical_rows
    matched = 0
    live = 0
    for batch in fragment.scanner(columns=[id_column], with_row_address=True).to_batches():
        ids = batch.column(id_column).to_pylist()
        addresses = batch.column("_rowaddr").to_pylist()
        for row_id, address in zip(ids, addresses):
            live += 1
            caption = captions.get(row_id)
            if caption is not None:
                values[address & 0xFFFFFFFF] = caption
                matched += 1
    if matched != live:
        raise ValueError(f"fragment {fragment.fragment_id}: only {matched}/{live} live rows have a caption")
    return values, matched


def field_id(dataset: lance.LanceDataset, column: str) -> int:
    return {f.name(): f.id() for f in dataset.lance_schema.fields()}[column]


def fragment_has_field(dataset: lance.LanceDataset, fragment_id: int, fid: int) -> bool:
    return any(fid in f.field_ids() for f in dataset.get_fragment(fragment_id).data_files())


def write_data_file(table: str, column: str, values: list, version: str, options: dict[str, str]) -> tuple[str, int]:
    name = f"{uuidlib.uuid4().hex}-{column}.lance"
    schema = pa.schema([pa.field(column, pa.string())])
    writer = LanceFileWriter(f"{table.rstrip('/')}/data/{name}", schema, version=version, storage_options=options)
    for start in range(0, len(values), 4096):
        writer.write_batch(pa.record_batch([pa.array(values[start : start + 4096], pa.string())], schema=schema))
    writer.close()
    return name, len(values)


def snapshot(dataset: lance.LanceDataset, fragment_ids: list[int]) -> dict[int, tuple]:
    state = {}
    for fid in fragment_ids:
        fragment = dataset.get_fragment(fid)
        deletion = fragment.deletion_file()
        state[fid] = (fragment.physical_rows, str(deletion) if deletion else None)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True)
    parser.add_argument("--sources", required=True, nargs="+")
    parser.add_argument("--column", required=True)
    parser.add_argument("--fragment_id", required=True)
    parser.add_argument("--id_column", default="uuid")
    parser.add_argument("--file_version", default="2.1")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--commit_attempts", type=int, default=5)
    args = parser.parse_args()

    if int(os.environ.get("RANK", "0")) != 0:
        return
    logger.info(f"pylance {lance.__version__}")
    options = storage_options()
    fragment_ids = parse_fragment_ids(args.fragment_id)
    captions = load_captions(args.sources, args.column, args.id_column, options)

    dataset = lance.dataset(args.table, storage_options=options)
    logger.info(f"{args.table}: version {dataset.version}, data storage {dataset.data_storage_version}")
    aligned = {}
    for fid in fragment_ids:
        fragment = dataset.get_fragment(fid)
        if fragment is None:
            raise ValueError(f"fragment {fid} not found")
        aligned[fid], matched = align_fragment(fragment, captions, args.id_column)
        logger.info(f"fragment {fid}: {fragment.physical_rows} physical rows, {matched} captions")
    total = sum(sum(v is not None for v in values) for values in aligned.values())
    if total != len(captions):
        raise ValueError(f"{len(captions) - total} source captions are not in the target fragments")

    exists = args.column in dataset.schema.names
    if exists:
        fid_existing = field_id(dataset, args.column)
        written = [f for f in fragment_ids if fragment_has_field(dataset, f, fid_existing)]
        if written:
            raise ValueError(f"{args.column} already physically present on fragments {written}")
    logger.info(f"plan: column {args.column} ({'exists, all-null' if exists else 'new'}), {total} captions into fragments {fragment_ids}")
    if args.dry_run:
        logger.info("dry run, nothing written")
        return

    if not exists:
        dataset.add_columns(pa.field(args.column, pa.string(), nullable=True))
        dataset = lance.dataset(args.table, storage_options=options)
        logger.info(f"added all-null {args.column}, version {dataset.version}")
    fid = field_id(dataset, args.column)
    major, minor = (int(x) for x in args.file_version.split("."))
    read_state = snapshot(dataset, fragment_ids)
    read_version = dataset.version

    groups = []
    for frag_id in fragment_ids:
        name, rows = write_data_file(args.table, args.column, aligned[frag_id], args.file_version, options)
        data_file = DataFile(name, [fid], [0], major, minor)
        groups.append(lance.LanceOperation.DataReplacementGroup(frag_id, data_file))
        logger.info(f"fragment {frag_id}: wrote {name} ({rows} rows)")

    for attempt in range(1, args.commit_attempts + 1):
        try:
            lance.LanceDataset.commit(
                args.table,
                lance.LanceOperation.DataReplacement(groups),
                read_version=read_version,
                storage_options=options,
            )
            break
        except Exception as error:
            logger.warning(f"commit attempt {attempt} failed: {error}")
            if attempt == args.commit_attempts:
                raise
            time.sleep(10 * attempt)
            dataset = lance.dataset(args.table, storage_options=options)
            if snapshot(dataset, fragment_ids) != read_state:
                raise RuntimeError("target fragments changed since read, rerun the script") from error
            if any(fragment_has_field(dataset, f, fid) for f in fragment_ids):
                raise RuntimeError("column appeared on target fragments concurrently") from error
            read_version = dataset.version

    dataset = lance.dataset(args.table, storage_options=options)
    mismatched = 0
    filled = 0
    for frag_id in fragment_ids:
        table = dataset.get_fragment(frag_id).to_table(columns=[args.id_column, args.column])
        for row_id, value in zip(table.column(args.id_column).to_pylist(), table.column(args.column).to_pylist()):
            filled += value is not None
            mismatched += value != captions.get(row_id)
    logger.info(f"committed version {dataset.version}: {filled} non-null in target fragments, {mismatched} mismatched")
    if mismatched:
        raise RuntimeError(f"{mismatched} rows mismatched after commit")


if __name__ == "__main__":
    main()
