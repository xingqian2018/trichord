# Slice a LanceDB table by `source_dataset`

Use this when you have a big joint LanceDB table (e.g. `image_meta_table_full.lance`) and want to extract one or more per-dataset sub-tables, written out as new Lance datasets.

The script does **single scan, multi output** — one read pass over the input fan-outs into N independent output `.lance` datasets, paired by position with the dataset names. Streaming end-to-end: rows flow scanner → bounded queue → writer, with `--max_rows_per_file` controlling output file rotation. No full materialization.

Lives at `pipelines/image/text_rendering/slice_lancedb.py` in `imaginaire4_sila`.

## When to use

- You only need a subset of `source_dataset` values from a giant joint Lance table for downstream sharding / re-captioning / debugging.
- You want to materialize that subset as its own `.lance` so subsequent steps (e.g. `shard_full_dbinfo.py`) don't waste IO scanning rows they'll throw away.
- Bonus: extracting **several** subsets in one go — the input is only scanned once across all of them.

## Template — single slice

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 slice_<dataset_name> \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --output_lancedb_path gs://nv-00-10206-lancedb/prod/image/text_related/<dataset_name>.lance \
    --dataset_name <dataset_name> \
    --max_concurrency 32 \
    --batch_size 4096 \
    --max_rows_per_file 100000
```

## Template — multiple slices in one pass

`--dataset_name` and `--output_lancedb_path` both take `nargs="+"` and are paired positionally (1st ↔ 1st, 2nd ↔ 2nd, …). Lengths must match.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 slice_multi \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --dataset_name dsetA dsetB dsetC \
    --output_lancedb_path \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetA.lance \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetB.lance \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetC.lance \
    --max_concurrency 32 \
    --batch_size 4096
```

## Arguments

- `--input_lancedb_path` *(required)* — source Lance URI (use `gs://...`). `.lance` suffix is auto-appended. The input must have a `source_dataset` column (script errors out otherwise).
- `--output_lancedb_path` *(required, `nargs="+"`)* — one or more output Lance URIs, paired with `--dataset_name`.
- `--dataset_name` *(required, `nargs="+"`)* — one or more `source_dataset` values to slice. Duplicates are rejected; `len(dataset_name) == len(output_lancedb_path)` is enforced.
- `--max_concurrency` *(default 256)* — number of parallel scanner threads. Each owns one fragment per task; work-stealing across threads. Bump higher (e.g. 64) on cloud storage if your bandwidth allows.
- `--batch_size` *(default 100000)* — rows per scanner batch. Larger → fewer Python boundary crossings but coarser progress / higher peak memory per batch. 4096 is a good cloud-storage working point.
- `--max_rows_per_file` *(default 100000)* — rows per output Lance datafile. Lance rotates datafiles when this fills. Drop (e.g. 10k) if you want more downstream parallelism on the read; raise (e.g. 1M) for fewer files on huge slices.

## Architecture / why it's fast

```
                          ┌──────────► queue[A] ──► writer A ──► output_lance_A
N scanner threads ──split─┼──────────► queue[B] ──► writer B ──► output_lance_B
                          └──────────► queue[C] ──► writer C ──► output_lance_C
```

- Scanner-side parallelism: `--max_concurrency` threads each open their own `lance.dataset(...)` and run `scanner(filter="source_dataset IN (...)", fragments=[one_fragment], scan_in_order=False)`. True per-fragment IO parallelism.
- Per-batch routing: each scanner splits batches by `source_dataset` value (via `pyarrow.compute.equal` masks) and pushes sub-batches to per-output queues.
- Stream write: each output runs **one** `lance.write_dataset(reader=…)` call backed by a `pa.RecordBatchReader` over its queue. Lance pulls lazily and rotates datafiles per `--max_rows_per_file`. Read and write happen concurrently (Lance / pyarrow release the GIL during S3/GCS IO).
- Bounded queues (size `2 × max_concurrency`) provide backpressure both ways — scanners stall when writers are slow, writers stall when scanners are slow.

## Notes

- **Empty slices are tolerated.** If a particular `dataset_name` matches zero rows, the corresponding writer logs a warning and skips creating an output (no empty `.lance`). The job only fails if **all** slices come up empty.
- **Schema is preserved exactly** — no column projection. The output schema matches the input schema.
- **No `--max-rows` flag.** Slicing is supposed to be exhaustive over matching rows; if you want a sample, post-process with another scanner.
- **Single rank.** This script does not use `slaunch`'s distributed init; one task is enough since parallelism is internal. Do not request `1xN`.
- **Verification.** The completion log re-opens each output and reports `Rows written / Output rows / Datafiles / URI` per slice — quick sanity check that rotation and counts match.

## Default input — the big joint table

Unless otherwise specified, the table being sliced is the canonical joint image table:

```
gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance
```

Use this as `--input_lancedb_path` by default. Override only when slicing from a custom / experimental main table.

## Known slice outputs (2026-04-27 cut from `image_meta_table_full.lance`)

These are the canonical per-dataset slices that have already been carved out of the joint table. Reuse these URIs as `--input-lancedb-path` for downstream sharding instead of re-running the slice.

| `source_dataset` | Output Lance URI |
|---|---|
| `screen2words_rico` | `gs://nv-00-10206-lancedb/prod/image/text_related/screen2words_rico_slice_from_maintable_<YYYYmmdd>.lance/` |
| `slide_audit`       | `gs://nv-00-10206-lancedb/prod/image/text_related/slide_audit_slice_from_maintable_<YYYYmmdd>.lance/` |
| `voxel51_rico`      | `gs://nv-00-10206-lancedb/prod/image/text_related/voxel51_rico_slice_from_maintable_<YYYYmmdd>.lance/` |
| `zennodo10k`        | `gs://nv-00-10206-lancedb/prod/image/text_related/zennodo10k_slice_from_maintable_<YYYYmmdd>.lance/` |
| `synthetic_scene_text_v0`                     | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/` |
| `synthetic_chinese_scene_text_v0`             | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_chinese_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/` |
| `synthetic_traditional_chinese_scene_text_v0` | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_traditional_chinese_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/` |

Naming convention: `<dataset>_slice_from_maintable_<MMDD>.lance`. When you re-cut from a fresher main table, bump the date suffix — don't overwrite the previous slice in place.

## Quick example: extract `zennodo10k` only

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 slice_zennodo10k \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --output_lancedb_path gs://nv-00-10206-lancedb/prod/image/text_related/zennodo10k.lance \
    --dataset_name zennodo10k \
    --max_concurrency 32 \
    --batch_size 4096
```

After this completes, the resulting `zennodo10k.lance` is a drop-in input for `shard_full_dbinfo.py` (see `data_sharding_to_webdataset` cheatsheet) — and the downstream sharder no longer wastes a scan on every other dataset in the joint table.
