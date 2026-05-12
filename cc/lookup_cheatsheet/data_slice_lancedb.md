# Slice a LanceDB table by `source_dataset`

Use this when you have a big joint LanceDB table (e.g. `image_meta_table_full.lance`) and want to extract one or more per-dataset sub-tables, written out as new Lance datasets.

The script does **distributed single-pass slicing**: every rank scans a disjoint shard of input fragments, filters by `source_dataset IN (...)`, and accumulates rows in a per-rank `LanceDBBuffer` keyed by source_dataset. Every read round is followed by a collective `sync` that gathers all ranks' rows onto rank 0; rank 0 then drains as many full `MAX_ROW_PER_FILE`-sized segments as it can per output Lance. After the loop, rank 0 flushes any remainder rows.

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
    --lancedb_credential credentials/gcs.secret
```

## Template — multiple slices in one pass

`--dataset_name` and `--output_lancedb_path` both take `nargs="+"` and are paired positionally (1st ↔ 1st, 2nd ↔ 2nd, …). Lengths must match; duplicate dataset names are rejected.

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
    --lancedb_credential credentials/gcs.secret
```

## Arguments

Only four — everything else is a module-level constant.

- `--input_lancedb_path` *(required)* — source Lance URI (use `gs://...`). `.lance` suffix is auto-appended; `gcs://` is normalized to `gs://`. The input must have a `source_dataset` column.
- `--output_lancedb_path` *(required, `nargs="+"`)* — one or more output Lance URIs, paired positionally with `--dataset_name`. `len(...)` must match `--dataset_name`. Existing datasets at these URIs are wiped at startup.
- `--dataset_name` *(required, `nargs="+"`)* — one or more `source_dataset` values to slice. Duplicates are rejected.
- `--lancedb_credential` *(default `credentials/gcs.secret`)* — credential file passed into `setup_msc({"lancedb": ...})` for both the input read and the output writes.

## Module constants (edit the script to change)

- `MAX_ROW_PER_FILE = 100_000` — both the segment size used by rank 0 to decide when to flush a full chunk **and** the `max_rows_per_file` passed to `lance.write_dataset`. Each Lance write is exactly this many rows.
- `FRAGMENT_GROUPN = 8` — number of input fragments grouped into one read task. Ranks process disjoint groups of this size.

## Notes

- **Default to `cpu 1x1`. Do NOT scale to multi-worker.** A single segment read can take a long time, and while one rank is still reading the others stall at the next collective `sync` (`gather_object`). With multiple ranks the idle ranks frequently hit the collective timeout and the whole job dies. Stick with `1x1` unless you've verified read durations are uniform.
- **Output cleanup at startup.** Rank 0 calls `clean_lance_path` on every output URI before any task runs (recursive `fs.rm` via `fsspec`). Anything previously at those paths is gone.
- **Mode selection.** First write per output URI in this run uses `mode="create"`; subsequent writes use `mode="append"`. Implemented by probing `lance.dataset(uri)` — once cleanup wipes the path, the first probe fails so the first write is `create`.
- **Empty slices.** If a `dataset_name` matches zero rows, no output dataset is ever created at the corresponding URI — the buffer simply never gets a key for that name. No empty `.lance` placeholder.
- **Schema.** Preserved exactly — all columns from the input row are forwarded. Per-segment `pa.Table.from_pylist(rows)` is used (schema inferred per write); if you ever hit "schema mismatch" on append, the inferred type drifted between segments — fix by reading the input dataset's schema once and passing it explicitly.
- **Sync count must match across ranks.** `LanceDBBuffer.sync()` calls `distributed.gather_object`, which is collective. The schedule (`n_rounds` syncs on every rank) guarantees this; don't break it by editing the task list.
- **Failure swallowing.** A failing read task is logged and the loop continues — the matching sync still fires so the collective doesn't deadlock, but those rows are lost. Treat persistent task-failed log lines as a real bug.

## Default input — the big joint table

Unless otherwise specified, the table being sliced is the canonical joint image table:

```
gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance
```

Use this as `--input_lancedb_path` by default. Override only when slicing from a custom / experimental main table.

## Known slice outputs (2026-04-27 cut from `image_meta_table_full.lance`)

These are the canonical per-dataset slices that have already been carved out of the joint table. Reuse these URIs as `--input_lancedb_path` for downstream sharding instead of re-running the slice.

- **Check [`./data_common_root.md`](./data_common_root.md) for common root path settings** — the `<lancedb_image_root>` alias used below is defined there.
- `<table_postfix>` = `slice_from_maintable_YYYYmmdd` (e.g. `slice_from_maintable_20260506` for the latest cut). Bump the date when re-cutting from a fresher main table rather than overwriting in place.

### Text dataset (Real + SGD)

| `source_dataset`                              | Output Lance URI                                                                                         |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `screen2words_rico`                           | `<lancedb_image_root>/regular_text/screen2words_rico_<table_postfix>.lance/`                             |
| `slide_audit`                                 | `<lancedb_image_root>/regular_text/slide_audit_<table_postfix>.lance/`                                   |
| `voxel51_rico`                                | `<lancedb_image_root>/regular_text/voxel51_rico_<table_postfix>.lance/`                                  |
| `zennodo10k`                                  | `<lancedb_image_root>/regular_text/zennodo10k_<table_postfix>.lance/`                                    |
| `synthetic_scene_text_v0`                     | `<lancedb_image_root>/synthetic_text/synthetic_scene_text_v0_<table_postfix>.lance/`                     |
| `synthetic_chinese_scene_text_v0`             | `<lancedb_image_root>/synthetic_text/synthetic_chinese_scene_text_v0_<table_postfix>.lance/`             |
| `synthetic_traditional_chinese_scene_text_v0` | `<lancedb_image_root>/synthetic_text/synthetic_traditional_chinese_scene_text_v0_<table_postfix>.lance/` |

### Regular dataset (Real)

| `source_dataset` | Output Lance URI                                                |
|------------------|-----------------------------------------------------------------|
| `red`            | `<lancedb_image_root>/regular/red_<table_postfix>.lance/`       |
| `coyo_700m`      | `<lancedb_image_root>/regular/coyo_700m_<table_postfix>.lance/` |

### Synthetic dataset (SGD)

| `source_dataset`                                 | Output Lance URI                                                                                       |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `MMC4`                                           | `<lancedb_image_root>/synthetic/mmc4_<table_postfix>.lance/`                                           |
| `generations_qwen_image_2512_filtered_photoreal` | `<lancedb_image_root>/synthetic/generations_qwen_image_2512_filtered_photoreal_<table_postfix>.lance/` |
| `wordnet_captions_20260224`                      | `<lancedb_image_root>/synthetic/wordnet_captions_20260224_<table_postfix>.lance/`                      |
| `datacomp_1b`                                    | `<lancedb_image_root>/synthetic/datacomp_1b_<table_postfix>.lance/`                                    |
| `midjourney`                                     | `<lancedb_image_root>/synthetic/midjourney_<table_postfix>.lance/`                                     |
| `midjourney_v6_20240703`                         | `<lancedb_image_root>/synthetic/midjourney_v6_20240703_<table_postfix>.lance/`                         |
