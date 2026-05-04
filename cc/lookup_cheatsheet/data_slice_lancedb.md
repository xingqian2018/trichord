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
slaunch cpu 4x1 slice_<dataset_name> \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --output_lancedb_path gs://nv-00-10206-lancedb/prod/image/text_related/<dataset_name>.lance \
    --dataset_name <dataset_name>
```

## Template — multiple slices in one pass

`--dataset_name` and `--output_lancedb_path` both take `nargs="+"` and are paired positionally (1st ↔ 1st, 2nd ↔ 2nd, …). Lengths must match; duplicate dataset names are rejected.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x8 slice_multi \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --dataset_name dsetA dsetB dsetC \
    --output_lancedb_path \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetA.lance \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetB.lance \
        gs://nv-00-10206-lancedb/prod/image/text_related/dsetC.lance
```

## Arguments

That's it — only three. Everything else is a module-level constant.

- `--input_lancedb_path` *(required)* — source Lance URI (use `gs://...`). `.lance` suffix is auto-appended. The input must have a `source_dataset` column.
- `--dataset_name` *(required, `nargs="+"`)* — one or more `source_dataset` values to slice. Duplicates are rejected.
- `--output_lancedb_path` *(required, `nargs="+"`)* — one or more output Lance URIs, paired positionally with `--dataset_name`. `len(...)` must match `--dataset_name`. Existing datasets at these URIs are wiped at startup.

## Module constants (edit the script to change)

- `MAX_ROW_PER_FILE = 100_000` — both the segment size used by rank 0 to decide when to flush a full chunk **and** the `max_rows_per_file` passed to `lance.write_dataset`. Each Lance write is exactly this many rows.

## Architecture

```
                       per-rank read           collective                rank 0 only
                       (1 fragment / round)    sync                      drain
   ┌─ rank 0 ─────► LanceDBBuffer ──┐                                  ┌─ pop_fullsize_segment
   │   reads its                    │                                  │     (MAX_ROW_PER_FILE @ a time)
   │   fragment_ids                 │                                  │
   ├─ rank 1 ─────► LanceDBBuffer ──┼──► gather_object on rank 0 ──►  ─┤   ─► lance.write_dataset(...)
   │                                │       (in strict rank order)     │      mode=create | append
   ├─ rank 2 ─────► LanceDBBuffer ──┤                                  │      max_rows_per_file=
   │     ...                        │                                  │       MAX_ROW_PER_FILE
   └─ rank N-1 ───► LanceDBBuffer ──┘                                  └─ remainder kept in buffer
                                                                          for next round
```

Per-rank task schedule (every rank runs the same number of `sync` tasks → collective stays in lockstep):

```
<read fragment 0, sync, read fragment 1, sync, ..., read fragment K-1, sync, sync, sync ...>
                                                                       └─── padding syncs
                                                                            on ranks with
                                                                            fewer fragments
```

`n_rounds = ceil(N_fragments / world_size)`. Ranks short by ±1 fragment pad with sync-only rounds at the tail.

Why per-segment writes are exactly `MAX_ROW_PER_FILE`:
- `pop_fullsize_segment()` on rank 0 picks the webdsname with the most buffered rows and pops exactly `MAX_ROW_PER_FILE` of them, leaving the remainder. So every mid-run Lance write is one file of exactly `MAX_ROW_PER_FILE` rows.
- The final flush handles only the trailing under-`MAX_ROW_PER_FILE` remainder per webdsname, written once at end.

## Notes

- **Distributed.** Use `slaunch cpu 1xN` to scale read/sync throughput. Reading scales near-linearly in N (each rank handles its own fragments); writing is rank 0 only. With many small fragments and few ranks the read side dominates, so 1x8 or 1x16 is usually a sweet spot.
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

### Common Image LanceDB Root Path:

`<root>` = `gs://nv-00-10206-vfm/lancedb/image/`

### Text dataset (Real + SGD)

| `source_dataset`                              | Output Lance URI                                                                                                 |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `screen2words_rico`                           | `<root>/regular_text/screen2words_rico_slice_from_maintable_<YYYYmmdd>.lance/`                                   |
| `slide_audit`                                 | `<root>/regular_text/slide_audit_slice_from_maintable_<YYYYmmdd>.lance/`                                         |
| `voxel51_rico`                                | `<root>/regular_text/voxel51_rico_slice_from_maintable_<YYYYmmdd>.lance/`                                        |
| `zennodo10k`                                  | `<root>/regular_text/zennodo10k_slice_from_maintable_<YYYYmmdd>.lance/`                                          |
| `synthetic_scene_text_v0`                     | `<root>/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/`                     |
| `synthetic_chinese_scene_text_v0`             | `<root>/synthetic_text/synthetic_chinese_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/`             |
| `synthetic_traditional_chinese_scene_text_v0` | `<root>/synthetic_text/synthetic_traditional_chinese_scene_text_v0_slice_from_maintable_<YYYYmmdd>.lance/` |

### Regular dataset (Real)

| `source_dataset` | Output Lance URI                                                  |
|------------------|-------------------------------------------------------------------|
| `red`            | `<root>/regular/red_slice_from_maintable_<YYYYmmdd>.lance/`       |
| `coyo_700m`      | `<root>/regular/coyo_700m_slice_from_maintable_<YYYYmmdd>.lance/` |

### Synthetic dataset (SGD)

| `source_dataset`                                 | Output Lance URI                                                                                         |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `MMC4`                                           | `<root>/synthetic/mmc4_slice_from_maintable_<YYYYmmdd>.lance/`                                           |
| `generations_qwen_image_2512_filtered_photoreal` | `<root>/synthetic/generations_qwen_image_2512_filtered_photoreal_slice_from_maintable_<YYYYmmdd>.lance/` |
| `wordnet_captions_20260224`                      | `<root>/synthetic/wordnet_captions_20260224_slice_from_maintable_<YYYYmmdd>.lance/`                      |
| `datacomp_1b`                                    | `<root>/synthetic/datacomp_1b_slice_from_maintable_<YYYYmmdd>.lance/`                                    |
| `midjourney`                                     | `<root>/synthetic/midjourney_slice_from_maintable_<YYYYmmdd>.lance/`                                     |
| `midjourney_v6_20240703`                         | `<root>/synthetic/midjourney_v6_20240703_slice_from_maintable_<YYYYmmdd>.lance/`                         |

Naming convention: `<dataset>_slice_from_maintable_<YYYYmmdd>.lance`. When you re-cut from a fresher main table, bump the date suffix — don't overwrite the previous slice in place.
