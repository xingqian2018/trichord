# LanceDB Operations

Scripts live in `pipelines/image/text_rendering/` in `imaginaire4_sila`.

## `lancedb_count.py` — count rows, optionally filtered / broken down by dataset

- Scans every fragment in parallel across ranks, and for each row checks two things:
  - **total**: does it match `_TOTAL_FILTER` (only set if you restrict `IMAGE_CAPTION_V2_SOURCE_DATASETS` to specific dataset names — otherwise every row counts as "total")
  - **count**: does it also pass `DEFAULT_QUALITY_FILTER` (the `filtering_qwen3vl_fft_8b_v1_*` columns — aesthetic score, collage/nsfw/watermark/white-background flags)

- Results are synced and accumulated per-`source_dataset`, and written as JSON with `summary` (grand totals) and `details` (per-dataset breakdown).
- The run is resumable: if `--output_status_json` already exists, it reads `latest_fragment_id` and only processes fragments after that.
- Output stats_<YYYYMMDD>.json, need to auto fill with the current date.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 32x4 lancedb_count_<YYYYMMDD> \
    pipelines/image/text_rendering/lancedb_count.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --lancedb_credential credentials/gcs.secret \
    --output_status_json s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/image_data_stats/stats_<YYYYMMDD>.json \
    --output_credential credentials/gcs.secret
```

Notes:
- `--input_lancedb_path` is a `gs://` Lance URI. Not `s3://`
- Re-running with the same `--output_status_json` resumes from `latest_fragment_id` instead of recounting from scratch.

### Filter and Display Stat on a set of Dataset:

We only care about the following dataset:

```PYTHON
IMAGE_CAPTION_V2_SOURCE_DATASETS: list[str] = [
    # Real data
    "datacomp_12b",
    "coyo_700m",
    "nvcommercial_700m",
    "MMC4",
    "pexels_residual_trustedK1_v2",
    "human_sft",
    "laion_aesthetic",
    # Real data not used but need captioning
    "red",
    # Synthetic data
    "generations_qwen_image_2512_filtered_photoreal",
    "self_improving_synthetic_2026-02-09",
    "wordnet_captions_20260224",
    "gemini_3_pro_image_200k",
    "gpt_image_2_20260507",
    "gpt_image_2_20260515",
    # Real text data
    "voxel51_rico",
    "screen2words_rico",
    "slide_audit",
    "zennodo10k",
    # Synthetic text data
    "synthetic_scene_text_v0",
    "synthetic_chinese_scene_text_v0",
    "synthetic_traditional_chinese_scene_text_v0",
]
```

## `lancedb_sampler_datacomp_12b.py` — spot-check the datacomp_12b face-image fix

Purpose: one-off diagnostic script (no `slaunch`/distributed needed) to visually verify whether the `datacomp12b_face_image_s3_range` fix (a recovered image replacing a broken original for `datacomp_12b` rows) actually resolves the data issue.

What it does:
- Scans `image_meta_table_full.lance` fragment-by-fragment (not distributed — single process) filtering `source_dataset = 'datacomp_12b' AND datacomp12b_face_image_s3_range IS NOT NULL AND <caption_col> IS NOT NULL`, stopping once `SAMPLE_LIMIT` (default 100) samples are collected. `SKIP_PROB` (default 0.9) randomly drops otherwise-valid rows so the samples spread across the scan instead of clustering in the first few fragments.
- For each sampled row, downloads **both** the original image (`image_s3_range`) and the corrected image (`datacomp12b_face_image_s3_range`) via a direct byte-range GET (translates the PBSS `s3r://` URI to the migrated GCS bucket/path inline — see `_PBSS_TO_GCS`), plus the row's caption text.
- Uploads `<uuid>.<ext>` (original), `<uuid>_fixed.<ext>` (corrected), and `<uuid>.txt` (caption) to `OUTPUT_S3_PATH` so the original/fixed pair can be eyeballed side by side.

When to use it:
- After a data-fix pass touches `datacomp12b_face_image_s3_range`, to sanity-check a handful of before/after image pairs before trusting the fix at scale.
- Not for counting/coverage — use `lancedb_count.py` for that.

Single-process script — prefer `slaunch cpu 1x1` over running it directly, so it gets a proper job/log on the cluster:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 lancedb_sampler_datacomp_12b \
    pipelines/image/text_rendering/lancedb_sampler_datacomp_12b.py
```

Config is all hardcoded at the top of the file — edit before running:
- `SAMPLE_LIMIT` — how many corrected rows to sample (default 100)
- `SKIP_PROB` — probability of skipping a valid row, to spread samples across fragments (default 0.9)
- `SKIP_TO_FRAGMENT` — set > 0 to resume/skip the first N fragments
- `OUTPUT_S3_PATH` — where the original/fixed/caption files land (default `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/datacomp_12b_sample_exam`)
- `CAP_COL` — which caption column must be non-null for a row to count

Output layout under `OUTPUT_S3_PATH/`: `<uuid>.<ext>` (original), `<uuid>_fixed.<ext>` (corrected), `<uuid>.txt` (caption) — one triplet per sampled row.


## `lancedb_source_fragid_mapping.py` — map fragment_id ↔ source_dataset

Purpose: distributed scan that builds a bidirectional mapping between Lance **fragment IDs** and the `source_dataset` value(s) they contain. Useful for figuring out which fragment range(s) belong to a given dataset (e.g. to target a `--fragment_group_size`-style resume/rerun at just one dataset) or, conversely, which datasets live in a given fragment.

- Scans every fragment (one fragment at a time, `columns=["source_dataset"]` only — no row-level filter) across ranks, collecting the **distinct** set of `source_dataset` values seen per fragment.
- After sync, rank 0 inverts that into `dataset_to_fragments`: per dataset, `min_fragment_id`, `max_fragment_id`, `num_fragments`, whether the fragment IDs are `contiguous`, and a compressed `fragment_ids` range string (e.g. `"0-2, 300-301, 566, 789-790"`).
- Output JSON has both directions: `dataset_to_fragments` (summary per dataset) and `fragment_to_datasets` (raw `{fragment_id: [source_dataset, ...]}` mapping — a fragment can span multiple datasets if writes interleaved).
- Resumable the same way as `lancedb_count.py`: if `--output_status_json` already exists, resumes from its `latest_fragment_id`.

When to use it:
- To find the fragment range for one dataset (e.g. to hand-scope a targeted `lancedb_count.py`/`slice_lancedb.py` run instead of scanning the whole table).
- To check whether a dataset's fragments are contiguous (`contiguous: true`) — useful context before writing fragment-range-based tooling (e.g. `lancedb_sampler_datacomp_12b.py`'s `SKIP_TO_FRAGMENT`).

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 32x4 lancedb_fragid_mapping_<YYYYMMDD> \
    pipelines/image/text_rendering/lancedb_source_fragid_mapping.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --lancedb_credential credentials/gcs.secret \
    --output_status_json s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/image_data_stats/fragid_mapping_<YYYYMMDD>.json \
    --output_credential credentials/gcs.secret
```

Notes:
- `--input_lancedb_path` is a `gs://` Lance URI, not `s3://`.
- `--fragment_group_size` defaults to 32 here (vs. 8 for `lancedb_count.py`) since this scan is cheap (single column, no per-row quality/dedup/caption checks).

## `lancedb_show_schema.py` — quick schema/metadata inspection

Purpose: a small standalone script (no `slaunch`/distributed needed) that opens a Lance dataset and prints:
- source URI, dataset version, total row count, fragment count
- every column name + its Arrow type, in a simple table

When to use it:
- Before writing a filter expression (e.g. for `lancedb_count.py` or `slice_lancedb.py`), check exact column names and types.
- Quickly confirm a Lance table exists and how many rows/fragments it has, without a cluster job.

Run it locally (edit `TABLE_URI` at the top of the file, or copy the pattern into a one-off script):

```bash
cd /home/xingqianx/Project/imaginaire4_sila
python pipelines/image/text_rendering/lancedb_show_schema.py
```

`TABLE_URI` is currently hardcoded in the script — update it to point at the table you want to inspect before running, e.g.:

```python
TABLE_URI = "gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260504.lance"
```
