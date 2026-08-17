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
