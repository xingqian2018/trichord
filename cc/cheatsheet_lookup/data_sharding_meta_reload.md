# Reload meta into an existing WebDataset

Use this when the WebDS shards already exist and the LanceDB table has been refreshed (new captions, new tags, etc.) — and you want to write **only a new meta** alongside the existing shards, without re-rendering images.

For each existing `{webds_path}/resolution_*/aspect_ratio_*/{old_meta_key}/<shard>.tar`, the script writes a sibling `{webds_path}/resolution_*/aspect_ratio_*/{new_meta_key}/<shard>.tar` whose entries mirror the original tar's per-entry order (keyed by `<uuid>.json`) and embed the latest LanceDB row.

## Some live information:

- `<table_postfix>` = `0429` (the current table postfix we are working on)
- `<new_meta_key>` = `metas_20260429` (the current table postfix we are working on)

## Quick-check inputs

| Arg | Value | Notes |
|---|---|---|
| `--lancedb_path` | `<lancedb_to_work_on>` | Refreshed Lance table URI. `.lance` suffix auto-appended; `gcs://` normalized to `gs://`. Default joint table: `gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance`. |
| `--webds_path` | `<the_existed_webdataset_path>` | Existing WebDS root to walk. The script does NOT create new shards — only writes sidecar tars beside discovered meta dirs. |
| `--webds_credential` | `credentials/gcs.secret` | Credential file for the WebDS bucket. |
| `--dataset_name` | `<dataset_name_that_to_process>` | Filter — only rows where `source_dataset == <dataset_name>` are loaded into the lookup. |
| `--webds_sample_filekey_type` | `<uuid_or_sdg_original_index>` | Lance column used to key the in-tar entry name back to a Lance row. Default `uuid`. Use `sdg_original_index` for the synthetic SDG datasets where filenames are SDG indices, not UUIDs. |
| `--old_meta_key` | `metas` | Existing meta subdir under each `aspect_ratio_*/` to read entry order from. |
| `--new_meta_key` | **ASK USER** | New sibling meta subdir to write. Common patterns: `metas_<date>`, `metas_<feature>` (e.g. `metas_recap_v2`). |
| `--num_concurrency` | `4` | Per-task concurrency for download/upload streams. |
| `--workers` | `4` | Outer worker thread count. |


## Known dataset lookup

| `--dataset_name` | `--lancedb_path` | `--webds_path` | `--webds_sample_filekey_type` |
|---|---|---|---|
| `screen2words_rico` | `gs://nv-00-10206-lancedb/prod/image/text_related/screen2words_rico_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_image_text_related/screen2words_rico/` | `uuid` |
| `slide_audit` | `gs://nv-00-10206-lancedb/prod/image/text_related/slide_audit_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_image_text_related/slide_audit/` | `uuid` |
| `voxel51_rico` | `gs://nv-00-10206-lancedb/prod/image/text_related/voxel51_rico_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_image_text_related/voxel51_rico/` | `uuid` |
| `zennodo10k` | `gs://nv-00-10206-lancedb/prod/image/text_related/zennodo10k_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_image_text_related/zennodo10k_slice/` | `uuid` |
| `synthetic_scene_text_v0` | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_scene_text_v0_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_scene_text_v0/` | `sdg_original_index` |
| `synthetic_chinese_scene_text_v0` | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_chinese_scene_text_v0/` | `sdg_original_index` |
| `synthetic_traditional_chinese_scene_text_v0` | `gs://nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_traditional_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_traditional_chinese_scene_text_v0/` | `sdg_original_index` |

When a row is filled in, the cheatsheet substitutes `--lancedb_path`, `--webds_path`, and `--webds_sample_filekey_type` directly — the user only has to supply `--new_meta_key`.

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 reload_meta_<dataset_name> \
    pipelines/image/text_rendering/shard_full_dbinfo_meta_reload.py \
    --lancedb_path <lancedb_to_work_on> \
    --webds_path <the_existed_webdataset_path> \
    --webds_credential credentials/gcs.secret \
    --logged_meta_credential credentials/gcs.secret \
    --dataset_name <dataset_name_that_to_process> \
    --webds_sample_filekey_type <uuid_or_sdg_original_index> \
    --old_meta_key metas \
    --new_meta_key <new_meta_key> \
    --num_concurrency 64 \
    --workers 32
```

## Notes
- Never run it directly. Show the command as a formatted response first and ask user's permission to run.
- Your command should follow the exact indent as the template shows.
- Entry order inside each rebuilt tar matches the existing `{old_meta_key}/<shard>.tar` exactly — only the JSON payload changes.
- UUIDs in the existing tar that are missing from the filtered LanceDB are **dropped** from the new tar, with a per-shard warning logging the missing count.
- Safe to re-run: the destination tars are overwritten on each run.
- No `--samples-per-shard` / `--max-rows` flags — shape is dictated by the existing webds.
- Image shards under `{old_meta_key}`'s sibling `images/` subdir are untouched.
