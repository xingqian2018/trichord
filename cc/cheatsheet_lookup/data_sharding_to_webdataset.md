# Shard a LanceDB table into WebDataset

Use this when you have a LanceDB table and need to convert it into sharded WebDataset tar files for downstream training/loading.

The job runs on CPU via `slaunch` (a single 1x1 task is enough — it parallelizes internally with `--workers`).

## Which script?

Three scripts live side-by-side in `pipelines/image/text_rendering/`:

- **`shard_logged_image.py`** — create webdataset to a new location, shards only `logged_images` and `logged_metas` with some ID-related lanceDB columes; ignores the rest of the LanceDB columes.
- **`shard_full_dbinfo.py`** — create webdataset to a new location, shards **images + full LanceDB info** into one combined meta per sample. Use when downstream needs the table entries (captions, tags, source URLs, etc.) alongside the image.
- **`shard_full_dbinfo_meta_reload.py`** — does **not** re-shard images. Walks an existing webds root and writes a **sidecar meta tar** per shard, embedding fresh LanceDB row info under a new top-level JSON key. Use when the images are already sharded and you only want to append/refresh a meta field (e.g. new captions, new tags) without re-rendering the dataset.

Ask user which type of sharding the user want.

## Template — `shard_logged_image.py` (logged_images + logged_metas only)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 shard_<dataset_name> \
    pipelines/image/text_rendering/shard_logged_image.py \
    --input-lancedb-path <gs_or_s3_lancedb_path>.lance \
    --output-webds-path <bucket>/<prefix>/<dataset_name> \
    --s3-profile <profile> \
    --samples-per-shard 1000 \
    --workers 16
```

## Template — `shard_full_dbinfo.py` (images + dbinfo)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 shard_<dataset_name> \
    pipelines/image/text_rendering/shard_full_dbinfo.py \
    --input-lancedb-path <gs_lancedb_path> \
    --dataset-name <dataset_name> \
    --output-webds-path <bucket>/<prefix>/<dataset_name> \
    --s3-profile <profile> \
    --samples-per-shard 1000 \
    --workers 16
```

Notes for `shard_full_dbinfo.py`:
- `--dataset-name` is a **filter** on the `source_dataset` column — only rows where `source_dataset == <dataset_name>` are sharded. Typical when running against the joint table `image_meta_table_full.lance`.
- The input table is expected to expose `image_s3_range`, `meta_s3_range`, `height`, `width`, etc.
- Output is bucketed by resolution and aspect ratio: `{output_webds_path}/resolution_<tag>/aspect_ratio_<tag>/{image,meta}/<shard>.tar`.
- Optional `--max-rows N` caps input rows — use for dev/testing.

## Arguments

- `--input-lancedb-path`: full URI to the source `.lance` table (e.g. `gs://nv-00-10206-lancedb/prod/image/text_related/zennodo10k.lance`).
- `--output-webds-path`: destination bucket + prefix where shards will be written. The profile resolves the storage backend.
- `--s3-profile`: matching profile in `~/.s3cfg` / s3_omni config — `team-gcs` for GCS buckets, `pbss` / `aws` for others. Defaults to `team-gcs` in `shard_full_dbinfo.py`.
- `--samples-per-shard`: number of records per `.tar` shard. 1000 is the default working point; bump up for small samples, down for large ones.
- `--workers`: parallel writer workers per task. Script default is 4; 16 is a good working point on the `cpu` queue.
- `--dataset-name` *(shard_full_dbinfo only)*: filter — only rows where `source_dataset == <dataset_name>` are sharded out of the input table.
- `--max-rows` *(shard_full_dbinfo only, optional)*: cap on input rows to process — handy for dev/testing.
- Our joint image lanceDB `--input-lancedb-path` is `gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance`, so by default we work with that.

## Template — `shard_full_dbinfo_meta_reload.py` (webdataset meta recreation)

Use this when the WebDS shards already exist and you only want to **refresh / add a new meta tar** alongside the existing meta subdir — e.g. after the LanceDB has been re-augmented with new caption fields and you want them surfaced into the webds layout without re-sharding the images.

For each existing `{root}/resolution_*/aspect_ratio_*/{old_meta_key}/<shard>.tar`, the script writes a sibling `{root}/resolution_*/aspect_ratio_*/{new_meta_key}/<shard>.tar` whose entries mirror the original tar's per-entry order (keyed by `<uuid>.json`) and embed the latest LanceDB row.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 reload_meta_<dataset_name> \
    pipelines/image/text_rendering/shard_full_dbinfo_meta_reload.py \
    --input_lancedb_path <gs_lancedb_path> \
    --output_webds_path <bucket>/<prefix>/<dataset_name> \
    --dataset_name <dataset_name> \
    --webds_credential credentials/gcp_checkpoint.secret \
    --s3_profile team-gcs \
    --old_meta_key metas \
    --new_meta_key metas_new \
    --num_concurrency 4 \
    --workers 16
```

### Arguments

- `--input_lancedb_path` *(required)*: Lance table URI to read fresh per-uuid metadata from. `.lance` suffix is auto-appended; `gcs://` is normalized to `gs://`.
- `--output_webds_path` *(required)*: existing WebDS root to walk. The script does **not** create new shards; it only writes sidecar tars beside discovered meta dirs.
- `--dataset_name` *(required)*: filter — only rows where `source_dataset == <dataset_name>` are loaded into the uuid lookup.
- `--webds_credential` *(default `credentials/gcp_checkpoint.secret`)*: credential file for the WebDS bucket.
- `--s3_profile` *(default `team-gcs`)*: S3/GCS profile name. Must allow overwrite/delete on the destination.
- `--old_meta_key` *(default `metas`)*: existing meta subdir under each `aspect_ratio_*/` to read entry order from.
- `--new_meta_key` *(default `metas_new`)*: new sibling meta subdir to write the rebuilt tars into.
- `--num_concurrency` *(default 4)*: per-task concurrency for download/upload streams.
- `--workers` *(default 4)*: outer worker thread count. 16 is a good working point on the `cpu` queue.

### Notes

- Entry order inside each rebuilt tar matches the existing `{old_meta_key}/<shard>.tar` exactly — only the JSON payload changes.
- UUIDs in the existing tar that are missing from the filtered LanceDB are **dropped** from the new tar, with a per-shard warning logging the missing count.
- Safe to re-run: the destination tars are overwritten on each run.
- No `--samples-per-shard` / `--max-rows` flags — shape is dictated by the existing webds.
