# Shard a LanceDB table into WebDataset

Use this when you have a LanceDB table and need to convert it into sharded WebDataset tar files for downstream training/loading.

The job runs on CPU via `slaunch` (a single 1x1 task is enough — it parallelizes internally with `--workers`).

## Which script?

Two scripts live side-by-side in `pipelines/image/text_rendering/`:

- **`shard_logged_image.py`** — shards using only `logged_images` and `logged_metas`; ignores the rest of the LanceDB row entries. Use when downstream training only needs the logged image + its logged meta.
- **`shard_full_dbinfo.py`** — shards **images + full LanceDB row info** into one combined meta per sample. Use when downstream needs the table entries (captions, tags, source URLs, etc.) alongside the image.

Pick `shard_full_dbinfo.py` whenever in doubt — it preserves more information.

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

## Example — logged_images + logged_metas only (zennodo10k)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 shard_zennodo10k \
    pipelines/image/text_rendering/shard_logged_image.py \
    --input-lancedb-path gs://nv-00-10206-lancedb/prod/image/text_related/zennodo10k.lance \
    --output-webds-path nv-00-10206-webdataset-images/webdataset_image_text_related/zennodo10k \
    --s3-profile team-gcs \
    --samples-per-shard 1000 \
    --workers 16
```

## Example — images + dbinfo (rico)

```bash
uv run python pipelines/image/text_rendering/shard_full_dbinfo.py \
    --input-lancedb-path gs://bucket/my_table \
    --dataset-name rico \
    --output-webds-path nv-00-10206-images/debug/webds/rico
```

(Wrap the same command in the `slaunch cpu 1x1 shard_<dataset_name> ...` template above when running at scale.)
