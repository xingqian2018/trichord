# Create caption-pickle tars from meta tars

Use this when an existing meta-only WebDS layout already carries per-sample LanceDB rows as `<uuid>.json`, and you need to produce a sibling caption-only WebDS layout where each sample is a `<uuid>.pickle` matching the cosmos-captioner shard schema (the format consumed by the captioner-aware data loaders).

For each existing `{webds_path}/<multiple/bucket/paths>/{meta_key}/<shard>.tar`, the script writes a sibling `{webds_path}/<multiple/bucket/paths>/{out_key}/<shard>.tar` whose entries are `<uuid>.pickle` files with this exact shape:

```python
{
  "key": "<uuid>",
  "caption": {
    "caption_cosmos_captioner_image": "<JSON string from table_meta>",
  },
}
```

The caption value is read verbatim from `meta["table_meta"]["captioning_cosmos_captioner_image_v1_full_caption_cosmos_captioner_image"]`. Samples missing that field are silently dropped from the output tar; tars that end up with zero usable samples are not uploaded.

Lives at `pipelines/image/text_rendering/create_caption_pickle.py` in `imaginaire4_sila`.

## Some live information

- `<meta_key>` = `metas_YYYYmmdd` (the source meta-tar subdir; ask user for the date if unsure — pick the latest)
- `<out_key>` — the caption-tar subdir to write into. Default `captions_cosmos_captioner_v1p1` (matches the cosmos-captioner v1.1 convention seen in production GCS).

## Quick-check inputs

| Arg                 | Value                           | Notes                                                                                                                                                                                |
|---------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--webds_path`      | `<the_existed_webdataset_path>` | Existing WebDS root. The script walks every `*.tar` under it and keeps the ones whose path contains `/{meta_key}/`. Output is written under the same root with the key swapped.     |
| `--webds_credential`| `credentials/gcs.secret`        | Credential file for the WebDS bucket. Default is `credentials/gcs.secret`.                                                                                                           |
| `--meta_key`        | **ASK USER**                    | Source key under which meta tars live, e.g. `metas_20260502`. Used both as the discovery filter (`/{meta_key}/`) and the substring to swap when computing the output path.           |
| `--out_key`         | `captions_cosmos_captioner_v1p1`| Destination key for the caption-pickle tars. Override only if the downstream loader expects a different folder name.                                                                 |
| `--num_concurrency` | `8`                             | MSC worker pool size. The main loop is single-threaded (one tar at a time); this only affects per-call download/upload concurrency.                                                  |
| `--overwrite`       | _flag_                          | Off by default — tars already present under `/{out_key}/` are skipped (safe to resume an interrupted run). Pass to force re-upload.                                                  |

## Known dataset lookup

Two tables, both keyed on `--dataset_name`. Table A fills `--webds_path`; Table B fills `--meta_key` with the latest dated meta tar set discovered on GCS. The slaunch job tag also reuses `<dataset_name>`.

### Table A — WebDS

| `--dataset_name`                              | `--webds_path`                                                                                      |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `screen2words_rico`                           | `s3://nv-00-10206-vfm/webdataset_image_regular_text/screen2words_rico/`                             |
| `slide_audit`                                 | `s3://nv-00-10206-vfm/webdataset_image_regular_text/slide_audit/`                                   |
| `voxel51_rico`                                | `s3://nv-00-10206-vfm/webdataset_image_regular_text/voxel51_rico/`                                  |
| `zennodo10k`                                  | `s3://nv-00-10206-vfm/webdataset_image_regular_text/zennodo10k/`                                    |
| `synthetic_scene_text_v0`                     | `s3://nv-00-10206-vfm/webdataset_image_synthetic_text/synthetic_scene_text_v0/`                     |
| `synthetic_chinese_scene_text_v0`             | `s3://nv-00-10206-vfm/webdataset_image_synthetic_text/synthetic_chinese_scene_text_v0/`             |
| `synthetic_traditional_chinese_scene_text_v0` | `s3://nv-00-10206-vfm/webdataset_image_synthetic_text/synthetic_traditional_chinese_scene_text_v0/` |

### Table B — latest `--meta_key`

Snapshot of the newest `metas_YYYYmmdd` key present per dataset (from `webds_tarcnt_by_key.py` against the `gcs:` form of each row above). Re-run the helper before launching to confirm — a newer captioner pass may have landed since this was written.

| `--dataset_name`                              | `--meta_key`     |
|-----------------------------------------------|------------------|
| `screen2words_rico`                           | `metas_20260501` |
| `slide_audit`                                 | `metas_20260501` |
| `voxel51_rico`                                | `metas_20260501` |
| `zennodo10k`                                  | `metas_20260501` |
| `synthetic_scene_text_v0`                     | `metas_20260501` |
| `synthetic_chinese_scene_text_v0`             | `metas_20260501` |
| `synthetic_traditional_chinese_scene_text_v0` | `metas_20260501` |

To refresh a row, convert `s3://` → `gcs:` and run:

```bash
python /home/xingqianx/Project/trichord/helper/webds_tarcnt_by_key.py gcs:nv-00-10206-vfm/<...>/<dataset_name>/
```

The latest dated `metas_*` key in the `Totals` block is the value to use.

When both rows are filled in, the cheatsheet substitutes `--webds_path` and `--meta_key` directly — the user only has to confirm `<dataset_name>` for the slaunch job tag.

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 caption_pickle_<dataset_name> \
    pipelines/image/text_rendering/create_caption_pickle.py \
    --webds_path <the_existed_webdataset_path> \
    --webds_credential credentials/gcs.secret \
    --meta_key <meta_key> \
    --out_key captions_cosmos_captioner_v1p1 \
    --num_concurrency 8
```

Add `--overwrite` to the slaunch command if you want to re-upload tars that already exist under `/{out_key}/`.

## How the work is split

- The script is **single-threaded by design** — the main loop processes one meta tar at a time. `--num_concurrency` only controls MSC's internal worker pool for the per-tar download and upload calls.
- A `cpu 1x1` slaunch shape is the right match: there is no distributed gather, so extra ranks would idle. Submitting via slaunch (instead of running on `n0` directly) is purely for queueing/log hygiene.
- Tar discovery is one `fs.find()` over `--webds_path`; existence-check for `/{out_key}/` is the same listing, so resumes are cheap.
- For each meta tar: download bytes → iterate `*.json` members → build `<uuid>.pickle` payloads in memory → write a fresh tar to a `BytesIO` → upload. No temp files on disk.

## Sanity check before launching

- Spot-check the source schema once with `s3 dl <one_meta_tar>` and `python -c 'import json,tarfile,io; ...'` — confirm `meta["table_meta"]["captioning_cosmos_captioner_image_v1_full_caption_cosmos_captioner_image"]` exists and is a string. If it is missing across the board, the run will produce zero output tars and warn `no caption found` for every shard.
- After a small smoke run, verify one output pickle:
  ```bash
  python ~/Project/bashrc/s3_omni.py dl <out_tar> /tmp/out.tar && \
      tar -xOf /tmp/out.tar $(tar -tf /tmp/out.tar | head -1) | \
      python -c 'import sys,pickle,pprint; pprint.pp(pickle.loads(sys.stdin.buffer.read()))'
  ```
  Expected keys: `key` (uuid) and `caption.caption_cosmos_captioner_image` (string).

## Progress / status check

Use the same helper as `data_shard_full_db`:

```bash
python /home/xingqianx/Project/trichord/helper/webds_tarcnt_by_key.py <dataset_name>
```

Progress = `{out_key}` total / `{meta_key}` total. Per-leaf zeros highlight which `resolution_*/aspect_ratio_*/` buckets the loop hasn't reached yet.

## Notes

- Never run it directly. Show the command as a formatted response first and ask user's permission to run.
- Your command should follow the exact indent as the template shows.
- The output pickle shape is fixed and matches what production caption tars carry (verified against `gcs:nv-00-10206-webdataset-images/.../captions_cosmos_captioner_v1p1/part_*/*.tar`). Do not rename the inner `caption_cosmos_captioner_image` key without also updating the data loader.
- A meta tar that has zero samples with the captioner field is **not** uploaded — this is intentional. If you expect every shard to produce output, this means the upstream captioner step has not populated `table_meta` yet for that shard; rerun the captioner before this script.
- Safe to re-run by default — already-uploaded tars under `/{out_key}/` are skipped at discovery time. Pass `--overwrite` only when you have changed the output schema and need to re-emit every shard.
- `--webds_path` must be in the repo's `s3://` form (consumed by `reformat_path_s3_to_msc`), not the `gcs:` shorthand used by `s3_omni`. Use `s3://nv-00-10206-webdataset-images/...` with `--webds_credential credentials/gcs.secret`.
- Single-thread is intentional for the v0 of this pipeline — the caption-tar payload is tiny (text only) and most runs are I/O-bound on the upload side. If a full corpus run takes too long, the natural next step is to wrap this loop in a distributed shard-by-rank pattern (see `shard_full_db.py`) and bump the slaunch shape, not to thread the inner loop.
