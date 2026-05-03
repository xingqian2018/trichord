# Create wdinfo for an existing WebDataset

Use this when WebDS shards already exist (image tars and one or more meta tars) and you need to (re)generate the `wdinfo.json` index files that downstream training/loading consume. The script walks the WebDS root, counts JSON entries inside each meta tar, and writes one `wdinfo.json` per leaf bucket group (any depth of intermediate dirs is supported, e.g. `resolution_*/aspect_ratio_*/`) plus a unified `wdinfo_unified.json` at the root.

## What it writes

The script auto-detects the bucket layout under `{webds_path}` — any number of intermediate dir levels is supported. A "group" is everything between `{webds_path}/` and the final `{key}/<shard>.tar` (e.g. `resolution_*/aspect_ratio_*/`, or just `bucket_*/`, or any other depth).

For each existing `{webds_path}/<multiple/bucket/paths>/{key}/<shard>.tar` group, the script:
- Counts JSON items inside each `{webds_path}/<multiple/bucket/paths>/metas/<shard>.tar` (the count anchor; all other selected `{key}/` subdirs in the same group must have matching tar names).
- Writes `{webds_path}/wdinfo/<multiple/bucket/paths>/wdinfo.json` listing `data_keys`, the sorted `data_list` of tarnames, `total_key_count`, `chunk_size`, and `root`.
- Writes a single `{webds_path}/wdinfo_unified.json` aggregating every group, with a per-group breakdown of tar → count.

## Some live information

- `<some_postfix>` = `YYYYmmdd` (the current metas postfix we are working on, can smartly look up the latest)
- `<webds_key>` — the list of subdir keys to include in wdinfo, it must include `images` and a type of metas (i.e. `metas_<some_postfix>`). Ask user.
- *currently* `--webds_key` = `images, metas_20260502`

## Quick-check inputs

| Arg | Value | Notes |
|---|---|---|
| `--webds_path` | `<the_existed_webdataset_path>` | Existing WebDS root to walk. Script writes `wdinfo*.json` in place; does NOT touch the data tars themselves. |
| `--webds_cred` | `credentials/gcs.secret` | Credential file for the WebDS bucket. Default is `credentials/gcs.secret`. |
| `--webds_key` | `<list_of_keys>` | Space-separated keys to keep in wdinfo (e.g. `images metas metas_<some_postfix>`). **Omit to auto-include every key discovered under each leaf bucket group** — the script logs the auto-detected list. |
| `--num_concurrency` | `256` | Async download concurrency for streaming meta tars. 256 is a good working point. |
| `--batch_size` | `256` | Per-rank batch size when fetching meta tars in each round. 256 keeps the network busy. |
| `--max_try` | `3` | Retries per failed meta-tar download. Job aborts with `RuntimeError` if any tar still fails after this many tries (no wdinfo is written). |

## Known dataset lookup

Single table keyed on `--dataset_name`. Look up the row to fill `--webds_path`; the slaunch job tag also reuses `<dataset_name>`.

### Table A — WebDS

| `--dataset_name` | `--webds_path` |
|---|---|
| `screen2words_rico` | `s3://nv-00-10206-vfm/webdataset_image_text_related/screen2words_rico/` |
| `slide_audit` | `s3://nv-00-10206-vfm/webdataset_image_text_related/slide_audit/` |
| `voxel51_rico` | `s3://nv-00-10206-vfm/webdataset_image_text_related/voxel51_rico/` |
| `zennodo10k` | `s3://nv-00-10206-vfm/webdataset_image_text_related/zennodo10k/` |
| `synthetic_scene_text_v0` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_scene_text_v0/` |
| `synthetic_chinese_scene_text_v0` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_chinese_scene_text_v0/` |
| `synthetic_traditional_chinese_scene_text_v0` | `s3://nv-00-10206-vfm/webdataset_synthetic/synthetic_traditional_chinese_scene_text_v0/` |

When a row is filled in, the cheatsheet substitutes `--webds_path` directly — the user only has to supply `--webds_key` (and confirm `<dataset_name>` for the slaunch job tag).

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 wdinfo_<dataset_name> \
    pipelines/image/text_rendering/create_wdinfo.py \
    --webds_path <the_existed_webdataset_path> \
    --webds_cred credentials/gcs.secret \
    --webds_key <list_of_keys> \
    --num_concurrency 256 \
    --batch_size 256 \
    --max_try 3
```

## Output paths

- Per-group: `{webds_path}/wdinfo/<multiple/bucket/paths>/wdinfo.json`
- Unified:  `{webds_path}/wdinfo_unified.json`

## Notes
- Never run it directly. Show the command as a formatted response first and ask user's permission to run.
- Your command should follow the exact indent as the template shows.
- The script anchors entry counts on the `metas` key. **All other selected keys must share the same tar names** under each leaf bucket group (i.e. each `<multiple/bucket/paths>/` prefix), otherwise it raises `ValueError: Tar name mismatch`.
- Omit `--webds_key` to let the script auto-detect every subdir as a data key. Most training setups want a curated subset (e.g. `images metas metas_<some_postfix>`), so ask the user before defaulting to auto-detect.
- A `1x1` shape is enough — the script parallelizes internally via async (`--num_concurrency`). It will also scale across ranks if you bump the slaunch shape, but it is rarely needed for index-only work.
- If any meta tar still fails after `--max_try` retries, the job raises `RuntimeError` and **no wdinfo is written** — fix the failing tars and rerun.
- Safe to re-run: each run rewrites `wdinfo*.json` from scratch based on the current set of tars; existing image/meta tars are never modified.
