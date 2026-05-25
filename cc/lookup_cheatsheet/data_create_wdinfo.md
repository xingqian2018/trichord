# Create wdinfo for an existing WebDataset

Use this when WebDS shards already exist (image tars and one or more meta tars) and you need to (re)generate the `wdinfo.json` index files that downstream training/loading consume. The script walks the WebDS root, counts JSON entries inside each meta tar, and writes one `wdinfo.json` per leaf bucket group plus a unified `wdinfo_unified.json` at the root.

Lives at `pipelines/image/text_rendering/create_wdinfo.py` in `imaginaire4_sila`.

## What it writes

For each existing `{webds_path}/<multiple/bucket/paths>/{key}/<shard>.tar` group, the script:
- Counts JSON items inside each `.../metas/<shard>.tar` (the count anchor; all other selected keys must have matching tar names).
- Writes `{webds_path}/wdinfo/<multiple/bucket/paths>/wdinfo.json` listing `data_keys`, sorted `data_list`, `total_key_count`, `chunk_size`, and `root`.
- Writes a single `{webds_path}/wdinfo_unified.json` aggregating every group.

## Step 1 — collect information

- **Check `./data_common_root.md` for the common root path and dataset name → `--webds_path` mapping**
- `<some_postfix>` = `YYYYmmdd` (the current metas postfix; can smartly look up the latest)
- `<webds_key>` — the list of subdir keys to include in wdinfo; must include `images` and a metas key (e.g. `metas_<some_postfix>`). Ask user.
- *currently* `--webds_key` = `images, metas_20260502`

### Input Information

| Arg                 | Value                           | Notes                                                                                                                                                                                                   |
|---------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--webds_path`      | `<the_existed_webdataset_path>` | Existing WebDS root. Look up in `./data_common_root.md`. Script writes `wdinfo*.json` in place; does NOT touch data tars.                                                                               |
| `--webds_cred`      | `credentials/gcs.secret`        | Credential file for the WebDS bucket. Default is `credentials/gcs.secret`.                                                                                                                              |
| `--webds_key`       | `<list_of_keys>`                | Space-separated keys to keep in wdinfo (e.g. `images metas metas_<some_postfix>`). **Omit to auto-include every key discovered under each leaf bucket group** — the script logs the auto-detected list. |
| `--num_concurrency` | `256`                           | Async download concurrency for streaming meta tars. 256 is a good working point.                                                                                                                        |
| `--batch_size`      | `256`                           | Per-rank batch size when fetching meta tars in each round. 256 keeps the network busy.                                                                                                                  |
| `--max_try`         | `3`                             | Retries per failed meta-tar download. Job aborts with `RuntimeError` if any tar still fails (no wdinfo is written).                                                                                     |

## Step 2 — compose the formatted command and show user for confirmation

### The `slaunch`-way launch template

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

## Step 3 — launch

- **No silent run by yourself, confirmation is always required!**
- Ask the user which cluster to launch the command on.
- Sanity check if the run is duplicated (usually with same run name); if duplicated, stop and inform the user.
- When user confirms and no duplication, use your skill `/ssh_run` to help launch the run.

## Notes

- The script anchors entry counts on the `metas` key. **All other selected keys must share the same tar names** under each leaf bucket group, otherwise it raises `ValueError: Tar name mismatch`.
- Omit `--webds_key` to let the script auto-detect every subdir as a data key. Most training setups want a curated subset (e.g. `images metas metas_<some_postfix>`), so ask the user before defaulting to auto-detect.
- A `1x1` shape is enough — the script parallelizes internally via async (`--num_concurrency`). It will also scale across ranks if you bump the slaunch shape, but is rarely needed for index-only work.
- If any meta tar still fails after `--max_try` retries, the job raises `RuntimeError` and **no wdinfo is written** — fix the failing tars and rerun.
- Safe to re-run: each run rewrites `wdinfo*.json` from scratch based on the current set of tars; existing image/meta tars are never modified.
