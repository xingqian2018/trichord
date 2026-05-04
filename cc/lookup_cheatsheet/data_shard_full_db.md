# Shard a full DB into a combined image+meta WebDataset

Use this when you want to materialize a **brand-new combined webds layout** — each output tar carries both the image bytes and the latest LanceDB row for every sample — by joining a refreshed LanceDB table with the original logged-image tars, while reusing the shape (per-tar uuid grouping) defined by an existing meta-only webds.

For each existing `{webds_path}/resolution_*/aspect_ratio_*/{old_meta_key}/<shard>.tar` (used purely as a shape reference), the script writes a sibling `{webds_path}/resolution_*/aspect_ratio_*/{new_meta_key}/<shard>.tar` whose entries are the union of `<uuid>.json` (LanceDB row, JSON-serialized) and `<uuid>.<ext>` (raw image bytes pulled from `--logged_image_path`).

Lives at `pipelines/image/text_rendering/shard_full_db.py` in `imaginaire4_sila`.

## Some live information:

- `<table_postfix>` = `YYYYmmdd` (the current table postfix we are working on, can smartly lookup for the latest)
- `<new_meta_key>` = the key under which combined image+meta tars are written, likely `metas_YYYYmmdd` or `combined_YYYYmmdd`, ask user.

## Quick-check inputs

| Arg                                  | Value                            | Notes                                                                                                                                                                                                                                      |
|--------------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--webds_path`                       | `<the_existed_webdataset_path>`  | Existing WebDS root. The script does NOT discover image shapes from scratch — it walks `{old_meta_key}/*.tar` here to learn which uuids belong in which output tar. The `{new_meta_key}` siblings are written under the same root.         |
| `--webds_credential`                 | `credentials/gcs.secret`         | Credential file for the WebDS bucket.                                                                                                                                                                                                      |
| `--lancedb_path`                     | `<lancedb_to_work_on>`           | Refreshed Lance table URI. `.lance` suffix auto-appended; `gcs://` normalized to `gs://`. Default joint table: `gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance`. Only the columns in the script's `COLUMN` list are read. |
| `--lancedb_webds_filekey_lookup_col` | `<uuid_or_sdg_original_index>`   | Lance column that keys each row to the in-tar member name. Default `uuid`. Use `sdg_original_index` for synthetic SDG datasets where filenames are SDG indices, not UUIDs.                                                                 |
| `--logged_image_path`                | `<s3_or_gs_logged_image_root>`   | Root URI to walk for `*.tar` files of raw images. Each tar member's basename (sans extension) must match a Lance row's filekey-lookup-col value, or it's silently dropped.                                                                 |
| `--logged_image_credential`          | _required_                       | Credential file for the logged-image bucket (e.g. `credentials/s3_image_pbss.secret` or `credentials/gcs.secret`). No default — must be supplied.                                                                                          |
| `--dataset_name`                     | `<dataset_name_that_to_process>` | Filter — only Lance rows where `source_dataset == <dataset_name>` are loaded.                                                                                                                                                              |
| `--old_meta_key`                     | `metas`                          | Existing meta subdir under each `aspect_ratio_*/`. Used only to read the per-tar uuid grouping; payload contents are irrelevant.                                                                                                           |
| `--new_meta_key`                     | **ASK USER**                     | New sibling subdir to write the combined image+meta tars into. Common patterns: `metas_<date>`, `combined_<date>`.                                                                                                                         |
| `--mode`                             | `append`                         | `append` (default): skip tars already present under `/{new_meta_key}/` — safe to resume an interrupted run. `replace`: delete everything under `/{new_meta_key}/` first, then re-do all shards.                                            |
| `--num_concurrency`                  | `4`                              | Per-task concurrency for download/upload streams. Bump to 32-64 on cloud queues.                                                                                                                                                           |

## How the work is split

- World ranks share two sources: **Lance fragments** (`get_fragments()`, sorted by `fragment_id`) and **logged-image tars** (full filesystem walk, sorted). Each is consecutively sharded across ranks.
- Within a rank, `lance_fragment_read_task` and `logged_image_read_task` are interleaved (1:1 by group), each followed by a `sync_and_upload_task`. Cross-rank gather catches uuids whose meta lands on rank A but image lands on rank B.
- A tar is uploaded only when **all** of its uuids have both meta and image filled. Stragglers get one final `sync_and_upload_task` after the queue drains; if any tar is still incomplete, the job fails with a `RuntimeError` listing how many tars and how many missing uuids per tar.

## Known dataset lookup

Three tables, all keyed on `--dataset_name`. Table A is the WebDS path; Table B is the LanceDB config; Table C is the logged-image root + credential. Look up the same row in all three when filling the template.

### Common Root Paths

| Alias                    | Path                                                   |
|--------------------------|--------------------------------------------------------|
| `<webds_image_reg_text>` | `s3://nv-00-10206-vfm/webdataset_image_regular_text`   |
| `<webds_image_sgd_text>` | `s3://nv-00-10206-vfm/webdataset_image_synthetic_text` |
| `<webds_image_reg>`      | `s3://nv-00-10206-vfm/webdataset_image_regular`        |
| `<webds_image_sgd>`      | `s3://nv-00-10206-vfm/webdataset_image_synthetic`      |
| `<webds_image_sgd>`      | `s3://nv-00-10206-vfm/webdataset_image_synthetic`      |
| `<lancedb_image_root>`   | `gs://nv-00-10206-vfm/lancedb/image`                   |
| `<logged_image_root>`    | `s3://nv-00-10206-images/logged_images`                |

### Table A — WebDS

| `--dataset_name`                                 | `--webds_path`                                                        |
|--------------------------------------------------|-----------------------------------------------------------------------|
| **Text dataset (Real + SGD)**                    |                                                                       |
| `screen2words_rico`                              | `<webds_image_reg_text>/screen2words_rico/`                           |
| `slide_audit`                                    | `<webds_image_reg_text>/slide_audit/`                                 |
| `voxel51_rico`                                   | `<webds_image_reg_text>/voxel51_rico/`                                |
| `zennodo10k`                                     | `<webds_image_reg_text>/zennodo10k/`                                  |
| `synthetic_scene_text_v0`                        | `<webds_image_sgd_text>/synthetic_scene_text_v0/`                     |
| `synthetic_chinese_scene_text_v0`                | `<webds_image_sgd_text>/synthetic_chinese_scene_text_v0/`             |
| `synthetic_traditional_chinese_scene_text_v0`    | `<webds_image_sgd_text>/synthetic_traditional_chinese_scene_text_v0/` |
| **Regular dataset (Real)**                       |                                                                       |
| `red`                                            | `<webds_image_reg>/red`                                               |
| `coyo_700m`                                      | `<webds_image_reg>/coyo_700m`                                         |
| **Synthetic dataset (SGD)**                      |                                                                       |
| `MMC4`                                           | `<webds_image_sgd>/mmc4`                                              |
| `generations_qwen_image_2512_filtered_photoreal` | `<webds_image_sgd>/generations_qwen_image_2512_filtered_photoreal`    |
| `wordnet_captions_20260224`                      | `<webds_image_sgd>/wordnet_captions_20260224`                         |
| `datacomp_1b`                                    | `<webds_image_sgd>/datacomp_1b`                                       |
| `midjourney`                                     | `<webds_image_sgd>/midjourney`                                        |
| `midjourney_v6_20240703`                         | `<webds_image_sgd>/midjourney_v6_20240703`                            |

### Table B — LanceDB

| `--dataset_name`                                 | `--lancedb_path`                                                                                                             |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Text dataset (Real + SGD)**                    |                                                                                                                              |
| `screen2words_rico`                              | `<lancedb_root>/regular_text/screen2words_rico_slice_from_maintable_<table_postfix>.lance`                                   |
| `slide_audit`                                    | `<lancedb_root>/regular_text/slide_audit_slice_from_maintable_<table_postfix>.lance`                                         |
| `voxel51_rico`                                   | `<lancedb_root>/regular_text/voxel51_rico_slice_from_maintable_<table_postfix>.lance`                                        |
| `zennodo10k`                                     | `<lancedb_root>/regular_text/zennodo10k_slice_from_maintable_<table_postfix>.lance`                                          |
| `synthetic_scene_text_v0`                        | `<lancedb_root>/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_<table_postfix>.lance`                     |
| `synthetic_chinese_scene_text_v0`                | `<lancedb_root>/synthetic_text/synthetic_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance`             |
| `synthetic_traditional_chinese_scene_text_v0`    | `<lancedb_root>/synthetic_text/synthetic_traditional_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance` |
|                                                  |                                                                                                                              |
| **Regular dataset (Real)**                       |                                                                                                                              |
| `red`                                            | `<lancedb_root>/regular/red_slice_from_maintable_<table_postfix>.lance`                                                      |
| `coyo_700m`                                      | `<lancedb_root>/regular/coyo_700m_slice_from_maintable_<table_postfix>.lance`                                                |
|                                                  |                                                                                                                              |
| **Synthetic dataset (SGD)**                      |                                                                                                                              |
| `MMC4`                                           | `<lancedb_root>/synthetic/MMC4_slice_from_maintable_<table_postfix>.lance`                                                   |
| `generations_qwen_image_2512_filtered_photoreal` | `<lancedb_root>/synthetic/generations_qwen_image_2512_filtered_photoreal_slice_from_maintable_<table_postfix>.lance`         |
| `wordnet_captions_20260224`                      | `<lancedb_root>/synthetic/wordnet_captions_20260224_slice_from_maintable_<table_postfix>.lance`                              |
| `datacomp_1b`                                    | `<lancedb_root>/synthetic/datacomp_1b_slice_from_maintable_<table_postfix>.lance`                                            |
| `midjourney`                                     | `<lancedb_root>/synthetic/midjourney_slice_from_maintable_<table_postfix>.lance`                                             |
| `midjourney_v6_20240703`                         | `<lancedb_root>/synthetic/midjourney_v6_20240703_slice_from_maintable_<table_postfix>.lance`                                 |

### Table C — logged image

| `--dataset_name`                                 | `--logged_image_path`                                                 | `--logged_image_credential`   |
|--------------------------------------------------|-----------------------------------------------------------------------|-------------------------------|
| **Text dataset (Real + SGD)**                    |                                                                       |                               |
| `screen2words_rico`                              | `<logged_image_root>/screen2words_rico/`                              | `credentials/gcs.secret`      |
| `slide_audit`                                    | `<logged_image_root>/slide_audit/`                                    | `credentials/gcs.secret`      |
| `voxel51_rico`                                   | `<logged_image_root>/voxel51_rico/`                                   | `credentials/gcs.secret`      |
| `zennodo10k`                                     | `<logged_image_root>/zennodo10k/`                                     | `credentials/gcs.secret`      |
| `synthetic_scene_text_v0`                        | `<logged_image_root>/synthetic_scene_text_v0/`                        | `credentials/gcs.secret`      |
| `synthetic_chinese_scene_text_v0`                | `<logged_image_root>/synthetic_chinese_scene_text_v0/`                | `credentials/gcs.secret`      |
| `synthetic_traditional_chinese_scene_text_v0`    | `<logged_image_root>/synthetic_traditional_chinese_scene_text_v0/`    | `credentials/gcs.secret`      |
|                                                  |                                                                       |                               |
| **Regular dataset (Real)**                       |                                                                       |                               |
| `red`                                            | `<logged_image_root>/red/`                                            | `credentials/gcs.secret`      |
| `coyo_700m`                                      | `<logged_image_root>/coyo_700m/`                                      | `credentials/gcs.secret`      |
|                                                  |                                                                       |                               |
| **Synthetic dataset (SGD)**                      |                                                                       |                               |
| `MMC4`                                           | `<logged_image_root>/MMC4/`                                           | `credentials/gcs.secret`      |
| `generations_qwen_image_2512_filtered_photoreal` | `<logged_image_root>/generations_qwen_image_2512_filtered_photoreal/` | `credentials/gcs.secret`      |
| `wordnet_captions_20260224`                      | `<logged_image_root>/wordnet_captions_20260224/`                      | `credentials/gcs.secret`      |
| `datacomp_1b`                                    | `<logged_image_root>/datacomp_1b/`                                    | `credentials/gcs.secret`      |
| `midjourney`                                     | `<logged_image_root>/midjourney/`                                     | `credentials/gcs.secret`      |
| `midjourney_v6_20240703`                         | `<logged_image_root>/<not_sure>/`                                     | `credentials/gcs.secret`      |

> Table C `--logged_image_path` values for the original 7 datasets are derived from the canonical convention `gcs:nv-00-10206-images/logged_images/<dataset_name>/` (per `s3path` cheatsheet) — verify with `python ~/Project/bashrc/s3_omni.py ls gcs:nv-00-10206-images/logged_images/` before launching if unsure. The four new datasets (`red`, `MMC4`, `generations_qwen_image_2512_filtered_photoreal`, `wordnet_captions_20260224`) have no recorded logged-image root yet — ask the user the first time, then fill the table in place. Same for their `--lancedb_webds_filekey_lookup_col`: the slice cheatsheet only records the Lance URI, not which column the WebDS filenames map to.

When a row is filled in, the cheatsheet substitutes `--webds_path`, `--lancedb_path`, `--lancedb_webds_filekey_lookup_col`, `--logged_image_path`, and `--logged_image_credential` directly — the user only has to supply `--new_meta_key`.

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 2x4 shard_full_db_<dataset_name> \
    pipelines/image/text_rendering/shard_full_db.py \
    --webds_path <the_existed_webdataset_path> \
    --webds_credential credentials/gcs.secret \
    --lancedb_path <lancedb_to_work_on> \
    --lancedb_webds_filekey_lookup_col <uuid_or_sdg_original_index> \
    --logged_image_path <logged_image_path> \
    --logged_image_credential <logged_image_credential> \
    --dataset_name <dataset_name_that_to_process> \
    --old_meta_key metas \
    --new_meta_key <new_meta_key> \
    --mode append \
    --num_concurrency 64
```

## Progress / status check

When the user asks for the **status of the dataset** (or "how far along is the shard", "count progress", "what's done so far", etc.), this is what they mean: compare tar counts between the old `metas` key and the new `<new_meta_key>` key. Use the helper script — do NOT roll your own with `s3 cnt` (that op only counts immediate children, not nested tars, and silently undercounts).

```bash
python /home/xingqianx/Project/trichord/helper/webds_tarcnt_by_key.py <dataset_name>
```

- Accepts a bare dataset name (e.g. `synthetic_scene_text_v0`) or a full `gcs:...` path.
- Output gives per-key totals (`images`, `metas`, `metas_<date>`, ...) and a per-leaf breakdown across `resolution_*/aspect_ratio_*/`.
- Progress = `<new_meta_key>` total / `metas` total. Per-leaf zeros highlight which buckets the job hasn't reached yet.

## Notes
- Never run it directly. Show the command as a formatted response first and ask user's permission to run.
- Your command should follow the exact indent as the template shows.
- The script **requires** an existing `{old_meta_key}/` layout under `--webds_path` — that's where the per-tar uuid grouping comes from. If you don't have one yet, run `data_sharding_meta_reload` first (or the original meta-shard pipeline) to bootstrap the shape.
- Each output tar contains both the JSON meta (`<uuid>.json`) and the raw image (`<uuid>.<ext>` from the logged-image tar member name). This is heavier than the meta-only sidecar layout — expect bandwidth roughly equal to the full image corpus per run.
- A logged-image member whose basename is **not** in the filtered LanceDB is dropped without warning; a uuid that is in LanceDB but missing from the logged-image tars will leave its tar incomplete, and the job will raise at the end with the missing count.
- Safe to re-run in default `--mode append`: tars already present under `/{new_meta_key}/` are skipped at init time and removed from each rank's task list. Use `--mode replace` to wipe and rebuild from scratch.
- No `--samples-per-shard` / `--max-rows` flags — shape is dictated by the existing webds.
- World size: this script is genuinely distributed (cross-rank gather of stragglers). `2x4` (2 nodes × 4 ranks) is the default working point; bump up if the logged-image bucket can sustain the bandwidth.
