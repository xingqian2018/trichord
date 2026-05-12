# Shard a full DB into split image+meta WebDataset tars

Use this to materialize a **brand-new WebDataset layout** for a dataset, planned from scratch off a refreshed LanceDB table joined with the original logged-image tars.

The script first does a Lance prefetch pass to bucket every filtered row by **(resolution, aspect ratio)** and pack 1000 samples per tar. Then it streams Lance fragments and logged-image tars in parallel, fills sample slots cross-rank, and emits two siblings per tar:

```
{webds_path}/<prekey_path_usually_buckets>/images/<postkey_path>/<NNNNNNNNN>.tar   # raw images (transcoded to a uniform format)
{webds_path}/<prekey_path_usually_buckets>/metas/<postkey_path>/<NNNNNNNNN>.tar    # JSON metas (one .json per uuid)
```

Lives at `pipelines/image/text_rendering/shard_full_db.py` in `imaginaire4_sila`.

## Some live information:

- `<table_postfix>` = `YYYYmmdd` (the current table postfix we are working on, can smartly lookup for the latest)

## Quick-check inputs

| Arg                         | Value                            | Notes                                                                                |
|-----------------------------|----------------------------------|--------------------------------------------------------------------------------------|
| `--webds_path`              | `<the_existed_webdataset_path>`  | Output root. Tars land under `resolution_*/aspect_ratio_*/images/` + `metas/`.       |
| `--webds_credential`        | `credentials/gcs.secret`         | Credential for the WebDS bucket.                                                     |
| `--lancedb_path`            | `<lancedb_to_work_on>`           | Lance table URI. `.lance` auto-appended; `gcs://` → `gs://`.                         |
| `--logged_image_credential` | `credentials/gcs.secret`         | Credential for the logged-image bucket.                                              |
| `--dataset_name`            | `<dataset_name_that_to_process>` | Filters Lance rows by `source_dataset`.                                              |
| `--num_concurrency`         | `4`                              | Per-task download/upload streams (default 4).                                        |
| `--target_image_format`     | `jpg` / `jpeg` / `webp`          | **Required.** Output image format. jpg/jpeg → quality 95; webp → quality 85 method 5.|

> **No `--logged_image_path`**: the script derives the logged-image tar path from each row's `image_s3_range` column in Lance — no explicit root arg needed.

## Known dataset lookup

Two tables, both keyed on `--dataset_name`. Table A is the WebDS path; Table B is the LanceDB config.

- **Check `./data_common_root.md` for some common root path settings** 
- `<table_postfix>` = usually `slice_from_maintable_YYYYmmdd` the newest slice, you may need to look up it.


### Table A — WebDS

| `--dataset_name`                                 | `--webds_path`                                                        | `--target_image_format` |
|--------------------------------------------------|-----------------------------------------------------------------------|-------------------------|
| **Text dataset (Real + SGD)**                    |                                                                       |                         |
| `screen2words_rico`                              | `<webds_image_reg_text>/screen2words_rico/`                           | `webp`                  |
| `slide_audit`                                    | `<webds_image_reg_text>/slide_audit/`                                 | `webp`                  |
| `voxel51_rico`                                   | `<webds_image_reg_text>/voxel51_rico/`                                | `webp`                  |
| `zennodo10k`                                     | `<webds_image_reg_text>/zennodo10k/`                                  | `webp`                  |
| `synthetic_scene_text_v0`                        | `<webds_image_sgd_text>/synthetic_scene_text_v0/`                     | `webp`                  |
| `synthetic_chinese_scene_text_v0`                | `<webds_image_sgd_text>/synthetic_chinese_scene_text_v0/`             | `webp`                  |
| `synthetic_traditional_chinese_scene_text_v0`    | `<webds_image_sgd_text>/synthetic_traditional_chinese_scene_text_v0/` | `webp`                  |
| **Regular dataset (Real)**                       |                                                                       |                         |
| `red`                                            | `<webds_image_reg>/red`                                               | `jpg`                   |
| `coyo_700m`                                      | `<webds_image_reg>/coyo_700m`                                         | `webp`                  |
| **Synthetic dataset (SGD)**                      |                                                                       |                         |
| `MMC4`                                           | `<webds_image_sgd>/mmc4`                                              | `webp`                  |
| `generations_qwen_image_2512_filtered_photoreal` | `<webds_image_sgd>/generations_qwen_image_2512_filtered_photoreal`    | `webp`                  |
| `wordnet_captions_20260224`                      | `<webds_image_sgd>/wordnet_captions_20260224`                         | `webp`                  |
| `datacomp_1b`                                    | `<webds_image_sgd>/datacomp_1b`                                       | `webp`                  |
| `midjourney`                                     | `<webds_image_sgd>/midjourney`                                        | `webp`                  |
| `midjourney_v6_20240703`                         | `<webds_image_sgd>/midjourney_v6_20240703`                            | `webp`                  |

### Table B — LanceDB

| `--dataset_name`                                 | `--lancedb_path`                                                                                                       |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Text dataset (Real + SGD)**                    |                                                                                                                        |
| `screen2words_rico`                              | `<lancedb_root>/regular_text/screen2words_rico_slice_from_maintable_<table_postfix>.lance`                             |
| `slide_audit`                                    | `<lancedb_root>/regular_text/slide_audit_slice_from_maintable_<table_postfix>.lance`                                   |
| `voxel51_rico`                                   | `<lancedb_root>/regular_text/voxel51_rico_slice_from_maintable_<table_postfix>.lance`                                  |
| `zennodo10k`                                     | `<lancedb_root>/regular_text/zennodo10k_slice_from_maintable_<table_postfix>.lance`                                    |
| `synthetic_scene_text_v0`                        | `<lancedb_root>/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_<table_postfix>.lance`                     |
| `synthetic_chinese_scene_text_v0`                | `<lancedb_root>/synthetic_text/synthetic_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance`             |
| `synthetic_traditional_chinese_scene_text_v0`    | `<lancedb_root>/synthetic_text/synthetic_traditional_chinese_scene_text_v0_slice_from_maintable_<table_postfix>.lance` |
|                                                  |                                                                                                                        |
| **Regular dataset (Real)**                       |                                                                                                                        |
| `red`                                            | `<lancedb_root>/regular/red_slice_from_maintable_<table_postfix>.lance`                                                |
| `coyo_700m`                                      | `<lancedb_root>/regular/coyo_700m_slice_from_maintable_<table_postfix>.lance`                                          |
|                                                  |                                                                                                                        |
| **Synthetic dataset (SGD)**                      |                                                                                                                        |
| `MMC4`                                           | `<lancedb_root>/synthetic/MMC4_slice_from_maintable_<table_postfix>.lance`                                             |
| `generations_qwen_image_2512_filtered_photoreal` | `<lancedb_root>/synthetic/generations_qwen_image_2512_filtered_photoreal_slice_from_maintable_<table_postfix>.lance`   |
| `wordnet_captions_20260224`                      | `<lancedb_root>/synthetic/wordnet_captions_20260224_slice_from_maintable_<table_postfix>.lance`                        |
| `datacomp_1b`                                    | `<lancedb_root>/synthetic/datacomp_1b_slice_from_maintable_<table_postfix>.lance`                                      |
| `midjourney`                                     | `<lancedb_root>/synthetic/midjourney_slice_from_maintable_<table_postfix>.lance`                                       |
| `midjourney_v6_20240703`                         | `<lancedb_root>/synthetic/midjourney_v6_20240703_slice_from_maintable_<table_postfix>.lance`                           |

When a row is filled in, the cheatsheet substitutes `--webds_path`, `--lancedb_path`, and `--logged_image_credential` directly — no per-launch user input is required for the new pipeline.

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 2x4 shard_full_db_<dataset_name> \
    pipelines/image/text_rendering/shard_full_db.py \
    --webds_path <the_existed_webdataset_path> \
    --webds_credential credentials/gcs.secret \
    --lancedb_path <lancedb_to_work_on> \
    --logged_image_credential credentials/gcs.secret \
    --dataset_name <dataset_name_that_to_process> \
    --target_image_format <target_image_format> \
    --num_concurrency 64
```

> **Cleanup on start**: rank 0 always finds and deletes all existing files under `--webds_path` before the run begins. This is unconditional — there is no `--mode` flag.

## Progress / status check

When the user asks for the **status of the dataset** (or "how far along is the shard", "count progress", etc.), compare tar counts under the two output keys (`images` and `metas`) summed across all `resolution_*/aspect_ratio_*/` leaves.

```bash
python /home/xingqianx/Project/trichord/helper/webds_tarcnt_by_key.py <dataset_name>
```

- Accepts a bare dataset name (e.g. `synthetic_scene_text_v0`) or a full `gcs:...` path.
- Keys to look for: `images` and `metas`. They should track each other — a healthy run has equal counts. A persistent gap means some tars finished one side but not the other (rare; would also surface as the tail RuntimeError).
- The helper is `wdinfo.json`-rooted; if a leaf has no `wdinfo.json` (this pipeline does not write one) the helper falls back to the dataset root and the per-leaf breakdown will be coarser. If you need a precise count, fall back to `python ~/Project/bashrc/s3_omni.py ls -r <webds_path> | grep '\.tar$' | wc -l`.

## Notes
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
- *No prerequisite layout.* Unlike the previous version, this script does **not** require an existing subtree under `--webds_path` — it plans tars purely from the Lance prefetch pass.
- Output is *split* into `images/` and `metas/` siblings. Images are transcoded to the format set by `--target_image_format` (jpg/jpeg → quality 95, webp → quality 85 method 5); the meta JSON gets `format_original` carrying the source format and `format` set to the target format whenever transcoding fired.
- Samples are dropped if they fail the per-bin minimum-resolution gate, if `(height, width)` are missing/invalid, or if PIL cannot verify/transcode the image bytes.
- A logged-image tar member whose basename is **not** in the filtered LanceDB is dropped without warning; a uuid that is in LanceDB but missing from the logged-image tars (or that fails image verify) leaves its tar incomplete, and the job raises `RuntimeError("Pipeline finished with incomplete tail")` at the very end.
- The script always wipes the output root before running (see cleanup note above). Re-running a job will overwrite from scratch.
- Tars are fixed at 1000 samples (`MetaSampleBuffer(nsample_each_tar=1000)`) and named `part000000/<NNNNNNNNN>.tar`. There are no `--samples-per-shard` / `--max-rows` flags.
- World size: this script is genuinely distributed (cross-rank gather of stragglers). `2x4` (2 nodes × 4 ranks) is the default working point; bump up if the logged-image bucket can sustain the bandwidth.
