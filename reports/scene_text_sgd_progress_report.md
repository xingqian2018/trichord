# Scene Text SGD Synthetic Data Progress Report

**RawData Base Path:** `gcs:nv-00-10206-vfm/debug/xingqianx/synthetic_data`
**WebDS Base Path:** `gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text`
**Last Updated:** 2026-07-16 (refreshed x35)

---

## SGD Main Image Generation Progress

### How to Update This Report
When user is asking to update the report, it usually means to update this main progress report. And a full step 1 to 4 should be run

**Step 1 — Update Prompts & Images counts**
- Run `s3_omni.py ls` (profile `gcs`) on both `prompt/` and `image/` sub-paths under `RawData Base Path/<dataset>/`.
- Each dataset has sub-partitions `partXXXXXX/`. List every part.
- In `prompt/partXXXXXX/`, files are named `XXXXXXXXX.json` — the numeric stem is the global index of that 1000-sample chunk.
- In `image/partXXXXXX/`, entries are folders named `XXXXXXXXX/` — same global index, each holding 1000 images.
- Group indices by floor-of-100 (index 0–99 → group "0-100", 100–199 → "100-200", …) and count.
- Update the **Prompts** and **Images** columns. Skip rows where RawData is already ✅ — unless asked to recount.

**Step 2 — Update RawData column**
- Mark ✅ if Prompts = 100 and Images = 100 for that row. Exit-0 alone does not qualify.

**Step 3 — Update ImageGen Job ID**
- Run `ssh gcpcode 'squeue -u xingqianx -o "%.10i"'`.
- If a Job ID is no longer in the queue, replace it with `—`.
- If it becomes `—` and RawData is not ✅, mark it ⚠️ to flag unfinished work with no running job.

**Step 4 — Update WebDS column**
- Run `python helper/count_tar_groups.py <dataset_name>` for each dataset.
- Mark ✅ if the tar count for that group equals 100. Skip rows where WebDS is already ✅ — unless asked to recount.

**Step 5 — Request by user, show run command for image gen, image edit, or WebDS creation**
- If asked about **image gen**, read `cc/lookup_cheatsheet/data_scene_text_gen_image_gen.md`.
- If asked about **image edit**, read `cc/lookup_cheatsheet/data_scene_text_gen_image_edit.md`.
- If asked about **WebDataset / sharding (base images)**, read `cc/lookup_cheatsheet/data_shard_customized_sgd_db.md`.
- If asked about **WebDataset / sharding (edit images, i.e. `image_distorted_text`)**, read `cc/lookup_cheatsheet/data_shard_customized_sgd_db.md` — use the edit template with `--input_image_path` pointing to `image_distorted_text/`, `--output_webds_image_key images_distorted_text`, and `--output_webds_meta_key NULL or meta_ebc (entity-based-caption)`.

---

### The main report (finished)

| Dataset                             | Range     | Part       | Prompts | Images | ImageGen Job ID | RawData | WebDS |
|-------------------------------------|-----------|------------|--------:|-------:|-----------------|--------|-------|
| synthetic_scene_text_v1             | 0-100     | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 100-200   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 200-300   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 300-400   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 400-500   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 500-600   | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 600-700   | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 700-800   | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 800-900   | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 900-1000  | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 1000-1100 | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 1100-1200 | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 1200-1300 | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 1300-1400 | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1             | 1400-1500 | part000001 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1_phi         | 0-100     | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1_phi         | 100-200   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1_phi         | 200-300   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1_phi         | 300-400   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_v1_phi         | 400-500   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1     | 0-100     | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1     | 100-200   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1     | 200-300   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1     | 300-400   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1     | 400-500   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1_phi | 0-100     | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1_phi | 100-200   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1_phi | 200-300   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1_phi | 300-400   | part000000 |     100 |    100 | —               | ✅     | ✅    |
| synthetic_scene_text_chinese_v1_phi | 400-500   | part000000 |     100 |    100 | —               | ✅     | ✅    |

---

## SGD Special Image Editing Progress

Tracks add-on image variants generated by the edit pipeline on top of the main images. Each extra column corresponds to one edit purpose (i.e. one `image_<purpose>/` subfolder under the dataset path).
This table is only updated upon special request.

### How to Update This Section

**Step A — Sync Prompts & Images from main table**
- Copy the **Prompts** and **Images** values for each matching (Dataset, Range, Part) row from the main report table above.
- Do not re-query GCS for these — they are already authoritative in the main table.

**Step B — Update `image_distorted_text` column**
- Run `s3_omni.py ls` (profile `gcs`) on `image_distorted_text/partXXXXXX/` under `RawData Base Path/<dataset>/`.
- Each entry is a folder `XXXXXXXXX/` (same global index as the main image folder). Group by floor-of-100 and count.
- Update the `image_distorted_text` column. Mark ✅ if count = 100.

**Step C — Update WebDS column**
- Count tars under the `images_distorted_text` key in the WebDS output for each dataset/range:
  ```
  python ~/Project/bashrc/s3_omni.py ls gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/<dataset>/<res>/<ar>/images_distorted_text/<part>/ | grep -c '.tar'
  ```
- Mark ✅ if count = 100 for that group.

**Step D — Create `captions_ebc`**
- Run `create_caption_pickle.py` from `imaginaire4_sila` to convert `metas_ebc` into `captions_ebc` pickle tars.
- Launch one job per dataset:
  ```bash
  CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
  bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 create_caption_pickle_<dataset> \
      pipelines/image/text_rendering/create_caption_pickle.py \
      --webds_path s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text/<dataset> \
      --webds_credential credentials/gcs.secret \
      --meta_key metas_ebc \
      --out_key captions_ebc \
      --num_concurrency 8 \
      --metadata_to_caption_pickle_type sgd_converted_ebc
  ```
- Once `captions_ebc` is complete across all datasets, regenerate `wdinfo_unified.json` (see `data_create_wdinfo.md`) including keys: `images`, `metas`, `images_distorted_text`, `metas_ebc`, `captions_ebc`.

### The report

| Dataset                             | Range     | Part       | Prompts | Images | Edits | image_distorted_text | WebDS |
|-------------------------------------|-----------|------------|--------:|-------:|------:|---------------------:|-------|
| synthetic_scene_text_v1             | 0-100     | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1             | 100-200   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1             | 200-300   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1             | 500-600   | part000001 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1             | 600-700   | part000001 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1             | 1000-1100 | part000001 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1             | 1100-1200 | part000001 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1             | 1200-1300 | part000001 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1             | 1300-1400 | part000001 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1             | 1400-1500 | part000001 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_v1_phi         | 0-100     | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1_phi         | 100-200   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_v1_phi         | 200-300   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_chinese_v1     | 0-100     | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1     | 100-200   | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1     | 200-300   | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1     | 300-400   | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1     | 400-500   | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1_phi | 0-100     | part000000 |     100 |    100 |   100 |                   ✅ | ✅    |
| synthetic_scene_text_chinese_v1_phi | 100-200   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_chinese_v1_phi | 200-300   | part000000 |     100 |    100 |       |                      |       |
| synthetic_scene_text_chinese_v1_phi | 300-400   | part000000 |     100 |    100 |       |                      |       |

