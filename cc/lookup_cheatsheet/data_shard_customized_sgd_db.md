# Shard a customized SGD DB from prompt+image JSONs into WebDataset tars

Use this to materialize a **WebDataset layout for synthetic/generated data (SGD)** whose metadata lives in prompt JSON files rather than a LanceDB table. Unlike `shard_full_db`, there is no Lance prefetch pass — the script discovers all `.json` files under `--input_prompt_json_path`, distributes them across ranks in batches of 4, and streams image bytes from paths embedded in each JSON entry.

Output layout is identical to `shard_full_db`:

```
{output_webds_path}/<prekey_path_usually_buckets>/images/<postkey_path>/<NNNNNNNNN>.tar   # raw images
{output_webds_path}/<prekey_path_usually_buckets>/metas/<postkey_path>/<NNNNNNNNN>.tar    # JSON metas
```

Bucketing by **(resolution, aspect ratio)** is still applied — samples that fail the minimum-resolution gate are dropped.

Lives at `pipelines/image/text_rendering/shard_customized_sgd_db.py` in `imaginaire4_sila`.

## Some live information:

- Jobs are launched in *groups of 100 partitions* at a time using `--prompt_json_range`. The range format is `<basename_start>.json:<basename_end>.json` (inclusive start, exclusive end), e.g.:
  - Batch 0: `000000000.json:000000100.json`
  - Batch 1: `000000100.json:000000200.json`
  - Batch 2: `000000200.json:000000300.json`
  - … and so on.

## Quick-check inputs

| Arg                              | Value                                                                        | Notes                                                                                        |
|----------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `--input_prompt_json_path`       | `<path_to_prompt_json_dir>`                                                  | Directory of `.json` files containing per-sample metadata.                                   |
| `--input_prompt_json_credential` | `credentials/gcs.secret`                                                     | Credential for the prompt JSON bucket.                                                       |
| `--prompt_json_range`            | `<basename_start>.json:<basename_end>.json`                                  | Optional. Filter JSONs by *basename* range `[start, end)`.                                   |
| `--input_image_path`             | `<path_to_image_dir>`                                                        | Root directory of image files (combined with `--part_of_after_key_path` to form image paths). |
| `--input_image_credential`       | `credentials/gcs.secret`                                                     | Credential for the image bucket.                                                             |
| `--output_webds_path`            | `<the_existed_webdataset_path>`                                              | Output root. Tars land under `resolution_*/aspect_ratio_*/images/` + `metas/`.               |
| `--output_webds_credential`      | `credentials/gcs.secret`                                                     | Credential for the output WebDS bucket.                                                      |
| `--metadata_convertion_type`     | `sgdv1_1:1` / `sgdv1_16:9` / `sgdv1_9:16` / `sgdv1_4:3` / `sgdv1_3:4`     | *Required.* Sets injected `width`×`height` for every sample (see resolution table below).   |
| `--part_of_after_key_path`       | `part000000`                                                                 | Subdir prefix in tar postkey path. Default `part000000`.                                     |
| `--num_concurrency`              | `4`                                                                          | Per-task download/upload streams (default 4).                                                |
| `--target_image_format`          | `jpg` / `jpeg` / `webp`                                                      | *Required.* jpg/jpeg → quality 95; webp → quality 85 method 5.                              |

*`--metadata_convertion_type` resolution table:*

| Value          | Width | Height |
|----------------|-------|--------|
| `sgdv1_1:1`    | 1328  | 1328   |
| `sgdv1_16:9`   | 1664  |  928   |
| `sgdv1_9:16`   |  928  | 1664   |
| `sgdv1_4:3`    | 1472  | 1104   |
| `sgdv1_3:4`    | 1104  | 1472   |

> **No `--lancedb_path` / `--dataset_name`**: metadata comes entirely from the prompt JSON files. `width`/`height` are *not* read from the JSON — they are injected by `--metadata_convertion_type`. Each entry must contain `image_s3_range` (buffer sample-id) and `uuid` (output filename stem).

## Known dataset lookup

- **Full alias list: see `./data_common_root.md`**
- Input paths are partitioned: each root contains `part000000/`, `part000001/`, … sub-dirs holding the actual `.json` files. Point `--input_*_json_path` at the *root* (without the `part<XXXXXX>` suffix) so the script discovers all partitions, or at a specific `part<XXXXXX>/` to process one partition.

### Useful root aliases for this script

| Alias                    | Path                                                                           | Used for                         |
|--------------------------|--------------------------------------------------------------------------------|----------------------------------|
| `<sgd_datagen_root>`     | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data`                         | Input prompt + image JSON roots  |
| `<webds_image_sgd_text>` | `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text`           | `--output_webds_path` target     |

### Table A — WebDS output paths

| Dataset                                           | `--output_webds_path`                                                          | `--part_of_after_key_path`       |
|---------------------------------------------------|--------------------------------------------------------------------------------|----------------------------------|
| `synthetic_scene_text_v1`                         | `<webds_image_sgd_text>/synthetic_scene_text_v1`                               | `part000000` / `part000001` (ask)|
| `synthetic_scene_text_v1_phi`                     | `<webds_image_sgd_text>/synthetic_scene_text_v1_phi`                           | `part000000`                     |
| `synthetic_scene_text_chinese_v1`                 | `<webds_image_sgd_text>/synthetic_scene_text_chinese_v1`                       | `part000000`                     |
| `synthetic_scene_text_chinese_v1_phi`             | `<webds_image_sgd_text>/synthetic_scene_text_chinese_v1_phi`                   | `part000000`                     |
| `synthetic_scene_text_traditional_chinese_v1`     | `<webds_image_sgd_text>/synthetic_scene_text_traditional_chinese_v1`           | `part000000`                     |
| `synthetic_scene_text_traditional_chinese_v1_phi` | `<webds_image_sgd_text>/synthetic_scene_text_traditional_chinese_v1_phi`       | `part000000`                     |

### Table B — Prompt JSON paths

| Dataset                                           | `--input_prompt_json_path`                                                          |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| `synthetic_scene_text_v1`                         | `<sgd_datagen_root>/synthetic_scene_text_v1/prompt/`                                |
| `synthetic_scene_text_v1_phi`                     | `<sgd_datagen_root>/synthetic_scene_text_v1_phi/prompt/`                            |
| `synthetic_scene_text_chinese_v1`                 | `<sgd_datagen_root>/synthetic_scene_text_chinese_v1/prompt/`                        |
| `synthetic_scene_text_chinese_v1_phi`             | `<sgd_datagen_root>/synthetic_scene_text_chinese_v1_phi/prompt/`                    |
| `synthetic_scene_text_traditional_chinese_v1`     | `<sgd_datagen_root>/synthetic_scene_text_traditional_chinese_v1/prompt/`            |
| `synthetic_scene_text_traditional_chinese_v1_phi` | `<sgd_datagen_root>/synthetic_scene_text_traditional_chinese_v1_phi/prompt/`        |

### Table C — Image paths

| Dataset                                           | `--input_image_path`                                                                | `--target_image_format` |
|---------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------|
| `synthetic_scene_text_v1`                         | `<sgd_datagen_root>/synthetic_scene_text_v1/image/`                                 | `webp`                  |
| `synthetic_scene_text_v1_phi`                     | `<sgd_datagen_root>/synthetic_scene_text_v1_phi/image/`                             | `webp`                  |
| `synthetic_scene_text_chinese_v1`                 | `<sgd_datagen_root>/synthetic_scene_text_chinese_v1/image/`                         | `webp`                  |
| `synthetic_scene_text_chinese_v1_phi`             | `<sgd_datagen_root>/synthetic_scene_text_chinese_v1_phi/image/`                     | `webp`                  |
| `synthetic_scene_text_traditional_chinese_v1`     | `<sgd_datagen_root>/synthetic_scene_text_traditional_chinese_v1/image/`             | `webp`                  |
| `synthetic_scene_text_traditional_chinese_v1_phi` | `<sgd_datagen_root>/synthetic_scene_text_traditional_chinese_v1_phi/image/`         | `webp`                  |

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x4 shard_customized_sgd_db_<dataset_name> \
    pipelines/image/text_rendering/shard_customized_sgd_db.py \
    --input_prompt_json_path <path_to_prompt_json_dir> \
    --input_prompt_json_credential credentials/gcs.secret \
    --input_image_path <path_to_image_dir> \
    --input_image_credential credentials/gcs.secret \
    --output_webds_path <the_existed_webdataset_path> \
    --output_webds_credential credentials/gcs.secret \
    --metadata_convertion_type <sgdv1_aspect_ratio> \
    --target_image_format <target_image_format> \
    --num_concurrency 64 \
    --prompt_json_range <prompt_json_range> \
    --metadata_convertion_type <metadata_convertion_type> \
    --part_of_after_key_path <part_of_after_key_path>
```

> **No cleanup on start**: unlike `shard_full_db`, this script does *not* wipe the output root. Instead it *refuses to overwrite* — if a target tar already exists it raises `RuntimeError`. Safe to resume a partial run by re-submitting the same range; completed tars are skipped automatically.

Jobs are always launched per 100-partition batch — append `--prompt_json_range` and `--metadata_convertion_type` with the appropriate window. Aspect ratios are assigned by round-robin cycle every 100 files:

| k | `--prompt_json_range`                         | `--metadata_convertion_type` |
|---|-----------------------------------------------|------------------------------|
| 0 | `000000000.json:000000100.json`               | `sgdv1_1:1`                  |
| 1 | `000000100.json:000000200.json`               | `sgdv1_4:3`                  |
| 2 | `000000200.json:000000300.json`               | `sgdv1_3:4`                  |
| 3 | `000000300.json:000000400.json`               | `sgdv1_16:9`                 |
| 4 | `000000400.json:000000500.json`               | `sgdv1_9:16`                 |
| … | `{k*100:09d}.json:{(k+1)*100:09d}.json`       | cycle repeats                |

```bash
    --prompt_json_range 000000000.json:000000100.json
```

## Progress / status check

Same helper as `shard_full_db` — count tars under `images` and `metas` leaves:

```bash
python /home/xingqianx/Project/trichord/helper/webds_tarcnt_by_key.py <dataset_name_or_full_gcs_path>
```

- Keys to look for: `images` and `metas`. They should track each other. A persistent gap means one side lagged.
- For a raw count fallback: `python ~/Project/bashrc/s3_omni.py ls -r <output_webds_path> | grep '\.tar$' | wc -l`

## Notes

- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
- *No LanceDB dependency.* Metadata bucketing uses `width`/`height` *injected* by `--metadata_convertion_type` (from `SGDV1_RESOLUTIONS`), not fields read from the JSON.
- *Sample IDs*: the buffer key is `image_s3_range`; the output tar member name stem is `uuid` (zero-padded to 12 digits from `original_sgd_index`).
- *Tar naming is deterministic*: tar filename = `original_sgd_index // 1000`, formatted as `{part_of_after_key_path}/{tar_id:09d}.tar`. Not a rolling counter — same input always produces the same tar name.
- Samples are dropped if they fail the per-bin minimum-resolution gate, or if PIL cannot verify the image bytes. Image dimensions are also verified against the injected `width`/`height` — mismatch raises `RuntimeError`.
- *No auto-wipe*: unlike `shard_full_db`, the script refuses to overwrite existing tars — raises `RuntimeError` on collision. Re-submitting the same range resumes safely; already-written tars are skipped.
- Tars are fixed at 1000 samples (`nsample_each_tar=1000`).
- JSON files are read in sorted order and split across ranks in groups of 4 (`read_batch_size = 4`). Use `--prompt_json_range` (filters by file *basename*) to target a specific window.
- World size: `2x4` (2 nodes × 4 ranks) is the default working point; bump up if the storage bucket can sustain the bandwidth.
