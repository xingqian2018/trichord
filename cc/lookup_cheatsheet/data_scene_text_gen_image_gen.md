# Scene Text Generation — Image Generation (Stage 2)

Shared stage 2 for all SGD pipelines (English, Chinese). Runs a diffusion model locally on GPU to generate images from the structured prompts produced by stage 1.

- Code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py` in `imaginaire4`.

---

## Live Parameters

### English

| Name                   | Path                                                                                                      |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/<partXXXXXX>`        |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image/<partXXXXXX>`         |

### English Physical Domain

| Name                   | Path                                                                                                      |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1_phi/prompt/<partXXXXXX>`    |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1_phi/image/<partXXXXXX>`     |

### Simplified Chinese

| Name                   | Path                                                                                                              |
|------------------------|-------------------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1/prompt/<partXXXXXX>`        |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1/image/<partXXXXXX>`         |

### Simplified Chinese Physical Domain

| Name                   | Path                                                                                                              |
|------------------------|-------------------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1_phi/prompt/<partXXXXXX>`    |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1_phi/image/<partXXXXXX>`     |

### Traditional Chinese

| Name                   | Path                                                                                                                    |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1/prompt/<partXXXXXX>` |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1/image/<partXXXXXX>`  |

### Traditional Chinese Physical Domain

| Name                   | Path                                                                                                                        |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1_phi/prompt/<partXXXXXX>` |
| Output Image Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1_phi/image/<partXXXXXX>`  |

---

## Arguments

| Arg                     | Required | Default                  | Notes                                                                    |
|-------------------------|----------|--------------------------|--------------------------------------------------------------------------|
| `--input_folder`        | yes      | see live parameters      | S3/GCS path to the prompt folder (stage 1 output)                        |
| `--output_folder`       | yes      | see live parameters      | S3/GCS path to write generated images                                    |
| `--input_credential`    | no       | `credentials/gcs.secret` | Credential for the input bucket                                          |
| `--output_credential`   | no       | `credentials/gcs.secret` | Credential for the output bucket                                         |
| `--batch_size`          | no       | `2`                      | Images generated per rank per step                                       |
| `--max_retry`           | no       | `2`                      | Max generation retries per sample before dropping                        |
| `--guidance`            | no       | `4.0`                    | Classifier-free guidance scale                                           |
| `--aspect_ratio`        | no       | `1:1`                    | One of `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`               |
| `--num_inference_steps` | no       | `30`                     | Diffusion denoising steps                                                |
| `--num_concurrency`     | no       | `32`                     | Concurrent upload workers                                                |
| `--input_prompt_range`  | no       | all files                | Restrict to a file range, format: `start.json:end.json` (end exclusive) |
| `--disable_pbar`        | no       | off                      | Suppress progress bars for cleaner log output                            |

---

## Aspect Ratio Schedule

Aspect ratios are round-robined every 100 files (= 100K samples, since each JSON holds 1000 samples):

| k | File range (`start.json` : `end.json`, end exclusive) | Aspect ratio  |
|---|-------------------------------------------------------|---------------|
| 0 | `000000000.json` : `000000100.json`                   | `1:1`         |
| 1 | `000000100.json` : `000000200.json`                   | `4:3`         |
| 2 | `000000200.json` : `000000300.json`                   | `3:4`         |
| 3 | `000000300.json` : `000000400.json`                   | `16:9`        |
| 4 | `000000400.json` : `000000500.json`                   | `9:16`        |
| … | `{k*100:09d}.json` : `{(k+1)*100:09d}.json`           | cycle repeats |

---

## Run Templates

### Run template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch long <number_of_node_ask_user> scene_text_image_gen_<VERSION> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py \
    --input_folder <input_prompt_s3_path> \
    --output_folder <output_image_s3_path> \
    --input_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --aspect_ratio <aspect_ratio> \
    --batch_size 2 \
    --disable_pbar
```

> Path convention: always use `s3://` prefix for `--input_folder` and `--output_folder`, even though the bucket lives on GCS. If the user provides a `gcs://` or `gcs:` path, silently convert it to `s3://` before building the command.

### Run template — `user triggered local run`

```bash
PYTHONPATH=. \
.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py \
    --input_folder <input_prompt_s3_path> \
    --output_folder <output_image_s3_path> \
    --input_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --aspect_ratio <aspect_ratio> \
    --batch_size 4 \
    --disable_pbar
```

---

## Notes

- Requires GPU nodes — model (`Qwen/Qwen-Image-2512`) is loaded locally via diffusers, no API credential needed.
- Output images are saved as WEBP at `{output_folder}/{json_basename}/{key:012d}.webp`.
- The script auto-skips already-generated images on resume — just rerun with the same paths.
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
