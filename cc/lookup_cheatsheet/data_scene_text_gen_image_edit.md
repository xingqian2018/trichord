# Scene Text — Image Edit (Optional Add-on)

Distributed image-editing pipeline that takes the structured prompt JSONs (stage 1 output) and the generated images (stage 2 output) and produces edited versions using `Qwen/Qwen-Image-Edit-2511`.

This is an **optional add-on** that can run after the main scene-text generation pipeline for various purposes (e.g. style transfer, quality refinement, domain adaptation). Multiple independent edit runs can be layered on top of the same generated images for different goals.

```
stage 1 (prompt gen)  →  stage 2 (image gen)
                                    ↓
                         [image edit — this script]  (optional, repeatable, purpose-specific)
```

- Stage 1 cheatsheet: `data_scene_text_gen_english` / `data_scene_text_gen_chinese`
- Stage 2 cheatsheet: `data_scene_text_gen_english` (Stage 2 section)
- Edit code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_edit.py` in `imaginaire4`

---

## Live Parameters

| Name                   | Path                                                                                                              |
|------------------------|-------------------------------------------------------------------------------------------------------------------|
| Input Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/<dataset_name>/prompt/<partXXXXXX>`                        |
| Input Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/<dataset_name>/image/<partXXXXXX>`                         |
| Output Edit Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/<dataset_name>/image_<purpose_of_edit>/<partXXXXXX>`       |

### `<dataset_name>` values

| `<dataset_name>`                              | Description                        |
|-----------------------------------------------|------------------------------------|
| `synthetic_scene_text_v1`                     | English                            |
| `synthetic_scene_text_v1_phi`                 | English — Physical Domain          |
| `synthetic_scene_text_chinese_v1`             | Simplified Chinese                 |
| `synthetic_scene_text_chinese_v1_phi`         | Simplified Chinese — Physical Domain |

---

## Arguments

| Arg                     | Required | Default                  | Notes                                                                    |
|-------------------------|----------|--------------------------|--------------------------------------------------------------------------|
| `--input_folder`        | yes      | see live parameters      | S3/GCS path to the prompt folder (stage 1 output, JSON files)            |
| `--image_folder`        | yes      | see live parameters      | S3/GCS path to the source image folder (stage 2 output, WEBP files)     |
| `--output_folder`       | yes      | see live parameters      | S3/GCS path to write the edited images                                   |
| `--input_credential`    | no       | `credentials/gcs.secret` | Credential for the prompt input bucket                                   |
| `--image_credential`    | no       | `credentials/gcs.secret` | Credential for the source image bucket                                   |
| `--output_credential`   | no       | `credentials/gcs.secret` | Credential for the output bucket                                         |
| `--input_prompt_range`  | no       | all files                | Restrict to a file range, format: `start.json:end.json` (end exclusive)  |
| `--batch_size`          | no       | `1`                      | Images edited per rank per step (only batch size 1 supported)            |
| `--max_retry`           | no       | `2`                      | Max edit retries per sample before dropping                              |
| `--true_cfg_scale`      | no       | `4.0`                    | CFG scale passed to the edit pipeline                                    |
| `--num_inference_steps` | no       | `40`                     | Diffusion denoising steps                                                |
| `--num_concurrency`     | no       | `32`                     | Concurrent S3 upload/download workers                                    |
| `--disable_pbar`        | no       | off                      | Suppress progress bars for cleaner log output                            |
| `--prompt_modification_fn` | yes   | —                        | See `--prompt_modification_fn` choices below                             |

### `--prompt_modification_fn` choices

| Value                                    | Description                                                              |
|------------------------------------------|--------------------------------------------------------------------------|
| `dense_caption_english_text_degradation` | Replace quoted text with garbled/foreign-script variants; use for English-phi DPO bad-image generation |
| `dense_caption_chinese_text_degradation` | Replace quoted text with garbled/foreign-script variants; use for Chinese-phi DPO bad-image generation |

---

## Run Template

The `--input_prompt_range` should match the file range used in Stage 2 (aspect ratios are round-robined every 100 files):

| k | File range (`start.json` : `end.json`, end exclusive) | Aspect ratio |
|---|-------------------------------------------------------|--------------|
| 0 | `000000000.json` : `000000100.json`                   | `1:1`        |
| 1 | `000000100.json` : `000000200.json`                   | `4:3`        |
| 2 | `000000200.json` : `000000300.json`                   | `3:4`        |
| 3 | `000000300.json` : `000000400.json`                   | `16:9`       |
| 4 | `000000400.json` : `000000500.json`                   | `9:16`       |
| … | `{k*100:09d}.json` : `{(k+1)*100:09d}.json`           | cycle repeats |

### Run template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch long <number_of_node_ask_user> scene_text_image_edit_<lauguage_version_variation>_<purpose_of_edit> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_edit.py \
    --input_folder <input_prompt_s3_path> \
    --image_folder <input_image_s3_path> \
    --output_folder <output_edit_s3_path> \
    --input_credential credentials/gcs.secret \
    --image_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --batch_size 1 \
    --prompt_modification_fn <dense_caption_chinese_text_degradation> \
    --disable_pbar
```

### Run template — `user triggered local run`

```bash
PYTHONPATH=. \
.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_edit.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/<partXXXXXX> \
    --image_folder s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image/<partXXXXXX> \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image_edit/<partXXXXXX> \
    --input_credential credentials/gcs.secret \
    --image_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --batch_size 16 \
    --prompt_modification_fn <none|dense_caption_chinese_text_degradation> \
    --disable_pbar
```

> Path convention: always use `s3://` prefix for all three folder args, even though the bucket lives on GCS. If the user provides a `gcs://` or `gcs:` path, silently convert it to `s3://` before building the command.

---

## Notes

- Requires GPU nodes — `Qwen/Qwen-Image-Edit-2511` is loaded locally via diffusers, no API credential needed.
- Each source image is downloaded from `--image_folder` at `{json_basename}/{key:012d}.webp` and edited in-place; the result is written to `--output_folder` at the same relative path.
- Samples whose source image cannot be downloaded are retried up to `--max_retry` times then dropped with a warning.
- The script auto-skips already-edited images on resume — just rerun with the same paths.
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
