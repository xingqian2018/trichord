# Scene Text Generation (i.e. SGD) — English 

LLM pipeline that generates synthetic English scene-text structured captions for text-rendering datasets.

Output is saved as milestone JSON files: `{output_path}/{idx:09d}.json` (one file per `--milestone` samples).

This is a two stage run. (a) stage 1 generate prompt, (b) stage 2 generate images.

- Stage 1 code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py` in `imaginaire4`.
- Stage 2 code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py` in `imaginaire4`.

---

# Stage 1, Prompt Generation

## Live parameters — English

| Name                    | Path                                                                                              |
|-------------------------|---------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json`        |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image/<partXXXXXX>`  |

## Live parameters — English Physical Domain

| Name                    | Path                                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_physical_v1.json`   |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1_phi/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1_phi/image/<partXXXXXX>`  |

---

## Model & Credential

`--model` accepts three values:

| Value                                | Backend          | Credential needed        |
|--------------------------------------|------------------|--------------------------|
| `qwen3-235b-a22b-instruct`           | NVIDIA gateway   | none                     |
| `qwen3-235b-a22b-instruct-lepton`    | Lepton endpoint  | `LEPTON_API_QWEN3_235B`  |
| `qwen3-235b-a22b-instruct-lepton-2`  | Lepton endpoint  | `LEPTON_API_QWEN3_235B`  |

For either lepton variant, read the credential from `~/Project/trichord/credentials/gateway.json` (key `LEPTON_API_QWEN3_235B`) and pass it as the env var `LEPTON_API_QWEN3_235B`.

---

## Arguments

| Arg                      | Required | Default                    | Notes                                                                    |
|--------------------------|----------|----------------------------|--------------------------------------------------------------------------|
| `--taxonomy_json`        | yes      | see live parameters        | S3/GCS path to criteria taxonomy JSON                                    |
| `--taxonomy_credential`  | no       | `credentials/gcs.secret`   | Credential for the taxonomy bucket                                       |
| `--output_path`          | yes      | see live parameters        | Output directory for milestone JSON files                                |
| `--output_credential`    | no       | `credentials/gcs.secret`   | Credential for the output bucket                                         |
| `--num_target_total_gen` | yes      | ask user                   | Total number of prompts to generate                                      |
| `--model`                | no         | ask user                 | Check Model & Credential session for more info                                                     |
| `--num_concurrency`      | no       | `128`                      | Concurrent LLM requests per rank                                         |
| `--timeout`              | no       | `400`                      | Per-request timeout in seconds                                           |
| `--batch_size`           | no       | `1024`                     | Samples processed per batch per rank                                     |
| `--start_idx`            | no       | `0`                        | Resume from this index — must be milestone-aligned (mod `--milestone`)   |
| `--milestone`            | no       | `1000`                     | Write a JSON file every N successfully generated prompts                 |

---

## Run Template

### Run template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 scene_text_prompt_gen_english_<VERSION> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json <taxonomy_json_s3_path> \
    --taxonomy_credential credentials/gcs.secret \
    --output_path <output_s3_path> \
    --output_credential credentials/gcs.secret \
    --num_target_total_gen <total_count> \
    --num_concurrency 256 \
    --batch_size 1024 \
    --milestone 1000
```

### Run Template — `user triggered local run`

- When using `qwen3-235b-a22b-instruct`

```bash
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json \
    --taxonomy_credential credentials/gcs.secret \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/part000000 \
    --output_credential credentials/gcs.secret \
    --start_idx <ask_user> \
    --num_target_total_gen 1000000 \
    --num_concurrency 256 \
    --batch_size 1024 \
    --model qwen3-235b-a22b-instruct \
    --timeout 400 \
    --milestone 1000
```

- When using `qwen3-235b-a22b-instruct-lepton`

```bash
LEPTON_API_QWEN3_235B=<credential> \
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json \
    --taxonomy_credential credentials/gcs.secret \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/part000001 \
    --output_credential credentials/gcs.secret \
    --start_idx <ask_user> \
    --num_target_total_gen 2000000 \
    --num_concurrency 256 \
    --batch_size 1024 \
    --model qwen3-235b-a22b-instruct-lepton \
    --timeout 400 \
    --milestone 1000
```

- When user is not sure about the `--start_idx`, you can check the result folder and see what is the last file it reaches.
- `--start_idx` must be divisible by `--milestone`, otherwise the script asserts.

---

## Notes

- Each milestone file is named `{idx:09d}.json` and stores a dict keyed by integer index.
- The script is genuinely distributed — `1x4` (1 node × 4 ranks) is a sensible starting point; scale up for throughput.
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.

---

# Stage 2, Image Generation

## Live parameters

| Name                    | Path |
|-------------------------|------|
| Input Prompt Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image/` |

---

## Arguments

| Arg                      | Required | Default                  | Notes                                                                        |
|--------------------------|----------|--------------------------|------------------------------------------------------------------------------|
| `--input_folder`         | yes      | see live parameters      | S3/GCS path to the prompt folder (stage 1 output)                            |
| `--output_folder`        | yes      | see live parameters      | S3/GCS path to write generated images                                        |
| `--input_credential`     | no       | `credentials/gcs.secret` | Credential for the input bucket                                              |
| `--output_credential`    | no       | `credentials/gcs.secret` | Credential for the output bucket                                             |
| `--batch_size`           | no       | `2`                      | Images generated per rank per step                                           |
| `--max_retry`            | no       | `2`                      | Max generation retries per sample before dropping                            |
| `--guidance`             | no       | `4.0`                    | Classifier-free guidance scale                                               |
| `--aspect_ratio`         | no       | `1:1`                    | One of `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`                   |
| `--num_inference_steps`  | no       | `30`                     | Diffusion denoising steps                                                    |
| `--num_concurrency`      | no       | `32`                     | Concurrent upload workers                                                    |
| `--input_prompt_range`   | no       | all files                | Restrict to a file range, format: `start.json:end.json` (end exclusive)     |

---

## Run Template

Aspect ratios are round-robined every 100 files (= 100K samples, since each JSON holds 1000 samples):

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
slaunch small <number_of_node_ask_user> scene_text_image_gen_<VERSION> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py \
    --input_folder <input_prompt_s3_path> \
    --output_folder <output_image_s3_path> \
    --input_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --aspect_ratio <aspect_ratio> \
    --batch_size 2
```

### Run Template — `user triggered local run`

```bash
PYTHONPATH=. \
.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/image/ \
    --input_credential credentials/gcs.secret \
    --output_credential credentials/gcs.secret \
    --input_prompt_range <start>.json:<end>.json \
    --aspect_ratio <aspect_ratio> \
    --batch_size 4
```

---

## Notes

- Requires GPU nodes — model (`Qwen/Qwen-Image-2512`) is loaded locally via diffusers, no API credential needed.
- Output images are saved as WEBP at `{output_folder}/{json_basename}/{key:012d}.webp`.
- The script auto-skips already-generated images on resume — just rerun with the same paths.
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
