# Scene Text Generation (i.e. SGD) — Chinese

LLM pipeline that generates synthetic Chinese scene-text structured captions for text-rendering datasets.

Output is saved as milestone JSON files: `{output_path}/{idx:09d}.json` (one file per `--milestone` samples).

This is a two stage run. (a) stage 1 generate prompt, (b) stage 2 generate images.

- Stage 1 code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_chinese.py` in `imaginaire4`.
- Stage 2 code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_image_gen.py` in `imaginaire4`.

---

# Stage 1, Prompt Generation

## Live parameters — Simplified Chinese

| Name                    | Path                                                                                                       |
|-------------------------|------------------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_simplified_chinese_v1.json`      |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1/image/<partXXXXXX>`  |

## Live parameters — Simplified Chinese Physical Domain

| Name                    | Path                                                                                                           |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_simplified_chinese_physical_v1.json` |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1_phi/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_chinese_v1_phi/image/<partXXXXXX>`  |

## Live parameters — Traditional Chinese

| Name                    | Path                                                                                                                   |
|-------------------------|------------------------------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_traditional_chinese_v1.json`                 |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1/image/<partXXXXXX>`  |

## Live parameters — Traditional Chinese Physical Domain

| Name                    | Path                                                                                                                       |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Taxonomy JSON           | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_traditional_chinese_physical_v1.json`            |
| Output Prompt Root Path | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1_phi/prompt/<partXXXXXX>` |
| Output Image Root Path  | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_traditional_chinese_v1_phi/image/<partXXXXXX>`  |


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

| Arg                      | Required   | Default                  | Notes                                                                                              |
|--------------------------|------------|--------------------------|----------------------------------------------------------------------------------------------------|
| `--taxonomy_json`        | yes        | see live parameters      | S3/GCS path to criteria taxonomy JSON                                                              |
| `--taxonomy_credential`  | no         | `credentials/gcs.secret` | Credential for the taxonomy bucket                                                                 |
| `--output_path`          | yes        | see live parameters      | Output directory for milestone JSON files                                                          |
| `--output_credential`    | no         | `credentials/gcs.secret` | Credential for the output bucket                                                                   |
| `--num_target_total_gen` | yes        | ask user                 | Total number of prompts to generate                                                                |
| `--model`                | no         | ask user                 | Check Model & Credential session for more info                                                     |
| `--language_variation`   | no         | `simplified`             | `simplified` → Simplified Chinese system prompt; `traditional` → Traditional Chinese system prompt |
| `--num_concurrency`      | no         | `128`                    | Concurrent LLM requests per rank                                                                   |
| `--timeout`              | no         | `400`                    | Per-request timeout in seconds                                                                     |
| `--batch_size`           | no         | `1024`                   | Samples processed per batch per rank                                                               |
| `--start_idx`            | no         | `0`                      | Resume from this index — must be milestone-aligned (mod `--milestone`)                             |
| `--milestone`            | no         | `1000`                   | Write a JSON file every N successfully generated prompts                                           |

---

## Run Template

### Run template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 scene_text_prompt_gen_chinese_<VERSION> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_chinese.py \
    --taxonomy_json <taxonomy_json_s3_path> \
    --taxonomy_credential credentials/gcs.secret \
    --output_path <output_s3_path> \
    --output_credential credentials/gcs.secret \
    --num_target_total_gen <total_count> \
    --language_variation <simplified|traditional> \
    --num_concurrency 128 \
    --batch_size 1024 \
    --timeout 400 \
    --milestone 1000
```

### Run Template — `user triggered local run`

- When using `qwen3-235b-a22b-instruct`

```bash
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_chinese.py \
    --taxonomy_json <taxonomy_json_s3_path> \
    --taxonomy_credential credentials/gcs.secret \
    --output_path <output_s3_path>/part000000 \
    --output_credential credentials/gcs.secret \
    --start_idx <ask_user> \
    --num_target_total_gen <total_count> \
    --language_variation <simplified|traditional> \
    --model qwen3-235b-a22b-instruct \
    --num_concurrency 128 \
    --batch_size 1024 \
    --timeout 400 \
    --milestone 1000
```

- When using `qwen3-235b-a22b-instruct-lepton`

```bash
LEPTON_API_QWEN3_235B=<credential> \
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_chinese.py \
    --taxonomy_json <taxonomy_json_s3_path> \
    --taxonomy_credential credentials/gcs.secret \
    --output_path <output_s3_path>/part000001 \
    --output_credential credentials/gcs.secret \
    --start_idx <ask_user> \
    --num_target_total_gen <total_count> \
    --language_variation <simplified|traditional> \
    --model qwen3-235b-a22b-instruct-lepton \
    --num_concurrency 128 \
    --batch_size 1024 \
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

See shared cheatsheet: `data_scene_text_gen_image_gen` — use the **Simplified Chinese**, **Simplified Chinese Physical Domain**, **Traditional Chinese**, or **Traditional Chinese Physical Domain** live parameter sections as appropriate.
