# Scene Text Prompt Generation — English

Two-step LLM pipeline that generates synthetic English scene-text prompts for text-rendering datasets.

**Step 1 — Structured scene gen**: queries `qwen3-235b-a22b-instruct` to produce a YAML description of a scene with exactly N (2–5) text-bearing entities, each with a `text_and_signage` field.

**Step 2 — Prompt synthesis**: collapses the YAML into a prose paragraph of ≤40 words starting with a randomly sampled letter.

Output is saved as milestone JSON files: `{output_path}/{idx:09d}.json` (one file per `--milestone` samples). Each entry maps index → `{structured, prompt_short, model, num_text_region, timestamp}`.

Lives at `projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py` in `imaginaire4`.

---

## Credential

Needs `LEPTON_API_QWEN3_235B` — read it from `~/Project/trichord/credentials/gateway.json` (key `LEPTON_API_QWEN3_235B`).

---

## Live parameters

| Name            | Path |
|-----------------|------|
| Taxonomy JSON   | `s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json` |
| Output path     | `s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/` |

---

## Arguments

| Arg                      | Required | Default                    | Notes                                                                    |
|--------------------------|----------|----------------------------|--------------------------------------------------------------------------|
| `--taxonomy_json`        | yes      | see live parameters        | S3/GCS path to criteria taxonomy JSON                                    |
| `--taxonomy_credential`  | no       | `credentials/gcs.secret`   | Credential for the taxonomy bucket                                       |
| `--output_path`          | yes      | see live parameters        | Output directory for milestone JSON files                                |
| `--output_credential`    | no       | `credentials/gcs.secret`   | Credential for the output bucket                                         |
| `--num_target_total_gen` | yes      | ask user                   | Total number of prompts to generate                                      |
| `--model`                | no       | `qwen3-235b-a22b-instruct` | `qwen3-235b-a22b-instruct` → NVIDIA gateway (`nvidia/qwen/qwen-235b`); `qwen3-235b-a22b-instruct-lepton` → self-hosted vLLM (`LeptonTR/Qwen3-235B-A22B-Instruct-2507`) |
| `--num_concurrency`      | no       | `256`                      | Concurrent LLM requests per rank                                         |
| `--timeout`              | no       | `400`                      | Per-request timeout in seconds                                           |
| `--batch_size`           | no       | `256`                      | Samples processed per batch per rank                                     |
| `--start_idx`            | no       | `0`                        | Resume from this index — must be milestone-aligned (mod `--milestone`)   |
| `--milestone`            | no       | `1000`                     | Write a JSON file every N successfully generated prompts                 |

---

## Template — `slaunch`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 scene_text_prompt_gen_english_<VERSION> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json <taxonomy_json_s3_path> \
    --taxonomy_credential credentials/gcs.secret \
    --output_path <output_s3_path> \
    --output_credential credentials/gcs.secret \
    --num_target_total_gen <total_count> \
    --num_concurrency 16 \
    --batch_size 128 \
    --milestone 1000
```

## Template — `user triggered local run`

One example for using `qwen3-235b-a22b-instruct`

```bash
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json \
    --taxonomy_credential credentials/gcs.secret \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/part000000 \
    --output_credential credentials/gcs.secret \
    --start_idx 3000 \
    --num_target_total_gen 1000000 \
    --num_concurrency 256 \
    --batch_size 1024 \
    --model qwen3-235b-a22b-instruct \
    --timeout 400 \
    --milestone 1000
```


One example for using `qwen3-235b-a22b-instruct-lepton`

```bash
LEPTON_API_QWEN3_235B=<credential> \
.venv/bin/python \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/scene_text_prompt_gen_english.py \
    --taxonomy_json s3://nv-00-10206-vfm/debug/xingqianx/vfm_aid/datagen_taxonomy/scene_text_english_v1.json \
    --taxonomy_credential credentials/gcs.secret \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/part000001 \
    --output_credential credentials/gcs.secret \
    --start_idx 1003000 \
    --num_target_total_gen 2000000 \
    --num_concurrency 256 \
    --batch_size 1024 \
    --model qwen3-235b-a22b-instruct-lepton \
    --timeout 400 \
    --milestone 1000
```

To resume from a checkpoint, add `--start_idx <milestone_aligned_idx>` (e.g. `--start_idx 5000` if 5 milestone files already exist).

---

## Notes

- `--start_idx` must be divisible by `--milestone`, otherwise the script asserts.
- Each milestone file is named `{idx:09d}.json` and stores a dict keyed by integer index.
- The script is genuinely distributed — `1x4` (1 node × 4 ranks) is a sensible starting point; scale up for throughput.
- Never run it directly. Show the command as a formatted response first and ask the user's permission to run.
- Your command should follow the exact indent as the template shows.
