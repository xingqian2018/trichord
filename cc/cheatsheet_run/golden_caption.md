# Golden Caption Pipeline

Four-stage captioning pipeline (`stage1 → stage2 → stage3 → stage4`) plus a final conversion step that turns the entity list into dense + structured captions.

All stages launch through `slaunch cpu 1x1 <job_name>` against `imaginaire4`.

---

## Before Running — Ask the User First

The command blocks below use **angle-bracket placeholders** (`<VERSION>`, `<VERSION_SHORT>`, `<JUDGE_MODEL>`, `<GEN_MODEL>`, `<STAGE1_REF_VERSION>`) everywhere a per-run value is needed. Never launch with a placeholder still in the command — always resolve each one with the user first. **Do not guess or default silently.**

1. **Output folder version** (`<VERSION>`) — the long suffix used in the S3 output directory (e.g. `v10_mixg3p`, `v10p1_mixg3p`). Appears as `s3://.../CosCapBenchImage/golden_caption_<VERSION>/stageN/`.
2. **Slurm job short tag** (`<VERSION_SHORT>`) — a compact identifier for the slurm job name, which shows up in logs and should be short (e.g. `v10`, `v10p1`, `v5`). Appears as `golden_caption_<VERSION_SHORT>s1`, `...s2`, etc. Typically a prefix of `<VERSION>` but not required to match.
3. **Force models** — `--force_judge_model <JUDGE_MODEL>` and `--force_gen_model <GEN_MODEL>`. Ask per stage if the user has not said. Common options: `gemini-3.1-pro`, `gemini-3-flash`, `qwen3-vl-235b-a22b-instruct`.
4. **Stage 2 only — confirm the stage-1 input path** (`<STAGE1_REF_VERSION>`). Stage 2 reads `--input_entity_list_folder` from a *prior* stage-1 run, which may be a different version than the one you're writing to. Examples seen in practice: `v6_q235g3p`, or the same `<VERSION>` as the current run. Always confirm which stage-1 output to consume before launching stage 2.

Substitute every placeholder into the command before running.

---

## Stage 1 — Entity Search

Searches for entities in each image; produces per-image entity lists.

- **Job name:** `golden_caption_<VERSION_SHORT>s1`
- **Script:** `projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage1_entity_search.py`
- **Example models (ask user):** judge `gemini-3.1-pro`, gen `gemini-3-flash`; up to 5 battle rounds

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=$(jq -r 'LEPTON_API_QWEN3_VL_235B' credentials/gateway.json) \
slaunch cpu 1x1 golden_caption_<VERSION_SHORT>s1 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage1_entity_search.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage1/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --timeout 400 \
    --max_retry 3 \
    --max_battle_rounds 5 \
    --force_judge_model <JUDGE_MODEL> \
    --force_gen_model <GEN_MODEL>
```

---

## Stage 2 — Entity Structured Grounding

Takes the stage-1 entity list and produces structured groundings.

- **Job name:** `golden_caption_<VERSION_SHORT>s2`
- **Script:** `projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_entity_structured_grounding.py`
- **Example models (ask user):** judge `gemini-3.1-pro`, gen `gemini-3.1-pro`
- **Note:** reads `--input_entity_list_folder` from a **prior stage-1 run** (`<STAGE1_REF_VERSION>`), which may differ from `<VERSION>`. Confirm with the user which stage-1 output to consume.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=$(jq -r 'LEPTON_API_QWEN3_VL_235B' credentials/gateway.json) \
slaunch cpu 1x1 golden_caption_<VERSION_SHORT>s2 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_entity_structured_grounding.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_entity_list_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<STAGE1_REF_VERSION>/stage1/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage2/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --timeout 400 \
    --max_retry 3 \
    --max_battle_rounds 5 \
    --force_judge_model <JUDGE_MODEL> \
    --force_gen_model <GEN_MODEL>
```

---

## Stage 3 — Entity Dense Grounding

Dense grounding over the stage-2 structured output.

- **Job name:** `golden_caption_<VERSION_SHORT>s3`
- **Script:** `projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage3_entity_dense_grounding.py`
- **Example model (ask user):** gen `gemini-3.1-pro`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=$(jq -r 'LEPTON_API_QWEN3_VL_235B' credentials/gateway.json) \
slaunch cpu 1x1 golden_caption_<VERSION_SHORT>s3 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage3_entity_dense_grounding.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage2/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage3/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --max_retry 3 \
    --force_gen_model <GEN_MODEL>
```

---

## Stage 4 — Camera and Style

Adds camera and style captions using the stage-3 JSON plus the original images.

- **Job name:** `golden_caption_<VERSION_SHORT>s4`
- **Script:** `projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage4_camera_and_style.py`
- **Example model (ask user):** gen `gemini-3-flash`
- **Note:** takes two input folders — raw images (`V1/`) and stage-3 JSON

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=$(jq -r 'LEPTON_API_QWEN3_VL_235B' credentials/gateway.json) \
slaunch cpu 1x1 golden_caption_<VERSION_SHORT>s4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage4_camera_and_style.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_json_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage3/ \
    --input_json_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage4/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --max_retry 3 \
    --force_gen_model <GEN_MODEL>
```

---

## Conversion — Entity List → Dense + Structured Captions

Runs after the four stages; consumes the whole `golden_caption_<VERSION>/` tree.

- **Job name:** `golden_caption_convert_<VERSION_SHORT>`
- **Script:** `projects/cosmos3/vfm/evaluation/captioning/golden_caption/convertion.py`

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 golden_caption_convert_<VERSION_SHORT> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/convertion_v2.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION> \
    --input_credential credentials/gcs.secret \
    --num_concurrency 32
```

---

## Quick Reference

| Stage | Concurrency | Timeout | Gen model (example) | Judge model (example) |
|---|---|---|---|---|
| 1 | 32 | 400 | `gemini-3-flash` | `gemini-3.1-pro` |
| 2 | 32 | 400 | `gemini-3.1-pro` | `gemini-3.1-pro` |
| 3 | 32 | — | `gemini-3.1-pro` | — |
| 4 | 32 | — | `gemini-3-flash` | — |
| convert | 32 | — | — | — |
