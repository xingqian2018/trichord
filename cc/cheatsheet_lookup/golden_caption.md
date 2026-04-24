# Golden Caption Pipeline

Four-stage captioning pipeline (`stage1 → stage2 → stage3 → stage4`) plus a final conversion step that turns the entity list into dense + structured captions.

## General

Remote launch command is usually looks like:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
<some one time environment variables> \
slaunch cpu 1x1 <slurm_job_name> \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

For local-run command, launch through:

```bash
<some one time environment variables> \
.venv/bin/python \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

## Information Collection

This skill should collect some basic information **angle-bracket placeholders** (`<VERSION>`, `<VERSION_LONG>`, `<JUDGE_MODEL>`, `<GEN_MODEL>`, some times `<STAGE1_REF_VERSION>` for stage2) in able to create a final commend. **Do not guess or default silently.**
Enviornment variable (usually for credentials) should be resolved fully as plan text...
Go the credentials from JSON `credentials/gateway.json`, usually they key is the same name as the environment variable we want (i.e. LEPTON_API_QWEN3_VL_235B).


## Version and setting for fast lookup.

| Version | Version_Long | Judge (all stages VLM) | Gen (Stage 1,2,4 VLM) | Gen (Stage 3 LLM) | Specials |
|---|---|---|---|---|---|
| default |  | gemini-3.1-pro |  | If not mentioned, same as Gen (Stage 1,2,4 VLM) |  |
| v9 | golden_caption_v9_g3fg3p |  | gemini-3-flash |  | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v5_g3fg3p |
| v10 | v10_mixg3p |  | gemini-3-flash |  | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v6_q235g3p |
| v10p1 | v10p1_mixg3p |  | gemini-3.1-pro |  | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v6_q235g3p |
| v11 | v11_mixg3p |  | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v5_g3fg3p |
| v12 | v12_g3fg3p |  | gemini-3-flash |  | A newer pipeline of search and refinement prompting |
| v13 | v13_q235bg3p |  | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | |
| v14 | v14_mixg3p |  | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v12_g3fg3p |
| v15 | v15_mixg3p |  | gemini-3-flash |  | Only a stage 2-to-4 run, grabbing stage 1 result from golden_caption_v13_q235bg3p |

---

## Stage 1 — Entity Search Template

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>s1 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage1_entity_search.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage1/ \
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

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>s2 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_entity_structured_grounding.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_entity_list_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/<STAGE1_REF_VERSION>/stage1/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage2/ \
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

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>s3 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage3_entity_dense_grounding.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage2/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage3/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --max_retry 3 \
    --force_gen_model <STAGE3_GEN_MODEL>
```

---

## Stage 4 — Camera and Style

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>s4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage4_camera_and_style.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_json_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage3/ \
    --input_json_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/stage4/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --max_retry 3 \
    --force_gen_model <GEN_MODEL>
```

---

## Conversion — Entity List → Dense + Structured Captions

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 golden_caption_convert_<VERSION_SHORT> \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/convertion_v2.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION> \
    --input_credential credentials/gcs.secret \
    --num_concurrency 32
```

---