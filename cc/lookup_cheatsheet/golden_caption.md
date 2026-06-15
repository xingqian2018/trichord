# Golden Caption Pipeline

Four-stage captioning pipeline (`stage1 → stage2 → stage3 → stage4`) plus a final conversion step that turns the entity list into dense + structured captions.

## General


## Step 1: Information Collection

- Get the stage number the user is working with via user input, or ask user when uncertain.
- Common settings:
  - `<EXPNAME> = exp_20260612`
  - `<JUDGE_MODEL>`, `<GEN_MODEL>` when not mentioned, ask the user if we should use `gemini-3.1-pro`
- Collect the necessary argument from input, or ask user when uncertain.
  - Stage 1: `<VERSION>`, `<JUDGE_MODEL>`, `<GEN_MODEL>`
  - Stage 2: `<VERSION>`, `<JUDGE_MODEL>`, `<GEN_MODEL>` `<STAGE1_VERSION>`
  - Stage 3: 
  - Stage 4: 
- Understand whether the user want to launch remotely or locally

This skill should collect some basic information **angle-bracket placeholders** (`<VERSION>`, `<JUDGE_MODEL>`, `<GEN_MODEL>`, some times `<STAGE1_REF_VERSION>` for stage2) in able to create a final commend. **Do not guess or default silently.**
Enviornment variable (usually for credentials) should be resolved fully as plan text...
Go the credentials from JSON `~/Project/trichord/credentials/gateway.json`, usually they key is the same name as the environment variable we want (i.e. LEPTON_API_QWEN3_VL_235B).


## Step 2: Compose Command and Show User

Now we have all target information, we should show the user the command (cmd) it asked for. When there are still place uncertain, ask user.

### Template for `remote` vs. `local` Launch

- Remote launch command is usually looks like:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
<some one time environment variables> \
slaunch cpu 1x1 <slurm_job_name> \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

- For local-run command, launch through:

```bash
<some one time environment variables> \
.venv/bin/python \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

### Stage 1 — Entity Search Template

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>_s1 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage1_entity_search.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/<EXPNAME>/<VERSION>/stage1/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 5 \
    --force_judge_model <JUDGE_MODEL> \
    --force_gen_model <GEN_MODEL>
```


### Stage 2 — Entity Structured Captioning

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_<VERSION>s2 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_structured_captioning.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_entity_list_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/<EXPNAME>/<STAGE1_VERSION>/stage1/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/<EXPNAME>/<VERSION>/stage2/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 5 \
    --force_judge_model <JUDGE_MODEL> \
    --force_gen_model <GEN_MODEL>
```


### Stage 3 — Entity Dense Grounding

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
    --timeout 400 \
    --max_retry 3 \
    --force_gen_model <STAGE3_GEN_MODEL>
```


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

## Checking status:

The trick to check status don't need to go find program logs. The logs are long so too long don't read. A much reliable way are checking folder cnt with the following command.

Output as to user as concise as possible.

```bash
s3 s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/stage<...>/ cnt
```

_(P.S. check your s3io skill)_