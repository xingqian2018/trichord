# Golden Caption Production Pipeline

The golden caption production pipeline with a total of 5 stages.

## General

## Step 1: Information Collection

- Get the stage number the user is working with via user input, or ask user when uncertain.
- Read `/home/xingqianx/Project/trichord/cc/lookup_cheatsheet/data_common_root.md` to get an idea of common arguments and paths.
  - By reading this cheetsheet and combine with the users input dataset name, you can get <WEBDS_FOLDER>
  - <OUTPUT_FOLDER> is close to <WEBDS_FOLDER> but with a postfix, i.e. `<root>/<datasetname>` -> `<root>/<datasetname>_gcraw/stage<N>`
- If RW from s3 location is needed, follow your `s3io` skill.
- Common settings:
  - `<JUDGE_MODEL>`, `<GEN_MODEL>` when not mentioned, ask the user if we should use `gemini-3.1-pro@nvidia`
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
slaunch cpu 1x1 golden_caption_<VERSION>_s1 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage1_entity_search.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 1 \
    --force_judge_model <JUDGE_MODEL> \
    --force_gen_model <GEN_MODEL>
```


### Stage 2 — Entity Structured Captioning

<Under construction>


### Stage 3 — Entity Dense Captioning

<Under construction>


## Stage 4 — Camera Lighting and Style

<Under construction>


## Stage 5 — Merge into Final Structured Caption

<Under construction>


## Checking status:

<Under construction>

