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
  - Stage 3: `<VERSION>`, `<GEN_MODEL>`, `<STAGE2_OUTPUT_FOLDER>`
  - Stage 4: `<VERSION>`, `<GEN_MODEL>`
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
slaunch cpu 1x1 gcprod_<VERSION>_s1 \
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

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_s2 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage2_structured_captioning.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_entity_list_folder <STAGE1_OUTPUT_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 5 \
    --force_captioning_model <GEN_MODEL> \
    --force_judge_model <JUDGE_MODEL> \
    --force_refine_model <REFINE_MODEL>
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_red_s2 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage2_structured_captioning.py \
    --input_webds_folder       s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_entity_list_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder            s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s2/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 256 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 1 \
    --force_captioning_model gpt-5.5 \
    --force_judge_model gpt-5.5 \
    --force_refine_model gpt-5.5
```


### Stage 3 — Entity Dense Captioning

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_s3 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage3_dense_captioning.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_stage2_folder <STAGE2_OUTPUT_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300 \
    --timeout 200 \
    --max_retry 3 \
    --force_gen_model <GEN_MODEL>
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_red_s3 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage3_dense_captioning.py \
    --input_webds_folder s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_stage2_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s2/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s3/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_gen_model gemini-3.1-pro@nvidiak
```


### Stage 4 — Camera, Lighting, Style and Quality

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_s4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage4_camera_lighting_style_and_quality.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_gen_model <GEN_MODEL>
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_red_s4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage4_camera_lighting_style_and_quality.py \
    --input_webds_folder       s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder            s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s4/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_gen_model gpt-5.5 \
```


## Stage 5 — Merge into Final Structured Caption

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_s5 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage5_finalize_caption.py \
    --input_webds_folder  <WEBDS_FOLDER> \
    --input_stage1_folder <STAGE1_OUTPUT_FOLDER> \
    --input_stage2_folder <STAGE2_OUTPUT_FOLDER> \
    --input_stage3_folder <STAGE3_OUTPUT_FOLDER> \
    --input_stage4_folder <STAGE4_OUTPUT_FOLDER> \
    --input_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_red_s5 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stage5_finalize_caption.py \
    --input_webds_folder  s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_stage1_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw/ \
    --input_stage2_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s2/ \
    --input_stage3_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s3/ \
    --input_stage4_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s4/ \
    --input_credential credentials/gcs.secret \
    --output_folder       s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s5/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300
```


## MX Tier0 Direct Captioning (`gcprod_mx_tier0.py`)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_mx_tier0 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_mx_tier0.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 100 \
    --timeout 100 \
    --max_retry 3 \
    --force_model <GEN_MODEL>
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_mx_tier0 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_mx_tier0.py \
    --input_webds_folder s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/webdataset_image_regular/red_mxtier0/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 100 \
    --timeout 1000 \
    --max_retry 10 \
    --force_model seed-2.1-pro
```


## MX Tier1 Direct Captioning (`gcprod_mx_tier1.py`)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_mx_tier1 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_mx_tier1.py \
    --input_webds_folder <WEBDS_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --output_folder <OUTPUT_FOLDER> \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 100 \
    --timeout 100 \
    --max_retry 3 \
    --force_model <GEN_MODEL>
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_mx_tier1_nocap_red \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_mx_tier1.py \
    --input_webds_folder s3://nv-00-10206-vfm/webdataset_debug/dev/red/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/webdataset_debug/dev/red_mxtier1/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_model gemini-3.1-pro@nvidiams
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_mx_tier1_nocap_coyo \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_mx_tier1_filter2.py \
    --input_webds_folder s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1_high_quality/coyo_700m/ \
    --input_webds_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/webdataset_debug/dev/v1_high_quality_coyo700m_mxtier1/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_model gpt-5.5
```


## Stage X — Caption Collection into One Meta / One Pickle Caption

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_<VERSION>_stageX \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stageX_caption_collector.py \
    --io_webds_path              <WEBDS_FOLDER> \
    --input_mx_tier0_path        <MX_TIER0_OUTPUT_FOLDER> \
    --input_mx_tier1_path        <MX_TIER1_OUTPUT_FOLDER> \
    --input_golden_caption_path  <STAGE5_OUTPUT_FOLDER> \
    --input_webds_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 4000
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x1 gcprod_red_stageX \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/gcprod_stageX_caption_collector.py \
    --io_webds_path             s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --input_mx_tier0_path       s3://nv-00-10206-vfm/webdataset_image_regular/red_mxtier0/ \
    --input_mx_tier1_path       s3://nv-00-10206-vfm/webdataset_image_regular/red_mxtier1/ \
    --input_golden_caption_path s3://nv-00-10206-vfm/webdataset_image_regular/red_gcraw_s5/ \
    --input_webds_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --regenerate
```

## Wdinfo Creation

Script lives in `imaginaire4_sila` (not `imaginaire4`).

- `--webds_key` lists every key (subdir name) that must be present across all tars. Only tars matched across **all** keys are included. Use `metas` as the anchor — coverage stat is printed per key.
- Outputs write back into `--webds_path`: per-group `wdinfo/` JSONs + a single `wdinfo_unified.json`.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 wdinfo_<VERSION> \
    pipelines/image/text_rendering/create_wdinfo.py \
    --webds_path <WEBDS_FOLDER> \
    --webds_cred credentials/gcs.secret \
    --webds_key  images metas meta_golden caption_golden \
    --num_concurrency 128 \
    --batch_size 32
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 wdinfo_red \
    pipelines/image/text_rendering/create_wdinfo.py \
    --webds_path s3://nv-00-10206-vfm/webdataset_image_regular/red/ \
    --webds_cred credentials/gcs.secret \
    --webds_key images metas meta_golden caption_golden \
    --wdinfo_path s3://nv-00-10206-vfm/webdataset_image_regular/red/wdinfo_golden_caption/ \
    --wdinfo_unified_path s3://nv-00-10206-vfm/webdataset_image_regular/red/wdinfo_unified_golden_caption.json \
    --num_concurrency 128 \
    --batch_size 32
```

