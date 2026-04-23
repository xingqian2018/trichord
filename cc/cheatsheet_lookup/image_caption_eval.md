# CosCapBench Image Caption Evaluation

Two evaluation tasks (`precision` and `recall`) run against a golden-caption output directory. Each task is run once per caption variant (`stage3_dense`, `stage3_structured`, `stage4_structured`), producing three runs per task.

## General

Remote launch command usually looks like:

```bash
slaunch cpu 1x4 <slurm_job_name> \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

For local-run command, launch through:

```bash
.venv/bin/python \
    <python_relative_code_path> \
    <nicely_organized_argument>
```

## Information Collection

This skill should collect some basic information **angle-bracket placeholders** (`<VERSION>`, `<VERSION_LONG>`, `<CAPTION_STAGE>`, `<SIGNATURE>`, `<JUDGE_MODEL>`) in order to create a final command. **Do not guess or default silently.**
Environment variables (usually for credentials) should be resolved fully as plain text.
Get credentials from JSON `credentials/gateway.json`; usually the key matches the environment variable name.

**Version resolution — preemptively look up on S3.** When the user says a short version (e.g. `v9`, `v10p1`), do **not** guess `<VERSION_LONG>` or the caption sub-folder names. List the benchmark root on S3 and pick the exact folder that matches:

```bash
# list candidate golden_caption_<VERSION_LONG>/ folders
python ~/Project/bashrc/s3_omni.py ls s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/ \
    --cred credentials/gcs.secret | grep golden_caption_<VERSION>
```

Then list the picked folder to confirm which `<CAPTION_STAGE>` sub-folders actually exist (`stage3_dense_caption/`, `stage3_structured_caption/`, `stage4_structured_caption/`):

```bash
python ~/Project/bashrc/s3_omni.py ls s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/ \
    --cred credentials/gcs.secret
```

If more than one folder matches `<VERSION>`, ask the user which one. If none matches, stop and ask — do not invent a `<VERSION_LONG>`.

## Caption variants for fast lookup.

For each `<VERSION_LONG>`, three caption variants are evaluated. Each variant maps to a `<CAPTION_STAGE>` sub-folder and a `<SIGNATURE>` tag that feeds both the slurm job name and the `--signature` flag.

| Variant | `<CAPTION_STAGE>` | `<SIGNATURE>` |
|---|---|---|
| Stage 3 dense | `stage3_dense_caption` | `<VERSION>s3d` |
| Stage 3 structured | `stage3_structured_caption` | `<VERSION>s3s` |
| Stage 4 structured | `stage4_structured_caption` | `<VERSION>s4s` |

## Default settings for fast lookup.

| Version | Version_Long | Judge Model | Notes |
|---|---|---|---|
| default |  | gemini-3.1-pro | Precision: `num_concurrency=32`, `batch_size=64`, `max_claims_per_call=16`. Recall: `num_concurrency=32`, `batch_size=64`, `max_retry=3`, `max_assertions_per_call=16`. |
| v9 | v9_g3fg3p | gemini-3.1-pro |  |

---

## Precision — Image Caption

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x4 coscapbench_precision_<SIGNATURE> \
    projects/cosmos3/vfm/evaluation/captioning/evaluate_image_caption_precision.py \
    --image_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --image_cred credentials/gcs.secret \
    --caption_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/<CAPTION_STAGE>/ \
    --caption_cred credentials/gcs.secret \
    --output_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/results/ \
    --output_cred credentials/gcs.secret \
    --judge_model <JUDGE_MODEL> \
    --batch_size 64 \
    --num_concurrency 32 \
    --max_claims_per_call 16 \
    --signature <SIGNATURE>
```

---

## Recall — Image Caption

The command template is the following...

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
slaunch cpu 1x4 coscapbench_recall_<SIGNATURE> \
    projects/cosmos3/vfm/evaluation/captioning/evaluate_image_caption_recall.py \
    --assertion_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1_assertion/ \
    --assertion_cred credentials/gcs.secret \
    --caption_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/<CAPTION_STAGE>/ \
    --caption_cred credentials/gcs.secret \
    --output_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/results/ \
    --output_cred credentials/gcs.secret \
    --judge_model <JUDGE_MODEL> \
    --batch_size 64 \
    --num_concurrency 32 \
    --max_retry 3 \
    --max_assertions_per_call 16 \
    --signature <SIGNATURE>
```

> Note: the stage 4 structured recall run historically used `--num_concurrency 30` instead of 32. Lower it only if you hit rate limits.

---

## Local run

To run any of the above locally instead of through `slaunch`, swap the prefix:

```bash
slaunch cpu 1x4 <slurm_job_name> \
```

with:

```bash
.venv/bin/python \
```

All other arguments stay the same.

---
