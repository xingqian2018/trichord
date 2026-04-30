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

## Reporting performance / status

When the user asks "is X done?", "what's the result?", "check the precision/recall run", or any similar status/performance question, **read the result JSON from S3** — do **not** parse the slurm `.e` log tail.

The result JSONs live under the same `results/` folder the eval writes into:

```
s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/results/precision_eval_result_<SIGNATURE>.json
s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/results/recall_eval_result_<SIGNATURE>.json
```

Download via the `s3io` skill (`dl ... /tmp/s3io_<basename>`) and read with the `Read` tool. If a file is **missing**, the run hasn't reached the write step yet — report it as `running` / `queued` (cross-check with `squeue` on the cluster if needed) and **do not** fall back to log scraping.

### Precision — fields to report

From `precision_eval_result_<SIGNATURE>.json`:

1. **Precision** — `total_stats.accuracy`
2. **Precision claim count** — second pair in `success_cnt` (e.g. `success_cnt = "297/298, 1318/1318"` → claim count = `1318/1318`)
3. **Evaluation success image count** — first pair in `success_cnt` (e.g. `297/298`)

### Recall — fields to report

From `recall_eval_result_<SIGNATURE>.json`:

1. **Recall** — `recall`
2. **Recall success image count** — first pair in `success_cnt` (e.g. `success_cnt = "293/293, 2976/4780"` → image count = `293/293`)

### Report to file:

1. ~/Project/trichord/reports/golden_caption_analysis_long_version.md (long version)
2. ~/Project/trichord/reports/golden_caption_analysis.md (summary version)

### Rules

- **Be concise — one table is usually the entire response.** No preamble, no per-variant prose paragraphs, no recap of the commands that were launched. Just the table, plus at most one short line if something is genuinely off (e.g. a job failed).
- One compact table when multiple variants are reported together (`v*s3d`, `v*s3s`, `v*s4s`). Columns: variant / signature / the metric fields above / status.
- Mark missing-JSON variants as `running` (job in `squeue`) or `queued` (not yet started). Do not invent partial numbers.
- Do not paste the full JSON, the `Params:` block, progress bars, or per-rank `[R0]/[R1]/...` lines unless the user explicitly asks.
