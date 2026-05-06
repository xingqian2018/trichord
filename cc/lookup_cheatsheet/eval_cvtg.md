# CVTG (Complex Visual Text Generation) evaluation

Two stages: Stage 1 generates images on GPU, Stage 2 OCRs them with a VLM judge and scores text fidelity (GNED / PNED).

The skill needs to gather information from user when necessary.

## Benchmark name (`--benchmark_name`)

Default: `cvtg2kL`

Canonical (in `BENCHMARK_CHOICE` of `inference_cvtg_distributed.py`, prompts root `s3://datasets/cvtg/`):

| Key          | JSON path                                       | Prompt field used | OCR language | Notes              |
|--------------|-------------------------------------------------|-------------------|--------------|--------------------|
| `cvtg2k`     | `cvtg_2kl/testing_prompt_2kl.json`              | `prompt`          | English      | raw prompt         |
| `cvtg2kL`    | `cvtg_2kl/testing_prompt_2kl.json`              | `prompt_upsampled`| English      | default, upsampled |
| `cvtg500L`   | `cvtg_500l/testing_prompt_500l.json`            | `prompt_upsampled`| English      | upsampled          |
| `cvtg102ch`  | `cvtg_102ch/testing_prompt_102ch.json`          | `prompt`          | Chinese      | Chinese OCR        |

Note: Stage 2's language switch is automatic — `cvtg102ch` triggers Chinese OCR rules; everything else is English.


## Experiment Name (`--experiment_name`)
- Custom Cosmos3 checkpoint: any name (must come with `--checkpoint_path` + `--credential_path`).

Baseline options (`is_benchmark_exp=True`, no checkpoint needed):
- `sd_v3p5_large`
- `flux_1_kontext_dev`
- `flux_2_klein_9b`
- `qwen_image`
- `qwen_image_2512`
- `z_image_turbo`
- `hunyuan_image_3p0`
- `glm_image`
- `nano_banana`
- `nano_banana_pro`

Default parameters per baseline (use these unless the user overrides):

| Model                  | Guidance Scale       | Resolution (1:1) | Num Steps | Special Notes                                               |
|------------------------|----------------------|------------------|-----------|-------------------------------------------------------------|
| **glm_image**          | 1.5                  | 1024 × 1024      | 50        | Must be divisible by 32. Text in quotes for rendering       |
| **sd_v3p5_large**      | 3.5                  | 1024 × 1024      | 28        | Can use 4.5 for complex prompts. max_sequence_length=512    |
| **flux_1_kontext_dev** | 2.5                  | 1024 × 1024      | 30        | Flexible resolutions. max_sequence_length=512               |
| **flux_2_klein_9b**    | 1.0                  | 1024 × 1024      | 4         | Fast distilled model. Step-distilled to 4 steps             |
| **qwen_image**         | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | Different resolution!                                       |
| **qwen_image_2512**    | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | Different resolution! Bilingual (use this for `cvtg102ch`)  |
| **z_image_turbo**      | 0.0                  | 1024 × 1024      | 9         | Must be 0 for turbo. Results in 8 NFEs. Bilingual support   |


## Image width and height
`--width` / `--height` by resolution tier × aspect ratio (e.g. 720p 1:1 → 960×960 [w x h]):

| Tier    | 1:1        | 4:3        | 3:4        | 16:9       | 9:16       |
|---------|------------|------------|------------|------------|------------|
| 256     | 256×256    | 320×256    | 256×320    | 320×192    | 192×320    |
| 480     | 640×640    | 736×544    | 544×736    | 832×480    | 480×832    |
| 720     | 960×960    | 1104×832   | 832×1104   | 1280×720   | 720×1280   |
| 1080    | 1440×1440  | 1664×1248  | 1248×1664  | 1920×1080  | 1080×1920  |
| 1280    | 1712×1712  | 1968×1472  | 1472×1968  | 2272×1280  | 1280×2272  |
| 2048    | 2728×2728  | 3160×2368  | 2368×3160  | 3640×2048  | 2048×3640  |
| gt_2048 | 5464×5464  | 6304×4728  | 4728×6304  | 7280×4096  | 4096×7280  |


## Output path (`--output_path`) and folder layout

When `--output_path` is omitted, Stage 1 auto-derives:

```
s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/<benchmark_name>/<user>_<experiment_name>[_<iter>][_<signature>]/
```

For baselines (no checkpoint), `<iter>` is dropped (e.g. `xingqianx_qwen_image_2512`). For Cosmos3 checkpoints, `<iter>` is the parent folder name of the checkpoint (e.g. `iter_000100000`).

Example:

```bash
--checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_vfm_ablations/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1/checkpoints/iter_000100000/model/
--output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/cvtg2kL/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1_iter100k/
```


## Stage 1 — Inference Bash Template
Stage 1 needs a GPU cluster. Valid `slaunch` cluster: `small` | `small_aws` | `long` | `aws`.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch <cluster> 1 cvtg_gen_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_cvtg_distributed.py \
    --experiment_name <experiment_name> \
    --checkpoint_path <checkpoint_path> \
    --credential_path credentials/gcs.secret \
    --benchmark_name <benchmark_name> \
    --num_batch_size 32 \
    --guidance 4.0 \
    --num_inference_steps 30 \
    --height <height> \
    --width <width> \
    --use_ema \
    --output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/<benchmark_name>/<folder_close_to_model_name_and_iter> \
    --output_credential_path credentials/gcs.secret
```

For a baseline run (no checkpoint), drop `--checkpoint_path`, `--credential_path`, and `--use_ema`. Example matching the user's reference command:

```bash
slaunch small 1 cvtg0_qwen_image_2512 \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_cvtg_distributed.py \
    --experiment_name qwen_image_2512 \
    --num_batch_size 32 \
    --benchmark_name cvtg102ch \
    --height 1328 --width 1328 --guidance 4.0 --num_inference_steps 30
```

Note: `--output_credential_path credentials/gcs.secret` is required to write to the GCS-backed output bucket. The script's default (`credentials/gcp_checkpoint.secret`) lacks write permission and yields a 403 on the pre-write `easy_io.exists()` HEAD probe.

Note: `--regenerate` is *not* in the default template — add it explicitly only when you want to wipe and redo an existing output dir. Without it, an existing run is resumed (skipping already-written prompts).

Note: for `cvtg102ch`, prefer a model with strong Chinese-text rendering (`qwen_image_2512`, `z_image_turbo`, `hunyuan_image_3p0`).


## Stage 2 — Score Base Template
Stage 2 is OCR + scoring against ground-truth `text_list`. CPU is sufficient.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch cpu 1x1 cvtg_score_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_cvtg_metric.py \
    --io_folder <stage1_output_path> \
    --io_cred credentials/gcp_checkpoint.secret \
    --benchmark_name <benchmark_name> \
    --image_extension webp \
    --num_concurrency 32 \
    --batch_size 32 \
    --judge_model gemini-3.1-pro \
    --force_resize 640x640 \
    --max_retry 5 \
    --signature <short_judge_tag>
```

`--judge_model` choices: `gemini-3-flash`, `gemini-3.1-pro`. Suggested `--signature`: `g3f` for `gemini-3-flash`, `g3p1p` for `gemini-3.1-pro`.

Note: Stage 2 `--io_folder` and `<benchmark_name>` must match what Stage 1 wrote (otherwise the prompt-id ↔ image lookup will fail).

Note: if the user supplies a `gcs:<path>` for `--io_folder`, auto-convert it to `s3://<path>` before running.

Note: `--force_resize WxH` resizes the image before sending to the VLM judge. `640x640` is the de-facto default to keep VLM payload small and consistent across runs.

Note: the `cvtg102ch` benchmark automatically switches OCR prompt to Chinese rules — no extra flag needed.


## Reporting performance / status

When the user asks "is X done?", "what's the result?", or "check the cvtg run", **read the result JSON from S3** — do **not** parse the slurm log tail.

Result JSON path:

```
<io_folder>/result_cvtg_<benchmark_name>_<signature>.json
```

(or `result_cvtg_<benchmark_name>.json` if `--signature` was omitted).

Download via the `s3io` skill (`dl ... /tmp/s3io_<basename>`) and read with the `Read` tool. If a file is **missing**, the run hasn't reached the write step yet — report it as `running` / `queued` (cross-check with `squeue` if needed) and **do not** fall back to log scraping.

Fields to report (top-level `stats`):

1. **GNED** — `stats.gned` (Global Normalized Edit Distance, Hungarian-matched; 1.0 = perfect)
2. **PNED** — `stats.pned` (Paired Normalized Edit Distance over min-len pairs; 1.0 = perfect)

Per-sample detail lives under `breakdown[<prompt_id>_<sample_idx>]` with `gned`, `pned.value_sum`, `pned.count`, `pred_text`, `gt_text`. Don't paste this unless asked.
