# CVTG (Complex Visual Text Generation) evaluation

Two stages: Stage 1 generates images on GPU, Stage 2 OCRs them with a VLM judge and scores text fidelity (GNED / PNED).

The skill needs to gather information from user when necessary.

## Stage 1 — Inference Bash Template
Stage 1 needs a GPU cluster. Valid `slaunch` cluster: `small` | `small_aws` | `long` | `aws`.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch <cluster> 2 cvtg_gen_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_cvtg_distributed.py \
    --experiment_name <experiment_name> \
    --checkpoint_path <checkpoint_path> \
    --credential_path credentials/gcs.secret \
    --benchmark_name <benchmark_name> \
    --num_batch_size 32 \
    --guidance 4.0 \
    --num_inference_steps 50 \
    --height <height> \
    --width <width> \
    --use_ema \
    --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/<benchmark_name>/<folder_close_to_model_name_and_iter> \
    --output_credential_path credentials/gcs.secret
```
Notes:
- `--use_cosmos3_negative_prompt`: injects a long English Cosmos3-style negative prompt. Without it, an empty negative prompt is used. Only meaningful for generators that honor `neg_prompt` (Cosmos3 + diffusers backends like SD3.5 / Flux / GLM); ignored by the gateway baselines (`nano_banana*`).
- `--checkpoint_path`, when providing a in-house checkpoint, sometimes the users is lasy, so by default the checkpoint path should ends for `/model/`
- For a baseline run (no checkpoint), drop `--checkpoint_path`, `--credential_path`, and `--use_ema`. Example matching the user's reference command:

```bash
slaunch small 1 cvtg0_qwen_image_2512 \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_cvtg_distributed.py \
    --experiment_name qwen_image_2512 \
    --num_batch_size 32 \
    --benchmark_name cvtg102ch \
    --height 1328 --width 1328 --guidance 4.0 --num_inference_steps 30
```
- `--regenerate` is *not* in the default template — add it explicitly only when you want to wipe and redo an existing output dir. Without it, an existing run is resumed (skipping already-written prompts).
- See other `Stage 1 —` instructions below

## Stage 1 — Benchmark name (`--benchmark_name`)

Default: `cvtg500L_opus` for English and `cvtg102ch_opus` for Chinese

Canonical (in `BENCHMARK_CHOICE` of `inference_cvtg_distributed.py`, prompts root `s3://datasets/cvtg/`):

| Key               | JSON path                                                         | Prompt field used  | OCR language |
|-------------------|-------------------------------------------------------------------|--------------------|--------------|
| `cvtg2k`          | `cvtg_2kl/testing_prompt_2kl.json`                                | `prompt`           | English      |
| `cvtg2kL`         | `cvtg_2kl/testing_prompt_2kl.json`                                | `prompt_upsampled` | English      |
| `cvtg500L`        | `cvtg_500l/testing_prompt_500l.json`                              | `prompt_upsampled` | English      |
| `cvtg500L_opus`   | `cvtg_500l/testing_prompt_500l_opus_4p7_720p_1to1.json`           | `prompt_upsampled` | English      |
| `cvtg102ch`       | `cvtg_102ch/testing_prompt_102ch.json`                            | `prompt`           | Chinese      |
| `cvtg102ch_opus`  | `cvtg_102ch/testing_prompt_102ch_opus_4p7_720p_1to1.json`         | `prompt_upsampled` | Chinese      |

Note: Stage 2's language switch is automatic — any benchmark starting with `cvtg102ch` triggers Chinese OCR rules; everything else is English.


## Stage 1 — Experiment Name (`--experiment_name`)
A string tells us what experiment we are running, some are baseline models, some are our freshly trained checkpoints.

### Baseline models

When `--experiment_name` matches the following:
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

It means we are trying to inference a baseline model (`is_benchmark_exp=True`, no checkpoint needed).

Then we need to auto figure out the default inference parameters per baseline (use these unless the user overrides):

| Model                  | Guidance Scale       | Resolution (1:1) | Num Steps | Positive Magic                             | Negative Prompt                                                   | Special Notes                                               |
|------------------------|----------------------|------------------|-----------|--------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| **glm_image**          | 1.5                  | 1024 × 1024      | 50        | None                                       | None                                                              | Must be divisible by 32. Text in quotes for rendering       |
| **sd_v3p5_large**      | 3.5                  | 1024 × 1024      | 28        | None                                       | None                                                              | Can use 4.5 for complex prompts. max_sequence_length=512    |
| **flux_1_kontext_dev** | 2.5                  | 1024 × 1024      | 30        | None                                       | None                                                              | Flexible resolutions. max_sequence_length=512               |
| **flux_2_klein_9b**    | 1.0                  | 1024 × 1024      | 4         | None                                       | None                                                              | Fast distilled model. Step-distilled to 4 steps             |
| **qwen_image**         | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | `", Ultra HD, 4K, cinematic composition."` | `" "` (single space)                                              | Different resolution!                                       |
| **qwen_image_2512**    | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | `""` (empty)                               | `"低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"` | Different resolution! Bilingual (use this for `cvtg102ch`)  |
| **z_image_turbo**      | 0.0                  | 1024 × 1024      | 9         | None                                       | None                                                              | Must be 0 for turbo. Results in 8 NFEs. Bilingual support   |

For `cvtg102ch`, prefer a model with strong Chinese-text rendering (`qwen_image_2512`, `z_image_turbo`, `hunyuan_image_3p0`).

### Pretrained models (ours)

When the user mentions a checkpoint path, it means this is an inference on a pretrained model (must come with `--checkpoint_path` + `--credential_path`). Based on the name of the path, you can locate the correct model size and thus map to the correct `--experiment_name`:
- `cosmos3_ga_64bm32b_t2ionly_base` (default for 64bm32b model)
- `cosmos3_ga_16bm8b_t2ionly_base` (default for 16bm8b model)

If the user doesn't mention the resolution, you may ask the user for the detail resolution and aspect ratio. Then follow the table below to get the correct `--width` and `--height`.

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


## Stage 1 — Checkpoint path (`--checkpoint_path`) and output path (`--output_path`), and their credentials.

These inputs are only needed if the experiment is not a baseline model experiment.

*Note: if the user supplies a `gcs:<path>` for `--output_path`, auto-convert it to `s3://<path>` before running.*

Credential path should be:
```bash
--credential_path credentials/gcs.secret
--output_credential_path credentials/gcs.secret
```

When `--output_path` is omitted, Stage 1 auto-derives:

```
s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/<benchmark_name>/<user>_<experiment_name>[_<iter>][_<signature>]/
```

For baselines (no checkpoint), `<iter>` is dropped (e.g. `xingqianx_qwen_image_2512`). For Cosmos3 checkpoints, `<iter>` is the parent folder name of the checkpoint (e.g. `iter_000100000`).

An example of our naming rules are as following:

```bash
--checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_vfm_ablations/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1/checkpoints/iter_000100000/model/
--output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/cvtg2kL/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1_iter100k/
```


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
    --force_resize <wxh_matches_the_current_pretrained_model_target> \
    --max_retry 5 \
    --signature <short_judge_tag>
```

`--judge_model` choices: `gemini-3-flash`, `gemini-3.1-pro`. Suggested `--signature`: `g3f` for `gemini-3-flash`, `g3p1p` for `gemini-3.1-pro`.

Note: Stage 2 `--io_folder` and `<benchmark_name>` must match what Stage 1 wrote (otherwise the prompt-id ↔ image lookup will fail).

Note: if the user supplies a `gcs:<path>` for `--io_folder`, auto-convert it to `s3://<path>` before running.

Note: `--force_resize WxH` resizes the image before sending to the VLM judge. `640x640` for 480p checkpoint or "960x960" for 720p checkpoing, is the de-facto default to keep VLM payload small and consistent across runs. Or if don't know, confirm with user.

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
3. **Success Count** — The number of images out of total that are successfully evaluated. 

Per-sample detail lives under `breakdown[<prompt_id>_<sample_idx>]` with `gned`, `pned.value_sum`, `pned.count`, `pred_text`, `gt_text`. Don't paste this unless asked.
