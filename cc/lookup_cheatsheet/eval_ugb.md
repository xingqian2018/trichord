# UniGenBench (UGB) evaluation

Two stages: Stage 1 generates images on GPU, Stage 2 scores them with an VLM/LLM judge.

The skill need to gather information from user when necessary.

## Stage 1 — Inference Bash Template
Stage 1 needs a GPU cluster. Valid `slaunch` cluster: `small` | `small_aws` | `long` | `aws`.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch <cluster> 4 ugb_gen_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py \
    --experiment_name <experiment_name> \
    --checkpoint_path <checkpoint_path> \
    --credential_path <credential_path> \
    --benchmark_name <benchmark_name> \
    --num_batch_size 32 \
    --guidance 4.0 \
    --num_inference_steps 50 \
    --height <height> \
    --width <width> \
    --use_ema \
    --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/<benchmark_name>/<folder_close_to_model_name_and_iter> \
    --output_credential_path <output_credential_path>
```

Note: `--output_credential_path credentials/gcs.secret` is required to write to the GCS-backed output bucket. The script's default (`credentials/gcp_checkpoint.secret`) lacks write permission and yields a 403 on the pre-write `easy_io.exists()` HEAD probe.

Note: `--regenerate` is *not* in the default template — add it explicitly only when you want to wipe and redo an existing output dir. Without it, an existing run is resumed (skipping already-written prompts).

## Stage 1 — Benchmark name (`--benchmark_name`)

Default: `v2_1170L_opus`

Canonical (in `BENCHMARK_CHOICE` of `inference_unigenbench_distributed.py`, prompts root `s3://datasets/unigenbench/`):

| Key                | CSV path (relative to prompts root)               | Notes                          |
|--------------------|---------------------------------------------------|--------------------------------|
| `v1`               | `v1/unigenbench_v1.csv`                           |                                |
| `ori_600ls`        | `original/unigenbench_full600_long_and_short.csv` |                                |
| `ori_600l`         | `original/unigenbench_full600_long.csv`           |                                |
| `ori_600s`         | `original/unigenbench_full600_short.csv`          |                                |
| `v2_1170L_G3F`     | `v2/v2_1170L_G3F.csv`                             | default                        |
| `v2_1170Lsu_G3F`   | `v2/v2_1170L_G3F_8b_t2i_structured_upsampled.csv` | structure caption              |
| `v2_1170Lsurc_G3F` | `v2/v2_1170L_G3F_recaptioned.csv`                 | structure to dense recaption   |

Reasoner-upsampler v1 (apr27) extras (prompts root `team-cosmos-benchmark:datasets/unigenbench/v2/reasoner_upsampler_v1_apr27/`):

| Key                          | CSV file                                |
|------------------------------|-----------------------------------------|
| `v2_1170L_opus`              | `Opus_v2_1170L_G3F.csv`                 |
| `v2_1170L_qwen3vl8b`         | `Qwen3VL8B_v2_1170L_G3F.csv`            |
| `v2_1170L_qwen3vl32b`        | `Qwen3VL32B_v2_1170L_G3F.csv`           |
| `v2_1170L_preexp015ft8b`     | `pre_exp015_372_ft8b_v2_1170L_G3F.csv`  |
| `v2_1170L_preexp015ft32b`    | `pre_exp015_373_ft32b_v2_1170L_G3F.csv` |


## Stage 1 — Experiment Name (`--experiment_name`)
A string tells us what experiment we are running, some are baseline models, some are our freshing trained checkpoints

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
- `gemini_image`

It means we are trying to inference a baseline model.

The we need to auto figure out the default inference parameters per baseline (use these unless the user overrides):

| Model                  | Guidance Scale       | Resolution (1:1) | Num Steps | Positive Magic                             | Negative Prompt                                                   | Special Notes                                               |
|------------------------|----------------------|------------------|-----------|--------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| **glm_image**          | 1.5                  | 1024 × 1024      | 50        | None                                       | None                                                              | Must be divisible by 32. Text in quotes for rendering       |
| **sd_v3p5_large**      | 3.5                  | 1024 × 1024      | 28        | None                                       | None                                                              | Can use 4.5 for complex prompts. max_sequence_length=512    |
| **flux_1_kontext_dev** | 2.5                  | 1024 × 1024      | 30        | None                                       | None                                                              | Flexible resolutions. max_sequence_length=512               |
| **flux_2_klein_9b**    | 1.0                  | 1024 × 1024      | 4         | None                                       | None                                                              | Fast distilled model. Step-distilled to 4 steps             |
| **qwen_image**         | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | `", Ultra HD, 4K, cinematic composition."` | `" "` (single space)                                              | Different resolution!                                       |
| **qwen_image_2512**    | 4.0 (true_cfg_scale) | 1328 × 1328      | 50        | `""` (empty)                               | `"低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"` | Different resolution! Comprehensive Chinese negative prompt |
| **z_image_turbo**      | 0.0                  | 1024 × 1024      | 9         | None                                       | None                                                              | Must be 0 for turbo. Results in 8 NFEs. Bilingual support   |

### Pretrained models (ours)

When the user mentioned a checkpoint path, it means this is a inference on a pretrained model. Based on the name of the path, you can locate the correct model size and thus map to the correct `--experiment_name`:
- `cosmos3_ga_64bm32b_t2ionly_base` (default for 64bm32b model)
- `cosmos3_ga_64bm32b_t2ionly_base` (default for 64bm32b model)

If user doesn't mentioned the resolution, you may ask user for the detail resolution and aspect ratio.

Then follows the table below to get the correct `--width` and `--height`:

`--width` / `--height` by resolution tier × aspect ratio (e.g. 720p 1:1 → 960×960 [w x h] ):

| Tier    | 1:1        | 4:3        | 3:4        | 16:9       | 9:16       |
|---------|------------|------------|------------|------------|------------|
| 256     | 256×256    | 320×256    | 256×320    | 320×192    | 192×320    |
| 480     | 640×640    | 736×544    | 544×736    | 832×480    | 480×832    |
| 720     | 960×960    | 1104×832   | 832×1104   | 1280×720   | 720×1280   |
| 1080    | 1440×1440  | 1664×1248  | 1248×1664  | 1920×1080  | 1080×1920  |
| 1280    | 1712×1712  | 1968×1472  | 1472×1968  | 2272×1280  | 1280×2272  |
| 2048    | 2728×2728  | 3160×2368  | 2368×3160  | 3640×2048  | 2048×3640  |
| gt_2048 | 5464×5464  | 6304×4728  | 4728×6304  | 7280×4096  | 4096×7280  |

We always use `--guidance 4.0` and `--num_inference_steps 50` when evaluating our pretrained model.


## Stage 1 — Checkpoint path (`--checkpoint_path`) and output path (`--output_path`), and their credentials.

These inputs are only needed if the experiment is not a baseline model experiment.

*Note: if the user supplies a `gcs:<path>` for `--input_folder`, auto-convert it to `s3://<path>` before running.*

Credential path should be:
```bash
--credential_path credentials/gcs.secret
--output_credential_path credentials/gcs.secret
```

An example of our naming rules are as following:

```bash
--checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_vfm_ablations/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1/checkpoints/iter_000100000/model/
--output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170Lsurc_G3F/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1_iter100k/
```

## Stage 2 — Score Base Template

Stage 2 needs CPU, the default is already in the template

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch cpu 1x1 ugb_score_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_unigenbench_metric.py \
    --input_folder <stage1_output_path> \
    --s3_cred credentials/gcs.secret \
    --benchmark_name <benchmark_name> \
    --batch_size 1170 \
    --judge_model gemini-3.1-pro \
    --num_concurrency 128 \
    --signature gemini-3p1-pro_<small_captial_of_benchmark_name> \
    --extension webp \
    --force_rescore
```

- The `--input_folder` must match what Stage 1 wrote or provided by user.
- The `--benchmark_name`, same set of benchmarks like in stage 1. Default is `v2_1170L_G3F` unless user mentioned otherwise.


## Reporting performance / status

When the user asks "is X done?", "what's the result?", or "check the ugb run", **read the result JSON from S3** — do **not** parse the slurm log tail.

Result JSON path:

```
<input_folder>/unigenbench_result_<signature>.json
```

(or `unigenbench_result.json` if `--signature` was omitted).

Download via the `s3io` skill (`dl ... /tmp/s3io_<basename>`) and read with the `Read` tool. If a file is **missing**, the run hasn't reached the write step yet — report it as `running` / `queued` (cross-check with `squeue` if needed) and **do not** fall back to log scraping.

Fields to report (top-level):

1. **Overall accuracy** — `stats.all.overall_accuracy` (combined across orig + phi prompts; `total_correct / total_count`)
2. **Orig accuracy** — `stats.orig.overall_accuracy` (original-prompt subset only)
3. **Phi accuracy** — `stats.phi.overall_accuracy` (phi/upsampled-prompt subset only)
4. **Success count** — `success_count` (e.g. `"1170/1170"` — how many prompts the judge actually scored)
5. **Judge model** — `judge_model`

For a per-dimension breakdown, each `stats.<split>` also contains `big_class_stats` (primary dimensions) and `small_class_stats` (sub-dimensions), each with `correct`, `total`, `accuracy`. Per-sample detail lives under `breakdown[<prompt_id>]`. Don't paste these unless asked.
