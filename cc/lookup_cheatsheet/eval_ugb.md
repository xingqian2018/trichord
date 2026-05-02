# UniGenBench (UGB) evaluation

Two stages: Stage 1 generates images on GPU, Stage 2 scores them with an LLM judge.

The skill need to gather information from user when necessary.

## Benchmark name (`--benchmark_name`)
- `v2_1170L_G3F`      (default)
- `v2_1170Lsu_G3F`    (structure caption)
- `v2_1170Lsurc_G3F`  (structure to dense recaption)


## Experiment Name (`--experiment_name`)
- `t2i_mot_expDAITR005_debug` (default)


## Image width and height
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


## Checkpoint path (`--checkpoint_path`) and output path (`--output_path`)

An example of our naming rules are as following:

```bash
--checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_vfm_ablations/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1/checkpoints/iter_000100000/model/
--output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170Lsurc_G3F/cosmos3_ga_16bm8b_v1_image_only_json_prompts_resume1_iter100k/
```


## Stage 1 — Inference Bash Template
Stage 1 needs a GPU cluster. Valid `slaunch` cluster: `small` | `small_aws` | `long` | `aws`.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch <cluster> 1 ugb_gen_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py \
    --experiment_name <experiment_name> \
    --checkpoint_path <checkpoint_path> \
    --credential_path credentials/gcp_checkpoint.secret \
    --benchmark_name <benchmark_name> \
    --num_batch_size 32 \
    --guidance 4.0 \
    --num_inference_steps 50 \
    --height <height> \
    --width <width> \
    --use_ema \
    --regenerate \
    --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/<benchmark_name>/<folder_close_to_model_name_and_iter>
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
    --signature gemini-3p1-pro \
    --extension webp \
    --force_rescore
```

Note: Stage 2 `--input_folder` and `<benchmark_name>` must match what Stage 1 wrote.

Note: if the user supplies a `gcs:<path>` for `--input_folder`, auto-convert it to `s3://<path>` before running.

## Some quick tables:

### Stage-1 image folders in training

Path template:
```
gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2i_mot_expDAITR/<run>/EveryNEvalUGB/unigenbench-on-<benchmark>/iter_<NNNNNNNNN>/
```

Available runs under `t2i_mot_expDAITR/`:

| Run | Notes |
|---|---|
| `t2i_mot_expDAITR005_000_mixv0` | mix v0 |

Available iters for `t2i_mot_expDAITR005_000_mixv0` × `v2_1170L_G3F`: `0`, `20000`, `40000`, `60000`, `80000`, `100000` (zero-padded to 9 digits in path, e.g. `iter_000080000`).

Example — feed into Stage 2 as `--input_folder`:
```
gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2i_mot_expDAITR/t2i_mot_expDAITR005_000_mixv0/EveryNEvalUGB/unigenbench-on-v2_1170L_G3F/iter_000080000/
```

