# PickScore Report — NFT Ablation

Benchmark: `pickscore` (2048 prompts, `pickscore_test.txt`), also `pickscore_opus4p7_c3upsampler` (2048, upsampled). Output is `<id>_0.png` per prompt plus `completed.txt` / `config.json`. This script only generates images; scoring is a separate step.


## Generation CMD (same form as `inference_ugb_i4.py`)

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 pickgen_<somename>_iter<N> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_pickscore_i4.py \
    --experiment_name cosmos3_ga_64bm32b_t2ionly_base_mini \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/<somecheckpoint>/checkpoints/iter_<NNNNNNNNN>/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name pickscore \
    --benchmark_credential_path credentials/gcs.secret \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/pickscore_for_nft/pickscore/<run_name> \
    --output_credential_path credentials/gcs.secret
```

- `s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5/checkpoints/iter_000031000/model` is an example model path, also this is where the baseline checkpoint stored.
- Remember evaluating a LoRA checkpoint, DON'T USE `--use_ema`, and MAKE `--experiment_name` = `cosmos3_ga_64bm32b_t2ionly_base_mini_lora` (rank 16 / alpha 32) or `cosmos3_ga_64bm32b_t2ionly_base_mini_lora_rank32` (rank 32 / alpha 64, for the exp005 `rank32` runs)


## Scoring CMD (`compute_pickscore_i4.py`, gateway-based, no GPU)

Scores `<input_folder>` against the Lepton PickScore gateway (`credentials/lepton_reward.secret`), writes `pickscore_result.json` into the same folder. Benchmark name is auto-read from the folder's `config.json`. Run on `gb300` with `small 1` (no cpu mode there).

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 1 pickscore_<run_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/compute_pickscore_i4.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/pickscore_for_nft/pickscore/<run_name> \
    --input_cred credentials/gcs.secret \
    --benchmark_cred credentials/gcs.secret \
    --num_concurrency 64 --batch_size 1024 \
    --extension png
```

- `--benchmark_cred` must be `credentials/gcs.secret`; the default `pdx_benchmark.secret` cannot read the prompt files in `nv-00-10206-vfm`
- Result keys: `stats.all.mean_pickscore`, `stats.all.std_pickscore`, `success_count`
- `--force_rescore` to overwrite an existing `pickscore_result.json`; `--signature <tag>` to write `pickscore_result_<tag>.json` instead

## pickscore

| Run                                                                                  | Images   | mean      | std   | min   | max   | success   |
|--------------------------------------------------------------------------------------|----------|-----------|-------|-------|-------|-----------|
| baseline (exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k, leaderboard model)         | 2048 png | 21.91     | 1.71  | 16.31 | 27.67 | 2048/2048 |
| cosmos3plus_64bm32b_t2ionly_nft_exp001_000_pickscore_lora_n48_iter60                 | 2048 png | 22.83     | 1.67  | 17.31 | 29.02 | 2048/2048 |
| ----------------------------------------------------------------------------         | -------- | --------- | ----  | ----- | ----- | --------- |
| cosmos3plus_64bm32b_t2ionly_nft_exp004_000_pickscore_lora_iter30                     | 2048 png | **23.02** | 1.68  | 16.87 | 29.70 | 2048/2048 |
| cosmos3plus_64bm32b_t2ionly_nft_exp004_001_pickscore_lora_onecycleoff_iter30         | 2048 png | 22.96     | 1.67  | 17.02 | 29.63 | 2048/2048 |
| cosmos3plus_64bm32b_t2ionly_nft_exp004_002_pickscore_lora_r12_n24_iter30             | 2048 png | 22.95     | 1.68  | 16.92 | 29.57 | 2048/2048 |
| cosmos3plus_64bm32b_t2ionly_nft_exp004_003_pickscore_lora_r12_onecycleoff_n24_iter30 | 2048 png | 22.91     | 1.65  | 17.29 | 29.53 | 2048/2048 |
| cosmos3plus_64bm32b_t2ionly_nft_exp004_004_pickscore_lora_r8_onecycleoff_n24_iter30  | 2048 png | 22.74     | 1.67  | 16.93 | 29.52 | 2048/2048 |

Result files: `<output_path>/pickscore_result.json` (keys `stats.all.mean_pickscore`, `std_pickscore`, `min_pickscore`, `max_pickscore`; per-image scores under `breakdown`). Scored 2026-09-12 via the Lepton PickScore gateway, 1024x1024 generations, guidance 4.0, 50 steps.

## pickscore — exp005_006 sweep (every 10 iters)

Checkpoint: `cosmos3plus_64bm32b_t2ionly_nft_exp005_006_pickscore_lora_r12_onecycleoff_rank32_epochstd_sq_n24` (LoRA rank 32 / alpha 64, inference via `base_mini_lora_rank32`, no EMA). Baseline mean 21.91 for reference.

| Iter    | Images   | mean      | std   | min   | max   | success   |
|---------|----------|-----------|-------|-------|-------|-----------|
| iter0   | 2048 png | 21.91     | 1.71  | 16.32 | 27.67 | 2048/2048 |
| iter10  | 2048 png | 22.14     | 1.67  | 16.72 | 28.74 | 2048/2048 |
| iter20  | 2048 png | 22.53     | 1.66  | 17.32 | 29.42 | 2048/2048 |
| iter30  | 2048 png | 22.86     | 1.67  | 17.4  | 29.8  | 2048/2048 |
| iter40  | 2048 png | 23.08     | 1.66  | 17.21 | 29.82 | 2048/2048 |
| iter50  | 2048 png | 23.21     | 1.66  | 17.46 | 29.67 | 2048/2048 |
| iter60  | 2048 png | 23.36     | 1.66  | 17.43 | 29.52 | 2048/2048 |
| iter70  | 2048 png | 23.39     | 1.68  | 17.53 | 29.3  | 2048/2048 |
| iter80  | 2048 png | 23.40     | 1.69  | 17.26 | 29.51 | 2048/2048 |
| iter90  | 2048 png | 23.46     | 1.68  | 17.35 | 29.55 | 2048/2048 |
| iter100 | 2048 png | 23.44     | 1.69  | 17.53 | 29.81 | 2048/2048 |
| iter110 | 2048 png | 23.42     | 1.68  | 17.47 | 29.24 | 2048/2048 |
| iter120 | 2048 png | 23.46     | 1.69  | 17.44 | 29.61 | 2048/2048 |
| iter130 | 2048 png | 23.55     | 1.69  | 17.66 | 29.73 | 2048/2048 |
| iter140 | 2048 png | 23.54     | 1.68  | 17.61 | 29.85 | 2048/2048 |
| iter150 | 2048 png | 23.60     | 1.67  | 17.62 | 29.89 | 2048/2048 |
| iter160 | 2048 png | 23.60     | 1.68  | 17.55 | 29.94 | 2048/2048 |
| iter170 | 2048 png | 23.47     | 1.66  | 17.42 | 29.86 | 2048/2048 |
| iter180 | 2048 png | 23.49     | 1.65  | 17.03 | 29.76 | 2048/2048 |
| iter190 | 2048 png | 23.44     | 1.65  | 17.13 | 29.67 | 2048/2048 |
| iter200 | 2048 png | 23.63     | 1.67  | 17.28 | 29.61 | 2048/2048 |
| iter210 | 2048 png | 23.72     | 1.68  | 17.13 | 30.14 | 2048/2048 |
| iter220 | 2048 png | 23.70     | 1.65  | 17.35 | 29.89 | 2048/2048 |
| iter230 | 2048 png | 23.65     | 1.65  | 17.35 | 29.8  | 2048/2048 |
| iter240 | 2048 png | 23.62     | 1.63  | 17.38 | 30.1  | 2048/2048 |
| iter250 | 2048 png | 23.65     | 1.62  | 16.75 | 30.14 | 2048/2048 |
| iter260 | 2048 png | 23.69     | 1.63  | 17.31 | 29.99 | 2048/2048 |
| iter270 | 2048 png | 23.60     | 1.62  | 17.51 | 29.94 | 2048/2048 |
| iter280 | 2048 png | 23.66     | 1.64  | 17.42 | 29.91 | 2048/2048 |
| iter290 | 2048 png | 23.69     | 1.65  | 17.57 | 29.77 | 2048/2048 |
| iter300 | 2048 png | 23.64     | 1.65  | 18.08 | 29.48 | 2048/2048 |
| iter310 | 2048 png | 23.54     | 1.65  | 17.92 | 29.36 | 2048/2048 |
| iter320 | 2048 png | 23.52     | 1.66  | 17.4  | 29.43 | 2048/2048 |
| iter330 | 2048 png | 23.30     | 1.68  | 17.55 | 29.72 | 2048/2048 |
| iter340 | 2048 png | 22.72     | 1.71  | 16.68 | 29.73 | 2048/2048 |
| iter350 | 2048 png | 23.62     | 1.66  | 17.65 | 29.61 | 2048/2048 |
| iter360 | 2048 png | **23.73** | 1.63  | 17.09 | 29.86 | 2048/2048 |
| iter370 | 2048 png | 23.47     | 1.65  | 17.38 | 29.62 | 2048/2048 |
| iter380 | 2048 png | 22.91     | 1.67  | 17.29 | 29.35 | 2048/2048 |
| iter390 | 2048 png | 23.05     | 1.64  | 18.15 | 29.61 | 2048/2048 |
| iter400 | 2048 png | 22.36     | 1.7   | 16.65 | 29.2  | 2048/2048 |

Notes (2026-09-14, iters 10–400 all scored 2048/2048):
- Fast climb iter10→60 (22.14→23.36), plateau iter70→190 (23.4–23.6), peak band iter200→290 (23.6–23.72), unstable after iter300.
- Nominal best is iter360 (23.73), one hundredth above iter210 (23.72); the two are tied within noise. iter210 sits in the stable 200–290 plateau and is the safer pick.
- Late phase oscillates rather than degrades: 23.30 (330) → 22.72 (340) → 23.62 (350) → 23.73 (360) → 22.36 (400), swings of ~1 point between consecutive checkpoints vs <0.1 inside 200–290.
- Reference: baseline 21.91; best exp004 (rank 16) 23.02 at iter30; this rank-32 run passes that by iter40.
- Backed up to `gcs:nv-00-10206-checkpoint/cosmos3_generator/cosmos3plus_t2ionly/<run>/checkpoints/`: iter100, 200, 210, 300, 360, 400, 500 (full DCP: model + optim + scheduler + trainer).
- iters 410–500 exist on GCS but are not evaluated.

## pickscore — exp006_000 (lance coyo700m prompts)

Checkpoint: `cosmos3plus_64bm32b_t2ionly_nft_exp006_000_pickscore_lance_coyo700m_n24` (LoRA rank 32 / alpha 64, inference via `base_mini_lora_rank32`, no EMA). Backed up to `gcs:nv-00-10206-checkpoint/cosmos3_generator/cosmos3plus_t2ionly/<run>/checkpoints/`: iter50, 100, 150, 200, 250, 300, 350.

| Iter    | Images   | mean   | std   | min   | max   | success   |
|---------|----------|--------|-------|-------|-------|-----------|
| iter200 | 2048 png | 23.11  | 1.75  | 16.08 | 29.72 | 2048/2048 |
