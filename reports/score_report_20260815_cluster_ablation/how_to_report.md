# Evaluate T2I Distill Model using UGB Score

## Step 1: Information Gathering

### Key Report Files (always update all 3)
- `/home/xingqianx/Project/trichord/reports/score_report_distill_202607/ugb.md` — contains our score tables and report, where you should log your final result.

### Some useful skill
- `s3io`
- `ssh_run`

### Cluster Rules

- Host: `gcpcode` (GCP)
- Script workflow: write locally → `scp` to `~/tmp/` → `ssh gcpcode 'bash ~/tmp/<name>.sh'`
- **Never use heredoc** — always scp
- Dedupe check before any submit: `ssh gcpcode 'squeue -u $USER -o "%i %j %T %R" | grep <job_name>'`
- **Never `scancel` without explicit user confirmation**


## Stage 2 — Generation Images

### Ask or figure out model path `<s3_path>`.
- Our model path is somethine like`<s3_path>=s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4/`
- Alternatively path is like `<s3_path>=s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3plus_t2ionly/cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_002_pretrain_n32xgpu4xbs4/` (our old runs)

### Arguments:
- `--num_batch_size 4 --guidance 4.0 --num_inference_steps 50`
- `--height 640 --width 640`
- `--use_ema --use_cosmos3_negative_prompt`

### Figure out `<run_name>`:
- Usually it should be matched with the model name, can be suggested from `<s3_path>`

### Run Image Generation

- Come up with a proper `<job_name>`
- The the command is the following

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 ugbgen_<somename>_iter<N> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_ugb_i4.py \
    --experiment_name cosmos3p5_ga_60bm30b_base_480 \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/<somecheckpoint>/checkpoints/iter_000050000/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name v2_1170L_opus \
    --benchmark_credential_path credentials/gcs.secret \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 640 --width 640 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cluster_balancing/v2_1170L_opus/cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4_iter50k \
    --output_credential_path credentials/gcs.secret
```
- 8 nodes for full run, 2 nodes for re-run (skip existing images)
- Aspect ratios: 16:9=1360×768, 4:3=1184×880, 3:4=880×1184
- v3_midtrain uses `cosmos3_ga_64bm32b_t2ionly_base_720` and 960×960
- aa_opus has 1567 prompts; v2_1170L has 1170; v2_1170L_G3F is different prompt set


## Stage 3 — Scoring

### UGB scorer (`cpu 1x1`)
```bash
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/compute_unigenbench_metric.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_distill/v2_1170L_opus/<run_name> \
    --s3_cred credentials/gcs.secret \
    --benchmark_name v2_1170L_G3F \
    --batch_size 1170 --judge_model gemini-3.1-pro --num_concurrency 64 \
    --extension png \
    --force_rescore
```
- Result file: `unigenbench_result.json`
- Keys: `stats.orig.overall_accuracy`, `stats.phi.overall_accuracy`, `stats.all.overall_accuracy`


## Stage 4 — Reporting and Tabling

### Naming Conventions

| Checkpoint type | Example run name |
|---|---|
| sft0 iterXXk | `cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iterXXk` |
| sft1 iterXXk | `cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iterXXk` |
| merged | `cosmos3_ga_64bm32b_t2ionly_merged_00N` |
| exp010 | `cosmos3_ga_64bm32b_t2ionly_exp010_sft0_union6_from_merge007_lr5em5_iterNk` |
| aspect ratio suffix | `_16to9`, `_4to3`, `_3to4` |
| PNG output suffix | `_png` |

Short names in baseline tables use abbreviated forms (e.g. `cosmos3_t2i_exp009_sft0_..._iterXXk`).

### Logging Rules

- **Bold**: only the single highest value per column in a table section
- **Image counts**: actual `.webp` / `.png` count from GCS (`cnt`), not scoring success rate
- **sft0 rows before sft1 rows** within same table section
- Add placeholder rows to all 3 files before scores are ready
- Fill in result JSON filename when score is confirmed
