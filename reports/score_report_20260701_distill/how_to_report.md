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

### Ask what is the `<s3_path>` for the model.
- Our baseline non distilled model is `<s3_path>=s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5/checkpoints/iter_000031000/`
- Our distilled model is `<s3_path>=s3://nv-00-10206-checkpoint-experiments/cosmos3_interactive/base_distill_32b_xx/base_distill_32b_xx_exp000_00_lr4em6lr8em6/<some_expname>/checkpoints/<some_iteration>`

### Ask the user `<num_steps>` should be used:
- For regular it should be 50 steps.
- For distill model it should be 4 steps.

### Ask the user whether we should use negative prompt.
- If yes (use negative prompt) `--use_cosmos3_negative_prompt`
- If no (don't negative prompt), cmd contains no `--use_cosmos3_negative_prompt`

### Ask the user `<run_name>`:
- Usually it should be matched with the model name, can be suggested from `<s3_path>`

### Run Image Generation

- Come up with a proper `<job_name>`
- The the command is the following

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_unigenbench_distributed.py \
    --experiment_name cosmos3_ga_64bm32b_t2ionly_base \
    --checkpoint_path <s3_path>/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name v2_1170L_opus4p7_ga \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps <num_steps> \
    --height 1024 --width 1024 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_distill/v2_1170L_opus/<run_name> \
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
