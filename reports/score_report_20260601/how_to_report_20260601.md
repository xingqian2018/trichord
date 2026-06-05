# How To: Eval Pipeline for Cosmos3 T2I

## Key Report Files (always update all 3)
- `reports/ugb_baseline.md` — UGB scores table
- `reports/cvtg_baseline.md` — CVTG scores tables (separate Cosmos3 sections)
- `reports/result_s3_location.md` — GCS paths, image counts, result JSONs

---

## Cluster

- Host: `gcpcode` (GCP)
- Script workflow: write locally → `scp` to `~/tmp/` → `ssh gcpcode 'bash ~/tmp/<name>.sh'`
- **Never use heredoc** — always scp
- Dedupe check before any submit: `ssh gcpcode 'squeue -u $USER -o "%i %j %T %R" | grep <job_name>'`
- **Never `scancel` without explicit user confirmation**

---

## Stage 1 — Image Generation

### UGB (v2_1170L_opus)
```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 <job_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py \
    --experiment_name cosmos3_ga_64bm32b_t2ionly_base \
    --checkpoint_path <s3_path>/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name v2_1170L_opus \   # or aa_opus, v2_1170L_G3F
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench/v2_1170L_opus/<run_name> \
    --output_credential_path credentials/gcs.secret
```
- 8 nodes for full run, 2 nodes for re-run (skip existing images)
- Aspect ratios: 16:9=1360×768, 4:3=1184×880, 3:4=880×1184
- v3_midtrain uses `cosmos3_ga_64bm32b_t2ionly_base_720` and 960×960
- aa_opus has 1567 prompts; v2_1170L has 1170; v2_1170L_G3F is different prompt set

### CVTG
```bash
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 <job_name> \  # 2 nodes for 102ch
    projects/cosmos3/vfm/evaluation/text_to_image/inference_cvtg_distributed.py \
    --benchmark_name cvtg500L_opus   # or cvtg102ch_opus_ascii
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg/cvtg500L_opus/<run_name>
    # same other args as UGB
```

---

## Stage 2 — Scoring

### UGB scorer (`cpu 1x1`)
```bash
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 <job_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_unigenbench_metric.py \
    --input_folder s3://.../unigenbench/v2_1170L_opus/<run_name> \
    --s3_cred credentials/gcs.secret \
    --benchmark_name v2_1170L_G3F \
    --batch_size 1170 --judge_model gemini-3.1-pro --num_concurrency 64 \
    --extension webp \    # or png if gen was in PNG mode
    --force_rescore
```
- Result file: `unigenbench_result.json`
- Keys: `stats.orig.overall_accuracy`, `stats.phi.overall_accuracy`, `stats.all.overall_accuracy`

### CVTG scorer (`cpu 1x1`)
```bash
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 <job_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_cvtg_metric.py \
    --io_folder s3://.../cvtg/cvtg500L_opus/<run_name> \
    --io_cred credentials/gcp_checkpoint.secret \
    --benchmark_name cvtg500L_opus \   # or cvtg102ch_opus_ascii
    --image_extension webp \
    --num_concurrency 32 --batch_size 32 --judge_model gemini-3.1-pro \
    --force_resize 960x960 --max_retry 5 --signature g3p1p
    # NO --force_rescore (not supported by CVTG scorer)
```
- Result file: `result_cvtg_cvtg500L_opus_g3p1p.json` / `result_cvtg_cvtg102ch_opus_ascii_g3p1p.json`
- Keys: `stats.gned`, `stats.pned`, `stats.success_count`

---

## GCS Paths

```
Base: gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/
  unigenbench/v2_1170L_opus/<run_name>/
  unigenbench/aa_opus/<run_name>/
  cvtg/cvtg500L_opus/<run_name>/
  cvtg/cvtg102ch_opus_ascii/<run_name>/

Checkpoints: gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/...
```

Use `python ~/Project/bashrc/s3_omni.py cnt gcs:<path>/` to count images.
Use `python ~/Project/bashrc/s3_omni.py autodl gcs:<path>` to download.

---

## Naming Conventions

| Checkpoint type | Example run name |
|---|---|
| sft0 iterXXk | `cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iterXXk` |
| sft1 iterXXk | `cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iterXXk` |
| merged | `cosmos3_ga_64bm32b_t2ionly_merged_00N` |
| exp010 | `cosmos3_ga_64bm32b_t2ionly_exp010_sft0_union6_from_merge007_lr5em5_iterNk` |
| aspect ratio suffix | `_16to9`, `_4to3`, `_3to4` |
| PNG output suffix | `_png` |

Short names in baseline tables use abbreviated forms (e.g. `cosmos3_t2i_exp009_sft0_..._iterXXk`).

---

## Logging Rules

- **Bold**: only the single highest value per column in a table section
- **Image counts**: actual `.webp` / `.png` count from GCS (`cnt`), not scoring success rate
- **sft0 rows before sft1 rows** within same table section
- Add placeholder rows to all 3 files before scores are ready
- Fill in result JSON filename when score is confirmed

---

## Common Gotchas

| Problem | Fix |
|---|---|
| CVTG scorer `--force_rescore` error | CVTG scorer doesn't support this flag — remove it |
| UGB scorer `--io_folder` error | Use `--input_folder` + `--s3_cred` (not `--io_folder` + `--credential_path`) |
| UGB score = 0/1170 | Gen output was PNG, scorer needs `--extension png` |
| Job SIGTERM'd (signal 15) | Node/NCCL issue — just relaunch |
| Container mount failure (`/nfs/dir`) | Node issue — relaunch, will hit different node |
| `iter_000031000` PNG output | Hardcode in `inference_unigenbench_distributed.py` was changed to PNG at some point |
