# Evaluate T2I DPO/SDPO Model using CVTG Score

## Step 1: Information Gathering

### Key Report Files
- `cvtg.md` — CVTG score tables (primary report for this series)
- `ugb.md` — UGB score tables (secondary; may be skipped if only CVTG is requested)

### Useful skills
- `s3io`
- `ssh_run`

### Cluster Rules

- **Generation**: `awscode`, partition `long`, 2 nodes (use `long 2`)
- **Scoring**: `gcpcode`, partition `cpu 1x1`
- Script workflow: write locally → `scp` to `~/tmp/sshrun/` → `ssh <cluster> 'bash ~/tmp/sshrun/<name>.sh'`
- **Never use heredoc** — always scp
- Dedupe check before any submit: `ssh <cluster> 'squeue -u $USER -o "%i %j %T %R" | grep <job_name>'`
- **Never `scancel` without explicit user confirmation**
- Autodl must run **locally** (not on gcpcode)


## Stage 1 — Checkpoints

### Checkpoint source path pattern
```
s3://nv-00-10206-checkpoint-experiments/cosmos3_generation/cosmos3plus_t2ionly/<exp_name>/checkpoints/iter_<NNNNNN>/model
```

### Checkpoint backup (10k only, after eval is done)
Source → Destination:
```
gcs:nv-00-10206-checkpoint-experiments/cosmos3_generation/cosmos3plus_t2ionly/<exp>/checkpoints/iter_000010000/
→
gcs:nv-00-10206-checkpoint/cosmos3_generation/cosmos3plus_t2ionly/<exp>/checkpoints/iter_000010000/
```
Use `s3_omni.py cp` with `gcs:` prefix (not `s3://`) for backup operations.


## Stage 2 — CVTG Image Generation

### Benchmarks
| Benchmark | Prompts | Notes |
|---|---|---|
| `cvtg500L_gc` | 500 | gc = golden caption; primary benchmark |
| `cvtg102ch_gc` | 102 | Chinese prompts with gc |
| `cvtg500L_opus` | 500 | opus prompts (secondary) |
| `cvtg102ch` | 102 | Chinese prompts (secondary) |

For DPO series, evaluate **gc variants first** (cvtg500L_gc, cvtg102ch_gc).

### Generation command (awscode, long 2)
```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh long 2 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_cvtg_i4.py \
    --experiment_name cosmos3_ga_64bm32b_t2ionly_base_mini \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_generation/cosmos3plus_t2ionly/<exp>/checkpoints/iter_<NNNNNN>/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name <benchmark_name> \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg_for_sdpo/<benchmark_name>/<exp>_iter<Nk> \
    --output_credential_path credentials/gcs.secret
```

- No `--benchmark_credential_path` needed for CVTG
- Output images are `.webp`
- Job name convention: `cvtggen_<short_exp>_<bench_short>_i<Nk>` (e.g. `cvtggen_007_500L_gc_i10k`)

### awscode disk issue
Nodes sometimes fail with "No space left on device" (enroot/pyxis issue). Fix: relaunch — it will skip already-generated images.


## Stage 3 — CVTG Scoring

### Scorer (gcpcode, cpu 1x1)
```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/compute_cvtg_metric.py \
    --io_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg_for_sdpo/<benchmark_name>/<exp>_iter<Nk> \
    --io_cred credentials/gcs.secret \
    --benchmark_name <scorer_benchmark_name> \
    --judge_model gemini-3.1-pro@<gateway> \
    --num_concurrency 128 \
    --image_extension webp
```

### Critical rules for scoring
- **`CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt`** — required; `compute_cvtg_metric.py` only exists under `imaginaire4_alt`, not the default `imaginaire4`. Omitting this fails with "No such file or directory" (found 2026-08-25).
- **Flag names are `--io_folder` / `--io_cred` / `--image_extension`**, NOT `--input_folder` / `--s3_cred` / `--extension` — the doc previously had the wrong names (found 2026-08-25 via `argparse` in `compute_cvtg_metric.py`). Using the wrong names fails with "the following arguments are required: --io_folder".
- **`--num_concurrency 128`** — always; default 4 is too slow
- **`--judge_model`** — must include gateway suffix: `@nvidia`, `@nvidiak`, or `@nvidiak`; bare `gemini-3.1-pro` causes KeyError
- **`@nvidiams` out of quota** (as of 2026-08-10) — round-robin only `@nvidia` and `@nvidiak`
- **Benchmark name mapping** — drop `_gc` suffix for scorer:
  - `cvtg500L_gc` folder → `--benchmark_name cvtg500L`
  - `cvtg102ch_gc` folder → `--benchmark_name cvtg102ch`
  - `cvtg500L` → `--benchmark_name cvtg500L`
  - `cvtg102ch` → `--benchmark_name cvtg102ch`
- **No `--force_rescore`** — flag is not supported
- Scorer always re-evaluates all images; no caching

### Gateway balancing for batch jobs
Split evenly across `@nvidia` and `@nvidiak` (2 gateways active).

### Result file
`result_cvtg_<benchmark>.json`

Keys to log:
- `stats.gned` → gned column
- `stats.pned` → pned column
- `stats.success_count` → success column (format as `N/total`)


## Stage 4 — UGB Image Generation (if requested)

### Benchmarks
| Benchmark | Prompts | Notes |
|---|---|---|
| `v2_1170L_opus4p7_gc` | 1170 | gc = golden caption; primary |
| `v2_1170L_opus` | 1170 | opus prompts (secondary) |

### Generation command (awscode, long 2)
```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh long 2 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_ugb_i4.py \
    --experiment_name cosmos3_ga_64bm32b_t2ionly_base_mini \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_generation/cosmos3plus_t2ionly/<exp>/checkpoints/iter_<NNNNNN>/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name <benchmark_name> \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_sdpo/<benchmark_name>/<exp>_iter<Nk> \
    --benchmark_credential_path credentials/gcs.secret \
    --output_credential_path credentials/gcs.secret
```

- Output images are `.png`
- Use `credentials/gcs.secret` for **both** `--benchmark_credential_path` and `--output_credential_path`


## Stage 5 — UGB Scoring (if requested)

### Scorer (gcpcode, cpu 1x1)
```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 <job_name> \
    projects/cosmos3/cosmos3/evaluation/text_to_image/compute_unigenbench_metric.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_sdpo/<benchmark_name>/<exp>_iter<Nk> \
    --s3_cred credentials/gcs.secret \
    --benchmark_name v2_1170L \
    --batch_size 1170 \
    --judge_model gemini-3.1-pro@<gateway> \
    --num_concurrency 64 \
    --extension png \
    --force_rescore
```

- **Always use `--benchmark_name v2_1170L`** regardless of whether input is opus or gc variant
- **Always use `--extension png`**
- **`CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt`** — required; `compute_unigenbench_metric.py` only exists under `imaginaire4_alt`, not the default `imaginaire4` (found 2026-08-25).

### Result file
`unigenbench_result.json`

Keys to log:
- `stats.all.overall_accuracy` → all column
- `stats.orig.overall_accuracy` → orig column
- `stats.phi.overall_accuracy` → phi column
- `stats.success_count` → success column


## Stage 6 — Reporting

### cvtg.md table structure
Four sections: `cvtg500L_opus`, `cvtg500L_gc`, `cvtg102ch`, `cvtg102ch_gc`

Each section columns: `Run | Guidance | Images | gned | pned | success`

Each section with enough rows gets a `Selected results` trimmed sub-table.

### ugb.md table structure
Two sections: `v2_1170L_opus`, `v2_1170L_opus4p7_gc`

Columns: `Run | Guidance | Images | all | orig | phi | success`

### Logging rules
- Image count: actual `.webp` or `.png` count from GCS
- Placeholder `—` for scores not yet computed
- After filling scores, run `nicetable` skill on the file to align columns
- **Bold** the single highest value per column in a table section
- Run name should reflect exp short name + iter (e.g. `cosmos3plus_64bm32b_t2ionly_sdpo_exp000_007_..._iter10k`)
