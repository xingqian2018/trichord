---
name: local_run
description: Launch a local 1x1 CPU/GPU run on the current machine using torch.distributed.run with nproc_per_node=1. Use when the job is small enough that a single local process suffices — no Slurm, no remote cluster needed.
user_invocable: true
---

# local_run — Single-Process Local Launch

Use this skill when the user wants to run a script locally (no Slurm, no SSH) with a single process — the typical case for 1x1 CPU/GPU evaluation or debug runs.

---

## Step 1: Understand the Command Shape

The canonical local run pattern:

```bash
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> \
    <script_path> \
    <script args ...>
```

Key parameters:
- `--nproc_per_node=1` — single process, no multi-GPU fan-out
- `--master_port=<PORT>` — pick any free port (e.g. 24813); if user doesn't specify, use 24813
- `PYTHONPATH` — always set to `/home/xingqianx/Project/imaginaire4` unless user overrides

---

## Step 2: Build the Command from User Input

The user will supply:
- The script path (relative to the project root, e.g. `projects/cosmos3/vfm/evaluation/...`)
- Script-specific flags (`--input_folder`, `--output_folder`, etc.)

If the user pastes an existing command, use it verbatim — only substitute values they explicitly changed.

---

## Step 3: Run It

Run the command directly in the local shell via Bash. Do **not** SSH to a remote host.

Working directory should be `/home/xingqianx/Project/imaginaire4` unless the script path implies otherwise.

```bash
cd /home/xingqianx/Project/imaginaire4 && \
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=24813 \
    <script_path> \
    <args>
```

Run with `run_in_background=true` for long-running jobs so the terminal stays free. Report the background PID.

---

## Step 4: Report

After launching:

```
Launched locally: <short description>
PID:  <pid>
Log:  (stdout/stderr streaming to terminal, or redirect to ~/log/ if background)
```

If the process exits non-zero, report the exit code and last few lines of stderr. Do not retry automatically.

---

## Reference: Full Example

```bash
cd /home/xingqianx/Project/imaginaire4 && \
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=24813 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_structured_captioning.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_entity_list_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage1_ablation_battle_2/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage2_ablation_battle_2/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 800 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 2 \
    --force_judge_model gemini-3.1-pro \
    --force_gen_model gemini-3.1-pro
```
