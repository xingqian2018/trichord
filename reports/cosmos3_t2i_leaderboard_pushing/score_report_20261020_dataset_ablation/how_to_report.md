# How To: UGB for the exp003 Dataset Ablation (Cosmos3.5 60B/30B)

## Report files
- `reports/score_report_20261020_dataset_ablation/ugb.md` — score tables and findings; log every new result here.
- `reports/score_report_20261020_dataset_ablation/gather_ugb_scores.py` — rebuilds the tables from each run's `unigenbench_result.json`; add new runs to `RUNS`.
- `scripts/visualizer/ugb_s3_viewer.py` — side-by-side images + judge verdicts; add new result folders to `RESULT_DIRS`.

## Cluster rules
- Host `gcpcode`; write scripts locally → `scp` to `~/tmp/` → `ssh gcpcode 'bash ~/tmp/<name>.sh'` (login shell is csh, never heredoc).
- Dedupe with `squeue -u $USER -h -o "%j"` before any submit; never `scancel` without explicit confirmation.

## Stage 1 — Generation (2 nodes, ~47 min)
Checkpoint: `s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/<job_name>_n16/checkpoints/iter_000025000/`.

```bash
NAME=ugb_gen_c3p5_<exp>_iter25000_hamid_noneg
cd ~/Project/imaginaire4_alt
export CONTAINER_WORKDIR=$HOME/Project/imaginaire4_alt
sbatch --job-name=$NAME --nodes=2 --gres=gpu:4 --ntasks-per-node=1 \
  -o $HOME/log/slurm/$NAME.%j.o -e $HOME/log/slurm/$NAME.%j.e \
  -p batch -t 4:00:00 --time-min 4:00:00 \
  $HOME/Project/bashrc/sbatch_launch/sbatch_launch_gcp.sh \
  projects/cosmos3/cosmos3/evaluation/text_to_image/inference_ugb_i4.py \
  --experiment_name cosmos3p5_ga_60bm30b_base_768 \
  --checkpoint_path <checkpoint> \
  --credential_path credentials/gcs.secret \
  --benchmark_name v2_1170L_gpt5p5_cosmos3p5_dense_hamid \
  --benchmark_credential_path credentials/gcs.secret \
  --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
  --height 1024 --width 1024 --use_ema \
  --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_<exp>_iter25000_noneg \
  --output_credential_path credentials/gcs.secret
```
- 1024×1024 is the "768" tier: use `cosmos3p5_ga_60bm30b_base_768`; plain `cosmos3p5_ga_60bm30b_base` dies with `KeyError: '768'`.
- `neg` variant: add `--use_cosmos3_negative_prompt` and drop `_noneg` from the job name and output folder.
- The gcpcode checkout under `~/Project/imaginaire4_alt` must be synced to the commit that has `base_768`.

## Stage 2 — Scoring (~6 min)
```bash
slaunch cpu 1 ugbscore_<tag> \
  projects/cosmos3/cosmos3/evaluation/text_to_image/compute_unigenbench_metric.py \
  --input_folder <output_path from stage 1> \
  --s3_cred credentials/gcs.secret \
  --benchmark_name v2_1170L \
  --judge_model gemini-3.1-pro@nvidia
```
- Always score against the original `v2_1170L` prompts, whatever prompts were used for generation.
- Judge is `gemini-3.1-pro@nvidia`; the bare `gemini-3.1-pro` is ambiguous across gateways and raises `KeyError`.
- Check `Success: 1170/1170` at the end of the log before logging the result.

## Stage 3 — Logging
- Run `gather_ugb_scores.py`, copy the numbers into `ugb.md`.
- Bold only the single highest value per column within a table section.
- Note any setting that differs from the rest of its section (resolution, experiment config, neg/noneg, judge gateway).
