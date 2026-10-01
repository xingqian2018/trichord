# Make Checkpoint Report

Writes a `state_dict_report` (key, shape, abs-sum, dtype) for a Cosmos3 VFM DCP checkpoint. Config-agnostic — no experiment name or model instantiation needed, reads the checkpoint's `model/` DCP shard directly (local or S3).

- Script: `projects/cosmos3/cosmos3/scripts/make_checkpoint_report.py`
- Repo: `imaginaire4_alt` (`/home/xingqianx/Project/imaginaire4_alt`)
- Distributed: each rank owns `1/world_size` of the parameter keys, so wall-clock scales down with world_size for large (30B/235B MoE) checkpoints. Runs fine single-process too (`world_size=1`).
- Output is **S3-only** — rank 0 builds the report in memory (`io.StringIO` → `io.BytesIO`) and uploads it via `easy_io.put`. There is no local-file write path; `--output_path` not starting with `s3://` raises immediately.

## Arguments

- `--checkpoint_path` (required) — base checkpoint dir, **not** including `model/`; the script appends `model` itself (`os.path.join(checkpoint_path, "model")`). e.g. `s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/3p5_production_runs/60bm30b_v2p1_pretrain/checkpoints/iter_000435000`
- `--output_path` (required) — must be `s3://...`, e.g. `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/checkpoint_report/<name>_report.txt`
- `--credential_path` (default and current standard `credentials/gcs.secret`) — S3 credentials for reading `--checkpoint_path`.
- `--output_credential_path` (default and current standard `credentials/gcs.secret`) — S3 credentials for writing `--output_path`; matches the `nv-00-10206-vfm` GCS-style bucket used for reports.

## Remote launch (via `ssh_run` + `sbatch_launch`)

Consult the `ssh_run` skill for host selection (`awscode` / `gcpcode`) and dispatch. `sbatch_launch/main.sh small 1 ...` on GCP already wraps the command in `torch.distributed.run --nproc_per_node=4` (8 on AWS) automatically — no need to invoke `torchrun` yourself.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 1 checkpoint_report_<name> \
    projects/cosmos3/cosmos3/scripts/make_checkpoint_report.py \
    --checkpoint_path <checkpoint_path> \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/checkpoint_report/<name>_report.txt \
    --credential_path credentials/gcs.secret \
    --output_credential_path credentials/gcs.secret
```

- `small 1` = 1 node, all GPUs on that node (4 on GCP, 8 on AWS) — sufficient for most checkpoints.
- For very large (30B/235B MoE) checkpoints, bump the node count (e.g. `small 2`) to shard keys further.

## Local / single-process (no torchrun)

```bash
.venv/bin/python \
    projects/cosmos3/cosmos3/scripts/make_checkpoint_report.py \
    --checkpoint_path <checkpoint_path> \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/checkpoint_report/<name>_report.txt \
    --credential_path credentials/gcs.secret \
    --output_credential_path credentials/gcs.secret
```

## Notes

- Report is written only by rank 0.
- `--checkpoint_path` takes the base `iter_NNNNNN` dir, not the `model/` subpath — passing `.../model` yourself will double it to `.../model/model` and fail metadata read.
