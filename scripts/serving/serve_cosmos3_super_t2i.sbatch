#!/bin/bash
# Cosmos3-Super-Text2Image service on oci-jhb (gb300), vLLM-Omni, CFG2 x Ulysses2, HSDP shard 4 per node.
# One job = a dispatcher on node 0 + R single-node replicas; clients use ONE URL, $U/run/c3_serve/broker_url (+ /v1), and
# every replica pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r serves on port 20000 + jobid % 20000 + 1 + r. Same flags as the Lepton deployment
# (LeptonTR/Cosmos3-Super-Text2Image), plus --no-guardrails: the guardrail models are never loaded (our requests
# send guardrails=False anyway, and loading them needs HF downloads that fail offline; jobs 990000-990004, 2026-10-13).
# Weights: nvidia/Cosmos3-Super-Text2Image (public, 132.5 GB) in the team-shared HF hub, downloaded on first run.
# Image: docker://vllm/vllm-omni:cosmos3-arm64 imported once into $U/containers/vllm-omni+cosmos3-arm64.sqsh.
# API key: SLURM_COSMOS3_API_KEY in ~/Project/trichord/credentials/gateway.json.
# Submit (bash): bash scripts/serving/serve.sh cosmos3_super_t2i <replicas>
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --wckey=p1
#SBATCH --no-kill
#SBATCH --segment=1
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J c3-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/slurm-%j.log
set -euo pipefail
umask 022
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init c3_serve 1

IMAGE=$U/containers/vllm-omni+cosmos3-arm64.sqsh
export C3_REPO=nvidia/Cosmos3-Super-Text2Image C3_REVISION=daf3d374804be4c512c2135568a7cb95d4341d79
# team-shared HF hub when it holds this revision, else our own hub (on gcpcode the shared Cosmos3 lock dir is not writable for us)
export HF_HUB_CACHE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
[[ -s "$HF_HUB_CACHE/models--nvidia--Cosmos3-Super-Text2Image/snapshots/$C3_REVISION/model_index.json" ]] || export HF_HUB_CACHE=$U/huggingface_cache/hub
export C3_MODEL=$HF_HUB_CACHE/models--nvidia--Cosmos3-Super-Text2Image/snapshots/$C3_REVISION
test -s "$IMAGE"
export VLLM_API_KEY=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/Project/trichord/credentials/gateway.json")))["SLURM_COSMOS3_API_KEY"].strip())')
[[ -n "$VLLM_API_KEY" ]]
export OMP_NUM_THREADS=8
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/c3_cache_$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm HF_HOME=$XDG_CACHE_HOME/hf

launch_replica() {
    local r=$1
    export C3_PORT=$((PORT_BASE + 1 + r))
    # first run downloads the weights into the shared hub (resumable); later runs only verify and start offline
    srun --nodes=1 --ntasks=1 --nodelist="$(pool_hosts "$r")" --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'mkdir -p $XDG_CACHE_HOME; \
             [[ -s "$C3_MODEL/model_index.json" ]] || python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"C3_REPO\"], revision=os.environ[\"C3_REVISION\"], max_workers=16)"; \
             export HF_HUB_OFFLINE=1; \
             exec vllm serve "$C3_MODEL" --omni \
                 --served-model-name LeptonTR/Cosmos3-Super-Text2Image \
                 --host 0.0.0.0 --port "$C3_PORT" --api-key "$VLLM_API_KEY" \
                 --cfg-parallel-size 2 --ulysses-degree 2 --tensor-parallel-size 1 \
                 --use-hsdp --hsdp-shard-size 4 --init-timeout 1800 --no-guardrails' >> "$STATE/server-$r.log" 2>&1 &
    REPLICA_PID[$r]=$!
    pool_attach "$r" "http://$(pool_head "$r"):$C3_PORT" LeptonTR/Cosmos3-Super-Text2Image SLURM_COSMOS3_API_KEY 2 "${REPLICA_PID[$r]}"
}

pool_start_broker
pool_run launch_replica
