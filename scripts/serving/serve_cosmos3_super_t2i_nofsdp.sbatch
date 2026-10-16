#!/bin/bash
# Cosmos3-Super-Text2Image service on oci-jhb (gb300), vLLM-Omni, no FSDP: 4 single-GPU servers per node (no CFG/Ulysses
# parallel, no HSDP sharding), trading per-image latency for throughput; the whole model (132.5 GB bf16) fits on one GPU.
# Experiment against serve_cosmos3_super_t2i.sbatch (one 4-GPU server per node); same model name, key and request API,
# its own job name (c3n-xqx) and dispatcher URL ($U/run/c3n_serve/broker_url).
# One job = a dispatcher on node 0 + R single-node replicas; clients use ONE URL, $U/run/c3n_serve/broker_url (+ /v1), and
# every replica pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r, GPU i serves on port 20000 + jobid % 20000 + 1 + 4r + i. Same model flags as the Lepton deployment except
# the parallelism, plus --no-guardrails: the guardrail models are never loaded (our requests
# send guardrails=False anyway, and loading them needs HF downloads that fail offline; jobs 990000-990004, 2026-10-13).
# Weights: nvidia/Cosmos3-Super-Text2Image (public, 132.5 GB) in the team-shared HF hub, downloaded on first run.
# Image: docker://vllm/vllm-omni:cosmos3-arm64 imported once into $U/containers/vllm-omni+cosmos3-arm64.sqsh.
# API key: SLURM_COSMOS3_API_KEY in ~/Project/trichord/credentials/gateway.json.
# Submit (bash): bash scripts/serving/serve.sh cosmos3_super_t2i_nofsdp <replicas>
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
#SBATCH -J c3n-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3n_serve/slurm-%j.log
set -euo pipefail
umask 022
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init c3n_serve 1

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
export XDG_CACHE_HOME=/tmp/c3n_cache_$SLURM_JOB_ID
export HF_HOME=$XDG_CACHE_HOME/hf

launch_replica() {
    local r=$1
    export C3_PORT=$((PORT_BASE + 1 + 4 * r)) C3_LOGS=$STATE/server-$r
    # first run downloads the weights into the shared hub (resumable); then one offline server per GPU
    srun --nodes=1 --ntasks=1 --nodelist="$(pool_hosts "$r")" --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'mkdir -p $XDG_CACHE_HOME; \
             [[ -s "$C3_MODEL/model_index.json" ]] || python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"C3_REPO\"], revision=os.environ[\"C3_REVISION\"], max_workers=16)"; \
             export HF_HUB_OFFLINE=1; \
             for i in 0 1 2 3; do \
                 CUDA_VISIBLE_DEVICES=$i VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm_$i vllm serve "$C3_MODEL" --omni \
                     --served-model-name LeptonTR/Cosmos3-Super-Text2Image \
                     --host 0.0.0.0 --port $((C3_PORT + i)) --api-key "$VLLM_API_KEY" \
                     --cfg-parallel-size 1 --ulysses-degree 1 --tensor-parallel-size 1 \
                     --init-timeout 1800 --no-guardrails >> "$C3_LOGS-gpu$i.log" 2>&1 & \
             done; \
             wait -n; exit 1' >> "$STATE/server-$r.log" 2>&1 &
    REPLICA_PID[$r]=$!
    for i in 0 1 2 3; do
        pool_attach "$r" "http://$(pool_head "$r"):$((C3_PORT + i))" LeptonTR/Cosmos3-Super-Text2Image SLURM_COSMOS3_API_KEY 2 "${REPLICA_PID[$r]}"
    done
}

pool_start_broker
pool_run launch_replica
