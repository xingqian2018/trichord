#!/bin/bash
# Qwen-Image-2.1 service on oci-jhb (gb300), vLLM-Omni. The model fits on one GPU (~34 GB bf16, vLLM recipe), so each
# node runs 4 single-GPU servers with step-level batching.
# One job = a dispatcher on node 0 + R single-node replicas; clients use ONE URL, $U/run/qi_serve/broker_url (+ /v1), and
# every replica pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r, GPU i serves on port 20000 + jobid % 20000 + 1 + 4r + i.
# Weights: Qwen/Qwen-Image-2.1 (public, 33.1 GB) in the team-shared HF hub, downloaded on first run.
# Image: docker://vllm/vllm-omni:qwen-image21-arm64-cu130 imported once into $U/containers/vllm-omni+qwen-image21-arm64-cu130.sqsh.
# API key: SLURM_QWEN_IMAGE_API_KEY in ~/Project/trichord/credentials/gateway.json.
# Submit (bash): bash scripts/serving/serve.sh qwen_image_2p1 <replicas>
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
#SBATCH -J qi-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qi_serve/slurm-%j.log
set -euo pipefail
umask 022
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init qi_serve 1

IMAGE=$U/containers/vllm-omni+qwen-image21-arm64-cu130.sqsh
export HF_HUB_CACHE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
export QI_REPO=Qwen/Qwen-Image-2.1 QI_REVISION=d26bb61231c349cf6b7896fa83353113880e1ba3
export QI_MODEL=$HF_HUB_CACHE/models--Qwen--Qwen-Image-2.1/snapshots/$QI_REVISION
test -s "$IMAGE"
export VLLM_API_KEY=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/Project/trichord/credentials/gateway.json")))["SLURM_QWEN_IMAGE_API_KEY"].strip())')
[[ -n "$VLLM_API_KEY" ]]
export OMP_NUM_THREADS=8
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/qi_cache_$SLURM_JOB_ID
export HF_HOME=$XDG_CACHE_HOME/hf

launch_replica() {
    local r=$1
    export QI_PORT=$((PORT_BASE + 1 + 4 * r)) QI_LOGS=$STATE/server-$r
    # first run downloads the weights into the shared hub (resumable); then one offline server per GPU
    srun --nodes=1 --ntasks=1 --nodelist="$(pool_hosts "$r")" --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'mkdir -p $XDG_CACHE_HOME; \
             [[ -s "$QI_MODEL/model_index.json" ]] || python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"QI_REPO\"], revision=os.environ[\"QI_REVISION\"], max_workers=16)"; \
             export HF_HUB_OFFLINE=1; \
             for i in 0 1 2 3; do \
                 CUDA_VISIBLE_DEVICES=$i VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm_$i vllm serve "$QI_MODEL" --omni \
                     --served-model-name Qwen/Qwen-Image-2.1 \
                     --host 0.0.0.0 --port $((QI_PORT + i)) --api-key "$VLLM_API_KEY" \
                     --step-execution --max-num-seqs 4 --init-timeout 1800 >> "$QI_LOGS-gpu$i.log" 2>&1 & \
             done; \
             wait -n; exit 1' >> "$STATE/server-$r.log" 2>&1 &
    REPLICA_PID[$r]=$!
    for i in 0 1 2 3; do
        pool_attach "$r" "http://$(pool_head "$r"):$((QI_PORT + i))" Qwen/Qwen-Image-2.1 SLURM_QWEN_IMAGE_API_KEY 4 "${REPLICA_PID[$r]}"
    done
}

pool_start_broker
pool_run launch_replica
