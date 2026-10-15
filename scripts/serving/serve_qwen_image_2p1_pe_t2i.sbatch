#!/bin/bash
# Qwen-Image-2.1-PE-T2I service on oci-jhb (gb300): the official prompt rewriter for Qwen-Image-2.1 (9B), vLLM with
# data parallel 4 per node. A short brief in, {"rewritten_prompt": ..., "wh_ratio": ...} out; the client sends the
# model's own system prompt (prompt/qwen_image_2p1_pe_t2i_system.txt in model_playground_v1).
# One job = a dispatcher on node 0 + R single-node replicas; clients use ONE URL, $U/run/qpe_serve/broker_url (+ /v1), and
# every replica pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r serves on port 20000 + jobid % 20000 + 1 + r.
# Weights: Qwen/Qwen-Image-2.1-PE-T2I (public, 18.8 GB) in the team-shared HF hub, downloaded on first run.
# Image: the same vllm-openai v0.31 image as the MiMo service (it registers Qwen3_5ForConditionalGeneration).
# API key: SLURM_QWEN_IMAGE_PE_API_KEY in ~/Project/trichord/credentials/gateway.json.
# Submit (bash): bash scripts/serving/serve.sh qwen_image_2p1_pe_t2i <replicas>
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --segment=1
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J qpe-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qpe_serve/slurm-%j.log
set -euo pipefail
umask 022
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init qpe_serve 1

IMAGE=$U/containers/vllm-openai+v0.31.0.sqsh
export HF_HUB_CACHE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
export PE_REPO=Qwen/Qwen-Image-2.1-PE-T2I PE_REVISION=f3ed7985c788ad75b3ab7223e0c4c51e2a43545b
export PE_MODEL=$HF_HUB_CACHE/models--Qwen--Qwen-Image-2.1-PE-T2I/snapshots/$PE_REVISION
test -s "$IMAGE"
export VLLM_API_KEY=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/Project/trichord/credentials/gateway.json")))["SLURM_QWEN_IMAGE_PE_API_KEY"].strip())')
[[ -n "$VLLM_API_KEY" ]]
export VLLM_ENGINE_READY_TIMEOUT_S=3600 OMP_NUM_THREADS=8
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/qpe_cache_$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm HF_HOME=$XDG_CACHE_HOME/hf

pool_start_broker
for ((r = 0; r < REPLICAS; r++)); do
    export PE_PORT=$((PORT_BASE + 1 + r))
    # first run downloads the weights into the shared hub (resumable); later runs only verify and start offline
    srun --nodes=1 --ntasks=1 --nodelist="$(pool_hosts "$r")" --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'unset PYTHONPATH; mkdir -p $XDG_CACHE_HOME; \
             [[ -s "$PE_MODEL/config.json" ]] || python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"PE_REPO\"], revision=os.environ[\"PE_REVISION\"], max_workers=16)"; \
             export HF_HUB_OFFLINE=1; \
             exec vllm serve "$PE_MODEL" --served-model-name Qwen/Qwen-Image-2.1-PE-T2I \
                 --host 0.0.0.0 --port "$PE_PORT" --api-key "$VLLM_API_KEY" \
                 --trust-remote-code --data-parallel-size 4 --tensor-parallel-size 1 \
                 --max-model-len 32768 --gpu-memory-utilization 0.85 \
                 --limit-mm-per-prompt "{\"image\":0,\"video\":0}" --generation-config vllm' > "$STATE/server-$r.log" 2>&1 &
    PIDS+=($!)
    pool_attach "$r" "http://$(pool_head "$r"):$PE_PORT" Qwen/Qwen-Image-2.1-PE-T2I SLURM_QWEN_IMAGE_PE_API_KEY 64 "$!"
done
pool_wait
