#!/bin/bash
# MiMo-V2.6-Pro-RL service on oci-jhb (gb300): vLLM TP4 per node, max context 1048576 (job 912109 auto-fit already
# chose 1048576, KV cache 5.59M tokens). Copied from Dense-Pretraining-LMs/.claude/scripts/vllm_serve_mimo_pro.sbatch.
# One job = a dispatcher on node 0 + R single-node replicas; clients use ONE URL, $U/run/mimo_pro_serve/broker_url (+ /v1), and
# every replica pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r serves on port 20000 + jobid % 20000 + 1 + r.
# API key: SLURM_MIMO_API_KEY in ~/Project/trichord/credentials/gateway.json (same value as the old
# $U/run/mimo_pro_serve/api_key, so existing MiMo clients keep working).
# Submit (bash): bash scripts/serving/serve.sh mimo_v2p6_pro_rl <replicas>
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --segment=1
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J mimo-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/mimo_pro_serve/slurm-%j.log
set -euo pipefail
umask 077
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init mimo_pro_serve 1

IMAGE=$U/containers/vllm-openai+v0.31.0.sqsh
export MIMO_MODEL=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/73875d00b30a89ef8cc353a0b60b0e9f9561952d
test -s "$IMAGE"; test -s "$MIMO_MODEL/config.json"
export VLLM_API_KEY=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/Project/trichord/credentials/gateway.json")))["SLURM_MIMO_API_KEY"].strip())')
[[ -n "$VLLM_API_KEY" ]]
export MIMO_MAX_MODEL_LEN=1048576 VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
# node-local caches: the 2026-10-08 smoke died with flashinfer cubin lock failures on the scratch cache
export XDG_CACHE_HOME=/tmp/mimo_cache_$SLURM_JOB_ID FLASHINFER_WORKSPACE_BASE=/tmp/mimo_cache_$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm DG_JIT_CACHE_DIR=$XDG_CACHE_HOME/deep_gemm
# DeepGEMM JIT needs an nvcc whose major.minor == CUDART (13.0): the system /usr/local/cuda inside vllm-openai:v0.31.0
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc

pool_start_broker
for ((r = 0; r < REPLICAS; r++)); do
    export MIMO_PORT=$((PORT_BASE + 1 + r))
    srun --nodes=1 --ntasks=1 --nodelist="$(pool_hosts "$r")" --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'mkdir -p $XDG_CACHE_HOME; exec vllm serve "$MIMO_MODEL" \
             --served-model-name mimo-v2.6-pro-rl \
             --host 0.0.0.0 --port "$MIMO_PORT" --api-key "$VLLM_API_KEY" \
             --trust-remote-code --tensor-parallel-size 4 \
             --gpu-memory-utilization 0.80 --max-model-len "$MIMO_MAX_MODEL_LEN" \
             --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
             --generation-config vllm' > "$STATE/server-$r.log" 2>&1 &
    PIDS+=($!)
    pool_attach "$r" "http://$(pool_head "$r"):$MIMO_PORT" mimo-v2.6-pro-rl SLURM_MIMO_API_KEY 32 "$!"
done
pool_wait
