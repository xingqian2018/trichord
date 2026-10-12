#!/bin/bash
# MiMo-V2.6-Pro-RL vLLM endpoint on oci-jhb (gb300): 1 GB300 node, TP4, max context 1048576.
# Copied from Dense-Pretraining-LMs/.claude/scripts/vllm_serve_mimo_pro.sbatch; only --max-model-len pinned to
# max_position_embeddings (job 912109 auto-fit already chose 1048576, KV cache 5.59M tokens).
# Port = 20000 + jobid % 20000; endpoint.env is written once /v1/models answers.
# API key: $VLLM_API_KEY, else $U/run/mimo_pro_serve/api_key.
# Submit (bash): sbatch -J mimo-xqx-1 scripts/serving/serve_mimo_v2p6_pro_rl.sbatch
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

U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
IMAGE=$U/containers/vllm-openai+v0.31.0.sqsh
MODEL=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/73875d00b30a89ef8cc353a0b60b0e9f9561952d
STATE=$U/run/mimo_pro_serve/job-$SLURM_JOB_ID
mkdir -p "$STATE"
test -s "$IMAGE"; test -s "$MODEL/config.json"
export VLLM_API_KEY=${VLLM_API_KEY:-$(cat "$U/run/mimo_pro_serve/api_key")}
[[ -n "$VLLM_API_KEY" ]]

export MIMO_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MIMO_HOST=$(hostname)
export MIMO_MODEL=$MODEL MIMO_MAX_MODEL_LEN=1048576
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
# node-local caches: the 2026-10-08 smoke died with flashinfer cubin lock failures on the scratch cache
export XDG_CACHE_HOME=/tmp/mimo_cache_$SLURM_JOB_ID FLASHINFER_WORKSPACE_BASE=/tmp/mimo_cache_$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm DG_JIT_CACHE_DIR=$XDG_CACHE_HOME/deep_gemm
# DeepGEMM JIT needs an nvcc whose major.minor == CUDART (13.0): the system /usr/local/cuda inside vllm-openai:v0.31.0
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc

server_pid=""
cleanup() { code=$?; [[ -n "$server_pid" ]] && kill -TERM "$server_pid" 2>/dev/null || true; rm -f "$STATE/endpoint.env"; echo "exit_code=$code" > "$STATE/finished"; }
trap cleanup EXIT
trap 'exit 143' TERM INT

srun --nodes=1 --ntasks=1 --container-image="$IMAGE" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --no-container-mount-home --container-workdir=/tmp --export=ALL \
     bash -c 'mkdir -p $XDG_CACHE_HOME; exec vllm serve "$MIMO_MODEL" \
         --served-model-name mimo-v2.6-pro-rl \
         --host 0.0.0.0 --port "$MIMO_PORT" --api-key "$VLLM_API_KEY" \
         --trust-remote-code --tensor-parallel-size 4 \
         --gpu-memory-utilization 0.80 --max-model-len "$MIMO_MAX_MODEL_LEN" \
         --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
         --generation-config vllm' > "$STATE/server.log" 2>&1 &
server_pid=$!

for ((i = 0; i < 240; i++)); do   # up to 2 h: weight loading alone was 3-8 min in the smokes
    kill -0 "$server_pid" 2>/dev/null || { wait "$server_pid"; exit 1; }
    if curl -fsS --max-time 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://$MIMO_HOST:$MIMO_PORT/v1/models" > "$STATE/models.json" 2>/dev/null; then
        printf 'job_id=%s\nmodel=mimo-v2.6-pro-rl\nbase_url=http://%s:%s/v1\n' "$SLURM_JOB_ID" "$MIMO_HOST" "$MIMO_PORT" > "$STATE/endpoint.env"
        echo "READY $(tr '\n' ' ' < "$STATE/endpoint.env")"
        break
    fi
    sleep 30
done
wait "$server_pid"
exit 1
