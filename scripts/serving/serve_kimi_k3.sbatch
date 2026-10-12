#!/bin/bash
# Kimi-K3 vLLM endpoint on oci-jhb (gb300): 2 GB300 nodes per replica, TP8 / EP8 / DCP8, max context 1048576.
# Adapted from sdharur's k3_scale2048 serve_continuation.sbatch + serve_node.sh, merged into one file in the
# style of Dense-Pretraining-LMs/.claude/scripts/vllm_serve_mimo_pro.sbatch.
# Port = 20000 + jobid % 20000 on the first node; endpoint.env is written once /v1/models answers.
# Discovery for the poster pipeline: python3 k3_endpoints.py --user xingqianx --job-prefix k3-xqx
# Submit (bash): mkdir -p /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve
#                sbatch -J k3-xqx-1 scripts/serving/serve_kimi_k3.sbatch
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J k3-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/slurm-%j.log
set -euo pipefail
umask 022

U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
IMAGE=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/kimi-k3-serving-v0.30.0-20260928.sqsh
MODEL=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/sdharur/models/moonshotai/Kimi-K3-f831ab66
STATE=$U/run/k3_serve/job-$SLURM_JOB_ID
mkdir -p "$STATE"
test -s "$IMAGE"; test -s "$MODEL/config.json"
[[ "$(cat "$MODEL/.stage_complete")" == f831ab66814297da540d832a5235f8e904f29d06 ]]
[[ "$SLURM_JOB_NUM_NODES" == 2 ]]

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export K3_HEAD_HOST=${hosts[0]}
export K3_PORT=$((20000 + SLURM_JOB_ID % 20000))
export K3_MASTER_PORT=$((40000 + SLURM_JOB_ID % 15000))
export K3_MODEL=$MODEL K3_MAX_MODEL_LEN=1048576
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1 NCCL_CUMEM_ENABLE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/k3_cache_$SLURM_JOB_ID HF_HOME=/tmp/k3_cache_$SLURM_JOB_ID/hf
printf 'image=%s\nmodel=%s\nnodes=%s\nbase_url=http://%s:%s/v1\nmax_model_len=%s\n' \
    "$IMAGE" "$MODEL" "$SLURM_JOB_NODELIST" "$K3_HEAD_HOST" "$K3_PORT" "$K3_MAX_MODEL_LEN" > "$STATE/config.txt"

server_pid=""
cleanup() { code=$?; [[ -n "$server_pid" ]] && kill -TERM "$server_pid" 2>/dev/null || true; rm -f "$STATE/endpoint.env"; echo "exit_code=$code" > "$STATE/finished"; }
trap cleanup EXIT
trap 'exit 143' TERM INT

# one task per node: rank 0 serves the API on the head node, rank 1 joins headless
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4 \
     --mem=0 --kill-on-bad-exit=1 --label --container-image="$IMAGE" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch,/dev/shm:/dev/shm \
     --no-container-mount-home --container-workdir=/tmp --export=ALL \
     bash -c 'unset PYTHONPATH; mkdir -p "$XDG_CACHE_HOME"; \
         headless=""; [[ "$SLURM_PROCID" == 1 ]] && headless="--headless"; \
         exec vllm serve "$K3_MODEL" --served-model-name moonshotai/Kimi-K3 \
             --host 0.0.0.0 --port "$K3_PORT" --trust-remote-code \
             --tensor-parallel-size 8 --enable-expert-parallel \
             --decode-context-parallel-size 8 --dcp-comm-backend a2a \
             --nnodes 2 --node-rank "$SLURM_PROCID" \
             --master-addr "$K3_HEAD_HOST" --master-port "$K3_MASTER_PORT" \
             --distributed-executor-backend mp --gpu-memory-utilization 0.92 \
             --max-model-len "$K3_MAX_MODEL_LEN" --max-num-seqs 512 --max-num-batched-tokens 16384 \
             --load-format fastsafetensors --no-enable-flashinfer-autotune --kv-cache-dtype fp8 \
             --attention-config "{\"use_prefill_query_quantization\":true,\"mla_prefill_backend\":\"TRTLLM_RAGGED\"}" \
             --attention-backend TOKENSPEED_MLA --enable-prefix-caching --prefix-match-unit 128 \
             --reasoning-parser kimi_k3 --mm-processor-cache-gb 0 $headless' > "$STATE/server.log" 2>&1 &
server_pid=$!

for ((i = 0; i < 240; i++)); do   # up to 2 h: loading 1.5 TB of weights plus graph capture
    kill -0 "$server_pid" 2>/dev/null || { wait "$server_pid"; exit 1; }
    if curl -fsS --max-time 5 "http://$K3_HEAD_HOST:$K3_PORT/v1/models" > "$STATE/models.json" 2>/dev/null \
        && grep -q 'moonshotai/Kimi-K3' "$STATE/models.json"; then
        printf 'job_id=%s\nmodel=moonshotai/Kimi-K3\nbase_url=http://%s:%s/v1\n' "$SLURM_JOB_ID" "$K3_HEAD_HOST" "$K3_PORT" > "$STATE/endpoint.env"
        echo "READY $(tr '\n' ' ' < "$STATE/endpoint.env")"
        break
    fi
    sleep 30
done
wait "$server_pid"
exit 1
