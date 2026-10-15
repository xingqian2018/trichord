#!/bin/bash
# Kimi-K3 service on gcp-iad-cs-001 (gcpcode, GB200): one job = a dispatcher on node 0 + R replicas of 4 GB200 nodes each
# (TP16 / EP16 / DCP16, max context 1048576). 4 nodes because the 1.56 TB of weights do not fit in 8 GB200s (~186 GB each).
# Weights: moonshotai/Kimi-K3 revision f831ab66 in the team-shared HF hub (same revision as sdharur's gb300 copy).
# Image: kimi-k3-serving-v0.30.0-20260928.sqsh rebuilt on gcpcode with xuali's build_kimi_k3_sqsh.sh (vllm-openai v0.30.0-aarch64).
# Clients use ONE URL, $U/run/k3_serve/broker_url (+ /v1); every replica
# pulls requests from the dispatcher through its sidecar (serving/broker_worker.py), see serving/pool_lib.sh.
# Replica r serves on port 20000 + jobid % 20000 + 1 + r of its first node.
# vLLM settings adapted from sdharur's k3_scale2048 serving; --max-model-len raised from 32768 to 1048576.
# API key: SLURM_KIMI_K3_API_KEY in ~/Project/trichord/credentials/gateway.json (dispatcher and replicas alike).
# Submit (bash): bash scripts/serving/serve.sh kimi_k3_gb200 <replicas>      # requests 4 x <replicas> nodes
# gb300 version: serve_kimi_k3.sbatch.
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --wckey=p1
#SBATCH --no-kill
#SBATCH -N 4
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
source "$HOME/Project/trichord/scripts/serving/pool_lib.sh"
pool_init k3_serve 4

IMAGE=$U/containers/kimi-k3-serving-v0.30.0-20260928.sqsh
MODEL=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06
test -s "$IMAGE"; test -s "$MODEL/config.json"
export VLLM_API_KEY=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/Project/trichord/credentials/gateway.json")))["SLURM_KIMI_K3_API_KEY"].strip())')
[[ -n "$VLLM_API_KEY" ]]
export K3_MODEL=$MODEL K3_MAX_MODEL_LEN=1048576
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1 NCCL_CUMEM_ENABLE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/k3_cache_$SLURM_JOB_ID HF_HOME=/tmp/k3_cache_$SLURM_JOB_ID/hf

launch_replica() {
    local r=$1
    export K3_HEAD_HOST=$(pool_head "$r") K3_PORT=$((PORT_BASE + 1 + r)) K3_MASTER_PORT=$((40000 + SLURM_JOB_ID % 15000 + r))
    # one task per node: rank 0 serves the API on the replica's first node, ranks 1-3 join headless
    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --nodelist="$(pool_hosts "$r")" --cpus-per-task=128 --gpus-per-node=4 \
         --mem=0 --kill-on-bad-exit=1 --label --container-image="$IMAGE" \
         --container-mounts=/lustre:/lustre,/scratch:/scratch,/dev/shm:/dev/shm \
         --no-container-mount-home --container-workdir=/tmp --export=ALL \
         bash -c 'unset PYTHONPATH; mkdir -p "$XDG_CACHE_HOME"; \
             headless=""; [[ "$SLURM_PROCID" != 0 ]] && headless="--headless"; \
             exec vllm serve "$K3_MODEL" --served-model-name moonshotai/Kimi-K3 \
                 --host 0.0.0.0 --port "$K3_PORT" --api-key "$VLLM_API_KEY" --trust-remote-code \
                 --tensor-parallel-size 16 --enable-expert-parallel \
                 --decode-context-parallel-size 16 --dcp-comm-backend a2a \
                 --nnodes 4 --node-rank "$SLURM_PROCID" \
                 --master-addr "$K3_HEAD_HOST" --master-port "$K3_MASTER_PORT" \
                 --distributed-executor-backend mp --gpu-memory-utilization 0.92 \
                 --max-model-len "$K3_MAX_MODEL_LEN" --max-num-seqs 512 --max-num-batched-tokens 16384 \
                 --load-format fastsafetensors --no-enable-flashinfer-autotune --kv-cache-dtype fp8 \
                 --attention-config "{\"use_prefill_query_quantization\":true,\"mla_prefill_backend\":\"TRTLLM_RAGGED\"}" \
                 --attention-backend TOKENSPEED_MLA --enable-prefix-caching --prefix-match-unit 128 \
                 --reasoning-parser kimi_k3 --enable-auto-tool-choice --tool-call-parser kimi_k3 \
                 --mm-processor-cache-gb 0 $headless' >> "$STATE/server-$r.log" 2>&1 &
    REPLICA_PID[$r]=$!
    pool_attach "$r" "http://$K3_HEAD_HOST:$K3_PORT" moonshotai/Kimi-K3 SLURM_KIMI_K3_API_KEY 48 "${REPLICA_PID[$r]}"
}

pool_start_broker
pool_run launch_replica
