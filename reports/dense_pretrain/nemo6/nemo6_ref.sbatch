#!/bin/bash
# reports_xx/nemo6/nemo6_ref.sbatch - the nemo6 benchmark suite (nemo6_report.yml: 40 tasks / 75 metrics, ADLR lm-eval harness) for an
# external reference model (user 11-04: "所有的外部参考" on the full suite, beyond the 7-set bpb / val_loss eval).
#   server: `vllm serve` (OpenAI API) in the model's vLLM image, the same serving settings as its bpb ref eval (reports_xx/mimo_doc_eval.sbatch)
#   client: Boxiang's aarch64 lm-eval venv (frozen harness @17abcb7) in his pytorch 25.08 image -> reports_xx/nemo6/run_tasks.py (lanes)
#
#   sbatch -N 1 -t 12:00:00 --export=ALL,NAME=QWEN3_30B_A3B_BASE,MODEL=<snapshot>,TP=1 reports_xx/nemo6/nemo6_ref.sbatch
# env: NAME MODEL [IMAGE] [TP=4] [PP=1] [DP=nodes*4/(TP*PP), single node only] [MML=16384] [MNS=256] [SERVE_EXTRA=<vllm serve flags>]
#      [LANES=4] [CONC=0] [PBUDGET=2048] [TASKS=<names or indices>] [TOKENIZER=MODEL]
#      [CONFIG=nemo6_report.yml] (full 40-task suite, user 11-04 "全测"; nemo6_nomulti.yml = 36 tasks without multilingual, unused)
#   multi-node: TP x PP spans the nodes (vLLM mp backend: head = API server, others `--headless`), as in mimo_doc_eval.sbatch.
# decoding: --generation-config vllm (the task's own sampling settings only; no model generation_config.json defaults such as top_k 20).
# Output: $U/runs/nemo6_eval/<NAME>/ (resumable: finished tasks are skipped; --requeue after a normal-QOS preemption).
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos=normal
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J xxnemo6
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/nemo6_eval/sbatch/xxnemo6_%j.log
set -euo pipefail
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
REPO=$U/Project/cosmos_llm_scaling_ladder
N6=$REPO/reports_xx/nemo6
RT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval
export LME_SRC_DIR=$RT/src VENV=$RT/env
CIMAGE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/container/nvidia+pytorch+25.08-py3.sqsh
IMAGE=${IMAGE:-$U/containers/vllm-openai+v0.31.0.sqsh}
TP=${TP:-4}; PP=${PP:-1}; MML=${MML:-16384}; MNS=${MNS:-256}; LANES=${LANES:-4}
CONC=${CONC:-0}; PBUDGET=${PBUDGET:-2048}   # CONC 0: per task, conc = clamp(PBUDGET / batch_size, 4, 128) prompts in flight per lane
TOKENIZER=${TOKENIZER:-${MODEL:?}}
NN=$SLURM_NNODES
DP=${DP:-$(( NN == 1 ? 4 / (TP * PP) : 1 ))}
OUTROOT=$U/runs/nemo6_eval
JUDGE_ENV=${JUDGE_ENV:-$U/private/nemo6_judge.env}
export HF_HOME=$U/data/nemo6_hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=3600
C=/tmp/nemo6_cache_$SLURM_JOB_ID
export XDG_CACHE_HOME=$C FLASHINFER_WORKSPACE_BASE=$C TRITON_CACHE_DIR=$C/triton VLLM_CACHE_ROOT=$C/vllm DG_JIT_CACHE_DIR=$C/deep_gemm
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA="=rdma_vf_rail0,rdma_vf_rail1,rdma_vf_rail2,rdma_vf_rail3" NCCL_IB_ADDR_FAMILY=AF_INET6 \
       NCCL_IB_TC=96 NCCL_IB_TIMEOUT=20 NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_MNNVL_ENABLE=2 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=1
mkdir -p "$OUTROOT/$NAME"
# continuation / requeued jobs: nothing to do once every report metric exists (no model load)
if python3 "$N6/collect_nemo6.py" --check "$NAME" --config "$N6/${CONFIG:-nemo6_report.yml}" 2>/dev/null; then
    echo "=== $(date) $NAME: all tasks already complete, nothing to run"; exit 0
fi
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); head=${nodes[0]}
addr=$(getent hosts "$head" | awk '{print $1}'); PORT=8000; MPORT=$((20000 + SLURM_JOB_ID % 20000))
echo "=== $(date)  NAME $NAME  model $MODEL  image $(basename "$IMAGE")  nodes $NN ($head)  TP $TP PP $PP DP $DP  MML $MML  lanes $LANES conc $CONC"

SCT=(--container-image="$IMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch --no-container-mount-home --container-workdir=/tmp --export=ALL)
COMMON="--tensor-parallel-size $TP --pipeline-parallel-size $PP --max-model-len $MML --max-num-seqs $MNS --gpu-memory-utilization 0.90 \
  --max-logprobs 100 --no-enable-prefix-caching --trust-remote-code --generation-config vllm --seed 0 ${SERVE_EXTRA:-}"
MULTI=""; (( NN > 1 )) && MULTI="--nnodes $NN --master-addr $addr --master-port $MPORT"
for ((r = 1; r < NN; r++)); do
    srun --nodes=1 --ntasks=1 -w "${nodes[$r]}" "${SCT[@]}" -o "$OUTROOT/$NAME/server_${SLURM_JOB_ID}_n$r.log" bash -c \
        "mkdir -p \$XDG_CACHE_HOME; exec vllm serve '$MODEL' --headless $MULTI --node-rank $r $COMMON" &
done
srun --nodes=1 --ntasks=1 -w "$head" "${SCT[@]}" -o "$OUTROOT/$NAME/server_${SLURM_JOB_ID}.log" bash -c \
    "mkdir -p \$XDG_CACHE_HOME; exec vllm serve '$MODEL' --served-model-name '$NAME' --host 0.0.0.0 --port $PORT \
     ${MULTI:+$MULTI --node-rank 0} $( (( DP > 1 )) && echo --data-parallel-size $DP ) $COMMON" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for i in $(seq 1 720); do                                   # up to 60 min (1-2 TB checkpoints load slowly)
    curl -sf "http://$addr:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { echo "server exited before ready (see $OUTROOT/$NAME/server_${SLURM_JOB_ID}.log)"; exit 1; }
    sleep 5
done
curl -sf "http://$addr:$PORT/health" >/dev/null || { echo "server not ready after 60 min"; exit 1; }
echo "=== server ready $(date)"

export NAME MODEL IMAGE CIMAGE TP PP DP NN MML MNS SERVE_EXTRA="${SERVE_EXTRA:-}" SERVE_ENV="${SERVE_ENV:-}"
MAN=$(python3 -c 'import json, os
print(json.dumps({k.lower(): os.environ.get(k, "") for k in "NAME MODEL IMAGE CIMAGE TP PP DP NN MML MNS SERVE_EXTRA SERVE_ENV".split()}))' 2>/dev/null || echo '{}')
export NEMO6_LCB_WORKERS=${NEMO6_LCB_WORKERS:-32} NEMO6_EXEC_DATA_GB=${NEMO6_EXEC_DATA_GB:-16}   # client_compat/nemo6_patches.py
CLIENT=(--overlap --nodes=1 --ntasks=1 -w "$head" --container-image="$CIMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch
        --container-workdir=/tmp --export=ALL,PYTHONPATH=$N6/client_compat:$LME_SRC_DIR)
# preflight: client tokenizer ids == server /tokenize ids (likelihood requests are tokenized client-side), else no task is run
if [[ "${SKIP_TOKCHECK:-0}" != 1 ]]; then
    trc=0
    srun "${CLIENT[@]}" bash -c 'source $VENV/bin/activate; exec python -u "$@"' _ "$N6/tok_check.py" \
        --tokenizer "$TOKENIZER" --host "$addr" --port "$PORT" --model "$NAME" || trc=$?
    (( trc == 0 )) || { echo "=== tokenizer check failed (rc $trc): not running any task"; exit 4; }
fi
# the client relaunches (finished tasks are skipped) if its step dies, e.g. an OOM kill, while the server is still healthy
for attempt in 1 2 3; do
    rc=0
    srun --overlap --nodes=1 --ntasks=1 -w "$head" --container-image="$CIMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch \
        --container-workdir=/tmp --export=ALL,PYTHONPATH=$N6/client_compat:$LME_SRC_DIR bash -c \
        'mkdir -p $XDG_CACHE_HOME; source $VENV/bin/activate; cd $LME_SRC_DIR; exec python -u "$@"' _ \
        "$N6/run_tasks.py" --config "$N6/${CONFIG:-nemo6_report.yml}" --results-dir "$OUTROOT" --model-name "$NAME" --tokenizer "$TOKENIZER" \
        --host "$addr" --port "$PORT" --lanes "$LANES" --conc "$CONC" --prompt-budget "$PBUDGET" ${TASKS:+--tasks "$TASKS"} \
        --judge-env "$JUDGE_ENV" --manifest "$MAN" || rc=$?
    echo "=== client attempt $attempt exit $rc $(date)"
    (( rc == 0 || rc == 3 )) && break                      # 0 all ok, 3 = some task failed / skipped (rerun later): not a crash
    curl -sf "http://$addr:$PORT/health" >/dev/null || { echo "server down, not relaunching the client"; break; }
done
exit $rc
