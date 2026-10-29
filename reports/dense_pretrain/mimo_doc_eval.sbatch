#!/bin/bash
# reports_xx/mimo_doc_eval.sbatch - reports_xx/mimo_doc_eval.py (the ladder's doc-isolated per-split eval) for a MiMo HF checkpoint,
# offline vLLM in the vllm-openai v0.31.0 container (the one that served MiMo-V2.6-Pro-RL and produced the teacher cache).
#   MiMo-V2.6-Pro-RL   (mxfp4, 566 GB):  1 node,  TP 4
#   MiMo-V2.5-Pro-Base (fp8, 1.03 TB):   2 nodes, TP 4 x PP 2 (vLLM mp multi-node, headless workers) (does not fit one 4 x GB300 node with room for activations)
#
#   sbatch -N 1 -t 08:00:00 --export=ALL,NAME=MIMO_V26_PRO_RL,MODEL=<snapshot>,PP=1 reports_xx/mimo_doc_eval.sbatch
#   sbatch -N 2 -t 08:00:00 --export=ALL,NAME=MIMO_V25_PRO_BASE,MODEL=<snapshot>,PP=2 reports_xx/mimo_doc_eval.sbatch
#   smoke: add EXTRA="--only cc --max-chunks 256"
#   other tokenizers (re-tokenised original text, BOS as context): HFO='{}' EXTRA="--text-dir ~/data/eval_ppl_decon --bos"
#     DeepSeek-V4-Flash-Base (fp8, 295 GB): 1 node; Kimi-K3 (1.56 TB): 2 nodes, PP 2
# Output: $WORK_DIR/<NAME>-ref-<job>/ (resumable: pass OUT=<an earlier dir> to continue it).
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos=hero-res
#SBATCH --reservation=sla_res_cosmos
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -J xxmimo
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval/sbatch/xxmimo_%j.log
set -euo pipefail
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
WORK_DIR=$U/runs/ladder_doc_eval
REPO=$U/Project/cosmos_llm_scaling_ladder
IMAGE=$U/containers/vllm-openai+v0.31.0.sqsh
REF=${REF:-$(ls -d $WORK_DIR/OPT_MUONH-q0-* | tail -1)}          # any ladder doc eval: same 115 splits for every checkpoint
OUT=${OUT:-$WORK_DIR/${NAME:?}-ref-$SLURM_JOB_ID}
PP=${PP:-1}
HFO=${HFO:-'{"head_dtype": "float32"}'}                        # MiMo: fp32 LM head (teacher cache); other models: '{}'
export HFO
mkdir -p "$OUT"
echo "model $MODEL  nodes $SLURM_NNODES  PP $PP  ref $REF  out $OUT  $(date)"

export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=3600
export XDG_CACHE_HOME=/tmp/mimo_cache_$SLURM_JOB_ID FLASHINFER_WORKSPACE_BASE=/tmp/mimo_cache_$SLURM_JOB_ID   # node-local (flashinfer lock failures on scratch)
export VLLM_CACHE_ROOT=/tmp/mimo_cache_$SLURM_JOB_ID/vllm DG_JIT_CACHE_DIR=/tmp/mimo_cache_$SLURM_JOB_ID/deep_gemm DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export PYTHONPATH=$REPO/code/scaling_ladder
CT=(--container-image="$IMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch,/home/xingqianx/data:/home/xingqianx/data --no-container-mount-home --container-workdir=/tmp --export=ALL)
RUN=(python3 -u "$REPO/reports_xx/mimo_doc_eval.py" --model "$MODEL" --ref "$REF" --out "$OUT" --tp 4 --pp "$PP" --hf-overrides "$HFO" ${EXTRA:-})

if (( SLURM_NNODES == 1 )); then
    srun --nodes=1 --ntasks=1 "${CT[@]}" bash -c 'mkdir -p $XDG_CACHE_HOME; exec "$@"' _ "${RUN[@]}"
    exit
fi
# multi-node (vLLM's mp backend, no Ray in this image): node 0 = the eval driver (LLM(..., nnodes, node_rank 0)), the other nodes run
# headless `vllm serve` workers with the SAME model / parallel settings as mimo_doc_eval.py's LLM(...)
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); head=${nodes[0]}
addr=$(getent hosts "$head" | awk '{print $1}'); port=$((20000 + SLURM_JOB_ID % 20000))
for ((r = 1; r < ${#nodes[@]}; r++)); do
    srun --nodes=1 --ntasks=1 -w "${nodes[$r]}" "${CT[@]}" bash -c "mkdir -p \$XDG_CACHE_HOME; exec vllm serve '$MODEL' --headless \
        --nnodes ${#nodes[@]} --node-rank $r --master-addr $addr --master-port $port \
        --tensor-parallel-size 4 --pipeline-parallel-size $PP --trust-remote-code --dtype bfloat16 --max-model-len 8200 \
        --gpu-memory-utilization 0.90 --max-logprobs 1 --no-enable-prefix-caching --seed 0 --hf-overrides \"\$HFO\"" &
done
srun --nodes=1 --ntasks=1 -w "$head" "${CT[@]}" bash -c 'mkdir -p $XDG_CACHE_HOME; exec "$@"' _ "${RUN[@]}" \
    --nnodes ${#nodes[@]} --master-addr "$addr" --master-port "$port"
