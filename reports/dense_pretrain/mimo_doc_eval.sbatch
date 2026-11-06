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
#   Qwen3-30B-A3B-Base (61 GB), Qwen3-235B-A22B (470 GB): Qwen3 ids, HFO='{}', 1 node
#   Mistral-Large-4-1T-A52B (FP8, 1.06 TB, own tokenizer + BOS, VL, consolidated weights only): 2 nodes, PP 2 + VLLM_PP_LAYER_PARTITION=35,37
#     (layers 39, 43, .. 71 reuse layer 35's KV cache: 35-71 must share a stage), IMAGE=<vllm-openai:mistral-large-4>,
#     LLMKW='{"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral", "language_model_only": true}' and the same as
#     WORKER_EXTRA='--tokenizer-mode mistral --config-format mistral --load-format mistral --language-model-only'
#   Qwen3.8-Flash-Next-FP8 (173 GB, own tokenizer, no BOS, VL): HFO='{}' IMAGE=<nightly> EXTRA="--text-dir ~/data/eval_ppl_decon"
#     LLMKW='{"limit_mm_per_prompt": {"image": 0, "video": 0}}' (text only), 1 node
#   RecursiveSI/flint (40B-A2.5B BF16, own tokenizer + BOS, no vLLM support; SGLang kit image built by $U/tmp/flint/build.sbatch): 1 node,
#     IMAGE=$U/containers/flint-sglang-v0.5.16-arm64.sqsh SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1
#     EXTRA="--backend sglang --text-dir ~/data/eval_ppl_decon --bos" SGLARGS = the kit's serve_b300.sh flags (launch_server syntax) + --tp-size 1 --dp-size 4
# 32K window (user 11-03): SEQ=32760 NAME=<NAME>_W32K - same targets, more context for documents longer than 8192.
# QOS: normal by default (sla_res_cosmos / hero-res is gone on slurm-1, 11-01); a smoke test: sbatch --qos=interactive.
# Output: $WORK_DIR/<NAME>-ref-<job>/ (resumable: pass OUT=<an earlier dir> to continue it).
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos=normal
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
IMAGE=${IMAGE:-$U/containers/vllm-openai+v0.31.0.sqsh}   # Qwen3.8-Flash-Next (qwen4_exp) needs vLLM nightly: IMAGE=<ypang's vllm-nightly-aarch64-8e2beaa.sqsh>
REF=${REF:-$(ls -d $WORK_DIR/OPT_MUONH-q0-* | tail -1)}          # any ladder doc eval: same 115 splits for every checkpoint
OUT=${OUT:-$WORK_DIR/${NAME:?}-ref-$SLURM_JOB_ID}
PP=${PP:-1}
HFO=${HFO:-'{"head_dtype": "float32"}'}                        # MiMo: fp32 LM head (teacher cache); other models: '{}'
LLMKW=${LLMKW:-'{}'}                                           # extra vllm.LLM arguments (JSON, one argument: not via EXTRA)
SEQ=${SEQ:-}                                                   # chunk length in predictions (empty = the ladder's 8192); 32760 = ~32K window
SEQ2=${SEQ2:-}                                                 # a second window from the SAME loaded model (user 11-04 "load 模型一次，出两个结果"),
OUT2=${OUT2:-$WORK_DIR/${NAME}_W32K-ref-$SLURM_JOB_ID}          # e.g. SEQ2=32760 -> <NAME>_W32K next to <NAME> (8K)
MML=$(( $( (( ${SEQ2:-0} > ${SEQ:-8192} )) && echo $SEQ2 || echo ${SEQ:-8192} ) + 8 ))   # = mimo_doc_eval.py's max_model_len (longest window; driver and workers agree)
WORKER_EXTRA=${WORKER_EXTRA:-}                                 # multi-node: the same extra arguments for the headless workers, as vllm serve flags
                                                               # (must match LLMKW, e.g. Mistral-Large-4: --tokenizer-mode mistral --config-format mistral ...)
export HFO
mkdir -p "$OUT"; [[ -n "$SEQ2" ]] && mkdir -p "$OUT2"
echo "model $MODEL  nodes $SLURM_NNODES  PP $PP  ref $REF  out $OUT  $(date)"

export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=3600
export XDG_CACHE_HOME=/tmp/mimo_cache_$SLURM_JOB_ID FLASHINFER_WORKSPACE_BASE=/tmp/mimo_cache_$SLURM_JOB_ID   # node-local (flashinfer lock failures on scratch)
export TRITON_CACHE_DIR=/tmp/mimo_cache_$SLURM_JOB_ID/triton   # SGLang (Flint): one Triton cache per job, node-local
export VLLM_CACHE_ROOT=/tmp/mimo_cache_$SLURM_JOB_ID/vllm DG_JIT_CACHE_DIR=/tmp/mimo_cache_$SLURM_JOB_ID/deep_gemm DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export PYTHONPATH=$REPO/code/scaling_ladder
CT=(--container-image="$IMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch,/home/xingqianx/data:/home/xingqianx/data --no-container-mount-home --container-workdir=/tmp --export=ALL)
RUN=(python3 -u "$REPO/reports_xx/mimo_doc_eval.py" --model "$MODEL" --ref "$REF" --out "$OUT" --tp 4 --pp "$PP" --hf-overrides "$HFO" --llm-kwargs "$LLMKW" ${SEQ:+--seq-len $SEQ} ${SEQ2:+--seq-len2 $SEQ2 --out2 $OUT2} ${SGLARGS:+--sgl-args "$SGLARGS"} ${EXTRA:-})

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
        --tensor-parallel-size 4 --pipeline-parallel-size $PP --trust-remote-code --dtype bfloat16 --max-model-len $MML \
        --gpu-memory-utilization 0.90 --max-logprobs 1 --no-enable-prefix-caching --seed 0 --hf-overrides \"\$HFO\" $WORKER_EXTRA" &
done
srun --nodes=1 --ntasks=1 -w "$head" "${CT[@]}" bash -c 'mkdir -p $XDG_CACHE_HOME; exec "$@"' _ "${RUN[@]}" \
    --nnodes ${#nodes[@]} --master-addr "$addr" --master-port "$port"
