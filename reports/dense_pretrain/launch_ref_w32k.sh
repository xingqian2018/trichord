#!/bin/bash
# reports_xx/launch_ref_w32k.sh [--dry] [--cap] - the external reference models again with a ~32K chunk window (SEQ=32760 predictions,
# max_model_len 32768; user 11-03 "所有的baseline ... 长文档切块 window = 32k"), same settings as their 8K runs otherwise. NAME = <8K NAME>_W32K.
#   --cap: keep <= 10 eval nodes: the second group waits (afterany) on jobs of the first.   Mistral-Large-4 is added once its 8K run works.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; CAP=0
for a in "$@"; do case $a in --dry) DRY=1 ;; --cap) CAP=1 ;; esac; done
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
H=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
TOK=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain
NIGHTLY=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ypang/containers/vllm-nightly-aarch64-8e2beaa.sqsh
V031=$U/containers/vllm-openai+v0.31.0.sqsh
TXT='{"limit_mm_per_prompt": {"image": 0, "video": 0}}'
declare -A J
# name | nodes | PP | time | image | HFO | LLMKW | EXTRA | model snapshot | --after (cap mode)
ROWS=(
"QWEN38_MAX|4|4|05:00:00|$NIGHTLY|{}|{}|--text-dir /home/xingqianx/data/eval_ppl_decon|models--Qwen--Qwen3.8-2.4T-A95B-FP8/snapshots/d2dc35658bcf77e66643428cb52e774cc3b5bd29|"
"KIMI_K3|2|2|06:00:00|$V031|{}|{}|--tok-dir $TOK/kimik3_tokenized/val_set/eval_ppl_decon --bos|models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06|"
"MIMO_V25_PRO_BASE|2|2|05:00:00|$V031|{\"head_dtype\": \"float32\"}|{}||models--XiaomiMiMo--MiMo-V2.5-Pro-Base/snapshots/1e94ccfda10a2e4dc62877139cd8b7d7b523a4be|"
"DEEPSEEK_V4_FLASH_BASE|1|1|05:00:00|$V031|{}|{}|--tok-dir $TOK/deepseekv4_tokenized/val_set/eval_ppl_decon --bos|models--deepseek-ai--DeepSeek-V4-Flash-Base/snapshots/8855555deef230a27a21a8d6f294b7b7497759b6|"
"QWEN38_FLASH_NEXT|1|1|06:00:00|$NIGHTLY|{}|$TXT|--text-dir /home/xingqianx/data/eval_ppl_decon|models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13|"
"QWEN3_30B_A3B_BASE|1|1|02:00:00|$V031|{}|{}||models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9|DEEPSEEK_V4_FLASH_BASE"
"QWEN3_235B_A22B|1|1|05:00:00|$V031|{}|{}||models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65|QWEN3_30B_A3B_BASE"
"MIMO_V26_PRO_RL|1|1|05:00:00|$V031|{\"head_dtype\": \"float32\"}|{}||models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/73875d00b30a89ef8cc353a0b60b0e9f9561952d|MIMO_V25_PRO_BASE"
"QWEN38_27B|1|1|04:00:00|$NIGHTLY|{}|$TXT|--text-dir /home/xingqianx/data/eval_ppl_decon|models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0|QWEN38_FLASH_NEXT"
)
for r in "${ROWS[@]}"; do
    IFS='|' read -r name nodes pp tlim image hfo llmkw extra model after <<< "$r"
    test -d "$H/$model" || { echo "missing $H/$model"; exit 1; }
    dep=(); if (( CAP )) && [[ -n "$after" ]]; then dep=(--dependency=afterany:${J[$after]}); fi
    if (( DRY )); then echo "[dry] ${name}_W32K  ${nodes}N PP$pp  $tlim  ${dep[*]:-}  $(basename "$image")  $extra"; J[$name]=dry; continue; fi
    J[$name]=$(NAME=${name}_W32K MODEL=$H/$model PP=$pp SEQ=32760 HFO="$hfo" LLMKW="$llmkw" IMAGE="$image" EXTRA="$extra" \
        sbatch --parsable -N "$nodes" -t "$tlim" "${dep[@]}" --export=ALL reports_xx/mimo_doc_eval.sbatch)
    echo "${name}_W32K: job ${J[$name]}  ${nodes}N  ${dep[*]:-}"
done
