#!/bin/bash
# reports_xx/nemo6/launch_refs.sh [--dry] [NAME ...] - nemo6 (full 40-task suite) for the external reference models
# (user 11-04: "把所有外部参考模型都跑了，除了flint"). Serving settings = the ones validated by their bpb ref evals
# (reports_xx/launch_ref_w32k.sh / mimo_doc_eval.sbatch); generation serving: TP x PP (+ DP on 1 node when TP < 4).
# Qwen3-30B-A3B-Base = the pilot (jobs 39048 / 39337 / 39594, done). Kimi-K3 / Mistral-Large-4 / Ripple-65B rows are added after their
# client-side smoke tests (tokenizer / backend). Flint: excluded (user). Ripple: no vLLM path yet (see memory).
set -euo pipefail
cd "$(dirname "$0")/../.."
DRY=0; ONLY=()
for a in "$@"; do case $a in --dry) DRY=1 ;; *) ONLY+=("$a") ;; esac; done
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
H=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
NIGHTLY=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ypang/containers/vllm-nightly-aarch64-8e2beaa.sqsh
V031=$U/containers/vllm-openai+v0.31.0.sqsh
MISTRAL=$U/containers/vllm-openai+mistral-large-4-arm64.sqsh   # official vllm/vllm-openai:mistral-large-4
TXT="--limit-mm-per-prompt '{\"image\": 0, \"video\": 0}'"
# max-num-seqs 1024 (11-04 23:55: at 256 every TP4 server sat at 256 running / ~6.5k waiting with 3-13 % KV used, 2-7k gen tok/s)
# DEPEND=afterany:<job> (env): continuation of a running job (resumes from its caches; exits at once if already complete)
# NAME | nodes | segment | TP | PP | max-num-seqs | time | image | extra vllm serve flags | snapshot | env (KEY=VAL ...)
ROWS=(
"QWEN3_235B_A22B|1|1|4|1|1024|24:00:00|$V031||models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65"
"QWEN38_FLASH_NEXT|1|1|4|1|1024|24:00:00|$NIGHTLY|$TXT|models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
"QWEN38_27B|1|1|1|1|512|24:00:00|$NIGHTLY|$TXT|models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
"QWEN38_MAX|4|4|4|4|1024|24:00:00|$NIGHTLY||models--Qwen--Qwen3.8-2.4T-A95B-FP8/snapshots/d2dc35658bcf77e66643428cb52e774cc3b5bd29"
"DEEPSEEK_V4_FLASH_BASE|1|1|4|1|1024|24:00:00|$V031||models--deepseek-ai--DeepSeek-V4-Flash-Base/snapshots/8855555deef230a27a21a8d6f294b7b7497759b6"
"MIMO_V26_PRO_RL|1|1|4|1|1024|24:00:00|$V031||models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/73875d00b30a89ef8cc353a0b60b0e9f9561952d"
"MIMO_V25_PRO_BASE|2|1|4|2|1024|24:00:00|$V031||models--XiaomiMiMo--MiMo-V2.5-Pro-Base/snapshots/1e94ccfda10a2e4dc62877139cd8b7d7b523a4be"
# own tokenizers (client-side fixes; the job's preflight tok_check.py stops it if client / server ids differ):
"KIMI_K3|2|1|4|2|1024|24:00:00|$V031||models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06|NEMO6_TRUST_REMOTE_CODE=1"
"MISTRAL_LARGE4|2|1|4|2|1024|24:00:00|$MISTRAL|--tokenizer-mode mistral --config-format mistral --load-format mistral --language-model-only|models--mistralai--Mistral-Large-4-1T-A52B/snapshots/af26e7869af601441d5cdf75a1272469ddc3d059|VLLM_PP_LAYER_PARTITION=35,37 NEMO6_ADD_BOS=1"
)
for r in "${ROWS[@]}"; do
    IFS='|' read -r name nodes seg tp pp mns tlim image extra model envs <<< "$r"
    if (( ${#ONLY[@]} )) && [[ ! " ${ONLY[*]} " =~ " $name " ]]; then continue; fi
    test -d "$H/$model" || { echo "missing $H/$model"; exit 1; }
    test -f "$image" || { echo "missing image $image"; exit 1; }
    if (( DRY )); then echo "[dry] $name  ${nodes}N seg$seg TP$tp PP$pp MNS$mns $tlim $(basename "$image")  $extra  ${envs:-}"; continue; fi
    j=$(env ${envs:-} NAME=$name MODEL=$H/$model TP=$tp PP=$pp MNS=$mns IMAGE=$image SERVE_EXTRA="$extra" \
        sbatch --parsable -N "$nodes" --segment="$seg" -t "$tlim" ${DEPEND:+--dependency=$DEPEND} --export=ALL reports_xx/nemo6/nemo6_ref.sbatch)
    echo "$name: job $j  ${nodes}N ${DEPEND:-}"
done
