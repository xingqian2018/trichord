#!/bin/bash
# reports_xx/nemo6/bcb_rescore.sh [--dry] [--qos Q] [NAME ...] - re-score BigCodeBench (complete + instruct greedy) of finished nemo6 runs with
# the OFFICIAL checker (11-06, user: "Bigcodebench 我们的实验的数字异常的高"): this harness revision never runs the unit tests (a program
# passes iff it exits 0; empty answers pass). One score_cached_cpu.sbatch job per model on a CPU node: the cached generations are re-judged by
# Boxiang's bcb_official/bcb_check.py in his BigCodeBench venv (same as his downstream-sheet rows), via client_compat/nemo6_patches.py
# (NEMO6_BCB_PYTHON); NEMO6_FORCE_TASKS=1 reruns the two tasks although they are done; the newer results file wins in collect_nemo6.py.
# The tokenizer / env of each model = the ones of its generation run (launch_refs.sh rows, nemo6_ladder.sbatch for ours).
set -euo pipefail
cd "$(dirname "$0")/../.."
DRY=0; QOS=cpu-normal; ONLY=()
while (( $# )); do case $1 in --dry) DRY=1 ;; --qos) QOS=$2; shift ;; *) ONLY+=("$1") ;; esac; shift; done
H=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
BCB_PY=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/envs/bcb_eval/bin/python
ROWS=(   # NAME | tokenizer (HF snapshot) | env
"PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_q3|models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137|"
"PROD1_DPSK_q4|models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137|"
"PROD1_DPSK_q5|models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137|"
"MIMO_V25_PRO_BASE|models--XiaomiMiMo--MiMo-V2.5-Pro-Base/snapshots/1e94ccfda10a2e4dc62877139cd8b7d7b523a4be|"
"MIMO_V26_PRO_RL|models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/73875d00b30a89ef8cc353a0b60b0e9f9561952d|"
"QWEN3_30B_A3B_BASE|models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9|"
"QWEN3_235B_A22B|models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65|"
"DEEPSEEK_V4_FLASH_BASE|models--deepseek-ai--DeepSeek-V4-Flash-Base/snapshots/8855555deef230a27a21a8d6f294b7b7497759b6|"
"KIMI_K3|models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06|NEMO6_TRUST_REMOTE_CODE=1"
"QWEN38_FLASH_NEXT|models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13|"
"QWEN38_27B|models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0|"
"QWEN38_MAX|models--Qwen--Qwen3.8-2.4T-A95B-FP8/snapshots/d2dc35658bcf77e66643428cb52e774cc3b5bd29|"
"MISTRAL_LARGE4|models--mistralai--Mistral-Large-4-1T-A52B/snapshots/af26e7869af601441d5cdf75a1272469ddc3d059|NEMO6_ADD_BOS=1"
)
for r in "${ROWS[@]}"; do
    IFS='|' read -r name tok envs <<< "$r"
    if (( ${#ONLY[@]} )) && [[ ! " ${ONLY[*]} " =~ " $name " ]]; then continue; fi
    test -d "$H/$tok" || { echo "missing $H/$tok"; exit 1; }
    if (( DRY )); then echo "[dry] $name  qos $QOS  tok $(basename "$(dirname "$(dirname "$tok")")")  ${envs:-}"; continue; fi
    j=$(env ${envs:-} NAME=$name MODEL=$H/$tok TASKS=bigcodebench_complete_greedy,bigcodebench_instruct_greedy NEMO6_BCB_PYTHON=$BCB_PY \
        NEMO6_FORCE_TASKS=1 NEMO6_BCB_WORKERS=${NEMO6_BCB_WORKERS:-32} \
        sbatch --parsable --export=ALL -J xxnemo6b --qos="$QOS" -t "${T:-08:00:00}" ${EXCLUDE:+--exclude=$EXCLUDE} ${NODELIST:+--nodelist=$NODELIST} reports_xx/nemo6/score_cached_cpu.sbatch)
    echo "$name: job $j ($QOS)"
done
