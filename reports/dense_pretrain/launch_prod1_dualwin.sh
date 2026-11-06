#!/bin/bash
# reports_xx/launch_prod1_dualwin.sh - our 115-split doc evals of the PROD v1 arms behind the "Latent MoE" and "NoPE / RoPE / partial RoPE"
# sections (user 11-04 "那先去测评！"): each checkpoint ONCE loaded, two windows (user: "load 模型一次，出两个结果"): its own 32K
# (<ID>_W32K) + 8K (<ID>) via run_doc_eval.sh EVAL_DUALWIN_SEQ=8192, 3 parallel 1-node parts per checkpoint, checkpoints as a serial chain.
# Each arm with the code tree it was trained with (EVAL_LADDER_CODE).
#   bash launch_prod1_dualwin.sh latent|rope [--after JOB]     prints the job ids; appends them to ~/tmp/reports_xx_autoeval/packplan_autoeval/jobs (auto-collect)
set -euo pipefail
cd "$(dirname "$0")"
WT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/share/cosmos_llm_scaling_ladder_wt_prod/code/scaling_ladder
PROD=$WT/prod/overlay; LATENT=$WT/prod/overlay_latent
PROPE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/code/scaling_ladder/prod_prope/overlay
case $1 in   # (CHANGE rung tree), lower rungs first
latent) PLAN=("PROD1_DPSK q0 $PROD" "PROD1_DPSK_LATENT2 q0 $LATENT" "PROD1_DPSK_LATENT2_E2X q0 $LATENT"
              "PROD1_DPSK q1 $PROD" "PROD1_DPSK_LATENT2 q1 $LATENT" "PROD1_DPSK_LATENT2_E2X q1 $LATENT"
              "PROD1_DPSK q2 $PROD" "PROD1_DPSK_LATENT2 q2 $LATENT" "PROD1_DPSK q3 $PROD" "PROD1_DPSK_LATENT2 q3 $LATENT") ;;
rope)   PLAN=("PROD1_DPSK_PROPE25 q0 $PROPE" "PROD1_DPSK_PROPE25G q0 $PROPE" "PROD1_DPSK_PROPE25 q1 $PROPE" "PROD1_DPSK_PROPE25G q1 $PROPE"
              "PROD1_DPSK_PROPE25 q2 $PROPE" "PROD1_DPSK_PROPE25G q2 $PROPE") ;;
*) echo "usage: $0 latent|rope [--after JOB]" >&2; exit 2 ;;
esac
prev=""; [[ "${2:-}" == --after ]] && prev="$3"
for p in "${PLAN[@]}"; do
    read -r ch q tree <<<"$p"
    out=$(EVAL_LADDER_CODE="$tree" EVAL_DUALWIN_SEQ=8192 EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= \
          bash run_doc_eval.sh ${prev:+--after $prev} "${ch}_W32K/$q=p:$ch")
    echo "$out"
    ids=$(echo "$out" | grep -o "job [0-9]*" | awk '{print $2}')
    echo "$ids" >> "$HOME/tmp/reports_xx_autoeval/packplan_autoeval/jobs"
    prev=$(echo "$ids" | tail -1)
done
