#!/bin/bash
# reports_xx/prepack_autoeval.sh - iso-doc evals of the E2X KD arms on the tc_full prepack data (user 11-06: A = PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL,
# B = PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL, q1 / q2 / q3, seed 1; jobs 44918-44931). Same contract as prodkd_autoeval.sh: every 10 min, a run whose
# final checkpoint exists gets both windows from ONE job set (EVAL_DUALWIN_SEQ=8192: <ID>_W32K = 32K, <ID> = 8K), 3 parallel 1-node parts, once (marker);
# whenever one of our eval jobs leaves the queue: collect + regenerate + mirror. Code tree = the one the latent arms train with (prod/overlay_latent);
# B's Megatron (qbx2) comes from its run.json (sh/eval_ckpt.sh). Exits when all 6 checkpoints are evaluated and collected.
# Run in tmux: tmux new -d -s prepack_autoeval "bash reports_xx/prepack_autoeval.sh"   Log: ~/tmp/reports_xx_autoeval/prepack_autoeval.log
cd "$(dirname "$0")"
ST=$HOME/tmp/reports_xx_autoeval; LOG=$ST/prepack_autoeval.log; MK=$ST/prepack_autoeval; mkdir -p "$MK"   # state + log outside the repo (user 11-05)
R=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1_prepack
TREE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/share/cosmos_llm_scaling_ladder_wt_prod/code/scaling_ladder/prod/overlay_latent
ARMS=("PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL|qwen3_tcfull_prepack_32k_e2x_kd_a0p8_s64y_mimo25"
      "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL|qwen3_tcfull_prepack_32k_e2x_qbx_kd_a0p8_s64y_mimo25")
declare -A ITERS=([q1]=15824 [q2]=36606 [q3]=70438)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start"
while :; do
    left=0
    held=$(squeue -u "$USER" -h -o '%i|%j|%r' | awk -F'|' '$2 ~ /^xxdoc-PROD1_DPSK_LATENT2_E2X_(QBX_)?KD_/ && tolower($3) ~ /requeued.held/ {print $1}' | paste -sd,)
    if [[ -n $held ]]; then scontrol release "$held" >> "$LOG" 2>&1; log "released held eval jobs: $held"; fi
    for q in q1 q2 q3; do
        for a in "${ARMS[@]}"; do
            id=${a%%|*}; run=${a##*|}; mk="$id-$q"; d=$R/$run-$q-iters${ITERS[$q]}-s1
            [[ -f $MK/$mk ]] && continue
            left=1
            [[ -f $d/argv.txt && -f $d/checkpoints/latest_checkpointed_iteration.txt ]] || continue
            tot=$(grep -A1 -x -- "--train-iters" "$d/argv.txt" | tail -1); it=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt")
            [[ "$it" == "$tot" ]] || continue
            busy=$(squeue -u "$USER" -h -o '%j %D' | awk '$1 ~ /^(xxdoc|xxmimo|xxflint)/ {n += $2} END {print n + 0}')
            if (( busy + 3 > 10 )); then log "waiting for eval nodes ($busy busy): $id $q"; continue; fi
            out=$(EVAL_LADDER_CODE=$TREE EVAL_DUALWIN_SEQ=8192 EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= bash run_doc_eval.sh "${id}_W32K/$q=run:$d" 2>&1)
            ids=$(echo "$out" | grep -o "job [0-9]*" | awk '{print $2}')
            if [[ -z $ids ]]; then log "submit FAILED $id $q: $(echo "$out" | tail -3 | tr '\n' ' ')"; continue; fi
            echo "$ids" > "$MK/$mk"; echo "$ids" >> "$MK/jobs"; log "submitted $id $q (both windows): $(echo $ids)"
        done
    done
    if [[ -f $MK/jobs ]]; then
        gone=0
        while read -r j; do [[ -f $MK/done-$j ]] && continue; if [[ -z $(squeue -h -j "$j" 2>/dev/null) ]]; then touch "$MK/done-$j"; gone=1; log "eval $j finished: $(sacct -n -X -j "$j" -o JobName%60,State,Elapsed | head -1)"; fi; done < "$MK/jobs"
        if (( gone )); then { python3 collect_doc_eval.py 2>&1 | tail -1; python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; fi
        pending=$(while read -r j; do [[ -f $MK/done-$j ]] || echo "$j"; done < "$MK/jobs" | wc -l)
    else pending=0; fi
    (( left == 0 && pending == 0 )) && { log "all done"; exit 0; }
    sleep 600
done
