#!/bin/bash
# reports_xx/prodkd_autoeval.sh - iso-doc evals of PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K (Runlong's PROD1_DPSK + 80B pack plan + KD MiMo-V2.5, user 11-04:
# "2.5 在 runlong 的 PROD1_DPSK 架构上的，只是更改到最新 PP32K 的训练，q0x3 q1/2/3"; jobs 38834-43). Same contract as packplan_autoeval.sh (which is left
# untouched while it runs): every 10 min, a run whose final checkpoint exists gets both windows once (marker file); whenever one of our eval jobs leaves
# the queue: collect + regenerate + mirror. Code tree = the one PROD1_DPSK trains with (prod/overlay, EVAL_LADDER_CODE), as launch_prod1_dualwin.sh.
#   seed 1: ONE job set evaluates both windows (EVAL_DUALWIN_SEQ=8192): <ID>_W32K (32K) + <ID> (8K)
#   seeds 2 / 3: two job sets, <ID>_S<k> (8K, EVAL_EXTRA_ARGS='--seq-length 8192') and <ID>_W32K_S<k> (32K) - the report's seed ids (gen_eval_html._seed_id)
# Exits when all 6 checkpoints (q0 s1-s3, q1, q2, q3) are evaluated and collected.
# Run in tmux: tmux new -d -s prodkd_autoeval "bash reports_xx/prodkd_autoeval.sh"   Log: ~/tmp/reports_xx_autoeval/prodkd_autoeval.log (markers: ~/tmp/reports_xx_autoeval/prodkd_autoeval/)
cd "$(dirname "$0")"
ST=$HOME/tmp/reports_xx_autoeval; LOG=$ST/prodkd_autoeval.log; MK=$ST/prodkd_autoeval; mkdir -p "$MK"   # state + log outside the repo (user 11-05)
R=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1_packplan
PROD=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/share/cosmos_llm_scaling_ladder_wt_prod/code/scaling_ladder/prod/overlay
ID=PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K; RUN=qwen3_e9r500_32k_prod1_dpsk_kd_a0p8_s64y_mimo25_packplan
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start"
while :; do
    left=0
    held=$(squeue -u "$USER" -h -o '%i|%j|%r' | awk -F'|' '$2 ~ /^xxdoc-PROD1_DPSK_KD_/ && tolower($3) ~ /requeued.held/ {print $1}' | paste -sd,)
    if [[ -n $held ]]; then scontrol release "$held" >> "$LOG" 2>&1; log "released held eval jobs: $held"; fi
    for qs in q0:1 q0:2 q0:3 q1:1 q2:1 q3:1; do
        q=${qs%:*}; sd=${qs#*:}; mk="$ID-$q-s$sd"; d=$R/$RUN-$q-tpp200-s$sd
        [[ -f $MK/$mk ]] && continue
        left=1
        [[ -f $d/argv.txt && -f $d/checkpoints/latest_checkpointed_iteration.txt ]] || continue
        tot=$(grep -A1 -x -- "--train-iters" "$d/argv.txt" | tail -1); it=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt")
        [[ "$it" == "$tot" ]] || continue
        need=3; (( sd > 1 )) && need=6
        busy=$(squeue -u "$USER" -h -o '%j %D' | awk '$1 ~ /^(xxdoc|xxmimo|xxflint)/ {n += $2} END {print n + 0}')
        if (( busy + need > 10 )); then log "waiting for eval nodes ($busy busy): $ID $q s$sd"; continue; fi
        if (( sd == 1 )); then
            out=$(EVAL_LADDER_CODE=$PROD EVAL_DUALWIN_SEQ=8192 EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= bash run_doc_eval.sh "${ID}_W32K/$q=run:$d" 2>&1)
        else
            out=$(EVAL_LADDER_CODE=$PROD EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS='--seq-length 8192' bash run_doc_eval.sh "${ID}_S$sd/$q=run:$d" 2>&1
                  EVAL_LADDER_CODE=$PROD EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= bash run_doc_eval.sh "${ID}_W32K_S$sd/$q=run:$d" 2>&1)
        fi
        ids=$(echo "$out" | grep -o "job [0-9]*" | awk '{print $2}')
        if [[ -z $ids ]]; then log "submit FAILED $ID $q s$sd: $(echo "$out" | tail -3 | tr '\n' ' ')"; continue; fi
        echo "$ids" > "$MK/$mk"; echo "$ids" >> "$MK/jobs"; log "submitted $ID $q s$sd: $(echo $ids)"
    done
    if [[ -f $MK/jobs ]]; then
        gone=0
        while read -r j; do [[ -f $MK/done-$j ]] && continue; if [[ -z $(squeue -h -j "$j" 2>/dev/null) ]]; then touch "$MK/done-$j"; gone=1; log "eval $j finished: $(sacct -n -X -j "$j" -o JobName%50,State,Elapsed | head -1)"; fi; done < "$MK/jobs"
        if (( gone )); then { python3 collect_doc_eval.py 2>&1 | tail -1; python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; fi
        pending=$(while read -r j; do [[ -f $MK/done-$j ]] || echo "$j"; done < "$MK/jobs" | wc -l)
    else pending=0; fi
    (( left == 0 && pending == 0 )) && { log "all done"; exit 0; }
    sleep 600
done
