#!/bin/bash
# reports_xx/router_autoeval.sh - our 115-split doc evals of Runlong's router / load-balancing arms on PROD1_DPSK_LATENT2_E2X (user 11-06
# "runlong有了moe routing QB的实验"): QB / QBX (quantile balancing), BIASONLY (no seq-aux), G5EM4 / G2P5EM4 (bias rate), RTADAMW / RTPIN (router optimizer); 11-08 QBK3 (q1-q4) / QBK3_WU (q1-q2).
# Every 10 min: a run whose final checkpoint exists (latest_checkpointed_iteration == --train-iters) gets ONE dual-window eval (32K -> <ID>_W32K,
# 8K -> <ID>; 3 one-node parts), same recipe as the E2X base (EVAL_LADDER_CODE = prod/overlay_latent; Megatron from the run's run.json, so QBX gets
# its patched copy); whenever one of these eval jobs leaves the queue: collect + regenerate + mirror. Exits when every listed cell is done.
# Run in tmux: tmux new -d -s rt_autoeval "bash reports_xx/router_autoeval.sh"   Log: ~/tmp/reports_xx_autoeval/router_autoeval.log
cd "$(dirname "$0")"
ST=$HOME/tmp/reports_xx_autoeval; LOG=$ST/router_autoeval.log; MK=$ST/router_autoeval; mkdir -p "$MK"
RB=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1
LATENT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/share/cosmos_llm_scaling_ladder_wt_prod/code/scaling_ladder/prod/overlay_latent
ROWBALL=${LATENT}_rowball                                  # RTPIN trained with prod/overlay_latent_rowball (HYPERP_ROUTER_ROWS=1)
# Change ID | run stem (run dir = <stem>-<q>-iters<N>-s1) | rungs | code tree (default $LATENT)
ARMS=("PROD1_DPSK_LATENT2_E2X_BIASONLY|qwen3_tpp200_prod1_dpsk_latent2_e2x_biasonly|q1 q2"
      "PROD1_DPSK_LATENT2_E2X_G5EM4|qwen3_tpp200_prod1_dpsk_latent2_e2x_g5em4|q1 q2"
      "PROD1_DPSK_LATENT2_E2X_G2P5EM4|qwen3_tpp200_prod1_dpsk_latent2_e2x_g2p5em4|q1 q2"
      "PROD1_DPSK_LATENT2_E2X_QB|qwen3_tpp200_prod1_dpsk_latent2_e2x_qb|q1 q2 q3"
      "PROD1_DPSK_LATENT2_E2X_QBX|qwen3_tpp200_prod1_dpsk_latent2_e2x_qbx|q1 q2 q3"
      "PROD1_DPSK_LATENT2_E2X_RTADAMW|qwen3_tpp200_prod1_dpsk_latent2_e2x_rtadamw|q1 q2 q3"
      "PROD1_DPSK_LATENT2_E2X_RTPIN|qwen3_tpp200_prod1_dpsk_latent2_e2x_rtpin|q1 q2 q3|$ROWBALL"
      "PROD1_DPSK_LATENT2_E2X_QBK3|qwen3_tpp200_prod1_dpsk_latent2_e2x_qbk3|q1 q2 q3 q4"        # 11-08: K3-exact QB (Megatron -qbk3 tree from run.json)
      "PROD1_DPSK_LATENT2_E2X_QBK3_WU|qwen3_tpp200_prod1_dpsk_latent2_e2x_qbk3_wu|q1 q2")       # 11-08: QBK3 + 1% LR warmup
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start"
while :; do
    left=0
    held=$(squeue -u "$USER" -h -o '%i|%j|%r' | awk -F'|' '$2 ~ /^(xxdoc|xxmimo|xxflint)/ && tolower($3) ~ /requeued.held/ {print $1}' | paste -sd,)
    if [[ -n $held ]]; then scontrol release "$held" >> "$LOG" 2>&1; log "released held eval jobs: $held"; fi
    for q in q1 q2 q3 q4; do                               # lower rungs of every arm first
        for a in "${ARMS[@]}"; do
            IFS='|' read -r id stem rungs tree <<<"$a"
            [[ " $rungs " == *" $q "* ]] || continue
            mk="$id-$q"; [[ -f $MK/$mk ]] && continue
            # evaluated elsewhere (e.g. QBK3 q1 by another session 11-08) or already queued: mark, never submit a duplicate
            if [[ -f ../results_xx/${id}_$q.json ]] || squeue -u "$USER" -h -o %j 2>/dev/null | grep -qx "xxdoc-${id}_W32K__P1of3-$q"; then
                echo external > "$MK/$mk"; log "already evaluated / queued elsewhere: $id $q"; continue; fi
            left=1
            d=$(ls -d "$RB/$stem-$q"-iters*-s1 2>/dev/null | head -1)
            [[ -n $d && -f $d/argv.txt && -f $d/checkpoints/latest_checkpointed_iteration.txt ]] || continue
            tot=$(grep -A1 -x -- "--train-iters" "$d/argv.txt" | tail -1); it=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt")
            [[ "$it" == "$tot" ]] || continue
            sq=$(squeue -u "$USER" -h -o '%j|%D|%T|%r' 2>/dev/null) || { log "squeue failed, retry next pass: $id $q"; continue; }
            busy=$(echo "$sq" | awk -F'|' '$1 ~ /^(xxdoc|xxmimo|xxflint)/ && !($3 == "PENDING" && $4 ~ /^Dependency/) {n += $2} END {print n + 0}')
            if (( busy + 3 > ${EVAL_CAP:-20} )); then log "waiting for eval nodes ($busy busy, cap ${EVAL_CAP:-20}): $id $q"; continue; fi
            j2=$(EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= EVAL_DUALWIN_SEQ=8192 EVAL_LADDER_CODE="${tree:-$LATENT}" bash run_doc_eval.sh "${id}_W32K/$q=run:$d" 2>&1)
            ids=$(echo "$j2" | grep -o "job [0-9]*" | awk '{print $2}')
            [[ -n "$ids" ]] || { log "submit FAILED $id $q: $(echo $j2 | tail -c 300)"; continue; }
            echo "$ids" > "$MK/$mk"; log "submitted $id $q (3 parts, both windows per job): $(echo $ids)"
            echo "$ids" >> "$MK/jobs"
        done
    done
    if [[ -f $MK/jobs ]]; then
        gone=0
        while read -r j; do [[ -f $MK/done-$j ]] && continue; st=$(sacct -n -X -j "$j" -o State%30 2>/dev/null | head -1 | awk '{print $1}'); if [[ $st =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)$ ]] || { [[ $st == PREEMPTED ]] && sq1=$(squeue -h -j "$j" -o %T 2>&1) && [[ -z $sq1 ]]; }; then touch "$MK/done-$j"; gone=1; log "eval $j finished: $(sacct -n -X -j "$j" -o JobName%45,State,Elapsed | head -1)"; fi; done < "$MK/jobs"
        # PREEMPTED counts as finished only once the job has left the queue: on slurm-2 a preempted eval is requeued and later completes
        # (11-08: QBK3 q2 P1 55217 was marked done at its preemption and its 8K result sat uncollected until 11-09)
        # hourly regeneration even without a finished eval (user 11-06 "记得把相关的结果存在html里"): the section's training-loss / router-health
        # tables come from the live training logs
        now=$(date +%s); last=$(cat "$MK/last_regen" 2>/dev/null || echo 0)
        if (( ! gone && now - last > 3600 )); then { python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; echo "$now" > "$MK/last_regen"; fi
        if (( gone )); then echo "$now" > "$MK/last_regen"; { python3 collect_doc_eval.py 2>&1 | tail -1; python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; fi
        pending=$(while read -r j; do [[ -f $MK/done-$j ]] || echo "$j"; done < "$MK/jobs" | wc -l)
    else pending=0; fi
    (( left == 0 && pending == 0 )) && { log "all done"; exit 0; }
    sleep 600
done
