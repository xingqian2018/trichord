#!/bin/bash
# reports_xx/packplan_autoeval.sh - data-derisk arms (80B pack plan, MUONH_FIT base: CE / KD 2.6 / KD 2.5, q0-q3), user 11-03:
# "记得这些新的实验，其实要evalute两份" + "如果有新的结果了, 数据 derisk的，记得update".
# Every 10 min: a run whose final checkpoint exists (latest_checkpointed_iteration == --train-iters) gets its two iso-doc evals
# (8K window: EVAL_EXTRA_ARGS='--seq-length 8192', <ID>; 32K window: the run's own 32768, <ID>_W32K), once (marker file);
# whenever one of our eval jobs leaves the queue: collect + regenerate + mirror. Exits when all 24 evals are done.
# Run in tmux: tmux new -d -s pp_autoeval "bash reports_xx/packplan_autoeval.sh"   Log: ~/tmp/reports_xx_autoeval/packplan_autoeval.log (markers: ~/tmp/reports_xx_autoeval/packplan_autoeval/)
cd "$(dirname "$0")"
ST=$HOME/tmp/reports_xx_autoeval; LOG=$ST/packplan_autoeval.log; MK=$ST/packplan_autoeval; mkdir -p "$MK"   # state + log outside the repo (user 11-05)
R=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/qwen3_tpp200
# ID|run stem|run root (default $R)|code tree the run trained with (default: run_doc_eval.sh's)
PP1=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1_packplan
LIVE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/code/scaling_ladder
ARMS=("MUONH_FIT_PP32K|qwen3_e9r500_32k_muonh_fit_packplan" "KD_MUONH_A0P8_S64Y_FIT_PP32K|qwen3_e9r500_32k_muonh_kd_a0p8_s64y_fit_packplan"
      "KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K|qwen3_e9r500_32k_muonh_kd_a0p8_s64y_fit_mimo25_packplan"
      "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K|qwen3_e9r500_32k_prod1_dpsk_kd_a0p8_s64y_mimo25_packplan|$PP1|$LIVE")   # 11-04: KD 2.5 on PROD1_DPSK (launched 15:53)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start"
while :; do
    left=0
    # 11-04 07:30: Slurm sometimes fails the job's env retrieval at launch and leaves it "requeued held" (Priority 0) forever:
    # 8 eval parts sat held 01:27-07:30 and their 8 nodes also blocked new submissions. Release such held jobs of ours (same args).
    held=$(squeue -u "$USER" -h -o '%i|%j|%r' | awk -F'|' '$2 ~ /^(xxdoc|xxmimo|xxflint)/ && tolower($3) ~ /requeued.held/ {print $1}' | paste -sd,)
    if [[ -n $held ]]; then scontrol release "$held" >> "$LOG" 2>&1; log "released held eval jobs: $held"; fi
    # rung/seed outer, arm inner (11-04): with the 10-node cap, lower rungs / q0 seeds of every arm go before higher rungs
    for qs in q0:1 q0:2 q0:3 q1:1 q2:1 q3:1; do          # q0 seeds 2 / 3 (user 11-03: 3-seed q0 means; change ids <ID>_S2, <ID>_W32K_S2)
        for a in "${ARMS[@]}"; do
            IFS='|' read -r id run root tree <<<"$a"; root=${root:-$R}
            q=${qs%:*}; sd=${qs#*:}; sfx=""; mk="$id-$q"; if (( sd > 1 )); then sfx="_S$sd"; mk="$id-$q-s$sd"; fi
            d=$root/$run-$q-tpp200-s$sd
            [[ -f $MK/$mk ]] && continue
            left=1
            [[ -f $d/argv.txt && -f $d/checkpoints/latest_checkpointed_iteration.txt ]] || continue
            tot=$(grep -A1 -x -- "--train-iters" "$d/argv.txt" | tail -1); it=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt")
            [[ "$it" == "$tot" ]] || continue
            # 11-03 23:15 (user "好慢啊"): 3 parallel 1-node jobs on disjoint suite groups (run_doc_eval.sh EVAL_SPLIT3=1, merged by collect_doc_eval.py).
            # 11-04 (user "load 模型一次，出两个结果"): each job loads the model ONCE and scores both windows (EVAL_DUALWIN_SEQ=8192: own 32K ->
            # <ID>_W32K, then 8K -> <ID>; validated bit-identical to separate runs). 3 nodes per checkpoint; at most EVAL_CAP (20) eval nodes.
            # nodes in use or about to start: RUNNING + PENDING except those waiting on a dependency (chained eval queues, 11-04)
            # 11-04 23:00: squeue socket timeouts (empty output) must not read as "0 busy": skip the submission this pass instead
            sq=$(squeue -u "$USER" -h -o '%j|%D|%T|%r' 2>/dev/null) || { log "squeue failed, retry next pass: $id$sfx $q"; continue; }
            busy=$(echo "$sq" | awk -F'|' '$1 ~ /^(xxdoc|xxmimo|xxflint)/ && !($3 == "PENDING" && $4 ~ /^Dependency/) {n += $2} END {print n + 0}')
            if (( busy + 3 > ${EVAL_CAP:-20} )); then log "waiting for eval nodes ($busy busy, cap ${EVAL_CAP:-20}): $id$sfx $q"; continue; fi   # cap 10 -> 20 (user 11-04 21:5x "我们derisk结果出的很慢啊")
            j2=$(EVAL_SPLIT3=1 EVAL_QOS=normal EVAL_EXTRA_ARGS= EVAL_DUALWIN_SEQ=8192 EVAL_LADDER_CODE="${tree:-}" bash run_doc_eval.sh "${id}_W32K$sfx/$q=run:$d" 2>&1)
            ids=$(echo "$j2" | grep -o "job [0-9]*" | awk '{print $2}')
            [[ -n "$ids" ]] || { log "submit FAILED $id$sfx $q: $(echo $j2 | tail -c 300)"; continue; }
            echo "$ids" > "$MK/$mk"; log "submitted $id$sfx $q (3 parts, both windows per job): $(echo $ids)"
            echo "$ids" >> "$MK/jobs"
        done
    done
    # collect when any of our eval jobs left the queue since the last pass (terminal sacct state; an empty squeue on a socket timeout is not "finished", 11-04)
    if [[ -f $MK/jobs ]]; then
        gone=0
        while read -r j; do [[ -f $MK/done-$j ]] && continue; st=$(sacct -n -X -j "$j" -o State 2>/dev/null | head -1 | awk '{print $1}'); if [[ $st =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)$ ]]; then touch "$MK/done-$j"; gone=1; log "eval $j finished: $(sacct -n -X -j "$j" -o JobName%45,State,Elapsed | head -1)"; fi; done < "$MK/jobs"
        if (( gone )); then { python3 collect_doc_eval.py 2>&1 | tail -1; python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; fi
        pending=$(while read -r j; do [[ -f $MK/done-$j ]] || echo "$j"; done < "$MK/jobs" | wc -l)
    else pending=0; fi
    (( left == 0 && pending == 0 )) && { log "all done"; exit 0; }
    sleep 600
done
