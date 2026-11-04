#!/bin/bash
# reports_xx/packplan_autoeval.sh - data-derisk arms (80B pack plan, MUONH_FIT base: CE / KD 2.6 / KD 2.5, q0-q3), user 11-03:
# "记得这些新的实验，其实要evalute两份" + "如果有新的结果了, 数据 derisk的，记得update".
# Every 10 min: a run whose final checkpoint exists (latest_checkpointed_iteration == --train-iters) gets its two iso-doc evals
# (8K window: EVAL_EXTRA_ARGS='--seq-length 8192', <ID>; 32K window: the run's own 32768, <ID>_W32K), once (marker file);
# whenever one of our eval jobs leaves the queue: collect + regenerate + mirror. Exits when all 24 evals are done.
# Run in tmux: tmux new -d -s pp_autoeval "bash reports_xx/packplan_autoeval.sh"   Log: reports_xx/packplan_autoeval.log
cd "$(dirname "$0")"
LOG=$PWD/packplan_autoeval.log; MK=$PWD/.packplan_autoeval; mkdir -p "$MK"
R=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/qwen3_tpp200
ARMS=("MUONH_FIT_PP32K|qwen3_e9r500_32k_muonh_fit_packplan" "KD_MUONH_A0P8_S64Y_FIT_PP32K|qwen3_e9r500_32k_muonh_kd_a0p8_s64y_fit_packplan"
      "KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K|qwen3_e9r500_32k_muonh_kd_a0p8_s64y_fit_mimo25_packplan")
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start"
while :; do
    left=0
    for a in "${ARMS[@]}"; do
        id=${a%%|*}; run=${a##*|}
        for q in q0 q1 q2 q3; do
            d=$R/$run-$q-tpp200-s1
            [[ -f $MK/$id-$q ]] && continue
            left=1
            [[ -f $d/argv.txt && -f $d/checkpoints/latest_checkpointed_iteration.txt ]] || continue
            tot=$(grep -A1 -x -- "--train-iters" "$d/argv.txt" | tail -1); it=$(cat "$d/checkpoints/latest_checkpointed_iteration.txt")
            [[ "$it" == "$tot" ]] || continue
            j8=$(EVAL_QOS=normal EVAL_EXTRA_ARGS='--seq-length 8192' bash run_doc_eval.sh "$id/$q=run:$d" 2>&1 | tail -1)
            j32=$(EVAL_QOS=normal EVAL_EXTRA_ARGS= bash run_doc_eval.sh "${id}_W32K/$q=run:$d" 2>&1 | tail -1)
            echo "$j8 / $j32" > "$MK/$id-$q"; log "submitted $id $q: $j8 / $j32"
            echo "$j8" "$j32" | grep -o "job [0-9]*" | awk '{print $2}' >> "$MK/jobs"
        done
    done
    # collect when any of our eval jobs left the queue since the last pass
    if [[ -f $MK/jobs ]]; then
        gone=0
        while read -r j; do [[ -f $MK/done-$j ]] && continue; if [[ -z $(squeue -h -j "$j" 2>/dev/null) ]]; then touch "$MK/done-$j"; gone=1; log "eval $j finished: $(sacct -n -X -j "$j" -o JobName%45,State,Elapsed | head -1)"; fi; done < "$MK/jobs"
        if (( gone )); then { python3 collect_doc_eval.py 2>&1 | tail -1; python3 gen_eval_md.py 2>&1 | tail -1; rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1; fi
        pending=$(while read -r j; do [[ -f $MK/done-$j ]] || echo "$j"; done < "$MK/jobs" | wc -l)
    else pending=0; fi
    (( left == 0 && pending == 0 )) && { log "all done"; exit 0; }
    sleep 600
done
