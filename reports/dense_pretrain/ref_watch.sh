#!/bin/bash
# reports_xx/ref_watch.sh JOBID... - background watcher (user 11-03 "搞个后台自己运行"): whenever one of the given Slurm jobs leaves the queue
# (or is stuck on a failed dependency), collect the doc evals, regenerate the report and mirror it. Exits when all jobs are gone.
# Run in tmux:  tmux new -d -s ref_watch "bash reports_xx/ref_watch.sh 35552 35810 ..."   Log: reports_xx/ref_watch.log
cd "$(dirname "$0")"
LOG=$PWD/ref_watch.log
left=" $* "
echo "$(date '+%F %T') watching:$left" >> "$LOG"
while [[ -n "${left// /}" ]]; do
    done_now=""
    for j in $left; do
        st=$(squeue -h -j "$j" -o "%T %r" 2>/dev/null)
        if [[ -z "$st" ]]; then
            done_now+=" $j"; echo "$(date '+%F %T') $j left the queue: $(sacct -n -X -j "$j" -o JobName%40,State,Elapsed 2>/dev/null | head -1)" >> "$LOG"
        elif [[ "$st" == *DependencyNeverSatisfied* ]]; then
            done_now+=" $j"; echo "$(date '+%F %T') $j dependency failed -> cancelled" >> "$LOG"; scancel "$j"
        fi
    done
    if [[ -n "$done_now" ]]; then
        for j in $done_now; do left="${left/ $j / }"; done
        { python3 collect_doc_eval.py 2>&1 | tail -2; python3 gen_eval_md.py 2>&1 | tail -1
          rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/ && echo mirrored; } >> "$LOG" 2>&1
        echo "$(date '+%F %T') still watching:$left" >> "$LOG"
    fi
    sleep 300
done
echo "$(date '+%F %T') all done" >> "$LOG"
