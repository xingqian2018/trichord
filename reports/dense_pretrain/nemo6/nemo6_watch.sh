#!/bin/bash
# reports_xx/nemo6/nemo6_watch.sh [interval_s=1800] - keep the eval report's nemo6 tables current while nemo6 jobs (xxnemo6) run:
# every interval: collect_nemo6.py (raw -> trichord/dense_pretrain_result/nemo6 -> shared copy -> results_xx/nemo6) + gen_eval_html.py
# (zh / en HTML + md mirror). One last round after the last xxnemo6 job leaves the queue, then exit. Run it in tmux `nemo6_watch`.
cd "$(dirname "$0")/.." || exit 1
IV=${1:-1800}
LOG=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/nemo6_eval/sbatch/nemo6_watch.log
while true; do
    n=$(squeue -h -u "$USER" -n xxnemo6 -o %i 2>/dev/null | wc -l)
    { echo "=== $(date '+%F %T') xxnemo6 jobs in queue: $n"
      python3 nemo6/collect_nemo6.py 2>&1 | grep -E "complete=|copied" | sed -E 's/  -> .*//'
      python3 gen_eval_html.py 2>&1 | tail -1 | cut -c1-160; } >> "$LOG" 2>&1
    (( n == 0 )) && { echo "=== $(date '+%F %T') no xxnemo6 job left, exit" >> "$LOG"; exit 0; }
    sleep "$IV"
done
