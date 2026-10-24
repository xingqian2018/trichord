#!/bin/bash
# reports_xx/auto_doc_eval.sh - every 20 min: submit the doc-isolated per-split eval (run_doc_eval.sh) for every checkpoint Boxiang has
# promoted (results/<Change ID>/ckpts/<rung>/) that we have neither evaluated nor queued. At most 10 of our eval jobs at a time: beyond
# that, new ones are chained (afterany) behind our newest job. Stops after 48 h or when $WORK_DIR/auto.stop exists.
#
#   nohup bash reports_xx/auto_doc_eval.sh > /dev/null 2>&1 &        log: $WORK_DIR/auto.log      stop: touch $WORK_DIR/auto.stop
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMOTED=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results
WORK_DIR=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval
LOG="${WORK_DIR}/auto.log"; MAX_JOBS=10; INTERVAL=1200; END=$(( $(date +%s) + 48 * 3600 ))
log() { echo "$(date '+%F %T') $*" >> "${LOG}"; }
rm -f "${WORK_DIR}/auto.stop"; log "start (pid $$)"
while (( $(date +%s) < END )) && [[ ! -e "${WORK_DIR}/auto.stop" ]]; do
    queued="$(squeue -u "${USER}" -h -o '%i %j' | awk '$2 ~ /^xxdoc-/ {sub(/^xxdoc-/, "", $2); print $1, $2}')"
    for d in "${PROMOTED}"/*/ckpts/q[0-9]; do
        change="$(basename "$(dirname "$(dirname "${d}")")")"; rung="$(basename "${d}")"
        [[ -r "${d}/latest_checkpointed_iteration.txt" && -r "${d}/argv.txt" ]] || continue
        it="$(cat "${d}/latest_checkpointed_iteration.txt")"; [[ -d "${d}/iter_$(printf %07d "${it}")" ]] || continue
        compgen -G "${WORK_DIR}/${change}-${rung}-*/eval/val_splits_*" > /dev/null && continue    # evaluated
        grep -q " ${change}-${rung}\$" <<<"${queued}" && continue                                 # queued / running
        n="$(grep -c . <<<"${queued}")"; after=()
        if (( n >= MAX_JOBS )); then after=(--after "$(awk '{print $1}' <<<"${queued}" | sort -n | tail -1)"); fi
        out="$(bash "${HERE}/run_doc_eval.sh" "${after[@]}" "${change}/${rung}" 2>&1)"; log "submit ${change}/${rung}: ${out}"
        queued="$(squeue -u "${USER}" -h -o '%i %j' | awk '$2 ~ /^xxdoc-/ {sub(/^xxdoc-/, "", $2); print $1, $2}')"
    done
    sleep "${INTERVAL}"
done
log "stop"
