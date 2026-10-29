#!/bin/bash
# reports_xx/auto_doc_eval.sh - every 20 min: submit the doc-isolated per-split eval (run_doc_eval.sh) for every final checkpoint of
# Boxiang's - finished runs not promoted yet (find_finished_runs.py) and promoted ones (results/<Change ID>/ckpts/<rung>/) - that we have
# neither evaluated nor queued. At most 10 of our eval jobs queued or running at a time: beyond that, nothing is submitted until a
# later round (chaining behind our newest job serialised everything). Stops after 48 h or when $WORK_DIR/auto.stop exists.
#
#   nohup bash reports_xx/auto_doc_eval.sh > /dev/null 2>&1 &        log: $WORK_DIR/auto.log      stop: touch $WORK_DIR/auto.stop
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMOTED=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results
WORK_DIR=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval
LOG="${WORK_DIR}/auto.log"; MAX_JOBS=10; INTERVAL=1200; END=$(( $(date +%s) + 48 * 3600 ))
log() { echo "$(date '+%F %T') $*" >> "${LOG}"; }
rm -f "${WORK_DIR}/auto.stop"; log "start (pid $$)"
seen() {   # evaluated or queued under this Change ID or its alias (Boxiang promotes the Muon grid as MUON_LR*, we evaluated it as OPT_MUON_LR*)
    local c
    for c in "$1" ${1/#MUON_LR/OPT_MUON_LR} ${1/#OPT_MUON_LR/MUON_LR} ${1/#MUONH_FIT_DSV3_SWA/SWA_FIT_DSV3} ${1/#SWA_FIT_DSV3/MUONH_FIT_DSV3_SWA} \
             ${1/#MUONH_FIT_SWA/SWA_FIT} ${1/#SWA_FIT/MUONH_FIT_SWA} \
             ${1/#MUONH_FIT_DSV3_NE_/ROUTER_NE_} ${1/#ROUTER_NE_/MUONH_FIT_DSV3_NE_}; do   # TODO 9: muonh_fit_dsv3_ne_* promoted as ROUTER_NE_*   # Boxiang promotes muonh_fit[_dsv3]_swa* as SWA_FIT[_DSV3]*
        compgen -G "${WORK_DIR}/${c}-$2-*/eval/val_splits_*" > /dev/null && return 0
        grep -q " ${c}-$2\$" <<<"${queued}" && return 0
    done
    return 1
}
while (( $(date +%s) < END )) && [[ ! -e "${WORK_DIR}/auto.stop" ]]; do
    queued="$(squeue -u "${USER}" -h -o '%i %j' | awk '$2 ~ /^xxdoc-/ {sub(/^xxdoc-/, "", $2); print $1, $2}')"
    # finished runs Boxiang has not promoted yet (evaluated straight from his run dir; skipped once evaluated / queued, promoted or not)
    while IFS= read -r spec; do
        key="${spec%%=*}"; change="${key%/*}"; rung="${key#*/}"
        seen "${change}" "${rung}" && continue
        [[ -d "${PROMOTED}/${change}/ckpts/${rung}" ]] && continue                               # promoted: handled by the loop below
        n="$(grep -c . <<<"${queued}")"; after=()
        if (( n >= MAX_JOBS )); then continue; fi             # cap reached: try again next round (no afterany chains: they serialise)
        out="$(bash "${HERE}/run_doc_eval.sh" "${after[@]}" "${spec}" 2>&1)"; log "submit ${spec}: ${out}"
        queued="$(squeue -u "${USER}" -h -o '%i %j' | awk '$2 ~ /^xxdoc-/ {sub(/^xxdoc-/, "", $2); print $1, $2}')"
    done < <(python3 "${HERE}/find_finished_runs.py")
    for d in "${PROMOTED}"/*/ckpts/q[0-9]; do
        change="$(basename "$(dirname "$(dirname "${d}")")")"; rung="$(basename "${d}")"
        [[ -r "${d}/latest_checkpointed_iteration.txt" && -r "${d}/argv.txt" ]] || continue
        it="$(cat "${d}/latest_checkpointed_iteration.txt")"; [[ -d "${d}/iter_$(printf %07d "${it}")" ]] || continue
        seen "${change}" "${rung}" && continue                                                    # evaluated / queued
        n="$(grep -c . <<<"${queued}")"; after=()
        if (( n >= MAX_JOBS )); then continue; fi             # cap reached: try again next round (no afterany chains: they serialise)
        out="$(bash "${HERE}/run_doc_eval.sh" "${after[@]}" "${change}/${rung}" 2>&1)"; log "submit ${change}/${rung}: ${out}"
        queued="$(squeue -u "${USER}" -h -o '%i %j' | awk '$2 ~ /^xxdoc-/ {sub(/^xxdoc-/, "", $2); print $1, $2}')"
    done
    sleep "${INTERVAL}"
done
log "stop"
