#!/bin/bash
# reports_xx/run_doc_eval.sh - document-isolated, PER-SPLIT eval (VAL_PER_SPLIT=1: the 9 decon suites incl. private_internal, 115 splits)
# of Boxiang's ladder checkpoints, one 1-node job each, via code/scaling_ladder/sh/eval_ckpt.sh <ckpt> doc.
# Nothing is written into Boxiang's directories: EVAL_DIR is ours (eval_ckpt.sh's default would be his $SL_WORK), and so is the data
# cache (EVAL_DATA_CACHE = $WORK_DIR/data_cache/corpus_v2: symlinks to his corpus_v2 cache files + the new per-split validation indices;
# his dir is read-only for us and the first attempt died on writing there). Build it once before the first launch:
#   find <his cache> -maxdepth 1 -type f -print0 | xargs -0 -n 5000 ln -s -t $WORK_DIR/data_cache/corpus_v2
#
#   bash reports_xx/run_doc_eval.sh [--dry] [--after JOBID] [CHANGE/rung[=run:<run name>] ...]   (default: every checkpoint in CKPTS below)
#
# Jobs run as a QUEUE, one node at a time: each job depends on the previous one (afterany); --after chains the first one too.
#
# Results: $WORK_DIR/<CHANGE>-<rung>-<job>/ (RESULTS.md, eval/val_splits_iter<N>_doc_isolated.json, logs/doc.log);
# python3 reports_xx/collect_doc_eval.py turns them into JSON under trichord/dense_pretrain_result/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LADDER_CODE="$(cd "${HERE}/../code/scaling_ladder" && pwd)"
PROMOTED=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results
RUNS=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200
WORK_DIR=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval
DRY=0; PREV=""
while [[ "${1:-}" == --* ]]; do case "$1" in --dry) DRY=1; shift ;; --after) PREV="$2"; shift 2 ;; *) echo "ERROR: unknown option $1" >&2; exit 2 ;; esac; done

# CHANGE/rung  ->  promoted dir under $PROMOTED, or "run:<run name>" for finals Boxiang has not promoted yet (staged with symlinks)
declare -A CKPTS=(
  [BASE/q0]=p [BASE/q1]=p [BASE/q2]=p [BASE/q3]=p [BASE/q4]=p
  [OPT_MUON/q0]=p [OPT_MUON/q1]=p [OPT_MUON/q2]=p [OPT_MUON/q3]=p
  [OPT_MUONH/q0]=p [OPT_MUONH/q1]=p [OPT_MUONH/q2]=p [OPT_MUONH/q3]=p
  [KD_MUONH_A0P7/q0]=p [KD_MUONH_A0P7/q1]=p [KD_MUONH_A0P7/q2]=p [KD_MUONH_A0P7/q3]=p
  [KD_MUONH_A0P8/q0]=p [KD_MUONH_A0P8/q1]=p [KD_MUONH_A0P8/q2]=p [KD_MUONH_A0P8/q3]=p
  [KD_MUONH_A0P9/q0]=p [KD_MUONH_A0P9/q1]=p [KD_MUONH_A0P9/q2]=p [KD_MUONH_A0P9/q3]=p
)
# ~2x the pooled doc-eval times of Boxiang's runs (q0 1.1 h, q1-q2 1.7 h, q3 2.1 h, q4 2.7 h): splitting adds no tokens, private_internal adds ~3.2M
declare -A LIMIT=([q0]=02:30:00 [q1]=03:30:00 [q2]=03:30:00 [q3]=04:30:00 [q4]=05:30:00 [q5]=08:00:00 [q6]=12:00:00)

stage() {   # unpromoted final: <WORK_DIR>/stage/<CHANGE>/ckpts/<rung>/ with argv.txt, run.json, latest_checkpointed_iteration.txt, iter_*
    local change="$1" rung="$2" run="$3" dst="${WORK_DIR}/stage/$1/ckpts/$2" src="${RUNS}/$3"
    local it; it="$(cat "${src}/checkpoints/latest_checkpointed_iteration.txt")"
    mkdir -p "${dst}"
    ln -sfn "${src}/argv.txt" "${dst}/argv.txt"; ln -sfn "${src}/run.json" "${dst}/run.json"
    ln -sfn "${src}/checkpoints/latest_checkpointed_iteration.txt" "${dst}/latest_checkpointed_iteration.txt"
    ln -sfn "${src}/checkpoints/iter_$(printf %07d "${it}")" "${dst}/iter_$(printf %07d "${it}")"
    echo "${dst}"
}

mkdir -p "${WORK_DIR}/sbatch"
keys=("$@"); if (( ${#keys[@]} == 0 )); then mapfile -t keys < <(printf '%s\n' "${!CKPTS[@]}" | sort); fi
for key in "${keys[@]}"; do
    spec=""; if [[ "${key}" == *=* ]]; then spec="${key#*=}"; key="${key%%=*}"; fi   # CHANGE/rung=run:<run name>: a finished run not promoted yet
    change="${key%/*}"; rung="${key#*/}"; src="${spec:-${CKPTS[${key}]:-p}}"   # any other CHANGE/rung: Boxiang's promoted dir
    if [[ "${src}" == p ]]; then ckpt="${PROMOTED}/${change}/ckpts/${rung}"; else ckpt="$(stage "${change}" "${rung}" "${src#run:}")"; fi
    [[ -f "${ckpt}/latest_checkpointed_iteration.txt" && -f "${ckpt}/argv.txt" ]] || { echo "ERROR: ${ckpt} is not a checkpoint dir" >&2; exit 1; }
    name="xxdoc-${change}-${rung}"
    dep=(); if [[ -n "${PREV}" ]]; then dep=(--dependency="afterany:${PREV}"); fi
    cmd=(sbatch --parsable "${dep[@]}" -A cosmos_base_training -p batch_long --qos=hero-res --reservation=sla_res_cosmos --segment=1 --nodes=1 --ntasks-per-node=4
         --gres=gpu:4 --exclusive --mem=0 --time="${LIMIT[${rung}]}" --job-name="${name}" --output="${WORK_DIR}/sbatch/${name}_%j.log"
         --export=ALL,VAL_PER_SPLIT=1,LADDER_DIR="${LADDER_CODE}",EVAL_DATA_CACHE="${WORK_DIR}/data_cache/corpus_v2"
         --wrap="export EVAL_DIR=${WORK_DIR}/${change}-${rung}-\${SLURM_JOB_ID}; exec bash ${LADDER_CODE}/sh/eval_ckpt.sh ${ckpt} doc")
    if (( DRY )); then echo "[dry] ${key}: ${ckpt} (${LIMIT[${rung}]})${PREV:+ after ${PREV}}"; PREV="dry"; continue; fi
    PREV="$("${cmd[@]}")"; PREV="${PREV%%;*}"
    echo "${key}: job ${PREV} (${LIMIT[${rung}]})${dep[*]:+ ${dep[*]}}"
done
