#!/bin/bash
# reports_xx/run_doc_eval.sh - document-isolated, PER-SPLIT eval (VAL_PER_SPLIT=1: the 9 decon suites incl. private_internal, 115 splits)
# of Boxiang's ladder checkpoints, one 1-node job each, via code/scaling_ladder/sh/eval_ckpt.sh <ckpt> doc.
# Nothing is written into Boxiang's directories: EVAL_DIR is ours (eval_ckpt.sh's default would be his $SL_WORK), and so is the data
# cache (EVAL_DATA_CACHE = $WORK_DIR/data_cache/corpus_v2: symlinks to his corpus_v2 cache files + the new per-split validation indices;
# his dir is read-only for us and the first attempt died on writing there). Build it once before the first launch:
#   find <his cache> -maxdepth 1 -type f -print0 | xargs -0 -n 5000 ln -s -t $WORK_DIR/data_cache/corpus_v2
#
#   bash reports_xx/run_doc_eval.sh [--dry] [--after JOBID] [CHANGE/rung[=run:<run name>] ...]   (default: every checkpoint in CKPTS below)
#   QOS: hero-res in the sla_res_cosmos reservation by default; EVAL_QOS=<qos> (e.g. normal) submits without the reservation.
#
# Jobs run as a QUEUE, one node at a time: each job depends on the previous one (afterany); --after chains the first one too.
#
# Results: $WORK_DIR/<CHANGE>-<rung>-<job>/ (RESULTS.md, eval/val_splits_iter<N>_doc_isolated.json, logs/doc.log);
# python3 reports_xx/collect_doc_eval.py turns them into JSON under trichord/dense_pretrain_result/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LADDER_CODE="$(cd "${EVAL_LADDER_CODE:-${HERE}/../code/scaling_ladder}" && pwd)"   # EVAL_LADDER_CODE: the tree a run was trained with (PROD1 overlays, 11-04)
# EVAL_DUALWIN_SEQ=8192 (user 11-04 "load 模型一次，出两个结果"): every CHANGE must end in _W32K and is evaluated at its own (larger) window;
# the SAME job / loaded model then evaluates window EVAL_DUALWIN_SEQ into <CHANGE without _W32K>[__P<i>of3]-<rung>-<job>
# (code/scaling_ladder/dualwin: a symlink farm of LADDER_CODE whose pretrain_doceval.py adds the second window)
DUALWIN="${EVAL_DUALWIN_SEQ:-}"
if [[ -n "${DUALWIN}" ]]; then LADDER_CODE="$(bash "${HERE}/../code/scaling_ladder/dualwin/make_tree.sh" "${LADDER_CODE}")"; fi
PROMOTED=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results
RUNS=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200
WORK_DIR=${EVAL_WORK_DIR:-/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval}   # EVAL_WORK_DIR: a test dir outside collect_doc_eval.py's glob
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
    if [[ "${run}" == /* ]]; then src="${run}"; fi           # run:/abs/path = a run outside Boxiang's group dir (e.g. our own q3 KD runs)
    local it; it="$(cat "${src}/checkpoints/latest_checkpointed_iteration.txt")"
    mkdir -p "${dst}"
    if grep -qx -- "--pack-plan" "${src}/argv.txt"; then      # pack-plan runs (code/scaling_ladder/packplan): training-only flags the eval entry rejects
        rm -f "${dst}/argv.txt"; awk 'skip { skip = 0; next } $0 == "--pack-plan" { skip = 1; next } $0 == "--packplan-eod-mask-loss" { next } { print }' \
            "${src}/argv.txt" > "${dst}/argv.txt"
    else ln -sfn "${src}/argv.txt" "${dst}/argv.txt"; fi
    ln -sfn "${src}/run.json" "${dst}/run.json"
    ln -sfn "${src}/checkpoints/latest_checkpointed_iteration.txt" "${dst}/latest_checkpointed_iteration.txt"
    ln -sfn "${src}/checkpoints/iter_$(printf %07d "${it}")" "${dst}/iter_$(printf %07d "${it}")"
    echo "${dst}"
}

mkdir -p "${WORK_DIR}/sbatch"
keys=("$@"); if (( ${#keys[@]} == 0 )); then mapfile -t keys < <(printf '%s\n' "${!CKPTS[@]}" | sort); fi
for key in "${keys[@]}"; do
    spec=""; if [[ "${key}" == *=* ]]; then spec="${key#*=}"; key="${key%%=*}"; fi   # CHANGE/rung=run:<run name>: a finished run not promoted yet
    change="${key%/*}"; rung="${key#*/}"; src="${spec:-${CKPTS[${key}]:-p}}"   # any other CHANGE/rung: Boxiang's promoted dir
    if [[ "${src}" == p ]]; then ckpt="${PROMOTED}/${change}/ckpts/${rung}"
    elif [[ "${src}" == p:* ]]; then ckpt="${PROMOTED}/${src#p:}/ckpts/${rung}"   # CHANGE/rung=p:<promoted CHANGE>: another eval id (e.g. <ID>_W32K) of a promoted run (11-04)
    else ckpt="$(stage "${change}" "${rung}" "${src#run:}")"; fi
    [[ -f "${ckpt}/latest_checkpointed_iteration.txt" && -f "${ckpt}/argv.txt" ]] || { echo "ERROR: ${ckpt} is not a checkpoint dir" >&2; exit 1; }
    limit="${LIMIT[${rung}]}"                                # off-ladder depth cells (_L<n>H<h>): ~4 min per layer (L 48 took 3:10)
    if [[ "${change}" =~ _L([0-9]+)H[0-9]+ ]]; then nl="${BASH_REMATCH[1]}"; h=$(( nl * 6 / 60 + 1 )); (( h > ${limit%%:*} )) && limit="$(printf %02d "${h}"):00:00"; fi
    if [[ "${change}" == *_W32K* ]]; then h=$(( 10#${limit%%:*} + 1 )); limit="$(printf %02d "${h}"):${limit#*:}"; fi   # 32K window: ~1.4x the 8K time (+1 h)
    if [[ -n "${DUALWIN}" ]]; then
        [[ "${change}" == *_W32K* ]] || { echo "ERROR: EVAL_DUALWIN_SEQ: ${change} must contain _W32K (window A = 32K, B = ${DUALWIN})" >&2; exit 1; }
        h=$(( 10#${limit%%:*} * 2 )); limit="$(printf %02d "${h}"):${limit#*:}"   # both windows in one job: the 32K limit + the 8K limit
    fi
    dep=(); if [[ -n "${PREV}" ]]; then dep=(--dependency="afterany:${PREV}"); fi
    qos=(--qos=hero-res --reservation=sla_res_cosmos); [[ "${EVAL_QOS:-hero-res}" != hero-res ]] && qos=(--qos="${EVAL_QOS}")   # EVAL_QOS=normal: sla_res_cosmos drained 10-29 (user 10-30)
    # EVAL_SPLIT3=1 (11-03, user "好慢啊"): three parallel 1-node jobs on disjoint suite groups (VAL_SUITES; samples at 8K: 60k / 47k / 33k of 141k),
    # ~2.5x shorter wall time, same node-hours; eval dirs <CHANGE>__P<i>of3-<rung>-<job>, merged into <CHANGE>_<rung>.json by collect_doc_eval.py
    parts=(""); if [[ "${EVAL_SPLIT3:-0}" == 1 ]]; then
        parts=("pile_test_decon_qwen3 pile_10k_decon_qwen3" "uncheatable_decon_qwen3 cosmos_val_v2_cc_decon_qwen3 cosmos_val_v2_python_decon_qwen3 cosmos_val_v2_arxiv_decon_qwen3"
               "codefresh_decon_qwen3 v3small_decon_qwen3 private_internal_decon_qwen3"); fi   # never ONE suite per job: then each split becomes its own set, no per-split JSON
    last=""
    for i in "${!parts[@]}"; do
        if [[ -n "${EVAL_ONLY_PARTS:-}" && ",${EVAL_ONLY_PARTS}," != *",$((i + 1)),"* ]]; then continue; fi   # EVAL_ONLY_PARTS=2: re-run one failed part
        ch="${change}"; [[ -n "${parts[$i]}" ]] && ch="${change}__P$((i + 1))of${#parts[@]}"
        name="xxdoc-${ch}-${rung}"
        dw=""; if [[ -n "${DUALWIN}" ]]; then chb="${change/_W32K/}"; [[ -n "${parts[$i]}" ]] && chb="${chb}__P$((i + 1))of${#parts[@]}"
            dw="export PYTHONDONTWRITEBYTECODE=1 DUALWIN_SEQ=${DUALWIN} DUALWIN_DIR=${WORK_DIR}/${chb}-${rung}-\${SLURM_JOB_ID}; "; fi
        cmd=(sbatch --parsable "${dep[@]}" -A cosmos_base_training -p batch_long "${qos[@]}" --segment=1 --nodes=1 --ntasks-per-node=4
             --gres=gpu:4 --exclusive --mem=0 --time="${limit}" --job-name="${name}" --output="${WORK_DIR}/sbatch/${name}_%j.log"
             --export=ALL,VAL_PER_SPLIT=1,LADDER_DIR="${LADDER_CODE}",EVAL_DATA_CACHE="${WORK_DIR}/data_cache/corpus_v2"
             --wrap="${dw}export EVAL_DIR=${WORK_DIR}/${ch}-${rung}-\${SLURM_JOB_ID}; exec bash ${LADDER_CODE}/sh/eval_ckpt.sh ${ckpt} doc")
        if (( DRY )); then echo "[dry] ${ch}/${rung}: ${ckpt} (${limit})${PREV:+ after ${PREV}}${parts[$i]:+ suites: ${parts[$i]}}${dw:+ + window ${DUALWIN} -> ${chb}} tree ${LADDER_CODE}"; last="dry"; continue; fi
        if [[ -n "${parts[$i]}" ]]; then last="$(VAL_SUITES="${parts[$i]}" "${cmd[@]}")"; else last="$("${cmd[@]}")"; fi
        last="${last%%;*}"
        echo "${ch}/${rung}: job ${last} (${limit})${dep[*]:+ ${dep[*]}}"
    done
    PREV="${last}"
done
