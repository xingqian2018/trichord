#!/bin/bash
# reports_xx/nemo6/attach_client.sh <jobid> <NAME> <MODEL> <tasks> [lanes] - run (re-run) some nemo6 tasks against the vLLM server of a
# RUNNING nemo6_ref.sbatch job (extra client step, srun --overlap): e.g. a task that failed for a client-side reason (judge rate limit).
# Same client image / venv / env / runner as nemo6_ref.sbatch. Finished tasks are skipped as usual.
set -euo pipefail
JOB=$1; NAME=$2; MODEL=$3; TASKS=$4; LANES=${5:-1}
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
N6=$U/Project/cosmos_llm_scaling_ladder/reports_xx/nemo6
RT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval
CIMAGE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/container/nvidia+pytorch+25.08-py3.sqsh
# the server runs on the batch script's nodes[0]: read it from the job log ("nodes N (<head>)"); squeue %N order is not that order
head=$(grep -m1 -oP "nodes \d+ \(\K[^)]+" "$U/runs/nemo6_eval/sbatch/xxnemo6_$JOB.log" 2>/dev/null || squeue -h -j "$JOB" -o %B)
addr=$(getent hosts "$head" | awk '{print $1}')
curl -sf "http://$addr:8000/health" >/dev/null || { echo "server of job $JOB ($head) not healthy"; exit 1; }
export LME_SRC_DIR=$RT/src VENV=$RT/env HF_HOME=$U/data/nemo6_hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 XDG_CACHE_HOME=/tmp/nemo6_attach_$$ NEMO6_LCB_WORKERS=${NEMO6_LCB_WORKERS:-32} NEMO6_EXEC_DATA_GB=${NEMO6_EXEC_DATA_GB:-16}
echo "$(date) attach to job $JOB ($head $addr): $NAME tasks $TASKS"
srun -A cosmos_base_training --jobid="$JOB" --overlap --nodes=1 --ntasks=1 -w "$head" --container-image="$CIMAGE" \
    --container-mounts=/lustre:/lustre,/scratch:/scratch --container-workdir=/tmp \
    --export=ALL,PYTHONPATH=$N6/client_compat:$LME_SRC_DIR bash -c \
    'mkdir -p $XDG_CACHE_HOME; source $VENV/bin/activate; cd $LME_SRC_DIR; exec python -u "$@"' _ \
    "$N6/run_tasks.py" --config "$N6/${CONFIG:-nemo6_report.yml}" --results-dir "$U/runs/nemo6_eval" --model-name "$NAME" --tokenizer "$MODEL" \
    --host "$addr" --port 8000 --lanes "$LANES" --tasks "$TASKS" --judge-env "$U/private/nemo6_judge.env" \
    --manifest "{\"attach_to_job\": \"$JOB\"}"
