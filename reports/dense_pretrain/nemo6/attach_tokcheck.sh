#!/bin/bash
# reports_xx/nemo6/attach_tokcheck.sh <jobid> <NAME> <TOKENIZER> - run tok_check.py against the server of a RUNNING nemo6_ref job
# (for jobs submitted before the preflight existed). Extra env for the client tokenizer, e.g. NEMO6_TRUST_REMOTE_CODE=1, is passed through.
set -euo pipefail
JOB=$1; NAME=$2; TOK=$3
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
N6=$U/Project/cosmos_llm_scaling_ladder/reports_xx/nemo6
RT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval
CIMAGE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/container/nvidia+pytorch+25.08-py3.sqsh
# the server runs on the batch script's nodes[0]: read it from the job log ("nodes N (<head>)"); squeue %N order is not that order
head=$(grep -m1 -oP "nodes \d+ \(\K[^)]+" "$U/runs/nemo6_eval/sbatch/xxnemo6_$JOB.log" 2>/dev/null || squeue -h -j "$JOB" -o %B)
addr=$(getent hosts "$head" | awk '{print $1}')
export VENV=$RT/env LME_SRC_DIR=$RT/src HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
srun -A cosmos_base_training --jobid="$JOB" --overlap --nodes=1 --ntasks=1 -w "$head" --container-image="$CIMAGE" \
    --container-mounts=/lustre:/lustre,/scratch:/scratch --container-workdir=/tmp --export=ALL,PYTHONPATH=$N6/client_compat:$LME_SRC_DIR \
    bash -c 'source $VENV/bin/activate; exec python -u "$@"' _ "$N6/tok_check.py" --tokenizer "$TOK" --host "$addr" --port 8000 --model "$NAME"
