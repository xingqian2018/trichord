#!/bin/bash
# reports_xx/nemo6/score_cached_cpu.sbatch - finish nemo6 tasks whose generations are ALL in the client's batch cache, on a CPU node, no server.
#   NAME=<NAME> MODEL=<hf snapshot (tokenizer)> TASKS=<task[,task]> sbatch reports_xx/nemo6/score_cached_cpu.sbatch
# Why (11-06): the scoring of mmlu_pro_0_shot_sampled (385,024 responses, CPU only) takes 30-50 min after the last generation. On a 4-node
# GPU job the idle GPUs got the job cancelled by svc-hwinf-cs-sched (uid 146504) ~30 min after the last generation, 3 times for Qwen3.8-Max
# (39840 / 40113 / 43992), always before results were written. Same client image / venv / env / runner / config as nemo6_ref.sbatch;
# --host 127.0.0.1 has no server, so a batch that is NOT cached fails with "connection refused" (task fails, no score written) - never a
# silently different result.
#SBATCH -A cosmos_base_training
#SBATCH -p cpu
#SBATCH --qos=cpu-interactive
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -J xxnemo6c
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/nemo6_eval/sbatch/xxnemo6c_%j.log
set -euo pipefail
: "${NAME:?}" "${MODEL:?}" "${TASKS:?}"
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
N6=$U/Project/cosmos_llm_scaling_ladder/reports_xx/nemo6
RT=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval
CIMAGE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/container/nvidia+pytorch+25.08-py3.sqsh
export LME_SRC_DIR=$RT/src VENV=$RT/env HF_HOME=$U/data/nemo6_hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
export NEMO6_NO_SERVER=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/tmp/nemo6_cpu_$SLURM_JOB_ID NEMO6_LCB_WORKERS=${NEMO6_LCB_WORKERS:-8} NEMO6_EXEC_DATA_GB=${NEMO6_EXEC_DATA_GB:-8}
# client = v2 only (user 11-08): the cache keys hold the seeded few-shot prompts of v2 runs; NEMO6_MAX_LENGTH = the generation run's
# server max sequence (only log-likelihood requests use it, and those come from the cache here)
export NEMO6_FIX=v2 NEMO6_FEWSHOT_SEED=${NEMO6_FEWSHOT_SEED:-1234} NEMO6_MAX_GEN_TOKS=${NEMO6_MAX_GEN_TOKS:-4096} NEMO6_MAX_LENGTH=${NEMO6_MAX_LENGTH:-16384}
echo "=== $(date) cached-only CPU scoring: $NAME tasks $TASKS on $(hostname)"
rc=0
srun --container-image="$CIMAGE" --container-mounts=/lustre:/lustre,/scratch:/scratch --container-workdir=/tmp \
    --export=ALL,PYTHONPATH=$N6/client_compat_v2:$N6/client_compat:$LME_SRC_DIR bash -c \
    'mkdir -p $XDG_CACHE_HOME; source $VENV/bin/activate; cd $LME_SRC_DIR; exec python -u "$@"' _ \
    "$N6/run_tasks.py" --config "$N6/${CONFIG:-nemo6_report.yml}" --results-dir "$U/runs/nemo6_eval" --model-name "$NAME" --tokenizer "$MODEL" \
    --host 127.0.0.1 --port 8000 --lanes 1 --tasks "$TASKS" --judge-env "$U/private/nemo6_judge.env" \
    --manifest "{\"cached_only_cpu\": true}" || rc=$?
echo "=== $(date) exit $rc"
exit $rc
