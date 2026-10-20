#!/bin/bash
set -e
RUN=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
C3_STATE=${C3_STATE:-c3_serve}
C3_SERVED_NAME=${C3_SERVED_NAME:-Slurm/Cosmos3-Super-Text2Image}
PROMPTS=${PROMPTS:-$HOME/data/t2i_data/agent_distilled_cosmos_captioner_full_prompts.jsonl}
OUTPUT=${OUTPUT:-s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_cosmos3_agent_distilled_prompt_v0}
test -f $playground/script/run_t2i_generation_cosmos3_from_prompts_v0.py || { echo "code not synced: git pull in ~/Project/trichord first"; exit 1; }
test -s $PROMPTS || { echo "missing prompts: $PROMPTS"; exit 1; }
mkdir -p $HOME/log/slurm
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_cosmos3_agent_distilled_prompt_v0 \
    --nodes=${NODES:-1} --exclusive --ntasks-per-node=${NTASKS:-32} --cpus-per-task=1 --time=2-00:00:00 \
    -o $HOME/log/slurm/t2i_cosmos3_agent_distilled_prompt_v0.%j.o -e $HOME/log/slurm/t2i_cosmos3_agent_distilled_prompt_v0.%j.e \
    --wrap="test -s $RUN/$C3_STATE/broker_url || { echo \"no dispatcher: $RUN/$C3_STATE/broker_url\"; exit 1; }; \
        export C3_ENDPOINTS=\$(cat $RUN/$C3_STATE/broker_url)/v1 C3_SERVED_NAME=$C3_SERVED_NAME PYTHONNOUSERSITE=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,C3_ENDPOINTS,C3_SERVED_NAME,PYTHONNOUSERSITE \
        python3 script/run_t2i_generation_cosmos3_from_prompts_v0.py --prompts $PROMPTS --output $OUTPUT"
