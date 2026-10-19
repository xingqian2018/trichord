#!/bin/bash
set -e
RUN=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
test -f $playground/script/run_t2i_generation_cosmos3_v0.py || { echo "code not synced: git pull in ~/Project/trichord first"; exit 1; }
mkdir -p $HOME/log/slurm
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_cosmos3_v0 \
    --nodes=${NODES:-1} --exclusive --ntasks-per-node=${NTASKS:-32} --cpus-per-task=1 --time=1-00:00:00 \
    -o $HOME/log/slurm/t2i_cosmos3_v0.%j.o -e $HOME/log/slurm/t2i_cosmos3_v0.%j.e \
    --wrap="for f in $RUN/k3_serve/broker_url $RUN/c3_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat $RUN/k3_serve/broker_url)/v1 C3_ENDPOINTS=\$(cat $RUN/c3_serve/broker_url)/v1 K3_NATIVE_TOOLS=1 PYTHONNOUSERSITE=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,C3_ENDPOINTS,K3_NATIVE_TOOLS,PYTHONNOUSERSITE \
        python3 script/run_t2i_generation_cosmos3_v0.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_cosmos3_kimik3_structured_v0 \
            --reasoning_effort random --max_samples_per_process 2000"
