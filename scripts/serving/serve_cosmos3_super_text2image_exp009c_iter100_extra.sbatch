#!/bin/bash
# Extra Cosmos3 x9c replicas (one node each) for the running c3x9c-xqx service, in attach mode (POOL_ATTACH=1, see
# pool_lib.sh): no dispatcher of its own, the sidecars pull from the dispatcher in $U/run/c3x9c_serve/broker_url, so
# running clients share the load with no change. Own job name (c3x9cx-xqx), cancellable on its own.
# Submit (bash): sbatch -N <replicas> scripts/serving/serve_cosmos3_super_text2image_exp009c_iter100_extra.sbatch
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --wckey=p1
#SBATCH --no-kill
#SBATCH --segment=1
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J c3x9cx-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3x9c_serve/slurm-%j.log
export POOL_ATTACH=1
source "$HOME/Project/trichord/scripts/serving/serve_cosmos3_super_text2image_exp009c_iter100.sbatch"
