#!/bin/bash
# Extra Kimi-K3 replicas on gcpcode (GB200) for the running kimi_k3_gb200 service: same replicas as
# serve_kimi_k3_gb200.sbatch, in attach mode (POOL_ATTACH=1, see pool_lib.sh): no dispatcher of its own, broker_url is
# left alone, and the sidecars pull from the dispatcher in $U/run/k3_serve/broker_url, so running clients share the load
# with no change. Own job name (k3x-xqx), so it runs next to the main k3-xqx job and can be cancelled on its own.
# Submit (bash): bash scripts/serving/serve.sh kimi_k3_gb200_extra <replicas>      # requests 4 x <replicas> nodes
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --wckey=p1
#SBATCH --no-kill
#SBATCH --segment=4
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J k3x-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/slurm-%j.log
export POOL_ATTACH=1
source "$HOME/Project/trichord/scripts/serving/serve_kimi_k3_gb200.sbatch"
