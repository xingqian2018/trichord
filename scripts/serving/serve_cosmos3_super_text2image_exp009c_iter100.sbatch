#!/bin/bash
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
#SBATCH -J c3x9c-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3x9c_serve/slurm-%j.log
export C3_STATE=c3x9c_serve
export C3_SERVED_NAME=Slurm/Cosmos3-Super-Text2Image-x9c
export C3_MODEL=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/models/cosmos3_super_text2image_exp009c_iter100
source "$HOME/Project/trichord/scripts/serving/serve_cosmos3_super_t2i.sbatch"
