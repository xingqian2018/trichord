#!/bin/bash
# reports_xx/nemo6/copy_hf_cache.sbatch - copy Boxiang's nemo6 dataset cache (HF datasets + hub, ~45 GB, incl. the gated GPQA / HLE he
# downloaded) to our own writable HF_HOME (datasets writes lock files: his dir is read-only for us). CPU node, cpu-interactive.
#SBATCH -A cosmos_base_training
#SBATCH -p cpu
#SBATCH --qos=cpu-interactive
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J xxnemo6cp
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/nemo6_eval/sbatch/xxnemo6cp_%j.log
set -euo pipefail
SRC=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval/hf_cache
DST=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/data/nemo6_hf_cache
mkdir -p "$DST"
echo "$(date) copy $SRC -> $DST"
for d in datasets hub xet; do
    [[ -d $SRC/$d ]] && rsync -a --no-owner --no-group --chmod=Du+w,Fu+w "$SRC/$d" "$DST/" &
done
wait
du -sh "$DST"/*
echo "$(date) done"
