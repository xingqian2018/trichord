"""
Docker build:

lustrepath=/lustre/fsw/portfolios/cosmos
mkdir -p $lustrepath/users/xingqianx/Container
srun --account=cosmos_base_training --nodes=1 --exclusive --ntasks-per-node=76 --cpus-per-task=1 --time=04:00:00 \
     --container-image=$lustrepath/projects/cosmos_base_training/containers/imaginaire4_v12.0.0.sqsh \
     --container-save=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
     --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
     --container-env=HOME \
     --container-remap-root \
     bash -c "pip install playwright && PLAYWRIGHT_BROWSERS_PATH=$HOME/Software/playwright-browsers python -m playwright install --with-deps chromium-headless-shell"

     
Run CMD (same docker/mounts as slaunch, but plain srun: ONE CONTAINER PER TASK, no torchrun):

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-long --job-name=poster_gen_v1 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=7-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v1.%j.o -e $HOME/log/slurm/poster_gen_v1.%j.e \
    --wrap="srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$HOME/Project/trichord/scripts/model_playground_v1 \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID \
        python3 script/run_poster_generation_v1.py \
            --model kimi-k3@nvidiak \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v1_kimi_cosmos3_kimi \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v1 import poster_generation_v1  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v1)
