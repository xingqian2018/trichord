r"""
v1p2 = v1 (cosmos3 t2i + render + visual critic) with the agent AND the critic on Slurm-hosted Kimi-K3
(`--model kimi-k3@k3slurm`, see gateway_kimik3.py). Endpoints are discovered on the batch host (squeue is not
available inside the container) and refreshed every 5 min into $K3_ENDPOINTS_FILE, which the gateway re-reads.

Docker build: same container as run_poster_generation_v1.py (no extra packages needed).

Run CMD:

mkdir -p $HOME/log/slurm $HOME/log/k3_endpoints
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=poster_gen_v1p2 \
    --nodes=2 --exclusive --ntasks-per-node=16 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v1p2.%j.o -e $HOME/log/slurm/poster_gen_v1p2.%j.e \
    --wrap="export K3_ENDPOINTS_FILE=$HOME/log/k3_endpoints/\$SLURM_JOB_ID.json; \
        python3 $playground/k3_endpoints.py --out \$K3_ENDPOINTS_FILE || exit 1; \
        python3 $playground/k3_endpoints.py --out \$K3_ENDPOINTS_FILE --every 300 > $HOME/log/k3_endpoints/\$SLURM_JOB_ID.log 2>&1 & \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS_FILE \
        bash script/isolated_tmp.sh python3 script/run_poster_generation_v1p2.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v1p2_k3slurm_cosmos3_k3slurm \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 510; \
        kill %1"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3  # noqa: E402

gateway_kimik3.install()  # before run_poster_generation imports build any Agent

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v1p2 import poster_generation_v1p2  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v1p2)
