r"""
v1p3 = v1p2 plus: (1) when a k3slurm request exceeds its context limit, that agent switches to kimi-k3@nvidiak
(gateway_kimik3_fallback.py; the count is recorded as fallback_requests in setting.json); (2) the critic gets
600s x 2 attempts and 16384 output tokens; (3) missing tool arguments and missing critic image paths are returned
to the model as ToolCallError instead of failing the sample.

v1p2 = v1 (cosmos3 t2i + render + visual critic) with the agent AND the critic on Slurm-hosted Kimi-K3
(`--model kimi-k3@k3slurm`, see gateway_kimik3.py). The command below is the v1p4 run: Kimi-K3 is our own service
(scripts/serving/serve_kimi_k3.sbatch), reached through its dispatcher URL ($U/run/k3_serve/broker_url) passed as
K3_ENDPOINTS. Start it first: bash ~/Project/trichord/scripts/serving/serve.sh kimi_k3 4

Docker build: same container as run_poster_generation_v1.py (no extra packages needed).

Run CMD:

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=poster_gen_v1p4 \
    --nodes=2 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v1p4.%j.o -e $HOME/log/slurm/poster_gen_v1p4.%j.e \
    --wrap="for f in /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f (bash ~/Project/trichord/scripts/serving/serve.sh status)\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS \
        bash script/isolated_tmp.sh python3 script/run_poster_generation_v1p3.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v1p4_kimik3_cosmos3_kimik3 \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()  # before run_poster_generation imports build any Agent

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v1p3 import poster_generation_v1p3  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v1p3)
