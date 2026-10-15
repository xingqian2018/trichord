r"""
v1p5 = v1p3/v1p4 with generate_image on our own Cosmos3-Super-Text2Image replicas instead of Lepton
(tool/generate_image_cosmos3_slurm.py, served by scripts/serving/serve_cosmos3_super_t2i.sbatch, job names c3-xqx*).
Same model, same 1024-tier sizes, guardrails still off; the model still sees the tool as generate_image.
Lepton is never called. Requests carry the bearer key SLURM_COSMOS3_API_KEY from credentials/gateway.json. With no replica
available the tool waits (re-reading the list every 30s, up to its 1800s timeout) instead of failing the sample.
Kimi-K3 (agent + critic) is unchanged from v1p4: our own k3-xqx* replicas.

Kimi-K3 and Cosmos3 are reached through their service dispatchers (scripts/serving/pool_lib.sh): one URL per service,
the replicas pull requests from it, so replicas can come and go without any endpoint list.
Start the services first (bash, on gb300); each is one Slurm job with a dispatcher and N replicas behind one URL:
    bash ~/Project/trichord/scripts/serving/serve.sh kimi_k3 4
    bash ~/Project/trichord/scripts/serving/serve.sh cosmos3_super_t2i 4
    bash ~/Project/trichord/scripts/serving/serve.sh status

Docker build: same container as run_poster_generation_v1.py (no extra packages needed).

Run CMD:

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=poster_gen_v1p5 \
    --nodes=2 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v1p5.%j.o -e $HOME/log/slurm/poster_gen_v1p5.%j.e \
    --wrap="for f in /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f (bash ~/Project/trichord/scripts/serving/serve.sh status)\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1 C3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url)/v1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,C3_ENDPOINTS \
        bash script/isolated_tmp.sh python3 script/run_poster_generation_v1p5.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v1p5_kimik3_cosmos3_kimik3 \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()  # before run_poster_generation imports build any Agent

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v1p5 import poster_generation_v1p5  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v1p5)
