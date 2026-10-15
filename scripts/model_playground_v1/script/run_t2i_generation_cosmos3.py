r"""
t2i Cosmos3: same loop as run_t2i_generation.py, with tool/t2i_generation_cosmos3.py and taxonomy/t2i_arena_v0.yaml.
Kimi-K3 turns the sampled topic into a Cosmos3 structured JSON prompt (the official Cosmos3-Super-Text2Image T2I
template, prompt/t2i_generation_cosmos3_v0.md) and calls generate_image once; the prompt goes to Cosmos3 as is, with
no rewriter (official settings: 50 steps, guidance 4.0, flow shift 3.0, guardrails off, 768-tier sizes). Each sample
keeps the full trace plus derived/image.png and derived/meta.json = {taxonomy, prompt}.

Start the services first (bash, on the cluster): kimi_k3_gb200 (or kimi_k3) and cosmos3_super_t2i via
    bash ~/Project/trichord/scripts/serving/serve.sh <service> <replicas>

Run CMD:

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_cosmos3_v0 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/t2i_cosmos3_v0.%j.o -e $HOME/log/slurm/t2i_cosmos3_v0.%j.e \
    --wrap="for f in /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f (bash ~/Project/trichord/scripts/serving/serve.sh status)\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1 C3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url)/v1 K3_NATIVE_TOOLS=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,C3_ENDPOINTS,K3_NATIVE_TOOLS \
        python3 script/run_t2i_generation_cosmos3.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_cosmos3_kimik3_structured_v0 \
            --reasoning_effort random --max_samples_per_process 2000"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_t2i_generation import main  # noqa: E402
from tool.t2i_generation_cosmos3 import t2i_generation_cosmos3  # noqa: E402

if __name__ == "__main__":
    main(t2i_generation_cosmos3, default_taxonomy="t2i_arena_v0")
