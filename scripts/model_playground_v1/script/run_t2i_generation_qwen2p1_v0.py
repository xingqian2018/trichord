r"""
t2i Qwen-Image-2.1 = run_t2i_generation.py with tool/t2i_generation_qwen2p1.py and the weighted arena taxonomy (taxonomy/t2i_arena_v0.yaml).
The rewriter is its own tool: Kimi-K3 writes a base prompt, calls upsample_prompt (Qwen-Image-2.1-PE-T2I), may fix the
expansion, and calls generate_image (Qwen-Image-2.1) once with a description and an aspect ratio; no review, no second
render. System prompt: prompt/t2i_generation_qwen2p1_v0.md. Each sample keeps the full trace plus derived/image.webp and
derived/meta.json = {taxonomy, prompt, upsampled_prompt, render_prompt}.

Start the services first (bash, on the cluster): kimi_k3_gb200 (or kimi_k3), qwen_image_2p1_pe_t2i, qwen_image_2p1 via
    bash ~/Project/trichord/scripts/serving/serve.sh <service> <replicas>

Run CMD (same as: env NODES=<n> NTASKS=<k> bash script/run_t2i_generation_qwen2p1_v0.sh on the cluster, n CPU nodes x k processes (default 1 x 32)):

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_qwen_image_2p1_v0 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/t2i_qwen_image_2p1_v0.%j.o -e $HOME/log/slurm/t2i_qwen_image_2p1_v0.%j.e \
    --wrap="for f in /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qpe_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qi_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f (bash ~/Project/trichord/scripts/serving/serve.sh status)\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1 QPE_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qpe_serve/broker_url)/v1 QI_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qi_serve/broker_url)/v1 K3_NATIVE_TOOLS=1 PYTHONNOUSERSITE=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,QPE_ENDPOINTS,QI_ENDPOINTS,K3_NATIVE_TOOLS,PYTHONNOUSERSITE \
        python3 script/run_t2i_generation_qwen2p1_v0.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_qwen_image_2p1_kimik3_pe_v0 \
            --reasoning_effort random --max_samples_per_process 2000"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_t2i_generation import main  # noqa: E402
from tool.t2i_generation_qwen2p1 import t2i_generation_qwen2p1  # noqa: E402

if __name__ == "__main__":
    main(t2i_generation_qwen2p1, default_taxonomy="t2i_arena_v0")
