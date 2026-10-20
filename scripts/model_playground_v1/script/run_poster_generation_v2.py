r"""
v2 = v1p5 with Kimi-K3 prompt upsampling in front of Cosmos3 (tool/poster_generation_v2.py). Like the Qwen-Image-2.1 t2i
pipeline, upsampling is its own tool: for each visual the poster agent calls upsample_prompt with a short request (a
separate Kimi-K3 conversation with the official Cosmos3 upsampling rules, tool/upsample_prompt_cosmos3.py, that returns the
template-valid structured JSON), fixes only real errors, and calls generate_image with that JSON and an output_path; the
image is rendered on our Cosmos3 replicas with the official settings (50 steps, guidance 4.0, flow shift 3.0, guardrails
off) and saved as a PNG. render_poster and poster_visual_critic are unchanged from v1p5.
System prompt: prompt/poster_generation_agentic_layout_t2i_required_v2.md (v1p5's, with the two-step image section).
K3_NATIVE_TOOLS=1: vLLM's kimi_k3 tool parser and tool_choice="required" for the upsampler (and the poster agent).
The upsampler conversation of each call is saved under tool/upsample_prompt/call/<id>/upsampler/.

Start the services first (bash, on the cluster):
    bash ~/Project/trichord/scripts/serving/serve.sh kimi_k3_gb200 <replicas>      (gcpcode; kimi_k3 on gb300)
    bash ~/Project/trichord/scripts/serving/serve.sh cosmos3_super_t2i <replicas>
    bash ~/Project/trichord/scripts/serving/serve.sh status

Run CMD (same as: env NODES=<n> NTASKS=<k> bash script/run_poster_generation_v2.sh on the cluster):

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=poster_gen_v2 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=1 --time=2-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v2.%j.o -e $HOME/log/slurm/poster_gen_v2.%j.e \
    --wrap="export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1 C3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url)/v1 K3_NATIVE_TOOLS=1 PYTHONNOUSERSITE=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,C3_ENDPOINTS,K3_NATIVE_TOOLS,PYTHONNOUSERSITE \
        bash script/isolated_tmp.sh python3 script/run_poster_generation_v2.py \
            --model kimi-k3@k3slurm \
            --prompt_preset poster_generation_agentic_layout_t2i_required_v2 \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v2_kimik3_cosmos3upsampled_kimik3 \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625"

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3_fallback  # noqa: E402

gateway_kimik3_fallback.install()

from run_poster_generation import main  # noqa: E402
from tool.poster_generation_v2 import poster_generation_v2  # noqa: E402

if __name__ == "__main__":
    main(poster_generation_v2)
