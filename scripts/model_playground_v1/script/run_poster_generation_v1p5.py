r"""
v1p5 = v1p3/v1p4 with generate_image on our own Cosmos3-Super-Text2Image replicas instead of Lepton
(tool/generate_image_cosmos3_slurm.py, served by scripts/serving/serve_cosmos3_super_t2i.sbatch, job names c3-xqx*).
Same model, same 1024-tier sizes, guardrails still off; the model still sees the tool as generate_image.
Lepton is never called. Requests carry the bearer key in ~/.config/cosmos3_serving/api_key. With no replica
available the tool waits (re-reading the list every 30s, up to its 1800s timeout) instead of failing the sample.
Kimi-K3 (agent + critic) is unchanged from v1p4: our own k3-xqx* replicas.

Start the replicas first and wait until both kinds show a URL (bash, on gb300):
    bash ~/Project/trichord/scripts/serving/serve.sh kimi_k3 2
    bash ~/Project/trichord/scripts/serving/serve.sh cosmos3_super_t2i 2
    bash ~/Project/trichord/scripts/serving/serve.sh status

Docker build: same container as run_poster_generation_v1.py (no extra packages needed).

Run CMD:

mkdir -p $HOME/log/slurm $HOME/log/k3_endpoints $HOME/log/c3_endpoints
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=poster_gen_v1p5 \
    --nodes=2 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/poster_gen_v1p5.%j.o -e $HOME/log/slurm/poster_gen_v1p5.%j.e \
    --wrap="export K3_ENDPOINTS_FILE=$HOME/log/k3_endpoints/\$SLURM_JOB_ID.json C3_ENDPOINTS_FILE=$HOME/log/c3_endpoints/\$SLURM_JOB_ID.json \
            C3_API_KEY_FILE=$HOME/.config/cosmos3_serving/api_key; \
        python3 $playground/k3_endpoints.py --out \$K3_ENDPOINTS_FILE --user xingqianx --job-prefix k3-xqx || exit 1; \
        python3 $playground/k3_endpoints.py --out \$C3_ENDPOINTS_FILE --user xingqianx --job-prefix c3-xqx \
            --model LeptonTR/Cosmos3-Super-Text2Image --api-key-file \$C3_API_KEY_FILE || exit 1; \
        python3 $playground/k3_endpoints.py --out \$K3_ENDPOINTS_FILE --user xingqianx --job-prefix k3-xqx --every 300 > $HOME/log/k3_endpoints/\$SLURM_JOB_ID.log 2>&1 & \
        python3 $playground/k3_endpoints.py --out \$C3_ENDPOINTS_FILE --user xingqianx --job-prefix c3-xqx \
            --model LeptonTR/Cosmos3-Super-Text2Image --api-key-file \$C3_API_KEY_FILE --every 300 > $HOME/log/c3_endpoints/\$SLURM_JOB_ID.log 2>&1 & \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS_FILE,C3_ENDPOINTS_FILE,C3_API_KEY_FILE \
        bash script/isolated_tmp.sh python3 script/run_poster_generation_v1p5.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_v1p5_kimik3_cosmos3slurm_kimik3 \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625; \
        kill %1 %2"

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
