r"""
Qwen-Image-2.1 text-to-image data: for each sample a taxonomy topic (taxonomy/t2i_topics_v0.yaml) goes to Kimi-K3,
which writes a short base prompt and makes exactly one generate_image call (tool/t2i_generation.py). Inside the tool,
Qwen-Image-2.1-PE-T2I rewrites it into the full prompt and picks the aspect ratio, and Qwen-Image-2.1 renders it
(40 steps, CFG off, official 2048-tier sizes). Each sample <output>/<run_id>/ keeps the full trace (setting.json,
chat.json, tool/generate_image/call/<id>/) plus derived/image.png and derived/meta.json = {taxonomy, prompt, upsampled_prompt}. All three models are our own Slurm services, each one URL (a
dispatcher with replicas behind it, scripts/serving/pool_lib.sh). Keys are SLURM_KIMI_K3_API_KEY,
SLURM_QWEN_IMAGE_PE_API_KEY and SLURM_QWEN_IMAGE_API_KEY in credentials/gateway.json. K3_NATIVE_TOOLS=1: Kimi runs vLLM's kimi_k3 tool parser, so the agent's
tool_choice="required" forces the generate_image call at decode time (see gateway_kimik3.py).

Start the services first (bash, on gb300); each is one Slurm job with a dispatcher and N replicas behind one URL:
    bash ~/Project/trichord/scripts/serving/serve.sh kimi_k3 4
    bash ~/Project/trichord/scripts/serving/serve.sh qwen_image_2p1_pe_t2i 1
    bash ~/Project/trichord/scripts/serving/serve.sh qwen_image_2p1 2
    bash ~/Project/trichord/scripts/serving/serve.sh status

Docker build: same container as run_poster_generation_v1.py (no extra packages needed).

Run CMD:

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_qwen_image_2p1_v0 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=1-00:00:00 \
    -o $HOME/log/slurm/t2i_qwen_image_2p1_v0.%j.o -e $HOME/log/slurm/t2i_qwen_image_2p1_v0.%j.e \
    --wrap="for f in /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qpe_serve/broker_url /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qi_serve/broker_url; do test -s \$f || { echo \"no dispatcher: \$f (bash ~/Project/trichord/scripts/serving/serve.sh status)\"; exit 1; }; done; \
        export K3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/k3_serve/broker_url)/v1 QPE_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qpe_serve/broker_url)/v1 QI_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/qi_serve/broker_url)/v1 K3_NATIVE_TOOLS=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,K3_ENDPOINTS,QPE_ENDPOINTS,QI_ENDPOINTS,K3_NATIVE_TOOLS \
        python3 script/run_t2i_generation.py \
            --model kimi-k3@k3slurm \
            --output s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_qwen_image_2p1_kimik3_pe_v0 \
            --reasoning_effort random --max_samples_per_process 2000"

"""

import argparse
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_kimik3  # noqa: E402

gateway_kimik3.install()  # before any Agent is built

from loguru import logger  # noqa: E402

from taxonomy.helper import Taxonomy  # noqa: E402
from tool.t2i_generation import DEFAULT_MODEL_NAME, t2i_generation  # noqa: E402
from utils import REASONING_EFFORT_LEVELS, random_reasoning_effort  # noqa: E402

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
RANDOM = "random"
MODEL_DEFAULT = "default"
PROGRESS_EVERY = 10
TOPIC_MESSAGE_TEMPLATE = "Topic:\n{topic}"


def parse_args(default_taxonomy: str = "t2i_topics_v0") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--taxonomy", default=default_taxonomy)
    parser.add_argument("--reasoning_effort", default=MODEL_DEFAULT, choices=[MODEL_DEFAULT, RANDOM, *REASONING_EFFORT_LEVELS])
    parser.add_argument("--max_samples_per_process", type=int, default=0, help="max samples generated per process; 0 or less loops until the process is stopped")
    return parser.parse_args()


def main(agent_cls: type = t2i_generation, default_taxonomy: str = "t2i_topics_v0") -> None:
    args = parse_args(default_taxonomy)
    if RANK != 0:
        logger.remove()
        logger.add(sys.stderr, filter=lambda record: record["extra"].get("progress", False))
    taxonomy = Taxonomy(args.taxonomy)
    total = args.max_samples_per_process if args.max_samples_per_process > 0 else "inf"

    sample = 0
    generated = 0
    while args.max_samples_per_process <= 0 or sample < args.max_samples_per_process:
        topic = taxonomy.random_get_one_topic()
        if args.reasoning_effort == MODEL_DEFAULT:
            reasoning_effort = None
        elif args.reasoning_effort == RANDOM:
            reasoning_effort = random_reasoning_effort(args.model)
        else:
            reasoning_effort = args.reasoning_effort

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        agent = agent_cls(model_name=args.model, reasoning_effort=reasoning_effort)

        started = time.time()
        try:
            agent.run({"prompt": TOPIC_MESSAGE_TEMPLATE.format(topic=topic), "taxonomy": topic})
            agent.save_history(os.path.join(args.output, run_id))
            output = agent.image_tool().output
            logger.info(f"rank {RANK}/{WORLD_SIZE} sample {sample} done in {time.time() - started:.1f}s | {output.get('wh_ratio') or output.get('aspect_ratio')} | {topic}")
            generated += 1
        except Exception as e:
            logger.warning(f"rank {RANK}/{WORLD_SIZE} sample {sample} run {run_id} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1)
        del agent
        sample += 1
        if sample % PROGRESS_EVERY == 0:
            logger.bind(progress=True).info(f"rank {RANK}/{WORLD_SIZE} generated {generated}/{total} ({sample - generated} failed)")


if __name__ == "__main__":
    main()
