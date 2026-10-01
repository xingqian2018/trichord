"""
Run CMD (same docker/mounts as slaunch, but plain srun: ONE CONTAINER PER TASK, no torchrun):
    export DOCKER_PATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.0.0.sqsh
    export CONTAINER_WORKDIR=$HOME/Project/trichord/scripts/model_playground
    lustrepath=/lustre/fsw/portfolios/cosmos
    srun --account=cosmos_base_training --partition=<cpu partition> --nodes=1 --ntasks=32 --cpus-per-task=2 \\
         --kill-on-bad-exit=0 \\
         --container-image=${DOCKER_PATH} \\
         --container-mounts $lustrepath:$lustrepath:rw,$HOME:$HOME:rw \\
         --container-workdir="$CONTAINER_WORKDIR" \\
         --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID \\
         .venv/bin/python script/run_poster_generation.py \\
             --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_kimi_nb2_kimi_v0 \\
             --aspect_ratio random

    Each srun task is its own container, so /tmp/poster_agent is private per agent process.
    Do NOT add --container-name (tasks would share one container) and do NOT wrap in torchrun.

One-time setup inside the container (the local .venv symlinks the host python and cannot be reused in the image):
    srun ... --ntasks=1 --container-image=${DOCKER_PATH} --container-mounts ... --container-workdir="$CONTAINER_WORKDIR" \\
         bash -c 'python3 -m venv .venv_container && .venv_container/bin/pip install -r requirements.txt'
    then launch with .venv_container/bin/python instead of .venv/bin/python

Check once before the real run:
    srun ... --ntasks=2 --container-image=${DOCKER_PATH} bash -c 'which google-chrome chromium; touch /tmp/probe_$SLURM_PROCID; ls /tmp/probe_*'
    each task must report a chrome binary and list only its own probe file
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.poster_generation import poster_generation  # noqa: E402
from taxonomy.helper import TOPIC_MESSAGE_TEMPLATE, Taxonomy  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE  # noqa: E402
from loguru import logger  # noqa: E402
from utils import REASONING_EFFORT_LEVELS, random_reasoning_effort  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
RANDOM = "random"
MODEL_DEFAULT = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--prompt_preset", default="agentic_poster_layout_t2i_required_lite")
    parser.add_argument("--aspect_ratio", default=RANDOM, choices=[RANDOM, *ASPECT_RATIO_TO_SIZE])
    parser.add_argument("--reasoning_effort", default=MODEL_DEFAULT, choices=[MODEL_DEFAULT, RANDOM, *REASONING_EFFORT_LEVELS])
    parser.add_argument("--max_samples_per_process", type=int, default=0, help="max samples generated per process; 0 or less loops until the process is stopped")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_prompt = (PROMPT_DIR / f"{args.prompt_preset}.md").read_text().strip()
    taxonomy = Taxonomy()

    sample = 0
    while args.max_samples_per_process <= 0 or sample < args.max_samples_per_process:
        topic = taxonomy.random_get_one_topic()
        aspect_ratio = random.choice(list(ASPECT_RATIO_TO_SIZE)) if args.aspect_ratio == RANDOM else args.aspect_ratio
        if args.reasoning_effort == MODEL_DEFAULT:
            reasoning_effort = None
        elif args.reasoning_effort == RANDOM:
            reasoning_effort = random_reasoning_effort(args.model)
        else:
            reasoning_effort = args.reasoning_effort

        agent_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        agent = poster_generation(agent_name, model_name=args.model, system_prompt=system_prompt, reasoning_effort=reasoning_effort)

        started = time.time()
        try:
            agent.run({"prompt": TOPIC_MESSAGE_TEMPLATE.format(topic=topic), "aspect_ratio": aspect_ratio})
            agent.save_history(args.output)
            logger.info(f"rank {RANK}/{WORLD_SIZE} sample {sample} done in {time.time() - started:.1f}s | {aspect_ratio} | {topic}")
        except Exception as e:
            logger.warning(f"rank {RANK}/{WORLD_SIZE} sample {sample} agent {agent_name} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        del agent
        sample += 1


if __name__ == "__main__":
    main()
