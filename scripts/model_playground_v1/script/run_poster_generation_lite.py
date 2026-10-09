"""
Lite poster generation: the agent thinks, writes one image prompt, makes exactly one generate_image call, and stops.

Run CMD (same docker/mounts as run_poster_generation.py, ONE CONTAINER PER TASK, no torchrun):

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-long --job-name=poster_generation_lite_v0 \
    --nodes=1 --exclusive --ntasks-per-node=32 --cpus-per-task=2 --time=7-00:00:00 \
    -o $HOME/log/slurm/poster_generation_lite_v0.%j.o -e $HOME/log/slurm/poster_generation_lite_v0.%j.e \
    --wrap="srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$HOME/Project/trichord/scripts/model_playground_v1 \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID \
        python3 script/run_poster_generation_lite.py \
            --model kimi-k3@nvidiak \
            --output s3://nv-00-10206-vfm/debug/xingqianx/agentic_data/poster_generation_lite_gpt2p5sunburst_v0 \
            --aspect_ratio random --reasoning_effort random --max_samples_per_process 625"

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

from tool.poster_generation_lite import DEFAULT_MODEL_NAME, poster_generation_lite  # noqa: E402
from taxonomy.helper import TOPIC_MESSAGE_TEMPLATE, Taxonomy  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE  # noqa: E402
from loguru import logger  # noqa: E402
from utils import REASONING_EFFORT_LEVELS, random_reasoning_effort  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
RANDOM = "random"
MODEL_DEFAULT = "default"
PROGRESS_EVERY = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt_preset", default="poster_generation_lite_v0")
    parser.add_argument("--taxonomy", default="poster_topics_v1")
    parser.add_argument("--aspect_ratio", default=RANDOM, choices=[RANDOM, *ASPECT_RATIO_TO_SIZE])
    parser.add_argument("--reasoning_effort", default=MODEL_DEFAULT, choices=[MODEL_DEFAULT, RANDOM, *REASONING_EFFORT_LEVELS])
    parser.add_argument("--max_samples_per_process", type=int, default=0, help="max samples generated per process; 0 or less loops until the process is stopped")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if RANK != 0:
        logger.remove()
        logger.add(sys.stderr, filter=lambda record: record["extra"].get("progress", False))
    system_prompt = (PROMPT_DIR / f"{args.prompt_preset}.md").read_text().strip()
    taxonomy = Taxonomy(args.taxonomy)
    total = args.max_samples_per_process if args.max_samples_per_process > 0 else "inf"

    sample = 0
    generated = 0
    while args.max_samples_per_process <= 0 or sample < args.max_samples_per_process:
        topic = taxonomy.random_get_one_topic()
        aspect_ratio = random.choice(list(ASPECT_RATIO_TO_SIZE)) if args.aspect_ratio == RANDOM else args.aspect_ratio
        if args.reasoning_effort == MODEL_DEFAULT:
            reasoning_effort = None
        elif args.reasoning_effort == RANDOM:
            reasoning_effort = random_reasoning_effort(args.model)
        else:
            reasoning_effort = args.reasoning_effort

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        agent = poster_generation_lite(model_name=args.model, system_prompt=system_prompt, reasoning_effort=reasoning_effort)

        started = time.time()
        try:
            result = agent.run({"prompt": TOPIC_MESSAGE_TEMPLATE.format(topic=topic), "aspect_ratio": aspect_ratio})
            if not any(record["files"] for instance in agent.tool_instances.values() for record in instance.calls):
                raise RuntimeError(f"no image produced: {result}")
            agent.save_history(os.path.join(args.output, run_id))
            logger.info(f"rank {RANK}/{WORLD_SIZE} sample {sample} done in {time.time() - started:.1f}s | {aspect_ratio} | {topic}")
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
