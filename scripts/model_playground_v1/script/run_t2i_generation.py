r"""
Shared loop for the text-to-image data pipelines: sample a topic from the taxonomy, run the agent on it, save the trace.
Launch one of the pipelines, each with its own Run CMD in its header:
    script/run_t2i_generation_qwen2p1_v0.py   Kimi-K3 -> upsample_prompt (Qwen-Image-2.1-PE-T2I) -> Qwen-Image-2.1
    script/run_t2i_generation_cosmos3_v0.py   Kimi-K3 writes a Cosmos3 structured prompt -> Cosmos3-Super-Text2Image
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
from utils import REASONING_EFFORT_LEVELS, random_reasoning_effort  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@k3slurm"
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
RANDOM = "random"
MODEL_DEFAULT = "default"
PROGRESS_EVERY = 10
TOPIC_MESSAGE_TEMPLATE = "Topic:\n{topic}"


def parse_args(default_taxonomy: str, default_exclude: tuple[str, ...] = ()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--taxonomy", default=default_taxonomy)
    parser.add_argument("--exclude_taxonomy", nargs="*", default=list(default_exclude),
                        help="top-level categories (e.g. \"UI & Digital\") whose topics are never sampled; the rest are renormalized")
    parser.add_argument("--reasoning_effort", default=MODEL_DEFAULT, choices=[MODEL_DEFAULT, RANDOM, *REASONING_EFFORT_LEVELS])
    parser.add_argument("--max_samples_per_process", type=int, default=0, help="max samples generated per process; 0 or less loops until the process is stopped")
    return parser.parse_args()


def sampling_pool(taxonomy: Taxonomy, exclude: list[str]) -> tuple[list[str], list[float]]:
    weights = taxonomy.weights if taxonomy.weighted else [1.0] * len(taxonomy.topics)
    keep = [(t, w) for t, w in zip(taxonomy.topics, weights) if t.split(taxonomy.sep)[0] not in exclude]
    if not keep:
        raise SystemExit(f"--exclude_taxonomy {exclude} removes every topic of {taxonomy.path.name}")
    return [t for t, _ in keep], [w for _, w in keep]


def main(agent_cls: type, default_taxonomy: str, default_exclude: tuple[str, ...] = ()) -> None:
    args = parse_args(default_taxonomy, default_exclude)
    if RANK != 0:
        logger.remove()
        logger.add(sys.stderr, filter=lambda record: record["extra"].get("progress", False))
    taxonomy = Taxonomy(args.taxonomy)
    topics, weights = sampling_pool(taxonomy, args.exclude_taxonomy)
    all_weight = sum(taxonomy.weights) if taxonomy.weighted else float(len(taxonomy))
    logger.info(f"taxonomy {taxonomy.path.name}: {len(topics)}/{len(taxonomy)} topics after excluding {args.exclude_taxonomy} "
                f"({sum(weights) / all_weight:.1%} of the sampling weight kept)")
    total = args.max_samples_per_process if args.max_samples_per_process > 0 else "inf"

    sample = 0
    generated = 0
    while args.max_samples_per_process <= 0 or sample < args.max_samples_per_process:
        topic = random.choices(topics, weights=weights)[0]
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
    raise SystemExit("run_t2i_generation.py is the shared loop; launch script/run_t2i_generation_qwen2p1_v0.py (Qwen-Image-2.1) or script/run_t2i_generation_cosmos3_v0.py (Cosmos3)")
