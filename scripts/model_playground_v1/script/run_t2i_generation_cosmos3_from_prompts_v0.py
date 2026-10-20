r"""
t2i Cosmos3 from existing structured prompts, no Kimi: every row of a prompt jsonl (one Cosmos3 structured JSON per row, e.g.
the cosmos_captioner "full" captions of the agent_distilled datasets in image_meta_table_full.lance) is rendered once by
tool/generate_image_cosmos3_with_strict_prompt_format_slurm.py with the official settings (50 steps, guidance 4.0, flow shift 3.0, guardrails
off, 768-tier sizes), at the template aspect ratio nearest to the source image.
Input rows: {"uuid", "source_dataset", "width", "height", "prompt": {<structured JSON>}}. A missing subject_details becomes {},
keys outside the template are dropped, and rows missing any other template key are skipped.
Each sample is <output>/<uuid>/ with setting.json, tool/generate_image/call/<id>/, derived/image.webp and
derived/meta.json = {uuid, source_dataset, source_size, aspect_ratio, upsampled_prompt}; a uuid whose meta.json exists is
skipped, so a relaunch resumes. Rank r of n takes rows r, r + n, r + 2n, ...

Start the service first (bash, on the cluster): bash ~/Project/trichord/scripts/serving/serve.sh cosmos3_super_t2i <replicas>

Run CMD (same as: env NODES=<n> NTASKS=<k> bash script/run_t2i_generation_cosmos3_from_prompts_v0.sh on the cluster):

mkdir -p $HOME/log/slurm
lustrepath=/lustre/fsw/portfolios/cosmos
playground=$HOME/Project/trichord/scripts/model_playground_v1
sbatch --account=cosmos_base_training --partition=cpu --qos=cpu-normal --job-name=t2i_cosmos3_agent_distilled_prompt_v0 \
    --nodes=2 --exclusive --ntasks-per-node=32 --cpus-per-task=1 --time=2-00:00:00 \
    -o $HOME/log/slurm/t2i_cosmos3_agent_distilled_prompt_v0.%j.o -e $HOME/log/slurm/t2i_cosmos3_agent_distilled_prompt_v0.%j.e \
    --wrap="export C3_ENDPOINTS=\$(cat /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/broker_url)/v1 PYTHONNOUSERSITE=1; \
        srun --kill-on-bad-exit=0 \
        --container-image=$lustrepath/users/xingqianx/Container/run_poster_generation.sqsh \
        --container-mounts=$lustrepath:$lustrepath:rw,$HOME:$HOME:rw \
        --container-workdir=$playground \
        --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID,C3_ENDPOINTS,PYTHONNOUSERSITE \
        python3 script/run_t2i_generation_cosmos3_from_prompts_v0.py \
            --prompts $HOME/data/t2i_data/agent_distilled_cosmos_captioner_full_prompts.jsonl \
            --output s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_cosmos3_agent_distilled_prompt_v0"
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from tool.generate_image_cosmos3_with_strict_prompt_format_slurm import ASPECT_RATIO_TO_SIZE, TEMPLATE_KEYS, generate_image_cosmos3_with_strict_prompt_format_slurm  # noqa: E402
from utils import S3_CREDENTIAL_FILE, put  # noqa: E402

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
PROGRESS_EVERY = 10
OPTIONAL_KEYS = {"subject_details": {}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, help="jsonl of {uuid, source_dataset, width, height, prompt}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples_per_process", type=int, default=0, help="0 or less renders every row of this rank")
    return parser.parse_args()


def make_exists(output: str):
    if urlparse(output).scheme != "s3":
        return lambda url: Path(url).is_file()
    import boto3
    from botocore.config import Config

    cred = json.loads(S3_CREDENTIAL_FILE.read_text())
    client = boto3.client("s3", endpoint_url=cred["endpoint_url"], region_name=cred["region_name"],
                          aws_access_key_id=cred["aws_access_key_id"], aws_secret_access_key=cred["aws_secret_access_key"],
                          config=Config(s3={"addressing_style": "path"}))

    def exists(url: str) -> bool:
        parsed = urlparse(url)
        try:
            client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            return True
        except client.exceptions.ClientError:
            return False

    return exists


def nearest_ratio(width: int, height: int) -> str:
    target = math.log(width / height)
    return min(ASPECT_RATIO_TO_SIZE, key=lambda r: abs(math.log(ASPECT_RATIO_TO_SIZE[r][0] / ASPECT_RATIO_TO_SIZE[r][1]) - target))


def normalize(prompt: dict) -> dict | None:
    prompt = {**OPTIONAL_KEYS, **{k: v for k, v in prompt.items() if k in TEMPLATE_KEYS}}
    if any(k not in prompt for k in TEMPLATE_KEYS):
        return None
    return {k: prompt[k] for k in TEMPLATE_KEYS}


def my_rows(path: str):
    with open(path) as f:
        for index, line in enumerate(f):
            if index % WORLD_SIZE == RANK and line.strip():
                yield json.loads(line)


def render(row: dict, prompt: dict, root: str) -> str:
    tool = generate_image_cosmos3_with_strict_prompt_format_slurm()
    ratio = nearest_ratio(row["width"], row["height"])
    result = tool.run({"request": row["uuid"], "prompt": prompt, "aspect_ratio": ratio}, tool_call_id="render")
    if tool.output is None:
        raise RuntimeError(result)
    image = tool.output
    tool.save_history(os.path.join(root, "tool", tool.function_name))
    meta = {"uuid": row["uuid"], "source_dataset": row["source_dataset"], "source_size": [row["width"], row["height"]],
            "aspect_ratio": ratio, "upsampled_prompt": image["prompt"]}
    put(json.dumps(tool.setting(), ensure_ascii=False, indent=4).encode("utf-8"), os.path.join(root, "setting.json"))
    put(image["image"], os.path.join(root, "derived", "image." + image["format"]))
    put(json.dumps(meta, ensure_ascii=False, indent=4).encode("utf-8"), os.path.join(root, "derived", "meta.json"))
    return ratio


def main() -> None:
    args = parse_args()
    if RANK != 0:
        logger.remove()
        logger.add(sys.stderr, filter=lambda record: record["extra"].get("progress", False))
    exists = make_exists(args.output)
    sample = generated = skipped = invalid = 0
    for row in my_rows(args.prompts):
        if 0 < args.max_samples_per_process <= sample:
            break
        root = os.path.join(args.output, row["uuid"])
        prompt = normalize(row["prompt"])
        if prompt is None:
            invalid += 1
            continue
        if exists(os.path.join(root, "derived", "meta.json")):
            skipped += 1
            continue
        started = time.time()
        try:
            ratio = render(row, prompt, root)
            logger.info(f"rank {RANK}/{WORLD_SIZE} sample {sample} done in {time.time() - started:.1f}s | {ratio} | {row['uuid']}")
            generated += 1
        except Exception as e:
            logger.warning(f"rank {RANK}/{WORLD_SIZE} sample {sample} uuid {row['uuid']} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1)
        sample += 1
        if sample % PROGRESS_EVERY == 0:
            logger.bind(progress=True).info(f"rank {RANK}/{WORLD_SIZE} generated {generated}/{sample} ({sample - generated} failed, {skipped} done before, {invalid} invalid)")
    logger.bind(progress=True).info(f"rank {RANK}/{WORLD_SIZE} finished: generated {generated}/{sample} ({sample - generated} failed, {skipped} done before, {invalid} invalid)")


if __name__ == "__main__":
    main()
