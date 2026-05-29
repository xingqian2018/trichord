import argparse
import asyncio
import atexit
import json
import logging
import os
import os.path as osp
import tempfile
from typing import Optional
from urllib.parse import urlparse

from multistorageclient.contrib.async_fs import MultiStorageAsyncFileSystem
from tabulate import tabulate
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CRED_PATHS = {
    "gcs": osp.expanduser("~/Project/imaginaire4/credentials/gcs.secret"),
}


def setup_msc(cred_dict: dict, max_workers: int = 64):
    os.environ["MSC_MAX_WORKERS"] = str(max_workers)

    config_dict = {
        "retry": {
            "attempts": 8,
            "delay": 0.05,
            "backoff_multiplier": 2,
        }
    }

    def _append_config_with_s3_credential_path(msc_config_dict, s3_credential_path, profile):
        with open(s3_credential_path, "r") as f:
            authinfo = json.load(f)

        msc_config_dict["profiles"] = msc_config_dict.get("profiles", {})
        msc_config_dict["profiles"][profile] = msc_config_dict["profiles"].get(profile, {})

        storage_provider_type = "s3"
        parsed_endpoint_url = urlparse(authinfo["endpoint_url"])

        if parsed_endpoint_url.hostname.endswith(".s8k.io"):
            storage_provider_type = "s8k"
        elif parsed_endpoint_url.hostname.startswith("storage.") and parsed_endpoint_url.hostname.endswith(
            ".googleapis.com"
        ):
            storage_provider_type = "gcs_s3"

        msc_config_dict["profiles"][profile]["storage_provider"] = {
            "type": storage_provider_type,
            "options": {
                "base_path": "",
                "endpoint_url": authinfo["endpoint_url"],
                "region_name": authinfo["region_name"],
            },
        }
        msc_config_dict["profiles"][profile]["credentials_provider"] = {
            "type": "S3Credentials",
            "options": {
                "access_key": authinfo["aws_access_key_id"],
                "secret_key": authinfo["aws_secret_access_key"],
            },
        }

    for profile, cred_path in cred_dict.items():
        if cred_path is not None:
            _append_config_with_s3_credential_path(config_dict, cred_path, profile)

    shared_tmp_dir = osp.expanduser("~/tmp")
    os.makedirs(shared_tmp_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".secret", delete=False, dir=shared_tmp_dir)
    json.dump(config_dict, tmp, indent=2)
    tmp.flush()
    config_path = tmp.name
    atexit.register(lambda p=config_path: os.remove(p) if osp.exists(p) else None)
    os.environ["MSC_CONFIG"] = config_path
    logger.info(f"MSC config written to {config_path}")


async def list_all_files(fs, msc_prefix: str) -> list[str]:
    all_files = []
    stack = [msc_prefix.rstrip("/")]
    while stack:
        current = stack.pop()
        try:
            entries = await fs._ls(current, detail=True)
        except Exception as e:
            logger.warning(f"Failed to list {current}: {e}")
            continue
        for entry in entries:
            name = entry.get("name", "")
            entry_type = entry.get("type", "").lower()
            if entry_type in ("directory", "prefix"):
                stack.append(name.rstrip("/"))
            else:
                all_files.append(name)
    return all_files


async def msc_download_many(
    fs,
    remote_paths: list[str],
    local_paths: list[str],
    pbar_desc: Optional[str] = None,
) -> list[bool]:
    max_concurrency = int(os.getenv("MSC_MAX_WORKERS", "32"))
    sem = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(remote_paths), desc=pbar_desc, disable=pbar_desc is None)

    async def _one(remote: str, local: str) -> bool:
        async with sem:
            try:
                data = await fs._cat_file(remote)
                os.makedirs(osp.dirname(local), exist_ok=True)
                with open(local, "wb") as f:
                    f.write(data)
                return True
            except Exception as e:
                tqdm.write(f"[FAIL] {remote}: {e}")
                return False
            finally:
                pbar.update(1)

    results = await asyncio.gather(*[asyncio.create_task(_one(r, l)) for r, l in zip(remote_paths, local_paths)])
    pbar.close()
    return list(results)


_AUTODL_CATEGORY = {
    ("gcs", "nv-00-10206-checkpoint-experiments"): "checkpoints",
    ("gcs", "nv-00-10206-checkpoint"):             "checkpoints",
    ("s3-training", "checkpoints-us-east-1"):       "checkpoints",
    ("team-cosmos", "cosmos_generation"):           "data",
    ("gcs", "nv-00-10206-vfm"):                    "data",
    ("gcs", "nv-00-10206-images"):                 "data",
    ("gcs", "nv-00-10206-webdataset-images"):      "data",
}


def parse_input_path(input_path: str):
    if ":" not in input_path:
        raise ValueError(f"Expected 'profile:bucket/key/' format, got: {input_path}")
    profile, rest = input_path.split(":", 1)
    rest = rest.lstrip("/")
    bucket = rest.split("/")[0]
    prefix = rest[len(bucket):].lstrip("/")
    return profile, bucket, prefix


def resolve_autodl_output(profile: str, bucket: str, prefix: str) -> str:
    if prefix.startswith("cosmos3_vfm/evaluation"):
        category = "results"
    elif profile == "team-cosmos-benchmark":
        category = "benchmark"
    else:
        category = _AUTODL_CATEGORY.get((profile, bucket))
        if category is None:
            raise ValueError(f"Cannot auto-detect output category for {profile}:{bucket}. Pass --output explicitly.")
    local_root = osp.join(os.getenv("HOME"), ".cache", "imaginaire4")
    return osp.join(local_root, category, prefix.rstrip("/"))


async def run(input_path: str, output_dir: Optional[str], max_workers: int, cred_path: Optional[str]):
    profile, bucket, prefix = parse_input_path(input_path)
    remote_prefix = f"{bucket}/{prefix}"

    resolved_cred = cred_path or DEFAULT_CRED_PATHS.get(profile)
    if resolved_cred is None:
        raise ValueError(f"No credential path for profile '{profile}'. Pass --cred_path explicitly.")

    if output_dir is None:
        output_dir = resolve_autodl_output(profile, bucket, prefix)

    params_disp = tabulate([
        ("INPUT",        input_path),
        ("PROFILE",      profile),
        ("BUCKET",       bucket),
        ("PREFIX",       prefix),
        ("OUTPUT DIR",   output_dir),
        ("MAX WORKERS",  max_workers),
        ("CRED PATH",    resolved_cred),
    ], tablefmt="simple")
    logger.info(f"\nParams:\n{params_disp}")

    setup_msc({profile: resolved_cred}, max_workers=max_workers)

    fs = MultiStorageAsyncFileSystem()
    msc_prefix = f"{profile}/{remote_prefix}"

    logger.info(f"Listing files under {msc_prefix} ...")
    all_files = await list_all_files(fs, msc_prefix)
    logger.info(f"Found {len(all_files)} total files")

    todo_remote = []
    todo_local = []
    skip_count = 0
    for remote_path in sorted(all_files):
        relative = remote_path[len(msc_prefix):].lstrip("/")
        local_path = osp.join(output_dir, relative) if relative else osp.join(output_dir, osp.basename(remote_path))
        if osp.exists(local_path):
            skip_count += 1
        else:
            todo_remote.append(remote_path)
            todo_local.append(local_path)

    logger.info(f"{skip_count} already exist, {len(todo_remote)} to download")

    if not todo_remote:
        logger.info("Nothing to do.")
        return

    results = await msc_download_many(fs, todo_remote, todo_local, pbar_desc="Downloading")

    success = sum(results)
    fail = len(results) - success
    logger.info(f"Done. {success} saved, {fail} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast parallel checkpoint download via MSC")
    parser.add_argument("--input", required=True, help="Remote path in 'profile:bucket/key/' format, e.g. gcs:nv-00-10206-checkpoint-experiments/path/to/ckpt/")
    parser.add_argument("--output", default=None, help="Local output directory (omit to mirror autodl: ~/.cache/imaginaire4/{category}/{prefix})")
    parser.add_argument("--max_workers", type=int, default=64, help="MSC concurrency (default: 64)")
    parser.add_argument("--cred_path", default=None, help="Override credential JSON path (default: ~/Project/imaginaire4/credentials/gcs.secret for gcs)")
    args = parser.parse_args()

    asyncio.run(run(args.input, args.output, args.max_workers, args.cred_path))
