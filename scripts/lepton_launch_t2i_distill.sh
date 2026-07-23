set -euo pipefail

# ---------------------------------------------------------------------------
# Step 1: Write GCP credentials from the Lepton secret to disk
# ---------------------------------------------------------------------------
mkdir -p /opt/imaginaire4/credentials

python -c '
import base64
import json
import os
import pathlib

env_name = "COSMOS_GCP_CHECKPOINT_CREDS"
output_paths = [
    pathlib.Path("/opt/imaginaire4/credentials/gcp_checkpoint.secret"),
    pathlib.Path("/opt/imaginaire4/credentials/gcp_training.secret"),
]

raw = os.environ.get(env_name, "")
if not raw:
    raise SystemExit(f"ERROR: {env_name} Lepton secret is empty or not mounted")

source_label = "env"
if raw.startswith("/") and len(raw) < 4096:
    source = pathlib.Path(raw)
    if source.is_file():
        raw = source.read_text()
        source_label = "file"

candidates = [(source_label, raw)]
try:
    decoded = base64.b64decode("".join(raw.split()), validate=True).decode()
    candidates.append(("base64", decoded))
except Exception:
    pass

errors = []
for label, text in candidates:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, str):
            loaded = json.loads(loaded)
        if not isinstance(loaded, dict):
            raise ValueError("credential JSON must be an object")
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        continue
    payload = json.dumps(loaded)
    for output_path in output_paths:
        output_path.write_text(payload)
        output_path.chmod(0o600)
    print(f"[credential] wrote valid JSON from {label} secret to {len(output_paths)} credential files")
    break
else:
    details = "; ".join(errors)
    raise SystemExit(f"ERROR: {env_name} did not contain JSON credentials ({details})")
' || exit 2

cd /opt/imaginaire4 || exit 2

# ---------------------------------------------------------------------------
# Step 2: Prefetch checkpoint from S3 to local cache (parallel, resumable)
# ---------------------------------------------------------------------------
# CKPT="cosmos3_vfm/t2w_mot_32b_sft_runs/t2w_mot_sft_exp305_000_qwen3_vl_32b_480p_alldatasets_32n_basefps16_sdg1_img/checkpoints/iter_000010000/model/"
CKPT="cosmos3_interactive/base_distill_32b_xx/base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999/checkpoints/iter_000006000/model/"
S3_CKPT="s3://nv-00-10206-checkpoint-experiments/$CKPT"
CACHE_DIR="/workspace/common/cosmos3/endpoint"
LOCAL_CKPT="$CACHE_DIR/checkpoints/$CKPT"

S3_CKPT="$S3_CKPT" LOCAL_CKPT="$LOCAL_CKPT" \
CRED=/opt/imaginaire4/credentials/gcp_checkpoint.secret \
python - <<'PY' || exit 2
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import boto3
from botocore.config import Config

s3_uri, dest, cred = os.environ["S3_CKPT"], Path(os.environ["LOCAL_CKPT"]), os.environ["CRED"]
WORKERS = 16
with open(cred) as f:
    creds = json.load(f)
s3 = boto3.client("s3", **creds, config=Config(
    signature_version="s3v4", s3={"addressing_style": "virtual"},
    max_pool_connections=WORKERS * 4, retries={"max_attempts": 10, "mode": "adaptive"}))

bucket, _, prefix = s3_uri[5:].partition("/")
dest.mkdir(parents=True, exist_ok=True)

todo, total = [], 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for o in page.get("Contents", ()):
        if o["Key"].endswith("/"):
            continue
        local = dest / o["Key"][len(prefix):].lstrip("/")
        if local.exists() and local.stat().st_size == o["Size"]:
            continue
        todo.append((o["Key"], o["Size"], local)); total += o["Size"]

print(f"[prefetch] {len(todo)} files ({total/1e9:.2f} GB) -> {dest}", flush=True)
if not todo:
    sys.exit(0)

def fetch(key, size, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    s3.download_file(bucket, key, str(tmp))
    os.replace(tmp, local)
    return size

t0, done_b, done_n = time.time(), 0, 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for fut in as_completed(pool.submit(fetch, *args) for args in todo):
        done_b += fut.result(); done_n += 1
        dt = time.time() - t0
        print(f"[prefetch] {done_n}/{len(todo)}  {done_b/1e9:5.2f}/{total/1e9:.2f} GB  "
              f"{done_b/1e6/max(dt,1e-3):6.0f} MB/s", flush=True)
print(f"[prefetch] done in {time.time()-t0:.1f}s", flush=True)
PY

# ---------------------------------------------------------------------------
# Step 3: Launch the inference server (use local checkpoint mirror)
# ---------------------------------------------------------------------------
# Loading model from /workspace/common/cosmos3/endpoint/checkpoints/cosmos3_vfm/t2w_mot_32b_sft_runs/t2w_mot_sft_exp305_000_qwen3_vl_32b_480p_alldatasets_32n_basefps16_sdg1_img/checkpoints/iter_000010000/model
export COSMOS_INTERNAL=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export COSMOS3_T2I_COORD_TIMEOUT_SECONDS=2592000
S3_OUTPUT_DIR=s3://nv-00-10206-vfm/cosmos3_image_sft/api_t2i_generations_distill/
PYTHONPATH="${PYTHONPATH:-}:packages/cosmos3" \
torchrun --nproc-per-node=2 -m \
    projects.cosmos3.cosmos3.evaluation.text_to_image.text2image_server \
    --checkpoint "$S3_CKPT" \
    --cache-dir "$CACHE_DIR" \
    --config-file projects/cosmos3/interactive/configs/config.py \
    --experiment base_distill_dmd2_ga_pt_32b_t2i_ablationV2 \
    --s3-credential-path /opt/imaginaire4/credentials/gcp_checkpoint.secret \
    --s3-output-dir $S3_OUTPUT_DIR \
    --host 0.0.0.0 \
    --port 8001 \
    --cfgp-size 1 \
    --cp-size 2 \
    --prompt-upsampler-endpoint-url https://inference-api.nvidia.com/v1/chat/completions \
    --prompt-upsampler-model aws/anthropic/bedrock-claude-opus-4-7 \
    --prompt-upsampler-api-key-env NVGATEWAY_API_KEY
