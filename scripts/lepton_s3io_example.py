Not runnable

##################
# upload to pbss #
##################

import boto3
from botocore.config import Config
from pathlib import Path

LOCAL_DIR = Path("/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert")
BUCKET = "yotta"
PREFIX = "model_weights/image_captioner/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert"

client = boto3.client(
    "s3",
    endpoint_url="https://pbss.s8k.io",
    aws_access_key_id="team-dir",
    aws_secret_access_key="---lookup-please---",
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

def remote_size(bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except client.exceptions.ClientError:
        return None

files = [f for f in LOCAL_DIR.rglob("*") if f.is_file()]
print(f"Found {len(files)} files...")

for i, f in enumerate(files):
    key = f"{PREFIX}/{f.relative_to(LOCAL_DIR)}"
    local_size = f.stat().st_size
    if remote_size(BUCKET, key) == local_size:
        print(f"[{i+1}/{len(files)}] SKIP {key}")
        continue
    client.upload_file(str(f), BUCKET, key)
    print(f"[{i+1}/{len(files)}] UP   {key}")

print("Done.")


#################
# upload to gcs #
#################


import boto3
from botocore.config import Config
from pathlib import Path

LOCAL_DIR = Path("/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert")
BUCKET = "nv-00-10206-checkpoint"
PREFIX = "cosmos_captioner/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert"

client = boto3.client(
    "s3",
    endpoint_url="https://storage.googleapis.com",
    aws_access_key_id="GOOG1EODU5KLGJKSRSZN6HTMMCLGRJQXKAFOLSJJCGGBRPCUWJDOEM25F5OQ4",
    aws_secret_access_key="xToiogNjSzj4pPrfVliWIJ0T/nwu8E36ll6iSISY",
    region_name="us-east4",
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

def remote_size(bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except client.exceptions.ClientError:
        return None

files = [f for f in LOCAL_DIR.rglob("*") if f.is_file()]
print(f"Found {len(files)} files...")

for i, f in enumerate(files):
    key = f"{PREFIX}/{f.relative_to(LOCAL_DIR)}"
    local_size = f.stat().st_size
    if remote_size(BUCKET, key) == local_size:
        print(f"[{i+1}/{len(files)}] SKIP {key}")
        continue
    client.upload_file(str(f), BUCKET, key)
    print(f"[{i+1}/{len(files)}] UP   {key}")

print("Done.")