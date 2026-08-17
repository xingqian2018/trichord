import boto3
from botocore.config import Config
from pathlib import Path

LOCALS = [
    "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted",
    "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted",
]
LOCAL = LOCALS[1]
DEST = "s3://yotta/model_weights/image_captioner/"

client = boto3.client(
    "s3",
    endpoint_url="https://pbss.s8k.io",
    aws_access_key_id="team-dir",
    aws_secret_access_key="86fe4e205c3511e4c9f2da3f070a0ecb",
    region_name="us-west-1",
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

_, dest_path = DEST.split("://", 1)
bucket, prefix = dest_path.split("/", 1)

def remote_size(key):
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except client.exceptions.ClientError:
        return None

local_dir = Path(LOCAL)
files = [f for f in local_dir.rglob("*") if f.is_file()]
print(f"Found {len(files)} files...")

for i, f in enumerate(files):
    key = prefix + str(f.relative_to(local_dir.parent))
    if remote_size(key) == f.stat().st_size:
        print(f"[{i+1}/{len(files)}] SKIP {key}")
        continue
    client.upload_file(str(f), bucket, key)
    print(f"[{i+1}/{len(files)}] UP   {key}")

print("Done.")
