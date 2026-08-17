import boto3
from botocore.client import Config
import os

LOCALS = [
    "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted",
    "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted",
]
LOCAL = LOCALS[0]
DEST = "s3://nv-00-10206-vfm/debug/xingqianx/customized_models/"

s3 = boto3.client(
    "s3",
    endpoint_url="https://storage.googleapis.com",
    aws_access_key_id="GOOG1EODU5KLGJKSRSZN6HTMMCLGRJQXKAFOLSJJCGGBRPCUWJDOEM25F5OQ4",
    aws_secret_access_key="xToiogNjSzj4pPrfVliWIJ0T/nwu8E36ll6iSISY",
    region_name="us-east4",
    config=Config(
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

_, dest_path = DEST.split("://", 1)
bucket, prefix = dest_path.split("/", 1)

if os.path.isfile(LOCAL):
    key = prefix + os.path.basename(LOCAL) if prefix.endswith("/") else prefix
    print(f"Uploading {LOCAL} -> {bucket}/{key}")
    s3.upload_file(LOCAL, bucket, key)
    print("Done.")
else:
    base = os.path.basename(LOCAL.rstrip("/"))
    for root, _, files in os.walk(LOCAL):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, os.path.dirname(LOCAL.rstrip("/")))
            key = prefix + rel if prefix.endswith("/") else prefix + "/" + rel
            print(f"Uploading {local_path} -> {bucket}/{key}")
            s3.upload_file(local_path, bucket, key)
    print("Done.")
