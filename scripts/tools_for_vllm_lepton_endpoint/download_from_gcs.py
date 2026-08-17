import boto3
from botocore.client import Config
import os

# INPUT = "s3://nv-00-10206-vfm/debug/xingqianx/scripts/vllm_checkpoint_converter_v2.py"
# OUTPUT = "/workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py"

INPUT = "s3://nv-00-10206-checkpoint/cosmos_captioner/qwen35_dense_27b_image_redcoyo_gpt55_all15_1to1_n1653242_1epoch_32n_20260812024436/safetensors/step_12915//"
OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V4-HamidSnah-MxTier1-VQAData-Mixed/"

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

_, path = INPUT.split("://", 1)
bucket, key = path.split("/", 1)

isfile = True
try:
    s3.head_object(Bucket=bucket, Key=key)
except Exception:
    isfile = False

if isfile:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    print(f"Downloading {key} -> {OUTPUT}")
    s3.download_file(bucket, key, OUTPUT)
else:
    prefix = key.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    os.makedirs(OUTPUT, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objkey = obj["Key"]
            relpath = objkey[len(prefix):]
            if not relpath:
                continue
            localpath = os.path.join(OUTPUT, relpath)
            os.makedirs(os.path.dirname(localpath), exist_ok=True)
            print(f"Downloading {objkey} -> {localpath}")
            s3.download_file(bucket, objkey, localpath)

print("Done.")
