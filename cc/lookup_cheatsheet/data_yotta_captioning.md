uv run yotta launch \
  --use-enroot-cache \
  --enroot-cache-always-pull \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=2 \
  --num-to-launch=1 \
  --dockerfile=pipelines/models/vlm/qwen3_vl_captioning.dockerfile \
  --base-conda-env=no_conda \
  --job-name=image-caption \
  --artifacts-storage-location=gcs \
  -- \
  python -m pipelines.sila.image.captioning.image_qwen3vl_captioning_pipeline_full \
  --dataset gs://nv-00-10206-lancedb/prod/image/text_related/screen2words_rico_slice_from_maintable_0429.lance/ \
  --pipeline-version cosmos_captioner_image_v1_full

vllm serve /tmp/local_model_weights/image_captioner/image-qwen3-vl-8b-lora-v3.2-merged \
    --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size <N_GPUS> \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --uvicorn-log-level warning \
    --disable-uvicorn-access-log

The model is at `gcs:nv-00-10206-dir/yotta/model_weights/image_captioner/image-qwen3-vl-8b-lora-v3.2-merged/`

## Captioner V2

### Download checkpoint to lepton

```PYTHON
import boto3
from botocore.client import Config
import os

# INPUT = "s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/cosmos_rl/qwen35_dense_27b_image_gpt55_n499989_1epoch_32n_20260812024401/safetensors/step_3906/"
# OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only/"

INPUT = "s3://nv-00-10206-vfm/debug/xingqianx/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted/"
OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted/"

# INPUT = "s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/cosmos_rl/qwen35_dense_27b_image_redcoyo_gpt55_n826622_1epoch_32n_20260812024418/safetensors/step_6457/"
# OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed/"

# INPUT = "s3://nv-00-10206-vfm/debug/xingqianx/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted/"
# OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted/"

# INPUT = "s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/cosmos_rl/qwen35_dense_27b_image_redcoyo_gpt55_all15_1to1_n1653242_1epoch_32n_20260812024436/safetensors/step_XXXX/"
# OUTPUT = "/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V4-HamidSnah-MxTier1-ReasonerData/"

s3 = boto3.client(
    "s3",
    endpoint_url="https://storage.googleapis.com",
    aws_access_key_id="<look it up>",
    aws_secret_access_key="<look it up>",
    region_name="us-east4",
    config=Config(
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

_, path = INPUT.split("://", 1)
bucket, prefix = path.split("/", 1)

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel = key[len(prefix):]
        local_path = os.path.join(OUTPUT, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {key} -> {local_path}")
        s3.download_file(bucket, key, local_path)
```

### Write checkpoint setting to leave a record

```
cat > Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed_setting.txt << 'EOF'
s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/cosmos_rl/qwen35_dense_27b_image_redcoyo_gpt55_n826622_1epoch_32n_20260812024418/safetensors/step_6457/
EOF
```


### Download and upload HF checkpoint

```PYTHON
import boto3, json, os
from botocore.client import Config

cred = json.load(open("/workspace/user/xingqianx/Project/imaginaire4/credentials/gcs.secret"))
s3 = boto3.client("s3", config=Config(s3={"addressing_style": "path"},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required"), **cred)

local_dir = "/workspace/user/xingqianx/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B"
bucket = "nv-00-10206-vfm"
prefix = "debug/xingqianx/huggingface/hub/models--Qwen--Qwen3.5-27B"

for root, _, files in os.walk(local_dir):
    for fname in files:
        local_path = os.path.join(root, fname)
        key = f"{prefix}/{os.path.relpath(local_path, local_dir)}"
        print(f"{local_path} -> s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
```



### Convert Qwen35-27B checkpoint

`/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only`
`/workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed/`

```
PYTHONPATH=/workspace/endpoints_code python3 projects/cosmos3/vlm/scripts/endpoints/convert_checkpoint.py \
  --checkpoint_path /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only \
  --model Qwen/Qwen3.5-27B \
  --output_dir /workspace/model \
  --skip_if_exists  \
  --remove_temp_dir 1 \
  --temp_prefix /tmp/checkpoints \
  --tokenizer_credential "$TOKENIZER_CRED_FILE" \
  --tokenizer_bucket nv-00-10206-checkpoint-experiments
```

```
cd /workspace/endpoints_code
PYTHONPATH=/workspace/endpoints_code python3 projects/cosmos3/vlm/scripts/endpoints/convert_checkpoint.py \
  --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/cosmos_rl/qwen35_dense_27b_image_redcoyo_gpt55_n826622_1epoch_32n_20260812024418/safetensors/step_6457/ \
  --model Qwen/Qwen3.5-27B \
  --output_dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --s3_credential /workspace/user/xingqianx/Project/imaginaire4/credentials/gcs.secret \
  --tokenizer_credential /workspace/user/xingqianx/Project/imaginaire4/credentials/gcs.secret \
  --tokenizer_bucket nv-00-10206-checkpoint
```

`New script run`

----------------

```LEPTON
.venv/bin/python /workspace/user/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only \
  --model_path      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted


python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted \
  --expected native

python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py prepare \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted \
  --summary-path /tmp/v3_prepare_summary.json

python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted \
  --expected vllm
```

----------------

```LEPTON
.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V2-HamidSnah-Only-Converted \
  --model_path      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted


python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --expected native

python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py prepare \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --summary-path /tmp/v3_prepare_summary.json

python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --expected vllm

```

```LOCAL
.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed \
  --model_path      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted

.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --expected native

.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter_v2.py prepare \
  --model-dir /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --summary-path /tmp/v3_prepare_summary.json

.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter_v2.py inspect \
  --model-dir /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted \
  --expected vllm
```

----------------

```
python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed \
  --model_path      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted
```

```
.venv/bin/python /home/xingqianx/Project/trichord/scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed \
  --model_path      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /home/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V3-HamidSnah-MxTier1-Mixed-Converted
```

```
python3 /workspace/user/xingqianx/vllm_scripts/vllm_checkpoint_converter.py \
  --checkpoint_path /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V4-HamidSnah-MxTier1-ReasonerData \
  --model_path      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
  --output_dir      /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V4-HamidSnah-MxTier1-ReasonerData-Converted
```



/workspace/user/xingqianx/vllm_scripts
