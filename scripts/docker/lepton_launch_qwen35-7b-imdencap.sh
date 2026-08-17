pip install tiktoken
vllm serve /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
    --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --mm-processor-cache-gb 0 \
    --mm-processor-kwargs '{"max_pixels": 16777216}' \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}' \
    --max-model-len 32768 \
    --served-model-name Lepton/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5

pip install tiktoken
vllm serve /workspace/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V1-Lr5em6-Convert \
    --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --mm-processor-cache-gb 0 \
    --mm-processor-kwargs '{"max_pixels": 16777216}' \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}' \
    --max-model-len 32768 \
    --served-model-name Lepton/Qwen3.5-27B-Image-Dense-Captioner-V1-Lr5em6


PYTHONPATH=/workspace/endpoints_code python3 /workspace/endpoints_code/projects/cosmos3/vlm/scripts/endpoints/convert_checkpoint.py \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/qwen35_red_coyo_paxfixv1_combined_image_27b_32n_bs128_1epoch_warmup100_lr1e5_densepromptv1/attempts/q35-red-coyo-paxfixv1-27b-32n-bs128-1epoch-densepromptv1_20260725084527/safetensors/step_2552/ \
    --model Qwen/Qwen3.5-27B \
    --s3_credential /workspace/endpoints_code/credentials/gcp_training.secret \
    --output_dir /workspace/cache/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V0-Lr1em5-Convert \
    --skip_if_exists \
    --remove_temp_dir 1 \
    --temp_prefix /tmp/checkpoints \
    --tokenizer_credential /workspace/endpoints_code/credentials/gcp_training.secret \
    --tokenizer_bucket nv-00-10206-checkpoint-experiments


PYTHONPATH=/workspace/endpoints_code python3 /workspace/endpoints_code/projects/cosmos3/vlm/scripts/endpoints/convert_checkpoint.py \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos_reason2/qwen35_red_coyo_paxfixv1_combined_image_27b_32n_bs128_1epoch_warmup100_lr5e6_densepromptv1/attempts/q35-red-coyo-paxfixv1-27b-32n-bs128-1epoch-lr5e6-densepromptv1_20260725084600/safetensors/step_2552/ \
    --model Qwen/Qwen3.5-27B \
    --s3_credential /workspace/endpoints_code/credentials/gcp_training.secret \
    --output_dir /workspace/cache/user/xingqianx/.cache/customized_models/Qwen3.5-27B-Image-Dense-Captioner-V1-Lr5em6-Convert \
    --skip_if_exists \
    --remove_temp_dir 1 \
    --temp_prefix /tmp/checkpoints \
    --tokenizer_credential /workspace/endpoints_code/credentials/gcp_training.secret \
    --tokenizer_bucket nv-00-10206-checkpoint-experiments