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

####

The model is at `gcs:nv-00-10206-dir/yotta/model_weights/image_captioner/image-qwen3-vl-8b-lora-v3.2-merged/`