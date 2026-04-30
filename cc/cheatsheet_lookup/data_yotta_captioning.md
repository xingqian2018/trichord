uv run yotta launch \
  --job-name=image-caption \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=8 \
  --num-to-launch=2 \
  --use-enroot-cache \
  --dockerfile=pipelines/models/vlm/qwen3_vl.dockerfile \
  --base-conda-env=no_conda \
  --artifacts-storage-location=gcs \
  -- \
  python -m pipelines.sila.image.captioning.image_qwen3vl_captioning_pipeline_full \
  --dataset gs://nv-00-10206-lancedb/prod/image/text_related/screen2words_rico_slice_from_maintable_20260427.lance/ \
  --pipeline-version cosmos_captioner_image_v1_full
