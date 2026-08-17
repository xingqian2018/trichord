# the cmd set up the multiplatform dokcer?

```bash
docker buildx create --name multiplatform --driver docker-container --use --bootstrap
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx inspect --bootstrap
```

docker login nvcr.io -u '$oauthtoken' -p 'asdfsadf'
format_type = ascii
org = 0970776711373753


```
uv run yotta cache rm
```


```
uv run yotta launch \
  --stream-logs \
  --use-enroot-cache \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=1 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --lepton-queue-priority=high \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-test" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3vl_captioning_pipeline_full \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260508.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator
```


```
uv run yotta launch \
  --stream-logs \
  --use-enroot-cache \
  --mode=slurm-ray \
  --cluster=iad \
  --partition=pool0_cosmos \
  --wckey=p2 \
  --team=cosmos_base_training \
  --replicas=1 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-test" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260504.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator
```


```
# Need to install enroot locally
sudo apt-get install -y /tmp/enroot_3.5.0-1_amd64.deb /tmp/enroot+caps_3.5.0-1_amd64.deb
sudo chmod 644 enroot_3.5.0-1_amd64.deb enroot+caps_3.5.0-1_amd64.deb
sudo apt-get install -y ./enroot_3.5.0-1_amd64.deb ./enroot+caps_3.5.0-1_amd64.deb
```

uv run yotta launch \
  --stream-logs \
  --replicas=1 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --replicas=1 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-test" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260508.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator


uv run yotta launch \
  --stream-logs \
  --replicas=1 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-prod-datacomb_1b@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/coyo_700m_slice_from_maintable_20260731.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator

# check status
uv run yotta slurm job log --cluster gcp-iad 1754469 -f


# Launch on gcp-iad for coyo_700m
```
uv run yotta launch \
  --replicas=64 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-prod-coyo_700m@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/coyo_700m_slice_from_maintable_20260731.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator
```

# Launch on gcp-iad for nvcommercial_700m
```
uv run yotta launch \
  --replicas=64 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-prod-nvcommercial_700m@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/nvcommercial_700m_slice_from_maintable_20260731.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator
```

## Test on red
```
uv run yotta launch \
  --stream-logs \
  --replicas=1 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-red-test@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260731.lance \
  --pipeline-version mengyaox_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator \
  --filter-passed-only
  --source-dataset coyo_700m \
  --source-dataset nvcommercial_700m \
  --source-dataset MMC4
```


# Log at
/lustre/fsw/portfolios/cosmos/users/xingqianx/logs/image-caption-v2-prod-coyo_700m

uv run yotta slurm job log --cluster gcp-iad <JOB_ID> -f

tail -n 10000000 image-caption-v2-prod-coyo_700m@p0-1754469.log | grep -v -E "Downloaded|processing" > status.log

# Launch on azure for debug
```
uv run yotta launch \
  --stream-logs \
  --use-enroot-cache \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=1 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --lepton-queue-priority=high \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="imcaptionv2-azure-test" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260731.lance \
  --pipeline-version mengyaox_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator \
  --filter-passed-only
```

### Newest debug on gcp-iad

```
uv run yotta launch \
  --stream-logs \
  --replicas=4 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-debug@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/coyo_700m_slice_from_maintable_20260731.lance \
  --pipeline-version hamid_snah_capbalance_nocot_dense \
  --endpoint-port 8067 \
  --filter-passed-only
```