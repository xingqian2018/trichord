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

## Debug on the caption v2 recover

- Debug table is `gs://nv-00-10206-vfm/lancedb/image/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_20260820.lance/`

`On azure`

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
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline_recover \
  --dataset gs://nv-00-10206-vfm/lancedb/image/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_20260820.lance/ \
  --pipeline-version hamid_snah_capbalance_nocot_dense_recover \
  --endpoint-port 8067 \
  --no-db-coordinator \
  --filter-passed-only
```

`On gcp-iad`

```
uv run yotta launch \
  --stream-logs \
  --replicas=2 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-debug@p0" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline_recover \
  --dataset gs://nv-00-10206-vfm/lancedb/image/synthetic_text/synthetic_scene_text_v0_slice_from_maintable_20260820.lance/ \
  --fragment-ids 5-7,10 \
  --pipeline-version hamid_snah_capbalance_nocot_dense_recover \
  --recover-from-pipeline-version hamid_snah_capbalance_nocot_dense \
  --endpoint-port 8067 \
```

`On azure official for filtered`

```
uv run yotta launch \
  --use-enroot-cache \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=4 \
  --num-to-launch 8 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --lepton-queue-priority=very-high \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="imcaptionv2-37173-40560-azure" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance/ \
  --fragment-ids 37173-40560 \
  --pipeline-version hamid_snah_capbalance_nocot_dense \
  --filter-passed-only \
  --max-fragments 10000 \
  --endpoint-port 8067
```


`On azure official for recover`

```
uv run yotta launch \
  --use-enroot-cache \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=4 \
  --num-to-launch 16 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --lepton-queue-priority=very-high \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="imcaptionv2-recover-azure" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline_recover \
  --dataset gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance/ \
  --fragment-ids 5073-5102,5194-5232,36952-36992,37046-37073,37165-37167 \
  --pipeline-version hamid_snah_capbalance_nocot_dense_recover \
  --recover-from-pipeline-version hamid_snah_capbalance_nocot_dense \
  --endpoint-port 8067
```

`On gcp-iad official`

```
uv run yotta launch \
  --stream-logs \
  --replicas=2 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="imcaptionv2-recover-gcpiad" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline_recover \
  --dataset gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance/ \
  --fragment-ids 5073-5102,5194-5232,36952-36992,37046-37073,37165-37167 \
  --pipeline-version hamid_snah_capbalance_nocot_dense_recover \
  --recover-from-pipeline-version hamid_snah_capbalance_nocot_dense \
  --endpoint-port 8067
```

`On azure official for recover midtrain`

- Targets the "Dataset we don't know" + "Dataset new!" datasets from `reports/image_data_stats/image_data_stats_20260915_full.md` — all currently 0% captioned. Fragment IDs looked up from `reports/image_data_stats/lancedb_fragid_mapping_20260915.md`:
  - `megalith-10m` — 37008-37044
  - `human_sft` — 37045
  - `pexels_residual_trustedK1_v2` — 37159
  - `LSDIR` — 37168
  - `Aesthetic-Train-V2` — 37169-37170
  - `unsplash_lite` — 37171
  - `photo-concept-bucket` — 40561-40566
  - `multiaspect-4k-1m` — 40567-40577
  - `flickr2k` — 40578

```
uv run yotta launch \
  --use-enroot-cache \
  --mode=lepton-ray \
  --cluster=azure \
  --replicas=4 \
  --num-to-launch 4 \
  --dockerfile=pipelines/models/vlm/qwen3p5_vl.dockerfile \
  --lepton-queue-priority=very-high \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="imcapv2rec-midt-azure" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline_recover \
  --dataset gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance/ \
  --fragment-ids 37008-37044,37045,37159,37168,37169-37170,37171,40561-40566,40567-40577,40578 \
  --pipeline-version hamid_snah_capbalance_nocot_dense_recover \
  --recover-from-pipeline-version hamid_snah_capbalance_nocot_dense \
  --endpoint-port 8067
```

