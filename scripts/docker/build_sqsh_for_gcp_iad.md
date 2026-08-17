# Building a sqsh File for gcp-iad-gb200

## Background

`gcp-iad-gb200` is an **ARM64 cluster**. `--use-enroot-cache` fails there because compute
nodes get a 401 from nvcr.io/Docker Hub during `enroot import` at job startup. The workaround
is to pre-build a `.sqsh` file locally and pass it via `--sqsh-file` instead.

---

## One-time Setup: enable buildx multiplatform (if not already done)

```bash
docker buildx create --name multiplatform --driver docker-container --use
docker buildx inspect --bootstrap
```

---

## Step 1 — Build the ARM64 Docker image locally

Run from the repo root (`/home/xingqianx/Project/imaginaire4_sila`).
Takes ~20–40 min the first time (QEMU emulation).

```bash
cd /home/xingqianx/Project/imaginaire4_sila

docker buildx build \
  --platform linux/arm64 \
  --load \
  -t qwen3p5_vl:arm64 \
  -f pipelines/models/vlm/qwen3p5_vl.dockerfile \
  .
```

```bash
cd /home/xingqianx/Project/imaginaire4_sila

docker buildx build \
  --platform linux/arm64 \
  --load \
  -t qwen3p5_vl:arm64 \
  -f /home/xingqianx/Project/trichord/scripts/docker/qwen3p5_vl_vllm_arm64.dockerfile \
  .
```



---

## Step 2 — Export to sqsh locally

```bash
docker create --name tmp_export --platform linux/arm64 qwen3p5_vl:arm64
docker export tmp_export | mksquashfs - /home/xingqianx/Project/qwen3p5_vl_vllm.sqsh -tar -noI -noX -noF -noappend
docker rm tmp_export
```

---

## Step 3 — Copy sqsh to Lustre

```bash
scp /home/xingqianx/Project/qwen3p5_vl_vllm.sqsh \
  gcp-iad-cs-001-login-002.nvidia.com:/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh
```

---

## Step 4 — Launch with --sqsh-file

```bash
cd /home/xingqianx/Project/imaginaire4_sila

uv run yotta launch \
  --stream-logs \
  --replicas=1 \
  --mode=slurm-ray \
  --cluster=gcp-iad \
  --partition=batch_long \
  --wckey=p0 \
  --team=cosmos_base_training \
  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/qwen3p5_vl_vllm.sqsh \
  --base-conda-env=no_conda \
  --artifacts-storage-location=pbss \
  --job-name="image-caption-v2-test" \
  -- python -m pipelines.sila.image.captioning_v2.image_qwen3p5vl_captioning_pipeline \
  --dataset gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260508.lance \
  --pipeline-version mx_tier1 \
  --endpoint-port 8067 \
  --no-db-coordinator
```


  --sqsh-file=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/qwen3p5_vl_vllm_arm64.sqsh \

---

## Notes

- `mksquashfs` must be installed locally: `sudo apt install squashfs-tools`
- The sqsh lives at a **stable Lustre path** — the whole team can reuse it once built.
- If the base image changes significantly, rebuild from Step 1.
- The sqsh is ~10–15 GB on Lustre. Verify with `ls -lh` on the login node after Step 3.
