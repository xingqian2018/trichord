# How to Make a .sqsh File from a Dockerfile

General cheatsheet for turning any Dockerfile into an enroot `.sqsh` image for Slurm
clusters (see also [build_sqsh_for_gcp_iad.md](build_sqsh_for_gcp_iad.md) for the
`gcp-iad-gb200` ARM64-specific workflow).

# Background

Slurm clusters run containers via [NVIDIA enroot](https://github.com/NVIDIA/enroot).
Docker images get implicitly converted to `.sqsh` at job start time, but pre-building
the `.sqsh` ahead of time makes jobs (especially large-scale ones) start faster, and
is required on clusters where compute nodes can't reach the registry directly
(e.g. `gcp-iad-gb200`, see the other doc).

There are two ways to produce the `.sqsh`, pick based on what's available:

- **`enroot import`** — simplest, needs `enroot` installed locally (or on the build node).
- **`docker export` + `mksquashfs`** — works with just Docker + `squashfs-tools`, no enroot needed.

--------------------------------------------

# HPSV3

## ARM64 BUILD

```bash
docker buildx build \
  --platform linux/arm64 \
  --load \
  -t hpsv3:arm64 \
  -f /home/xingqianx/Project/imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/hpsv3.dockerfile \
  .
docker create --name tmp_export_arm64 --platform linux/arm64 hpsv3:arm64
docker export tmp_export_arm64 | mksquashfs - /home/xingqianx/hpsv3_arm64.sqsh -tar -noI -noX -noF -noappend
docker rm tmp_export_arm64

s3a hpsv3_arm64.sqsh gcs:nv-00-10206-vfm/debug/xingqianx/docker/hpsv3_arm64.sqsh ul

```

## x86_64 BUILD

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t hpsv3:amd64 \
  -f /home/xingqianx/Project/imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/hpsv3.dockerfile \
  .

docker create --name tmp_export_amd64 --platform linux/amd64 hpsv3:amd64
docker export tmp_export_amd64 | mksquashfs - /home/xingqianx/hpsv3_amd64.sqsh -tar -noI -noX -noF -noappend
docker rm tmp_export_amd64

s3a hpsv3_arm64.sqsh gcs:nv-00-10206-vfm/debug/xingqianx/docker/hpsv3_amd64.sqsh ul
s3a gcs:nv-00-10206-vfm/debug/xingqianx/docker/hpsv3_amd64.sqsh ~/docker/ dl

docker tag hpsv3:amd64 nvcr.io/0970776711373753/xingqian_rm_hpsv3:amd64
docker push nvcr.io/0970776711373753/xingqian_rm_hpsv3:amd64
```


## x86_64 Test

```bash
docker images

docker run -it --rm --gpus all --platform linux/amd64 -v $HOME:$HOME -e HOME=$HOME -w $HOME hpsv3:amd64 bash

cd Project/imaginaire4_alt/
export HF_HOME=$HOME/.cache/huggingface
PYTHONPATH=. python \
    -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=24173 \
    projects/cosmos3/cosmos3/reward_service/main.py \
        --scorer_type HPSv3 \
        --redis_host localhost \
        --redis_port 6379 \
        --app_port 8080
```

--------------------------------------------

# PICKSCORE

## ARM64 BUILD

```bash
docker buildx build \
  --platform linux/arm64 \
  --load \
  -t pickscore:arm64 \
  -f /home/xingqianx/Project/imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/pickscore.dockerfile \
  .

docker create --name tmp_export_arm64 --platform linux/arm64 pickscore:arm64
docker export tmp_export_arm64 | mksquashfs - /home/xingqianx/pickscore_arm64.sqsh -tar -noI -noX -noF -noappend
docker rm tmp_export_arm64

s3a pickscore_arm64.sqsh gcs:nv-00-10206-vfm/debug/xingqianx/docker/pickscore_arm64.sqsh ul

```

## x86_64 BUILD

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t pickscore:amd64 \
  -f /home/xingqianx/Project/imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/pickscore.dockerfile \
  .

docker create --name tmp_export_amd64 --platform linux/amd64 pickscore:amd64
docker export tmp_export_amd64 | mksquashfs - /home/xingqianx/pickscore_amd64.sqsh -tar -noI -noX -noF -noappend
docker rm tmp_export_amd64

s3a pickscore_amd64.sqsh gcs:nv-00-10206-vfm/debug/xingqianx/docker/pickscore_amd64.sqsh ul
s3a gcs:nv-00-10206-vfm/debug/xingqianx/docker/pickscore_amd64.sqsh ~/docker/ dl
```


## x86_64 Test

```bash
docker images

docker run -it --rm --gpus all --platform linux/amd64 -v $HOME:$HOME -e HOME=$HOME -w $HOME pickscore:amd64 bash --noprofile --norc

cd Project/imaginaire4_alt/
export HF_HOME=$HOME/.cache/huggingface
PYTHONPATH=. python \
    -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=24173 \
    projects/cosmos3/cosmos3/reward_service/main.py \
        --scorer_type PickScore \
        --redis_host localhost \
        --redis_port 6379 \
        --app_port 8080
```

--------------------------------------------


