#!/bin/bash
# Cosmos3-Super-Text2Image vLLM-Omni endpoint on oci-jhb (gb300): 1 GB300 node, CFG2 x Ulysses2, HSDP shard 4.
# Same flags as the Lepton deployment (LeptonTR/Cosmos3-Super-Text2Image), so clients keep the same model name.
# Weights: nvidia/Cosmos3-Super-Text2Image (public, 132.5 GB) in the team-shared HF hub, downloaded on first run.
# Image: docker://vllm/vllm-omni:cosmos3-arm64 imported once with
#   enroot import -o $U/containers/vllm-omni+cosmos3-arm64.sqsh docker://vllm/vllm-omni:cosmos3-arm64
# Port = 20000 + jobid % 20000; endpoint.env is written once /v1/models answers.
# API key: every /v1 request needs "Authorization: Bearer <key>"; the key is read from $C3_API_KEY_FILE
# (default ~/.config/cosmos3_serving/api_key, mode 600), the same file the poster pipeline reads.
# Submit (bash): mkdir -p /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve
#                sbatch -J c3-xqx-1 scripts/serving/serve_cosmos3_super_t2i.sbatch
#SBATCH -A cosmos_base_training
#SBATCH -p batch_long
#SBATCH --qos normal
#SBATCH --segment=1
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J c3-xqx
#SBATCH -o /scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/c3_serve/slurm-%j.log
set -euo pipefail
umask 022

U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
IMAGE=$U/containers/vllm-omni+cosmos3-arm64.sqsh
export HF_HUB_CACHE=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub
export C3_REPO=nvidia/Cosmos3-Super-Text2Image C3_REVISION=daf3d374804be4c512c2135568a7cb95d4341d79
export C3_MODEL=$HF_HUB_CACHE/models--nvidia--Cosmos3-Super-Text2Image/snapshots/$C3_REVISION
STATE=$U/run/c3_serve/job-$SLURM_JOB_ID
mkdir -p "$STATE"
test -s "$IMAGE"
export VLLM_API_KEY=$(cat "${C3_API_KEY_FILE:-$HOME/.config/cosmos3_serving/api_key}")
[[ -n "$VLLM_API_KEY" ]]

export C3_PORT=$((20000 + SLURM_JOB_ID % 20000))
export C3_HOST=$(hostname)
export OMP_NUM_THREADS=8
# node-local caches, same reason as the mimo endpoint (cache lock failures on shared storage)
export XDG_CACHE_HOME=/tmp/c3_cache_$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm HF_HOME=$XDG_CACHE_HOME/hf
printf 'image=%s\nmodel=%s\nnode=%s\nbase_url=http://%s:%s/v1\n' "$IMAGE" "$C3_MODEL" "$C3_HOST" "$C3_HOST" "$C3_PORT" > "$STATE/config.txt"

server_pid=""
cleanup() { code=$?; [[ -n "$server_pid" ]] && kill -TERM "$server_pid" 2>/dev/null || true; rm -f "$STATE/endpoint.env"; echo "exit_code=$code" > "$STATE/finished"; }
trap cleanup EXIT
trap 'exit 143' TERM INT

# first run downloads the weights into the shared hub (resumable); later runs only verify and start offline
srun --nodes=1 --ntasks=1 --container-image="$IMAGE" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --no-container-mount-home --container-workdir=/tmp --export=ALL \
     bash -c 'mkdir -p $XDG_CACHE_HOME; \
         [[ -s "$C3_MODEL/model_index.json" ]] || python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"C3_REPO\"], revision=os.environ[\"C3_REVISION\"], max_workers=16)"; \
         export HF_HUB_OFFLINE=1; \
         exec vllm serve "$C3_MODEL" --omni \
             --served-model-name LeptonTR/Cosmos3-Super-Text2Image \
             --host 0.0.0.0 --port "$C3_PORT" --api-key "$VLLM_API_KEY" \
             --cfg-parallel-size 2 --ulysses-degree 2 --tensor-parallel-size 1 \
             --use-hsdp --hsdp-shard-size 4 --init-timeout 1800' > "$STATE/server.log" 2>&1 &
server_pid=$!

for ((i = 0; i < 240; i++)); do   # up to 2 h: first run includes the 132.5 GB download
    kill -0 "$server_pid" 2>/dev/null || { wait "$server_pid"; exit 1; }
    if curl -fsS --max-time 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://$C3_HOST:$C3_PORT/v1/models" > "$STATE/models.json" 2>/dev/null \
        && grep -q 'Cosmos3-Super-Text2Image' "$STATE/models.json"; then
        printf 'job_id=%s\nmodel=LeptonTR/Cosmos3-Super-Text2Image\nbase_url=http://%s:%s/v1\n' "$SLURM_JOB_ID" "$C3_HOST" "$C3_PORT" > "$STATE/endpoint.env"
        echo "READY $(tr '\n' ' ' < "$STATE/endpoint.env")"
        break
    fi
    sleep 30
done
wait "$server_pid"
exit 1
