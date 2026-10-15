#!/bin/bash
# One-line launcher for the self-hosted services on gb300 (run from any directory, in bash). Each service is ONE Slurm
# job: a dispatcher on its first node plus <replicas> model replicas that pull requests from it (serving/pool_lib.sh),
# so clients use one URL per service, like a Lepton endpoint.
#   bash scripts/serving/serve.sh kimi_k3 4                   # 4 Kimi-K3 replicas x 2 GB300 nodes = 8 nodes (gb300)
#   bash scripts/serving/serve.sh kimi_k3_gb200 4             # 4 Kimi-K3 replicas x 4 GB200 nodes = 16 nodes (gcpcode)
#   bash scripts/serving/serve.sh cosmos3_super_t2i 4         # 4 Cosmos3-Super-Text2Image replicas x 1 node
#   bash scripts/serving/serve.sh qwen_image_2p1 2            # 2 Qwen-Image-2.1 nodes, 4 single-GPU servers each
#   bash scripts/serving/serve.sh qwen_image_2p1_pe_t2i 1     # 1 Qwen-Image-2.1 prompt-rewriter node (DP4)
#   bash scripts/serving/serve.sh mimo_v2p6_pro_rl 1          # 1 MiMo-V2.6-Pro-RL node
#   bash scripts/serving/serve.sh cosmos3_super_t2i 1 batch   # batch partition, 4 h (quick test, usually starts sooner)
#   bash scripts/serving/serve.sh status                      # every service: job, URL, queue, live replicas
# To resize a service, cancel its job and submit again with a new replica count.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run

declare -A PREFIX=([kimi_k3]=k3-xqx [kimi_k3_gb200]=k3-xqx [mimo_v2p6_pro_rl]=mimo-xqx [cosmos3_super_t2i]=c3-xqx [qwen_image_2p1]=qi-xqx [qwen_image_2p1_pe_t2i]=qpe-xqx)
declare -A STATE=([kimi_k3]=k3_serve [kimi_k3_gb200]=k3_serve [mimo_v2p6_pro_rl]=mimo_pro_serve [cosmos3_super_t2i]=c3_serve [qwen_image_2p1]=qi_serve [qwen_image_2p1_pe_t2i]=qpe_serve)
declare -A KEY=([kimi_k3]=SLURM_KIMI_K3_API_KEY [kimi_k3_gb200]=SLURM_KIMI_K3_API_KEY [mimo_v2p6_pro_rl]=SLURM_MIMO_API_KEY [cosmos3_super_t2i]=SLURM_COSMOS3_API_KEY [qwen_image_2p1]=SLURM_QWEN_IMAGE_API_KEY [qwen_image_2p1_pe_t2i]=SLURM_QWEN_IMAGE_PE_API_KEY)
declare -A MODEL=([kimi_k3]=moonshotai/Kimi-K3 [kimi_k3_gb200]=moonshotai/Kimi-K3 [mimo_v2p6_pro_rl]=mimo-v2.6-pro-rl [cosmos3_super_t2i]=LeptonTR/Cosmos3-Super-Text2Image [qwen_image_2p1]=Qwen/Qwen-Image-2.1 [qwen_image_2p1_pe_t2i]=Qwen/Qwen-Image-2.1-PE-T2I)
declare -A NODES_PER_REPLICA=([kimi_k3]=2 [kimi_k3_gb200]=4 [mimo_v2p6_pro_rl]=1 [cosmos3_super_t2i]=1 [qwen_image_2p1]=1 [qwen_image_2p1_pe_t2i]=1)
SERVICES="kimi_k3 kimi_k3_gb200 mimo_v2p6_pro_rl cosmos3_super_t2i qwen_image_2p1 qwen_image_2p1_pe_t2i"

usage() { sed -n 2,14p "${BASH_SOURCE[0]}"; exit 1; }

status() {
    for svc in $SERVICES; do
        jobs=$(squeue -u "$USER" -h -o "%i %T %P %D %M/%l" -n "${PREFIX[$svc]}" | tr '\n' ' ')
        [[ -n "$jobs" ]] || continue
        echo "$svc  job: $jobs"
        url_file="$RUN/${STATE[$svc]}/broker_url"
        if [[ -s "$url_file" ]]; then
            b=$(cat "$url_file")
            echo "  url: $b/v1"
            curl -fsS --max-time 5 "$b/stats" | python3 -c 'import json, sys; v = json.load(sys.stdin)[sys.argv[1]]; print("  live_replica_servers={live_workers} queued={queued} in_flight={in_flight} served={served} failed={failed}".format(**v))' "${MODEL[$svc]}" || echo "  (dispatcher not answering)"
        else
            echo "  dispatcher not up yet (log: $RUN/${STATE[$svc]}/slurm-<jobid>.log)"
        fi
    done
}

[[ $# -ge 1 ]] || usage
if [[ "$1" == status ]]; then status; exit 0; fi

svc=$1; n=${2:-1}; part=${3:-batch_long}
[[ -n "${PREFIX[$svc]:-}" ]] || usage
[[ "$n" =~ ^[0-9]+$ && "$n" -ge 1 ]] || usage
case "$part" in
    batch) time=04:00:00 ;;
    batch_long) time=2-00:00:00 ;;
    *) usage ;;
esac
grep -q "\"${KEY[$svc]}\"" ~/Project/trichord/credentials/gateway.json || { echo "missing ${KEY[$svc]} in ~/Project/trichord/credentials/gateway.json"; exit 1; }
if squeue -u "$USER" -h -o "%i" -n "${PREFIX[$svc]}" | grep -q .; then
    echo "$svc already has a job ($(squeue -u "$USER" -h -o "%i %T" -n "${PREFIX[$svc]}" | tr '\n' ' ')); cancel it first to resize"
    exit 1
fi
mkdir -p "$RUN/${STATE[$svc]}"
nodes=$((n * NODES_PER_REPLICA[$svc]))
sbatch -p "$part" -t "$time" -N "$nodes" -J "${PREFIX[$svc]}" "$HERE/serve_$svc.sbatch" | sed "s/$/  ($svc: $n replicas, $nodes nodes, $part)/"
echo "url file: $RUN/${STATE[$svc]}/broker_url   check: bash $HERE/serve.sh status"
