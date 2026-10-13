#!/bin/bash
# One-line launcher for the self-hosted endpoints on gb300 (run from any directory, in bash):
#   bash scripts/serving/serve.sh cosmos3_super_t2i 2         # 2 Cosmos3-Super-Text2Image replicas, batch_long 48 h (1 node each)
#   bash scripts/serving/serve.sh kimi_k3 2                   # 2 Kimi-K3 replicas, batch_long 48 h (2 nodes each)
#   bash scripts/serving/serve.sh mimo_v2p6_pro_rl 1          # 1 MiMo-V2.6-Pro-RL replica, batch_long 48 h (1 node)
#   bash scripts/serving/serve.sh cosmos3_super_t2i 1 batch   # same, but batch 4 h (quick test, usually starts sooner)
#   bash scripts/serving/serve.sh status                      # every replica: state, and its URL once READY
# Each service runs serving/serve_<service>.sbatch; job names keep the short prefixes the pipeline scans for.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run

declare -A PREFIX=([kimi_k3]=k3-xqx [mimo_v2p6_pro_rl]=mimo-xqx [cosmos3_super_t2i]=c3-xqx)
declare -A STATE=([kimi_k3]=k3_serve [mimo_v2p6_pro_rl]=mimo_pro_serve [cosmos3_super_t2i]=c3_serve)
declare -A KEY=([kimi_k3]=SLURM_KIMI_K3_API_KEY [mimo_v2p6_pro_rl]=SLURM_MIMO_API_KEY [cosmos3_super_t2i]=SLURM_COSMOS3_API_KEY)

usage() { sed -n 2,8p "${BASH_SOURCE[0]}"; exit 1; }

status() {
    for svc in kimi_k3 mimo_v2p6_pro_rl cosmos3_super_t2i; do
        squeue -u "$USER" -h -o "%i %j %T %P %M %l" | awk -v p="${PREFIX[$svc]}" 'index($2, p) == 1' | sort -n | while read -r id name state part elapsed limit; do
            env="$RUN/${STATE[$svc]}/job-$id/endpoint.env"
            if [[ -s "$env" ]]; then url=$(grep base_url "$env" | cut -d= -f2-); else url="(not ready, log: $RUN/${STATE[$svc]}/slurm-$id.log)"; fi
            printf '%-18s %-9s %-12s %-8s %-11s %s/%s  %s\n' "$svc" "$id" "$name" "$state" "$part" "$elapsed" "$limit" "$url"
        done
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
mkdir -p "$RUN/${STATE[$svc]}"

last=$(squeue -u "$USER" -h -o "%j" | sed -n "s/^${PREFIX[$svc]}-\([0-9]\+\)$/\1/p" | sort -n | tail -1)
for ((i = 1; i <= n; i++)); do
    name="${PREFIX[$svc]}-$((${last:-0} + i))"
    sbatch -p "$part" -t "$time" -J "$name" "$HERE/serve_$svc.sbatch" | sed "s/$/  ($name, $part)/"
done
echo "check: bash $HERE/serve.sh status   (URL appears once the replica prints READY)"
