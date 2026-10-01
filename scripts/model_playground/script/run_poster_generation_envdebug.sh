#!/bin/bash
# bash /home/xingqianx/Project/trichord/scripts/model_playground/script/run_poster_generation_envdebug.sh

HOST=$(hostname)
if [ -z "$SLURM_PROCID" ] && [[ "$HOST" == aws-* || "$HOST" == gcp-* ]]; then
  if [[ "$HOST" == aws-* ]]; then
    DOCKER_PATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v11.2.3.sqsh
    PARTITION="${PARTITION:-cpu-big}"
  else
    DOCKER_PATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.0.0.sqsh
    PARTITION="${PARTITION:-cpu}"
  fi
  lustrepath=/lustre/fsw/portfolios/cosmos
  exec srun --account=cosmos_base_training --partition="$PARTITION" --nodes=1 --ntasks=2 --cpus-per-task=2 --time=00:10:00 \
       --container-image="$DOCKER_PATH" \
       --container-mounts "$lustrepath:$lustrepath:rw,$HOME:$HOME:rw" \
       --container-workdir="$HOME/Project/trichord/scripts/model_playground" \
       --container-env=HOME,SLURM_PROCID,SLURM_NTASKS,SLURM_JOB_ID \
       bash "$HOME/Project/trichord/scripts/model_playground/script/run_poster_generation_envdebug.sh"
fi

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
T="${SLURM_PROCID:-0}"
say() { echo "[task $T] $*"; }
PY=""
for cand in .venv_container/bin/python .venv/bin/python python3; do
  if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then PY="$cand"; break; fi
done

say "host=$(hostname) arch=$(uname -m) os=$(. /etc/os-release 2>/dev/null; echo "$NAME $VERSION_ID") glibc=$(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
say "cwd=$(pwd) python=$PY ($($PY --version 2>&1))"

say "--- python deps"
$PY - <<'PYEOF' 2>&1 | sed "s/^/[task ${SLURM_PROCID:-0}]   /"
import importlib
for m in ["openai", "PIL", "fsspec", "boto3", "yaml", "loguru", "tqdm"]:
    try:
        importlib.import_module(m); print(f"{m:8s} ok")
    except Exception as e:
        print(f"{m:8s} MISSING ({type(e).__name__})")
PYEOF

say "--- chrome (render_poster tool)"
WORK=$(mktemp -d /tmp/env_debug_XXXX)
$PY - "$WORK/t.png" <<'PYEOF' 2>&1 | tail -12 | sed "s/^/[task ${SLURM_PROCID:-0}]   /"
import subprocess
import sys
from PIL import Image
from tool.render_poster import CHROME_CANDIDATES, find_chrome_binary, render_html_to_png
chrome = find_chrome_binary()
if chrome is None:
    print(f"chrome: NOT FOUND (tried {CHROME_CANDIDATES})")
    sys.exit(0)
print(f"chrome: {chrome}")
missing = [line.split()[0] for line in subprocess.run(["ldd", chrome], capture_output=True, text=True).stdout.splitlines() if "not found" in line]
print(f"chrome system libs: {'OK' if not missing else 'NOT OK, missing ' + ' '.join(missing)} (before bundled lib folder)")
html = "<html><head><style>.poster{width:1080px;height:1440px;background:linear-gradient(teal,navy);color:white;font:bold 80px sans-serif}</style></head><body><div class='poster'>env debug</div></body></html>"
try:
    out = render_html_to_png(html, sys.argv[1])
    print(f"chrome render: OK {Image.open(out).size}")
except Exception as e:
    print(f"chrome render: NOT OK ({type(e).__name__}: {str(e)[-600:]})")
PYEOF
rm -rf "$WORK"

say "--- /tmp isolation across tasks"
touch "/tmp/env_debug_probe_$T"
sleep 5
SEEN=$(ls /tmp/env_debug_probe_* 2>/dev/null | wc -l)
if [ "$SEEN" -eq 1 ]; then say "/tmp private per task: OK"; else say "/tmp private per task: NOT OK ($SEEN probe files visible, tasks share /tmp)"; fi
sleep 2; rm -f "/tmp/env_debug_probe_$T"

say "--- unshare (per-process private /tmp)"
if unshare -Urm sh -c "mount -t tmpfs none /tmp && touch /tmp/ns_ok && ls /tmp/ns_ok" >/dev/null 2>&1; then say "unshare -Urm + tmpfs /tmp: OK"; else say "unshare -Urm + tmpfs /tmp: NOT OK"; fi

say "--- credentials and network"
for f in credentials/gateway.json credentials/gcs.secret; do
  [ -s "$HOME/Project/trichord/$f" ] && say "$f: OK" || say "$f: NOT OK (missing)"
done
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://inference-api.nvidia.com/v1/models 2>/dev/null)
case "$CODE" in 200|401|403) say "nvidia gateway reachable: OK (HTTP $CODE)";; *) say "nvidia gateway reachable: NOT OK (HTTP ${CODE:-none})";; esac
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://storage.googleapis.com 2>/dev/null)
case "$CODE" in 2*|3*|4*) say "gcs endpoint reachable: OK (HTTP $CODE)";; *) say "gcs endpoint reachable: NOT OK (HTTP ${CODE:-none})";; esac
