#!/bin/bash
# Run this on the awscode or gcpcode head node. It submits itself with srun (2 tasks, one container
# each, same image and mounts as slaunch) and runs the checks inside the containers.
# Override the CPU partition with PARTITION=<name> (default: cpu).

HOST=$(hostname)
if [ -z "$SLURM_PROCID" ] && [[ "$HOST" == aws-* || "$HOST" == gcp-* ]]; then
  if [[ "$HOST" == aws-* ]]; then
    DOCKER_PATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v11.2.3.sqsh
  else
    DOCKER_PATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.0.0.sqsh
  fi
  lustrepath=/lustre/fsw/portfolios/cosmos
  exec srun --account=cosmos_base_training --partition="${PARTITION:-cpu}" --nodes=1 --ntasks=2 --cpus-per-task=2 --time=00:10:00 \
       --container-image="$DOCKER_PATH" \
       --container-mounts "$lustrepath:$lustrepath:rw,$HOME:$HOME:rw" \
       --container-workdir="$HOME/Project/trichord/scripts/model_playground" \
       bash "$HOME/Project/trichord/scripts/model_playground/script/run_poster_generation_envdebug.sh"
fi

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

say "--- chrome"
CHROME=""
for c in "$HOME/Software/chrome/google-chrome" google-chrome google-chrome-stable chromium chromium-browser; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi
done
if [ -z "$CHROME" ]; then
  say "chrome: NOT FOUND"
else
  say "chrome: $CHROME ($($CHROME --version 2>/dev/null | tr -d '\n'))"
  MISSING=$(ldd "$(dirname "$(readlink -f "$CHROME")")/chrome" 2>/dev/null | grep -c "not found")
  say "chrome missing shared libs: $MISSING"
  WORK=$(mktemp -d /tmp/env_debug_XXXX)
  echo "<body style='background:tomato;width:400px;height:300px'><h1>ok</h1></body>" > "$WORK/t.html"
  "$CHROME" --headless=new --disable-gpu --no-sandbox --disable-dev-shm-usage --disable-crash-reporter \
            --user-data-dir="$WORK/profile" --window-size=400,300 --screenshot="$WORK/t.png" "file://$WORK/t.html" >"$WORK/chrome.log" 2>&1
  if [ -s "$WORK/t.png" ]; then say "chrome headless render: OK ($(stat -c %s "$WORK/t.png") bytes)"; else say "chrome headless render: FAILED"; grep -iE "error|required|not found" "$WORK/chrome.log" | head -3 | sed "s/^/[task $T]   /"; fi
  rm -rf "$WORK"
fi

say "--- /tmp isolation"
touch "/tmp/env_debug_probe_$T"
sleep 2
say "probe files visible: $(ls /tmp/env_debug_probe_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
rm -f "/tmp/env_debug_probe_$T"

say "--- credentials and network"
for f in credentials/gateway.json credentials/gcs.secret; do
  [ -s "$HOME/Project/trichord/$f" ] && say "$f: present" || say "$f: MISSING"
done
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://inference-api.nvidia.com/v1/models 2>/dev/null)
say "gateway reachable (HTTP ${CODE:-none}, 401/403 means reachable without key)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://storage.googleapis.com 2>/dev/null)
say "gcs endpoint reachable (HTTP ${CODE:-none})"
