# Shared by the serve_<service>.sbatch scripts: one Slurm job = one dispatcher + R model replicas.
#   node 0 (the batch host, "rank 0") runs the dispatcher (broker.py) and one sidecar per replica server
#   replica r runs on nodes [r*NPR, (r+1)*NPR) of the job; each sidecar pulls work from the dispatcher and feeds it
# Clients use one URL, written to $ROOT/broker_url:  http://<node0>:<20000 + jobid % 20000>/v1
# Replica servers listen on the following ports (pool_port r [i]).
SERVING=$HOME/Project/trichord/scripts/serving
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx

pool_init() {  # <run dir name> <nodes per replica>
    ROOT=$U/run/$1
    STATE=$ROOT/job-$SLURM_JOB_ID
    NPR=$2
    mkdir -p "$STATE"
    (( SLURM_JOB_NUM_NODES % NPR == 0 )) || { echo "job has $SLURM_JOB_NUM_NODES nodes, not a multiple of $NPR"; exit 2; }
    REPLICAS=$((SLURM_JOB_NUM_NODES / NPR))
    mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    PORT_BASE=$((20000 + SLURM_JOB_ID % 20000))
    BROKER_URL=http://${HOSTS[0]}:$PORT_BASE
    export BROKER_URL_FILE=$ROOT/broker_url
    PIDS=()
    trap pool_cleanup EXIT
    trap 'exit 143' TERM INT
    printf 'job=%s\nreplicas=%s\nnodes_per_replica=%s\nnodes=%s\nbroker_url=%s/v1\n' "$SLURM_JOB_ID" "$REPLICAS" "$NPR" "$SLURM_JOB_NODELIST" "$BROKER_URL" > "$STATE/config.txt"
}

pool_hosts() {  # <replica> -> comma-separated nodes of that replica
    local r=$1
    (IFS=,; echo "${HOSTS[*]:$((r * NPR)):$NPR}")
}

pool_head() {  # <replica> -> first node of that replica
    echo "${HOSTS[$(($1 * NPR))]}"
}

pool_start_broker() {
    python3 "$SERVING/broker.py" --port "$PORT_BASE" > "$STATE/broker.log" 2>&1 &
    PIDS+=($!)
    for ((i = 0; i < 60; i++)); do
        curl -fsS --max-time 3 "$BROKER_URL/health" > /dev/null 2>&1 && break
        sleep 2
    done
    echo "$BROKER_URL" > "$BROKER_URL_FILE.tmp" && mv "$BROKER_URL_FILE.tmp" "$BROKER_URL_FILE"
    echo "DISPATCHER $BROKER_URL/v1 replicas=$REPLICAS"
}

pool_attach() {  # <replica> <server url> <served model> <key name> <max inflight> <server pid>
    local r=$1 url=$2 model=$3 key_name=$4 n=$5 pid=$6
    local key
    key=$(python3 -c "import json, os; print(json.load(open(os.path.expanduser('~/Project/trichord/credentials/gateway.json')))['$key_name'].strip())")
    (
        for ((t = 0; t < 240; t++)); do   # up to 2 h: first run may download weights
            kill -0 "$pid" 2>/dev/null || { echo "replica $r ($url) exited before READY"; exit 1; }
            if curl -fsS --max-time 5 -H "Authorization: Bearer $key" "$url/v1/models" 2>/dev/null | grep -q "$model"; then
                echo "READY replica=$r $url"
                exec python3 "$SERVING/broker_worker.py" --model "$model" --backend "$url" --api-key-name "$key_name" --max-inflight "$n" > "$STATE/worker-${url##*:}.log" 2>&1
            fi
            sleep 30
        done
        echo "replica $r ($url) not ready after 2 h"
    ) &
}

pool_wait() {  # keep serving while the dispatcher and at least one replica are alive
    while kill -0 "${PIDS[0]}" 2>/dev/null; do
        local alive=0
        for pid in "${PIDS[@]:1}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        (( alive )) || { echo "all replicas exited"; exit 1; }
        sleep 30
    done
    echo "dispatcher exited"
    exit 1
}

pool_cleanup() {
    code=$?
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    [[ "$(cat "$BROKER_URL_FILE" 2>/dev/null)" == "$BROKER_URL" ]] && rm -f "$BROKER_URL_FILE"
    echo "exit_code=$code" > "$STATE/finished"
}
