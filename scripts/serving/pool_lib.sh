# Shared by the serve_<service>.sbatch scripts: one Slurm job = one dispatcher + R model replicas.
#   node 0 (the batch host, "rank 0") runs the dispatcher (broker.py) and one sidecar per replica server
#   replica r runs on nodes [r*NPR, (r+1)*NPR) of the job; each sidecar pulls work from the dispatcher and feeds it
# Clients use one URL, written to $ROOT/broker_url:  http://<node0>:<20000 + jobid % 20000>/v1
# Fault handling: a sidecar exits (releasing its jobs) when its replica process dies or its backend stops answering, and
# then ends that replica's srun; pool_run restarts a dead replica on the same nodes (MAX_RESTARTS times, after
# RESTART_DELAY seconds) and restarts a dead dispatcher on the same port. The serve scripts use --no-kill, so a failed
# node takes down only its own replica.
# A serve script defines launch_replica <r>: start replica r's srun in the background, set REPLICA_PID[r]=$!, and
# pool_attach its server(s); then it calls pool_start_broker and pool_run launch_replica.
# POOL_ATTACH=1 (attach mode): the job brings replicas only; it starts no dispatcher, never touches broker_url, and its
# sidecars pull from the dispatcher already named in $ROOT/broker_url, so clients get the extra capacity unchanged.
SERVING=$HOME/Project/trichord/scripts/serving
U=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_DELAY=${RESTART_DELAY:-60}
declare -A REPLICA_PID REPLICA_RESTARTS

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
    if [[ "${POOL_ATTACH:-0}" == 1 ]]; then
        until [[ -s "$BROKER_URL_FILE" ]] && curl -fsS --max-time 3 "$(cat "$BROKER_URL_FILE")/health" > /dev/null 2>&1; do
            echo "attach mode: waiting for the dispatcher in $BROKER_URL_FILE"
            sleep 30
        done
        BROKER_PID=""
        echo "ATTACHED to dispatcher $(cat "$BROKER_URL_FILE")/v1 replicas=$REPLICAS"
        return
    fi
    python3 "$SERVING/broker.py" --port "$PORT_BASE" >> "$STATE/broker.log" 2>&1 &
    BROKER_PID=$!
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
                python3 "$SERVING/broker_worker.py" --model "$model" --backend "$url" --api-key-name "$key_name" --max-inflight "$n" \
                    --watch-pid "$pid" >> "$STATE/worker-${url##*:}.log" 2>&1
                echo "sidecar for replica $r ($url) exited; stopping the replica so pool_run restarts it"
                kill -TERM "$pid" 2>/dev/null
                exit 0
            fi
            sleep 30
        done
        echo "replica $r ($url) not ready after 2 h; stopping it"
        kill -TERM "$pid" 2>/dev/null
    ) &
    PIDS+=($!)
}

pool_run() {  # <launch function>: start every replica, then keep the dispatcher and the replicas alive
    local launch=$1 r alive broker_restarts=0
    for ((r = 0; r < REPLICAS; r++)); do REPLICA_RESTARTS[$r]=0; "$launch" "$r"; PIDS+=("${REPLICA_PID[$r]}"); done
    while true; do
        sleep 30
        if [[ -n "$BROKER_PID" ]] && ! kill -0 "$BROKER_PID" 2>/dev/null; then
            (( broker_restarts < MAX_RESTARTS )) || { echo "dispatcher exited $broker_restarts times, giving up"; exit 1; }
            broker_restarts=$((broker_restarts + 1))
            echo "dispatcher exited; restart $broker_restarts/$MAX_RESTARTS"
            pool_start_broker
        fi
        alive=0
        for ((r = 0; r < REPLICAS; r++)); do
            if kill -0 "${REPLICA_PID[$r]}" 2>/dev/null; then alive=1; continue; fi
            (( REPLICA_RESTARTS[$r] < MAX_RESTARTS )) || continue
            alive=1
            REPLICA_RESTARTS[$r]=$((REPLICA_RESTARTS[$r] + 1))
            echo "replica $r exited; restart ${REPLICA_RESTARTS[$r]}/$MAX_RESTARTS in ${RESTART_DELAY}s"
            sleep "$RESTART_DELAY"
            "$launch" "$r"
            PIDS+=("${REPLICA_PID[$r]}")
        done
        (( alive )) || { echo "all replicas exited and used up their restarts"; exit 1; }
    done
}

pool_cleanup() {
    code=$?
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    [[ "$(cat "$BROKER_URL_FILE" 2>/dev/null)" == "$BROKER_URL" ]] && rm -f "$BROKER_URL_FILE"
    echo "exit_code=$code" > "$STATE/finished"
}
