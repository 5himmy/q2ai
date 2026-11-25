#!/bin/bash
# Q2AI Training Control Script
# Usage:
#   ./q2ai_train.sh start [N]  - Start N instances (default 1) + training
#   ./q2ai_train.sh stop       - Stop everything
#   ./q2ai_train.sh restart [N] - Stop + start N instances
#   ./q2ai_train.sh status     - Show running processes
#   ./q2ai_train.sh log        - Show training progress

DIR="/home/oky/q2ai"
SRC="/home/oky/Documents/OLD VSCODE/q2ai"
VENV="$SRC/python/.venv/bin/python"
BASE_PORT=27910

SYNC_PIDFILE=/tmp/q2ai_sync.pid

# Upload dashboards to 5himmy.com every 30 minutes.
# network.html is regenerated from the latest checkpoint before each upload.
sync_loop() {
    while true; do
        (cd "$SRC/python" && "$VENV" visualize_network.py > /dev/null 2>&1)
        for f in evolve.html network.html; do
            if [ -f "$SRC/$f" ]; then
                scp -q "$SRC/$f" oky@5himmy.com:/var/www/5himmy.com/html/$f 2>/dev/null \
                    && echo "$(date '+%F %T') uploaded $f" \
                    || echo "$(date '+%F %T') FAILED $f"
            fi
        done
        sleep 1800
    done
}

stop_sync() {
    if [ -f "$SYNC_PIDFILE" ]; then
        kill -9 "$(cat "$SYNC_PIDFILE")" 2>/dev/null
        rm -f "$SYNC_PIDFILE"
    fi
    pkill -9 -f "scp.*5himmy.com" 2>/dev/null
    pkill -9 -f "sleep 1800" 2>/dev/null
}

start_sync() {
    stop_sync
    sync_loop > /tmp/sync_evolve.log 2>&1 &
    echo $! > "$SYNC_PIDFILE"
    echo "  Sync PID: $! (evolve.html + network.html → 5himmy.com every 30min)"
}

stop_all() {
    echo "Stopping all Q2AI processes..."
    stop_sync
    pkill -9 -f "monitor.py" 2>/dev/null
    pkill -9 -f "train.py" 2>/dev/null
    pkill -9 -f "q2ai.*connect" 2>/dev/null
    pkill -9 -f "q2proded" 2>/dev/null
    sleep 1
    # Clean up shared memory and semaphores
    for i in $(seq 0 299); do
        rm -f /dev/shm/q2ai_state_$i 2>/dev/null
        rm -f /dev/shm/sem.q2ai_obs_$i 2>/dev/null
        rm -f /dev/shm/sem.q2ai_act_$i 2>/dev/null
    done
    echo "All stopped."
}

start_all() {
    NUM_ENVS=${1:-1}
    echo "Starting Q2AI training with $NUM_ENVS instance(s)..."

    cd "$DIR"

    # Start N server + KaaN pairs
    for i in $(seq 0 $((NUM_ENVS - 1))); do
        PORT=$((BASE_PORT + i))
        echo "  --- Instance $i (port $PORT) ---"

        # Server
        ./q2proded +set game opentdm +set g_gamemode 1 +set g_spawn_bots 1 \
            +set g_bot_skill 1 +set g_match_countdown 1 +map hop \
            +set g_tie_mode 0 +set g_match_time 1800 \
            +set g_bot_nomove 0 +set g_bot_noshoot 0 \
            +set dmflags 8192 \
            +set net_port $PORT +set sys_console 2 \
            +set rcon_password q2ai \
            > /tmp/server_${i}.log 2>&1 &
        echo "    Server PID: $!"
    done

    echo "  Waiting for servers to initialize..."
    sleep 3

    for i in $(seq 0 $((NUM_ENVS - 1))); do
        PORT=$((BASE_PORT + i))

        # KaaN headless client (cl_maxfps 30 = enough for 10Hz server, saves CPU)
        "$DIR/q2ai" +set logfile 0 +set name q2ai_bot +set q2ai_team b \
            +set q2ai_instance $i +set basedir "$DIR" \
            +set cl_maxfps 30 +set q2ai_nomove 1 \
            +connect 127.0.0.1:$PORT > /dev/null 2>&1 &
        echo "    KaaN[$i] PID: $! (instance=$i, port=$PORT)"
    done

    echo "  Waiting for matches to start..."
    sleep 15

    # Check match status
    READY=0
    for i in $(seq 0 $((NUM_ENVS - 1))); do
        if grep -q "Fight!" /tmp/server_${i}.log 2>/dev/null; then
            echo "    Instance $i: Match started!"
            READY=$((READY + 1))
        else
            echo "    Instance $i: Waiting..."
        fi
    done

    if [ $READY -lt $NUM_ENVS ]; then
        echo "  Some instances not ready yet, waiting 10 more seconds..."
        sleep 10
    fi

    # Training — resume from latest checkpoint if available, else BC model
    cd "$SRC/python"
    LATEST_CKPT=$(ls -t "${SRC}/python/checkpoints/"model_*.pt 2>/dev/null | head -1)
    TRAIN_ARGS=(--num-envs "$NUM_ENVS" --num-rollouts 100000 --log-interval 5
                --rollout-len 256 --save-interval 50)
    if [ -n "$LATEST_CKPT" ]; then
        TRAIN_ARGS+=(--resume "$LATEST_CKPT")
        echo "  Resuming from checkpoint: $(basename "$LATEST_CKPT")"
    fi
    # Extra trainer flags, e.g. TRAIN_EXTRA="--miss-penalty 0.05" ./q2ai_train.sh restart 100
    if [ -n "${TRAIN_EXTRA:-}" ]; then
        read -r -a _extra <<< "$TRAIN_EXTRA"
        TRAIN_ARGS+=("${_extra[@]}")
        echo "  Extra trainer args: $TRAIN_EXTRA"
    fi
    PYTHONUNBUFFERED=1 "$VENV" train.py "${TRAIN_ARGS[@]}" \
        > /tmp/training.log 2>&1 &
    echo "  Training PID: $! (num-envs=$NUM_ENVS)"

    # Monitor (auto-generates HTML dashboard)
    PYTHONUNBUFFERED=1 "$VENV" "$SRC/python/monitor.py" --interval 600 \
        > /tmp/monitor.log 2>&1 &
    echo "  Monitor PID: $!"

    # Sync evolve.html + network.html to web server every 30 minutes
    start_sync

    echo ""
    echo "All running! $NUM_ENVS server(s) + $NUM_ENVS KaaN(s) + trainer + monitor + sync"
    echo "Dashboard: $SRC/evolve.html"
    echo "Web: http://5himmy.com/evolve.html"
    echo "Use './q2ai_train.sh log' to check progress."
}

case "${1:-}" in
    start)
        start_all "${2:-16}"
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all "${2:-16}"
        ;;
    sync)
        start_sync
        ;;
    status)
        echo "=== Q2AI Processes ==="
        ps aux | grep -E "q2proded|q2ai.*connect|train.py" | grep -v grep || echo "Nothing running."
        ;;
    log)
        echo "=== Training Log (last 20 lines) ==="
        tail -20 /tmp/training.log 2>/dev/null || echo "No training log yet."
        echo ""
        echo "=== Server Logs (last 5 lines each) ==="
        for f in /tmp/server_*.log; do
            if [ -f "$f" ]; then
                echo "--- $(basename $f) ---"
                tail -5 "$f"
            fi
        done
        if [ ! -f /tmp/server_0.log ]; then
            # Fallback for old single-instance log
            tail -5 /tmp/server.log 2>/dev/null || echo "No server logs."
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log|sync} [num_instances]"
        echo "  sync       - (Re)start only the 5himmy.com dashboard upload loop"
        echo "  start 4   - Start 4 parallel server+KaaN pairs"
        echo "  stop       - Stop all processes"
        echo "  restart 2  - Restart with 2 instances"
        exit 1
        ;;
esac
