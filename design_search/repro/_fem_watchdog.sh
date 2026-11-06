#!/usr/bin/env bash
# RAM-monitored FEM stage runner.
#
# Launches `python -m design_search.landscape.fem` with $1 workers, watches
# RAM and swap every 30 s. If MemUsed/MemTotal exceeds $OOD_FEM_RAM_DANGER_PCT
# (default 80) OR swap usage grows for 3 consecutive samples, kills the FEM
# process and re-launches with the fallback worker count. landscape/fem.py
# is resumable (skips IDs already in fem_scores.jsonl) so the restart picks
# up where it left off.
#
# Usage:
#   bash design_search/repro/_fem_watchdog.sh \
#       --run-root <root> --stage walk \
#       --workers 150 --workers-fallback 100 \
#       --log <root>/logs/fem_walk.log
#
# Exit codes:
#   0 — FEM completed successfully (possibly after fallback)
#   2 — fallback also tripped; needs human attention
#   3 — the FEM process exited with an error that was not a watchdog trip
#       (no fallback is attempted; see the log)

set -u -o pipefail

RUN_ROOT=""
STAGE=""
WORKERS=150
WORKERS_FALLBACK=100
DANGER_PCT="${OOD_FEM_RAM_DANGER_PCT:-80}"
LOG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root)         RUN_ROOT="$2"; shift 2 ;;
        --stage)            STAGE="$2"; shift 2 ;;
        --workers)          WORKERS="$2"; shift 2 ;;
        --workers-fallback) WORKERS_FALLBACK="$2"; shift 2 ;;
        --log)              LOG="$2"; shift 2 ;;
        --)                 shift; EXTRA_ARGS+=("$@"); break ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$RUN_ROOT" || -z "$STAGE" ]]; then
    echo "ERROR: --run-root and --stage are required" >&2
    exit 1
fi
if [[ -z "$LOG" ]]; then
    LOG="$RUN_ROOT/logs/fem_${STAGE}.log"
fi
mkdir -p "$(dirname "$LOG")"

RAM_CSV="$RUN_ROOT/logs/fem_ram.csv"
if [[ ! -f "$RAM_CSV" ]]; then
    echo "timestamp,stage,n_workers,mem_used_pct,mem_avail_gb,swap_used_gb" >"$RAM_CSV"
fi

# ------------------------------------------------------------------
# launch_fem <n_workers> -> writes log; returns exit code from FEM
# Sets $FEM_PID on success; kills the process on watchdog trip.
# ------------------------------------------------------------------
launch_fem() {
    local nw="$1"
    echo "[$(date -Iseconds)] FEM start  stage=$STAGE workers=$nw  log=$LOG" | tee -a "$LOG"

    # setsid: the FEM process leads its own process group, so a watchdog
    # trip can kill it together with its spawned ProcessPoolExecutor workers
    # (killing the parent alone would orphan up to `workers` FEM processes).
    OMPI_MCA_btl=self,tcp \
    setsid python -m design_search.landscape.fem \
        --run-root "$RUN_ROOT" \
        --stage "$STAGE" \
        --workers "$nw" \
        "${EXTRA_ARGS[@]}" \
        >>"$LOG" 2>&1 &
    FEM_PID=$!
    echo "[$(date -Iseconds)] FEM pid=$FEM_PID" | tee -a "$LOG"
    # setsid only forks when the caller is already a group leader (job-control
    # shells); then $! would be a short-lived parent and the group kill would
    # miss. Verify the process leads its own group so a trip can kill it all.
    sleep 1
    if [[ "$(ps -o pgid= -p "$FEM_PID" 2>/dev/null | tr -d ' ')" != "$FEM_PID" ]]; then
        echo "[$(date -Iseconds)] WARNING: FEM pid $FEM_PID is not a process-group leader; a watchdog kill will reach the parent only" | tee -a "$LOG"
    fi
}

watch_loop() {
    # Watches FEM_PID; trips watchdog if RAM/swap hit danger thresholds.
    # Returns 0 if FEM completed cleanly, 1 if the watchdog tripped,
    # 3 if the FEM process exited non-zero on its own.
    local prev_swap_kb=0
    local swap_climbing=0
    while kill -0 "$FEM_PID" 2>/dev/null; do
        sleep 30
        local mem_total mem_avail mem_used_pct mem_avail_gb swap_used_kb swap_used_gb
        mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
        mem_avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        swap_used_kb=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)
        mem_used_pct=$(( (mem_total - mem_avail) * 100 / mem_total ))
        mem_avail_gb=$(( mem_avail / 1048576 ))
        swap_used_gb=$(( swap_used_kb / 1048576 ))

        echo "$(date -Iseconds),$STAGE,$1,$mem_used_pct,$mem_avail_gb,$swap_used_gb" >>"$RAM_CSV"

        if (( mem_used_pct > DANGER_PCT )); then
            echo "[$(date -Iseconds)] WATCHDOG: mem_used_pct=$mem_used_pct% > ${DANGER_PCT}% threshold; killing FEM pid=$FEM_PID" | tee -a "$LOG"
            kill_fem_group
            return 1
        fi

        if (( swap_used_kb > prev_swap_kb )); then
            swap_climbing=$((swap_climbing + 1))
        else
            swap_climbing=0
        fi
        prev_swap_kb=$swap_used_kb
        if (( swap_climbing >= 3 )); then
            echo "[$(date -Iseconds)] WATCHDOG: swap climbing for 3 samples (now ${swap_used_gb} GB); killing FEM pid=$FEM_PID" | tee -a "$LOG"
            kill_fem_group
            return 1
        fi
    done
    wait "$FEM_PID"
    local rc=$?
    if (( rc != 0 )); then
        echo "[$(date -Iseconds)] FEM exited with code $rc (not a watchdog trip)" | tee -a "$LOG"
        return 3
    fi
    return 0
}

kill_fem_group() {
    # Kill the FEM process group (parent + spawned workers); fall back to the
    # PID alone if the group kill is refused.
    kill -TERM -- "-$FEM_PID" 2>/dev/null || kill -TERM "$FEM_PID" 2>/dev/null || true
    sleep 30
    kill -KILL -- "-$FEM_PID" 2>/dev/null || kill -KILL "$FEM_PID" 2>/dev/null || true
    wait "$FEM_PID" 2>/dev/null || true
}

launch_fem "$WORKERS"
watch_loop "$WORKERS"; rc=$?
if (( rc == 0 )); then
    echo "[$(date -Iseconds)] FEM complete (workers=$WORKERS)" | tee -a "$LOG"
    exit 0
elif (( rc == 3 )); then
    echo "[$(date -Iseconds)] FEM failed on its own (see $LOG); not retrying" | tee -a "$LOG"
    exit 3
fi

# Watchdog tripped — fall back to fewer workers
echo "[$(date -Iseconds)] FALLBACK: re-launching at workers=$WORKERS_FALLBACK" | tee -a "$LOG"
launch_fem "$WORKERS_FALLBACK"
watch_loop "$WORKERS_FALLBACK"; rc=$?
if (( rc == 0 )); then
    echo "[$(date -Iseconds)] FEM complete after fallback (workers=$WORKERS_FALLBACK)" | tee -a "$LOG"
    exit 0
elif (( rc == 3 )); then
    echo "[$(date -Iseconds)] FEM failed on its own after fallback (see $LOG)" | tee -a "$LOG"
    exit 3
fi

echo "[$(date -Iseconds)] FEM watchdog tripped at fallback level — needs human attention" | tee -a "$LOG"
exit 2
