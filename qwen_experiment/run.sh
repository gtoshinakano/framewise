#!/usr/bin/env bash
# Launcher for the Qwen 3.6 vs Qwen 3.5 first-frame benchmark (see qwen_experiment/README.md).
#
#   run.sh start [--count N] [--episode NAME] [--models a,b]   pause the worker, run on the GPU, restart it afterwards
#   run.sh status                                              container, worker and VRAM
#   run.sh stop                                                stop the experiment (the worker comes back)
#   run.sh report                                              build report.html from results.json (host, no GPU)
#   run.sh dry [--count N]                                     rebuild every input on the CPU, no model, worker untouched
#
# Only one GPU process at a time: the inference worker keeps a Qwen model in VRAM while idle, and a second GPU
# process beside it has crashed this host's kernel before. So `start` stops the worker, waits for VRAM to drain
# and arms a detached watcher that brings the worker back however the run ends.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKER=video-analyzer-inference-worker-1
API=video-analyzer-video-analyzer-1
EXP=framewise-qwen-experiment
IMAGE=framewise-inference-rocm:local
VRAM_LIMIT_MIB=4096
COMPOSE="docker compose -f compose.yml -f compose.inference.rocm.yml -f compose.dev.yml"

vram_mib() {
  local max=0 value
  for file in /sys/class/drm/card*/device/mem_info_vram_used; do
    [ -r "$file" ] || continue
    value=$(( $(cat "$file") / 1048576 ))
    [ "$value" -gt "$max" ] && max=$value
  done
  echo "$max"
}

queue_idle() {
  docker exec -i "$API" python - <<'PY'
import sqlite3, sys
conn = sqlite3.connect("/data/app.sqlite3")
busy = conn.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
print("busy" if busy else "idle")
sys.exit(1 if busy else 0)
PY
}

start_worker() {
  if [ "$(docker inspect -f '{{.State.Running}}' "$WORKER" 2>/dev/null)" != "true" ]; then
    docker start "$WORKER" >/dev/null && echo "inference worker started again"
  fi
}

case "${1:-}" in
  start)
    shift
    if docker ps -a --format '{{.Names}}' | grep -qx "$EXP"; then
      echo "An experiment container already exists; run '$0 stop' first."; exit 1
    fi
    if ! state=$(queue_idle); then echo "queue: $state; the app is busy, try again when it is idle."; exit 1; fi
    echo "queue: $state"
    # The Qwen 3.6 vision projector must be present (download it with the companion script).
    docker exec "$WORKER" sh -c 'test -f "$QWEN35_MODEL_PATH" && test -f "$QWEN35_PROJECTOR_PATH" \
      && test -f /models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf \
      && test -f /models/qwen3.6-35b/mmproj-Qwen3.6-35B-A3B-BF16.gguf \
      && test -x "$QWEN35_SERVER_PATH"' \
      || { echo "A model file, the Qwen 3.6 mmproj, or llama-server is missing."; exit 1; }
    echo "VRAM in use before: $(vram_mib) MiB; stopping $WORKER ..."
    docker stop -t 60 "$WORKER" >/dev/null
    for _ in $(seq 1 60); do [ "$(vram_mib)" -lt "$VRAM_LIMIT_MIB" ] && break; sleep 2; done
    if [ "$(vram_mib)" -ge "$VRAM_LIMIT_MIB" ]; then echo "VRAM did not drain; the worker may still hold it."; exit 1; fi
    echo "VRAM in use now: $(vram_mib) MiB"
    $COMPOSE run -d --no-deps --name "$EXP" --entrypoint python -v "$PWD/qwen_experiment:/work:rw" \
      inference-worker /work/benchmark.py --out /work --keep-frames "$@" \
      || { echo "Could not start the experiment container."; exit 1; }
    # Watcher: after the run ends, restart the worker and build the report.
    nohup setsid bash -c "for _ in \$(seq 1 600); do docker inspect '$EXP' >/dev/null 2>&1 && break; sleep 1; done; docker wait '$EXP' >/dev/null 2>&1; sleep 5; docker start '$WORKER' >/dev/null 2>&1; python3 qwen_experiment/report.py --results qwen_experiment/results.json --out qwen_experiment/report.html >/dev/null 2>&1" \
      >"qwen_experiment/watcher.log" 2>&1 < /dev/null &
    docker update --restart=no "$EXP" >/dev/null
    echo
    echo "Experiment started on the GPU; the worker stays stopped until it finishes (then it starts by itself)."
    echo "Status: $0 status   Stop: $0 stop   Report: $0 report (after it finishes)"
    ;;
  status)
    state=$(docker inspect -f '{{.State.Status}} (exit {{.State.ExitCode}})' "$EXP" 2>/dev/null) || state=none
    echo "experiment container: $state"
    echo "inference worker:     $(docker inspect -f '{{.State.Status}}' "$WORKER" 2>/dev/null || echo unknown)"
    echo "VRAM in use:          $(vram_mib) MiB"
    if [ -f qwen_experiment/results.json ]; then
      python3 -c "import json; d=json.load(open('qwen_experiment/results.json')); [print(f'  cut {f[\"cut_index\"]}: ' + ', '.join(f'{k}={a[\"latency_seconds\"]:.1f}s' for k,a in f.get('answers',{}).items() if a.get('latency_seconds'))) for f in d['frames']]" 2>/dev/null || echo "  results.json present (not readable yet)"
    fi
    ;;
  stop)
    if docker ps --format '{{.Names}}' | grep -qx "$EXP"; then docker stop -t 30 "$EXP" >/dev/null && echo "experiment stopped"; else echo "no experiment is running"; fi
    for _ in $(seq 1 30); do [ "$(docker inspect -f '{{.State.Running}}' "$WORKER")" = "true" ] && break; sleep 2; done
    start_worker
    ;;
  report)
    cd qwen_experiment
    python3 report.py
    echo "Report: $(pwd)/report.html"
    ;;
  dry)
    shift
    $COMPOSE run --no-deps --rm --entrypoint python -v "$PWD/qwen_experiment:/work:rw" \
      inference-worker /work/benchmark.py --dry --out /work "$@"
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
