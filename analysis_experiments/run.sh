#!/usr/bin/env bash
# Launcher for the pose-detail experiment (see analysis_experiments/README.md).
#
#   run.sh start [--run-id ID] [experiment args]   pause the inference worker, run on the GPU, restart the worker afterwards
#   run.sh status                                  progress, container, worker and VRAM
#   run.sh stop                                    stop the experiment (the worker is restarted automatically)
#   run.sh dry-run                                 whole pipeline with fake models on the CPU (no GPU, worker untouched)
#
# Only one GPU process at a time: the inference worker keeps a Qwen model in VRAM while idle, and a second GPU
# process beside it once crashed this host's kernel (see the gpu-testing-hazard note). So `start` stops the
# worker, waits for the VRAM to drain and arms a detached watcher that starts the worker again however the run ends.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKER=video-analyzer-inference-worker-1
API=video-analyzer-video-analyzer-1
EXP=framewise-pose-experiment
RUNS=frontend/public/analysis-experiments/runs
IMAGE=framewise-inference-rocm:local
VRAM_LIMIT_MIB=4096
HOST_IP=$(grep -E '^APP_BIND_IP=' .env 2>/dev/null | cut -d= -f2 || true)
URL="http://${HOST_IP:-192.168.0.6}:${APP_PORT:-8080}/analysis-experiments/index.html"

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
c = sqlite3.connect('file:/data/app.sqlite3?mode=ro', uri=True)
busy = []
for table in ('jobs', 'assistant_turns', 'chat_gateway_requests'):
    if c.execute("select 1 from sqlite_master where name=?", (table,)).fetchone():
        n = c.execute(f"select count(*) from {table} where status in ('queued','running')").fetchone()[0]
        if n:
            busy.append(f'{table}: {n}')
print('busy (' + ', '.join(busy) + ')' if busy else 'idle')
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
    RUN_ID="pose-$(date +%Y%m%d-%H%M)"
    if [ "${1:-}" = "--run-id" ]; then RUN_ID="$2"; shift 2; fi
    if docker ps -a --format '{{.Names}}' | grep -qx "$EXP"; then
      if [ "$(docker inspect -f '{{.State.Running}}' "$EXP")" = "true" ]; then echo "An experiment is already running: $0 status"; exit 1; fi
      docker rm "$EXP" >/dev/null
    fi
    if ! state=$(queue_idle); then echo "queue: $state. Try again when it is idle."; exit 1; fi
    echo "queue: $state"
    docker exec "$WORKER" sh -c 'test -f "$QWEN35_MODEL_PATH" && test -f "$QWEN35_PROJECTOR_PATH" && test -f "$QWEN36_MODEL_PATH" && test -f "$QWEN3_VIDEO_MODEL_PATH/model.safetensors.index.json" && test -x "$QWEN35_SERVER_PATH"' \
      || { echo "A model file or llama-server is missing in the worker container."; exit 1; }
    mkdir -p "$RUNS"; chmod 777 "$RUNS" "$(dirname "$RUNS")"
    echo "VRAM in use before: $(vram_mib) MiB; stopping $WORKER ..."
    docker stop -t 60 "$WORKER" >/dev/null
    for _ in $(seq 1 60); do [ "$(vram_mib)" -lt "$VRAM_LIMIT_MIB" ] && break; sleep 2; done
    if [ "$(vram_mib)" -ge "$VRAM_LIMIT_MIB" ]; then
      echo "VRAM did not drain ($(vram_mib) MiB in use). Not starting; restarting the worker."
      start_worker; exit 1
    fi
    echo "VRAM in use now: $(vram_mib) MiB"
    # Armed before the container exists: the worker comes back when the container ends, or after 10 minutes if it never started.
    nohup setsid bash -c "for _ in \$(seq 1 600); do docker inspect '$EXP' >/dev/null 2>&1 && break; sleep 1; done; docker wait '$EXP' >/dev/null 2>&1; sleep 5; docker start '$WORKER' >/dev/null 2>&1" \
      >"analysis_experiments/watcher.log" 2>&1 < /dev/null &
    if ! docker compose run -d --no-deps --name "$EXP" --entrypoint python \
         -v "$PWD/analysis_experiments:/work:ro" -v "$PWD/$RUNS:/report/runs" \
         inference-worker /work/experiment.py all --run-id "$RUN_ID" "$@"; then
      echo "The experiment container did not start; the watcher will restart the worker."; exit 1
    fi
    docker update --restart=no "$EXP" >/dev/null
    echo
    echo "Run id:   $RUN_ID"
    echo "Report:   $URL?run=$RUN_ID"
    echo "Status:   $0 status        Stop: $0 stop"
    echo "The inference worker stays stopped until the run ends (then it starts by itself)."
    ;;
  status)
    state=$(docker inspect -f '{{.State.Status}} (exit {{.State.ExitCode}})' "$EXP" 2>/dev/null) || state=none
    echo "experiment container: $state"
    echo "inference worker:     $(docker inspect -f '{{.State.Status}}' "$WORKER" 2>/dev/null || echo unknown)"
    echo "VRAM in use:          $(vram_mib) MiB"
    latest=$(ls -t "$RUNS"/*/progress.json 2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then
      echo "latest run:           $(basename "$(dirname "$latest")")"
      python3 - "$latest" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print('status:', p.get('status'), '| updated', p.get('updated_at'))
for name, s in p.get('stages', {}).items():
    print(f"  {name:<9} {s.get('status'):<8} {s.get('done', 0)}/{s.get('total', 0)} {s.get('message') or ''}")
PY
      tail -n 4 "$(dirname "$latest")/log.txt" 2>/dev/null | cut -c1-200
    fi
    echo "report: $URL"
    ;;
  stop)
    if docker ps --format '{{.Names}}' | grep -qx "$EXP"; then docker stop -t 30 "$EXP" >/dev/null && echo "experiment stopped"; else echo "no experiment is running"; fi
    for _ in $(seq 1 30); do [ "$(docker inspect -f '{{.State.Running}}' "$WORKER")" = "true" ] && break; sleep 2; done
    start_worker
    ;;
  dry-run)
    mkdir -p "$RUNS"; chmod 777 "$RUNS" "$(dirname "$RUNS")"
    docker run --rm -v "$PWD/$RUNS:/runs" --entrypoint sh "$IMAGE" -c 'rm -rf /runs/dry-run'
    docker run --rm --entrypoint python -e DATA_DIR=/data -e TMPDIR=/data/tmp -v framewise_data:/data -v "$PWD/backend/app:/app/app:ro" \
      -v "$PWD/analysis_experiments:/work:ro" -v "$PWD/$RUNS:/report/runs" "$IMAGE" /work/experiment.py all --dry --run-id dry-run "${@:2}"
    echo "Report: $URL?run=dry-run"
    ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
