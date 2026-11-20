#!/usr/bin/env bash
# Launcher for the audio experiment (see audio_experiment/README.md).
#
#   run.sh start [--run-id ID | --resume ID] [--hours N]   copy the database, run every phase unattended (workers stopped for the GPU phases)
#   run.sh status                                          phase, containers, workers, VRAM, progress
#   run.sh stop                                            stop the run; both workers are started again
#   run.sh dry-run                                         whole pipeline with fake models on the CPU over a 3.5-minute excerpt (workers untouched)
#   run.sh report [ID]                                     rebuild report.html / report.md / spotcheck.html on the host (stdlib python3)
#   run.sh serve                                           serve the reports on the LAN (http.server, port 8095)
#   run.sh round2 [--base ID] [--run-id ID | --resume ID]  round 2 on top of a finished round-1 run (about 40 min; workers stopped for its GPU phases)
#   run.sh round2-dry                                      round 2 with fakes on the CPU, on top of the dry run
#   run.sh report2 [ID] [spotcheck-export.json ...]        rebuild a round-2 report on the host, with labels/*.json and any exports of its spot check
#
# One GPU process at a time: this host's kernel crashed once with two. `start` requires an idle queue, stops the inference worker AND the audio
# worker, waits for the VRAM to drain, and runs the phases one container after the other (audio models, then Qwen, ...). A detached watcher starts
# both workers again however the run ends.
set -uo pipefail
cd "$(dirname "$0")/.."

INFER=video-analyzer-inference-worker-1
AUDIO=video-analyzer-audio-worker-1
API=video-analyzer-video-analyzer-1
EXP=framewise-audio-experiment
IMAGE=framewise-audio-rocm:local
RUNS=frontend/public/audio-experiment/runs
WORK=audio_experiment/work
VRAM_LIMIT_MIB=4096
PORT=8095
HOST_IP=$(grep -E '^APP_BIND_IP=' .env 2>/dev/null | cut -d= -f2 || true)
HOST_IP=${HOST_IP:-192.168.0.6}
BASE="http://${HOST_IP}:${PORT}/runs"
PHASES_MINUTES="prepare:40 glossary:25 audio:130 qwen:240 second:40 lfm3:30 final:20"
PHASES2_MINUTES="music:30 voice:30 qwen2:120 final2:20"
DEFAULT_HOURS=6.9

vram_mib() {
  local max=0 value
  for file in /sys/class/drm/card*/device/mem_info_vram_used; do
    [ -r "$file" ] || continue
    value=$(( $(cat "$file") / 1048576 )); [ "$value" -gt "$max" ] && max=$value
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

start_workers() {
  for w in "$AUDIO" "$INFER"; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$w" 2>/dev/null)" != "true" ]; then docker start "$w" >/dev/null 2>&1 && echo "$(date +%T) started $w"; fi
  done
}

note() {  # note <run id> <kind> <text>: appended to the run's notes.json from the host (stdlib)
  python3 - "$RUNS/$1" "$2" "$3" <<'PY'
import json, sys, time, os
d, kind, text = sys.argv[1:4]
p = os.path.join(d, 'notes.json')
try: notes = json.load(open(p))
except Exception: notes = []
notes.append({'kind': kind, 'text': text, 'at': time.strftime('%Y-%m-%d %H:%M:%S')})
json.dump(notes, open(p, 'w'), ensure_ascii=False)
os.chmod(p, 0o666)
PY
}

docker_env() {
  printf -- "-e HF_HUB_OFFLINE=1 -e HF_HOME=/models/cache/huggingface -e PYANNOTE_METRICS_ENABLED=0 -e MPLCONFIGDIR=/tmp/mpl -e XDG_CACHE_HOME=/tmp/xdg -e XDG_CONFIG_HOME=/tmp/xdgc -e TMPDIR=/tmp -e DATA_DIR=/data -e APP_PARENT=/app -e PYTHONDONTWRITEBYTECODE=1 -e AEX_DB=/priv/app.sqlite3 -e AEX_RUNS=/report/runs -e QWEN35_SERVER_PATH=/models/llamacpp/llama-server -e QWEN36_MODEL_PATH=/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf -e LFM3_OMNI_MODEL_PATH=/models/lfm3-omni-2b/LFM3-Omni-2B-BF16.gguf -e LFM3_OMNI_PROJECTOR_PATH=/models/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf -e QWEN36_THINKING_BUDGET=2048 -e QWEN_PARALLEL=3"
}

docker_mounts() {  # docker_mounts <run id>
  printf -- "-v framewise_data:/data:ro -v framewise_models:/models/cache:ro -v %s/audio_experiment:/work:ro -v %s/backend/app:/app/app:ro -v %s/%s/%s:/priv:ro -v %s/%s:/report/runs -v %s/.models/llamacpp-runtime:/models/llamacpp:ro -v %s/.models/qwen3.6-35b:/models/qwen3.6-35b:ro -v %s/.models/lfm3-omni-2b:/models/lfm3-omni-2b:ro" \
    "$PWD" "$PWD" "$PWD" "$WORK" "$1" "$PWD" "$RUNS" "$PWD" "$PWD" "$PWD"
}

prepare_dirs() {
  mkdir -p "$RUNS" "$WORK/$1"
  chmod 777 "$RUNS" "$(dirname "$RUNS")" "$WORK" "$WORK/$1" 2>/dev/null || true
}

copy_db() {  # copy_db <run id>: a consistent copy through sqlite's backup API, source opened read-only; nothing is written to the live volume
  [ -s "$WORK/$1/app.sqlite3" ] && { echo "database copy already there"; return 0; }
  docker exec -i "$API" python - <<'PY' || return 1
import sqlite3
src = sqlite3.connect('file:/data/app.sqlite3?mode=ro', uri=True)
dst = sqlite3.connect('/tmp/aex-copy.sqlite3')
src.backup(dst); dst.execute('PRAGMA journal_mode=DELETE'); dst.close(); src.close()
PY
  docker cp "$API:/tmp/aex-copy.sqlite3" "$WORK/$1/app.sqlite3" && docker exec "$API" rm -f /tmp/aex-copy.sqlite3
  chmod 644 "$WORK/$1/app.sqlite3"
}

serve() {
  if ! curl -s -o /dev/null --max-time 2 "http://${HOST_IP}:${PORT}/"; then
    mkdir -p frontend/public/audio-experiment
    nohup setsid python3 -m http.server "$PORT" --bind "$HOST_IP" --directory frontend/public/audio-experiment >/tmp/audio-experiment-http.log 2>&1 < /dev/null &
    sleep 1; echo "serving $BASE on the LAN (the app on :8080 only shows it after an image rebuild)"
  fi
}

# ---------------------------------------------------------------- the detached driver
drive() {  # drive <run id> <end epoch> [excerpt args...]
  local id=$1 end=$2; shift 2
  local extra=("$@") state="$RUNS/$id/driver.state"
  trap 'docker rm -f '"$EXP"' >/dev/null 2>&1; start_workers; echo "$(date +%T) driver ended" >>"'"$RUNS/$id"'/log.txt"' EXIT
  say() { echo "$(date +%T) $*" | tee -a "$RUNS/$id/log.txt"; }
  local abort_gpu=0
  gpu_ids="--group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3)"
  for item in ${AEX_PHASES:-$PHASES_MINUTES}; do
    local phase=${item%%:*} cap=${item##*:} now deadline
    echo "$phase" >"$state"
    now=$(date +%s)
    if [ "$now" -ge "$end" ]; then say "phase $phase skipped: the time budget of the run is used up"; note "$id" skip "phase $phase skipped: the time budget of the run is used up"; continue; fi
    deadline=$(( now + cap * 60 )); [ "$deadline" -gt "$end" ] && deadline=$end
    local devices=""
    case "$phase" in
      prepare|final|final2)
        [ "$phase" != prepare ] && start_workers ;;
      *)
        if [ "$abort_gpu" = 1 ]; then say "phase $phase skipped: the GPU phases were aborted"; continue; fi
        devices="--device /dev/kfd --device /dev/dri $gpu_ids --security-opt seccomp=unconfined"
        if [ "$(docker inspect -f '{{.State.Running}}' "$INFER" 2>/dev/null)" = "true" ] || [ "$(docker inspect -f '{{.State.Running}}' "$AUDIO" 2>/dev/null)" = "true" ]; then
          say "waiting for an idle queue before stopping the workers"
          for _ in $(seq 1 180); do queue_idle >/dev/null && break; sleep 60; done
          if ! queue_idle >/dev/null; then say "the queue never became idle; aborting before the GPU phases"; note "$id" fatal "the queue never became idle; the GPU phases did not run"; abort_gpu=1; continue; fi
          docker stop -t 60 "$INFER" "$AUDIO" >/dev/null 2>&1
        fi
        for _ in $(seq 1 90); do [ "$(vram_mib)" -lt "$VRAM_LIMIT_MIB" ] && break; sleep 2; done
        if [ "$(vram_mib)" -ge "$VRAM_LIMIT_MIB" ]; then say "VRAM did not drain ($(vram_mib) MiB); aborting"; note "$id" fatal "VRAM did not drain; the GPU phases did not run"; abort_gpu=1; continue; fi ;;
    esac
    say "=== phase $phase (cap ${cap} min, VRAM $(vram_mib) MiB) ==="
    # shellcheck disable=SC2046
    timeout -s TERM $(( cap + 8 ))m docker run --rm --init --name "$EXP" --entrypoint python $devices $(docker_env) $(docker_mounts "$id") "$IMAGE" \
      /work/experiment.py phase "$phase" --run-id "$id" --deadline "$deadline" "${extra[@]}" >>"$RUNS/$id/driver.log" 2>&1
    local code=$?
    docker rm -f "$EXP" >/dev/null 2>&1
    if [ "$code" -ne 0 ]; then say "phase $phase exited with $code"; note "$id" warn "phase $phase exited with code $code (see driver.log); later phases ran with what existed"; fi
    sleep 5
  done
  echo done >"$state"
  say "all phases finished"
}

case "${1:-}" in
  start)
    shift
    RUN_ID="audio-$(date +%Y%m%d-%H%M)"; HOURS=$DEFAULT_HOURS; RESUME=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --run-id) RUN_ID=$2; shift 2 ;;
        --resume) RUN_ID=$2; RESUME=1; shift 2 ;;
        --hours) HOURS=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    if docker ps -a --format '{{.Names}}' | grep -qx "$EXP"; then
      [ "$(docker inspect -f '{{.State.Running}}' "$EXP")" = "true" ] && { echo "An experiment is already running: $0 status"; exit 1; }
      docker rm "$EXP" >/dev/null
    fi
    if [ -f "$RUNS/$RUN_ID/driver.pid" ] && kill -0 "$(cat "$RUNS/$RUN_ID/driver.pid")" 2>/dev/null; then echo "run $RUN_ID is still driven by pid $(cat "$RUNS/$RUN_ID/driver.pid")"; exit 1; fi
    if [ "$RESUME" = 0 ] && [ -d "$RUNS/$RUN_ID" ]; then echo "run id $RUN_ID exists; use --resume $RUN_ID"; exit 1; fi
    if ! state=$(queue_idle); then echo "queue: $state. Try again when it is idle."; exit 1; fi
    echo "queue: $state"
    docker run --rm --entrypoint sh -v "$PWD/.models:/m:ro" "$IMAGE" -c 'test -f /m/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf && test -x /m/llamacpp-runtime/llama-server' \
      || { echo "The Qwen3.6 model file or llama-server is missing under .models."; exit 1; }
    prepare_dirs "$RUN_ID"
    echo "copying the database (read-only backup) ..."
    copy_db "$RUN_ID" || { echo "could not copy the database"; exit 1; }
    mkdir -p "$RUNS/$RUN_ID"; chmod 777 "$RUNS/$RUN_ID"
    END=$(( $(date +%s) + $(python3 -c "print(int($HOURS*3600))") ))
    serve
    nohup setsid bash "$0" _drive "$RUN_ID" "$END" >>"$RUNS/$RUN_ID/driver.log" 2>&1 < /dev/null &
    DRIVER=$!
    echo "$DRIVER" >"$RUNS/$RUN_ID/driver.pid"; chmod 666 "$RUNS/$RUN_ID/driver.pid"
    # armed at once: both workers come back when the driver is gone, whatever way it ended
    nohup setsid bash -c "while kill -0 $DRIVER 2>/dev/null; do sleep 15; done; docker rm -f $EXP >/dev/null 2>&1; sleep 3; docker start $AUDIO $INFER >/dev/null 2>&1" \
      >"audio_experiment/watcher.log" 2>&1 < /dev/null &
    echo
    echo "Run id:    $RUN_ID"
    echo "Report:    $BASE/$RUN_ID/report.html      (spot check: $BASE/$RUN_ID/spotcheck.html)"
    echo "Hard stop: $(date -d "@$END" '+%Y-%m-%d %H:%M')   (expected to finish in about 4-5 h)"
    echo "Status:    $0 status        Stop: $0 stop        Resume: $0 start --resume $RUN_ID"
    echo "Both workers stay stopped during the GPU phases and are started again by the driver (and by a watcher if the driver dies)."
    ;;
  _drive)
    shift; drive "$@" ;;
  round2)
    shift
    RUN_ID="audio-r2-$(date +%Y%m%d-%H%M)"; HOURS=3; RESUME=0; BASE_ID=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --base) BASE_ID=$2; shift 2 ;;
        --run-id) RUN_ID=$2; shift 2 ;;
        --resume) RUN_ID=$2; RESUME=1; shift 2 ;;
        --hours) HOURS=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    [ -z "$BASE_ID" ] && [ -f "$RUNS/$RUN_ID/base.txt" ] && BASE_ID=$(cat "$RUNS/$RUN_ID/base.txt")
    [ -z "$BASE_ID" ] && BASE_ID=$(ls -t "$RUNS"/*/results.json 2>/dev/null | grep -v '/dry-run' | head -1 | xargs -r dirname | xargs -r basename)
    [ -n "$BASE_ID" ] && [ -f "$RUNS/$BASE_ID/prep.json" ] && [ -f "$RUNS/$BASE_ID/ep3/qwen/Q1.json" ] || { echo "no finished round-1 run to build on (--base ID)"; exit 1; }
    if docker ps -a --format '{{.Names}}' | grep -qx "$EXP"; then
      [ "$(docker inspect -f '{{.State.Running}}' "$EXP")" = "true" ] && { echo "An experiment is already running: $0 status"; exit 1; }
      docker rm "$EXP" >/dev/null
    fi
    if [ -f "$RUNS/$RUN_ID/driver.pid" ] && kill -0 "$(cat "$RUNS/$RUN_ID/driver.pid")" 2>/dev/null; then echo "run $RUN_ID is still driven by pid $(cat "$RUNS/$RUN_ID/driver.pid")"; exit 1; fi
    if [ "$RESUME" = 0 ] && [ -d "$RUNS/$RUN_ID" ]; then echo "run id $RUN_ID exists; use --resume $RUN_ID"; exit 1; fi
    if ! state=$(queue_idle); then echo "queue: $state. Try again when it is idle."; exit 1; fi
    echo "queue: $state"
    docker run --rm --entrypoint sh -v "$PWD/.models:/m:ro" "$IMAGE" -c 'test -f /m/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf && test -x /m/llamacpp-runtime/llama-server && test -f /m/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf' \
      || { echo "The Qwen3.6 or LFM3-Omni model file or llama-server is missing under .models."; exit 1; }
    prepare_dirs "$RUN_ID"
    mkdir -p "$RUNS/$RUN_ID"; chmod 777 "$RUNS/$RUN_ID"
    echo "$BASE_ID" >"$RUNS/$RUN_ID/base.txt"; chmod 666 "$RUNS/$RUN_ID/base.txt"
    END=$(( $(date +%s) + $(python3 -c "print(int($HOURS*3600))") ))
    serve
    AEX_PHASES="$PHASES2_MINUTES" nohup setsid bash "$0" _drive "$RUN_ID" "$END" --base "$BASE_ID" >>"$RUNS/$RUN_ID/driver.log" 2>&1 < /dev/null &
    DRIVER=$!
    echo "$DRIVER" >"$RUNS/$RUN_ID/driver.pid"; chmod 666 "$RUNS/$RUN_ID/driver.pid"
    nohup setsid bash -c "while kill -0 $DRIVER 2>/dev/null; do sleep 15; done; docker rm -f $EXP >/dev/null 2>&1; sleep 3; docker start $AUDIO $INFER >/dev/null 2>&1" \
      >"audio_experiment/watcher.log" 2>&1 < /dev/null &
    echo
    echo "Round 2:   $RUN_ID   (base $BASE_ID)"
    echo "Report:    $BASE/$RUN_ID/report.html      (spot check: $BASE/$RUN_ID/spotcheck.html)"
    echo "Hard stop: $(date -d "@$END" '+%Y-%m-%d %H:%M')   (expected to finish in about 40 min)"
    echo "Status:    $0 status        Stop: $0 stop        Resume: $0 round2 --resume $RUN_ID"
    ;;
  round2-dry)
    shift
    [ -f "$RUNS/dry-run/prep.json" ] || { echo "run $0 dry-run first: round 2 builds on it"; exit 1; }
    prepare_dirs dry-run-r2
    docker run --rm --entrypoint sh -v "$PWD/$RUNS:/runs" "$IMAGE" -c 'rm -rf /runs/dry-run-r2'
    mkdir -p "$RUNS/dry-run-r2"; chmod 777 "$RUNS/dry-run-r2"; echo dry-run >"$RUNS/dry-run-r2/base.txt"
    # shellcheck disable=SC2046
    docker run --rm --init --entrypoint python $(docker_env) $(docker_mounts dry-run-r2) "$IMAGE" /work/experiment.py all2 --dry --run-id dry-run-r2 --base dry-run "$@"
    code=$?
    serve
    echo "Report: $BASE/dry-run-r2/report.html"
    exit $code
    ;;
  report2)
    id=${2:-}
    [ -z "$id" ] && id=$(ls -t "$RUNS"/*/base.txt 2>/dev/null | head -1 | xargs -r dirname | xargs -r basename)
    [ -n "$id" ] && [ -f "$RUNS/$id/base.txt" ] || { echo "no round-2 run found"; exit 1; }
    args=(); for f in audio_experiment/labels/*.json; do [ -f "$f" ] && args+=(--labels "$f"); done
    for f in "${@:3}"; do args+=(--id-labels "$f"); done
    python3 audio_experiment/round2_analysis.py "$RUNS/$id" --base "$RUNS/$(cat "$RUNS/$id/base.txt")" "${args[@]}"
    serve
    echo "Report: $BASE/$id/report.html"
    ;;
  status)
    latest=$(ls -t "$RUNS"/*/progress.json 2>/dev/null | grep -v dry-run | head -1 || true)
    id=""; [ -n "$latest" ] && id=$(basename "$(dirname "$latest")")
    echo "experiment container: $(docker inspect -f '{{.State.Status}} (exit {{.State.ExitCode}})' "$EXP" 2>/dev/null || echo none)"
    echo "inference worker:     $(docker inspect -f '{{.State.Status}}' "$INFER" 2>/dev/null || echo unknown)"
    echo "audio worker:         $(docker inspect -f '{{.State.Status}}' "$AUDIO" 2>/dev/null || echo unknown)"
    echo "VRAM in use:          $(vram_mib) MiB"
    if [ -n "$id" ]; then
      pid=$(cat "$RUNS/$id/driver.pid" 2>/dev/null || true)
      echo "latest run:           $id   driver: $( [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "running (pid $pid)" || echo "not running")   phase: $(cat "$RUNS/$id/driver.state" 2>/dev/null || echo -)"
      python3 - "$latest" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print('status:', p.get('status'), '| updated', p.get('updated_at'), '| elapsed', round((p.get('elapsed_seconds') or 0) / 60), 'min')
stages = p.get('stages', {})
for name, s in stages.items():
    if s.get('status') != 'done' or '.' not in name:
        print(f"  {name:<22} {s.get('status'):<8} {s.get('done', 0)}/{s.get('total', 0)} {s.get('message') or ''}")
PY
      tail -n 5 "$RUNS/$id/log.txt" 2>/dev/null | cut -c1-200
      echo "report: $BASE/$id/report.html"
    fi
    ;;
  stop)
    for pidfile in "$RUNS"/*/driver.pid; do
      [ -f "$pidfile" ] || continue
      pid=$(cat "$pidfile"); kill -0 "$pid" 2>/dev/null && { echo "stopping driver $pid"; pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; }
    done
    docker ps --format '{{.Names}}' | grep -qx "$EXP" && docker stop -t 30 "$EXP" >/dev/null && echo "experiment container stopped"
    sleep 4; start_workers
    ;;
  dry-run)
    shift
    prepare_dirs dry-run
    docker run --rm --entrypoint sh -v "$PWD/$RUNS:/runs" "$IMAGE" -c 'rm -rf /runs/dry-run'
    mkdir -p "$RUNS/dry-run"; chmod 777 "$RUNS/dry-run"
    copy_db dry-run || exit 1
    # shellcheck disable=SC2046
    docker run --rm --init --entrypoint python $(docker_env) $(docker_mounts dry-run) "$IMAGE" /work/experiment.py all --dry --run-id dry-run --excerpt 20 230 "$@"
    code=$?
    serve
    echo "Report: $BASE/dry-run/report.html"
    exit $code
    ;;
  report)
    id=${2:-}
    [ -z "$id" ] && id=$(ls -t "$RUNS" | grep -v '^index' | head -1)
    python3 audio_experiment/report.py "$RUNS/$id" ${3:+--labels "$3"} && { [ "$id" = dry-run ] || python3 audio_experiment/findings_block.py "$RUNS/$id" audio_experiment/FINDINGS.md; }
    serve
    echo "Report: $BASE/$id/report.html"
    ;;
  serve)
    serve; echo "$BASE/index.html" ;;
  *)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
