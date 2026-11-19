#!/usr/bin/env bash
# Run the Phase A vision experiments with exclusive access to the inference GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
scratch=$(realpath "${1:?Pass the scratch directory containing capture.jsonl and images/}")
worker=${TARGET_EVAL_WORKER:-video-analyzer-inference-worker-1}
api=${TARGET_EVAL_API:-video-analyzer-video-analyzer-1}
repo=$PWD
benchmark=framewise-target-vision-eval-$$
compose=(docker compose -f compose.yml -f compose.inference.rocm.yml -f compose.dev.yml --profile inference)
test -r "$scratch/capture.jsonl"
test -w "$scratch"
test "$(docker inspect -f '{{.State.Running}}' "$worker")" = true

docker exec "$api" python -c '
import os, sqlite3, sys
with sqlite3.connect("file:" + os.environ.get("DATA_DIR", "/data") + "/app.sqlite3?mode=ro", uri=True) as db:
    active = []
    for table in ("jobs", "assistant_turns", "chat_gateway_requests"):
        if db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            count = db.execute(f"SELECT count(*) FROM {table} WHERE status IN (\x27queued\x27,\x27running\x27)").fetchone()[0]
            if count:
                active.append(f"{table} ({count})")
    if active:
        sys.exit("The inference queue is busy: " + ", ".join(active))
'

restore() {
    if [[ $(docker inspect -f '{{.State.Running}}' "$benchmark" 2>/dev/null || true) == true ]]; then
        docker stop -t 10 "$benchmark" >/dev/null
    fi
    docker start "$worker" >/dev/null
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker stop -t 60 "$worker" >/dev/null

for ((attempt=0; attempt<30; attempt++)); do
    enough=1
    for used in /sys/class/drm/card*/device/mem_info_vram_used; do
        if [[ -r $used ]] && (( $(<"$used") > 2147483648 )); then enough=0; fi
    done
    if (( enough )); then break; fi
    sleep 2
done
if (( ! enough )); then
    echo 'VRAM did not drain after stopping the inference worker.' >&2
    exit 1
fi

"${compose[@]}" run --rm --no-deps --name "$benchmark" --user "$(id -u):$(id -g)" \
    -v "$repo:/repo:ro" -v "$scratch:/eval" -e PYTHONPATH=/repo/backend -e TMPDIR=/tmp \
    --entrypoint python inference-worker -B /repo/scripts/target_vision_eval.py run --scratch /eval
