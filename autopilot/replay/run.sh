#!/bin/bash
# Run one replay scorer in a throwaway container of the API image, on a copy of the live database.
#   autopilot/replay/run.sh missed_cuts --project Seishun [--episodes "Episode 2,Episode 7"] [--sweep]
# The copy and the scene-score cache live in REPLAY_DIR (default /tmp/framewise-replay); media is mounted read-only;
# reports go to autopilot/replay/reports/. Nothing on the live service is written or restarted.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCORER=${1:?name a scorer, e.g. missed_cuts}; shift
WORK=${REPLAY_DIR:-/tmp/framewise-replay}
mkdir -p "$WORK/cache"
if [ ! -s "$WORK/app.sqlite3" ] || [ -n "${FRESH:-}" ]; then
  # A consistent copy through SQLite's backup API, made inside the API container, which can read the live file.
  docker compose -f "$ROOT/compose.yml" exec -T video-analyzer python3 -c "
import sqlite3; s=sqlite3.connect('file:/data/app.sqlite3?mode=ro',uri=True); d=sqlite3.connect('/tmp/replay-copy.sqlite3'); s.backup(d); d.close()"
  docker compose -f "$ROOT/compose.yml" cp video-analyzer:/tmp/replay-copy.sqlite3 "$WORK/app.sqlite3" >/dev/null
  docker compose -f "$ROOT/compose.yml" exec -T video-analyzer rm -f /tmp/replay-copy.sqlite3
fi
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)" \
  -v framewise_data:/data:ro -v "$ROOT/backend/app:/app/app:ro" -v "$ROOT/autopilot:/autopilot" -v "$WORK:/replay" \
  -w /app -e PYTHONPATH=/app --entrypoint python video-analyzer:local "/autopilot/replay/$SCORER.py" --out /autopilot/replay/reports "$@"
