#!/usr/bin/env bash
# Put an archived run back under frontend/public/multiframe-experiment/runs/ so the CPU stages and the report work on it:
#   multiframe_experiment/archive/restore.sh mf-20261118-final [new-run-id]
#   python3 multiframe_experiment/experiment.py rules --run-id <id> --root frontend/public/multiframe-experiment/runs --merge-with 'qwen35|observer|roster'
#   python3 multiframe_experiment/report.py frontend/public/multiframe-experiment/runs/<id>
# Frames (images) are not archived: the report shows no thumbnails, and `prepare` is needed again only to change the cuts or the keyframes.
set -euo pipefail
cd "$(dirname "$0")/../.."
src="multiframe_experiment/archive/runs/${1:?run id}"
dst="frontend/public/multiframe-experiment/runs/${2:-$1}"
[ -d "$src" ] || { echo "no archived run $1 (see multiframe_experiment/archive/README.md)"; exit 1; }
[ ! -e "$dst" ] || { echo "$dst exists; pass another run id"; exit 1; }
mkdir -p "$dst"
cp -r "$src"/. "$dst"/
python3 - "$dst" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
(d / 'progress.json').write_text(json.dumps({'run_id': d.name, 'status': 'done', 'stages': {}, 'started_at': '2026-11-18T00:00:00+00:00'}))
PY
chmod -R a+rwX "$dst"
echo "restored to $dst"
