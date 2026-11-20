#!/usr/bin/env python3
"""Copies a run's verdict and ranking table into FINDINGS.md between the markers (stdlib). `findings_block.py <run dir> <FINDINGS.md>`."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

START, END = '<!-- verdict:start -->', '<!-- verdict:end -->'


def main(run_dir, findings):
    results = json.load(open(Path(run_dir) / 'results.json'))
    doc = report.Doc()
    header, rows = report.ranking_table(results)
    doc.table(header, rows[:14])
    t = results.get('recommended_time') or {}
    text = (f"**Run `{results['run_id']}` ({results['generated_at']}).** " + report.verdict(results) + '\n\n' + doc.md() +
            f"\nRecommended-pipeline time per 25-minute episode (Episode 3): {json.dumps(t.get('parts'))}, total {t.get('total_seconds')} s; current pipeline re-run: {t.get('baseline_seconds')} s.\n"
            f"Full tables, samples with audio and the spot-check page: `frontend/public/audio-experiment/runs/{results['run_id']}/`.\n")
    path = Path(findings)
    body = path.read_text()
    if START not in body:
        print('no markers in', findings)
        return 1
    a, b = body.index(START) + len(START), body.index(END)
    path.write_text(body[:a] + '\n' + text + body[b:])
    print('FINDINGS.md updated from', results['run_id'])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
