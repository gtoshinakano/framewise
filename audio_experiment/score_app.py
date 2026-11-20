#!/usr/bin/env python3
"""Score the APP's current transcript against the experiment's labels (stdlib only, read-only HTTP).

  python3 audio_experiment/score_app.py [--api http://192.168.0.6:8080] [--labels audio_experiment/labels/round1.json]

For every timed label, the transcript rows inside its span vote by overlap for a character; no character counts as wrong (strict).
Use it before and after porting the new transcription flow: the experiment's round 2 reached 88 % overall (typical 97 %, hard 81 %);
the stored transcript of 2026-11-20 scores about 69 % (typical 94 %, hard 48 %) by the experiment's own measure.
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import names as N  # noqa: E402


def get(api, path):
    with urllib.request.urlopen(api.rstrip('/') + path, timeout=60) as response:
        return json.load(response)


def overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--api', default='http://192.168.0.6:8080')
    parser.add_argument('--labels', default=str(Path(__file__).resolve().parent / 'labels' / 'round1.json'))
    args = parser.parse_args()
    labels = json.load(open(args.labels))['labels']
    titles = {c['id']: c['title'] for c in get(args.api, f'/api/projects/{common.PROJECT_ID}/context') if c['kind'] == 'character'}
    totals = defaultdict(lambda: [0, 0])
    spelled = Counter()
    for n in sorted({l['episode'] for l in labels}):
        rows = [r for r in get(args.api, f'/api/episodes/{common.EPISODES[n]}/transcript') if not r.get('deleted') and not r.get('superseded_by')]
        for r in rows:
            spelled.update(N.hits(r['text']))
        for label in (l for l in labels if l['episode'] == n):
            votes = Counter()
            for r in rows:
                shared = overlap(label['start'], label['end'], r['start_time'], r['end_time'])
                if shared > 0:
                    votes[titles.get(r.get('character_context_id'))] += shared
            speaker = votes.most_common(1)[0][0] if votes else None
            for key in (label['set'], 'all'):
                totals[key][0] += 1
                totals[key][1] += speaker == label['speaker']
    for key, (n, right) in sorted(totals.items(), key=lambda kv: kv[0] == 'all'):
        print(f'{key:8} {right}/{n} = {right / n:.0%}')
    print('verified spellings in the transcript:', dict(spelled.most_common()))


if __name__ == '__main__':
    main()
