#!/usr/bin/env python3
"""Which hand-labelled pairs a config gets right or wrong, per body category (stdlib).

    python3 multiframe_experiment/tools/error_lists.py <run dir> [--config 'hybrid|observer|roster'] [-v] [-m]

Prints true positives, false positives and misses per category (walking, arms_hands, legs, body_pose, position), and with -v the false
claims (the character and the first description), with -m the missed pairs. Claims come from raw/answers-*.jsonl of the run (run the
`rules` stage first for hybrid and dwpose configs). A claim counts for a pair and category the same way scoring.ground_truth counts it.
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import schema  # noqa: E402

BODY = ('walking', 'arms_hands', 'legs', 'body_pose', 'position')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--config', default='hybrid|observer|roster')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-m', action='store_true')
    args = parser.parse_args()
    run = Path(args.run)
    labels = {}
    for path in glob.glob(str(run / 'labels' / '*.json')):
        data = json.loads(Path(path).read_text())
        for pair in data['pairs']:
            labels[(data['sample'], pair['from'], pair['to'])] = pair
    file = {'hybrid': 'hybrid', 'dwpose': 'dwpose'}.get(args.config.split('|')[0], args.config.split('|')[0])
    claims = {}
    for line in (run / 'raw' / f'answers-{file}.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record['config'] != args.config or not record.get('answer'):
            continue
        for segment in record['answer']['segments']:
            for person in segment['characters']:
                cats = ({'walking'} if person['walking'] == 'yes' else set()) | {schema.PART_CATEGORY[c['part']] for c in person['changes']}
                for cat in cats:
                    first = next((c['description'] for c in person['changes'] if schema.PART_CATEGORY[c['part']] == cat), '')
                    claims.setdefault((record['sample'], segment['from'], segment['to']), {}).setdefault(cat, []).append((person['name'], first[:80]))
    tp, fp, fn = (collections.Counter() for _ in range(3))
    for key, label in labels.items():
        for cat in BODY:
            got = cat in claims.get(key, {})
            truth = label['flags'][cat]
            if got and truth:
                tp[cat] += 1
            elif got:
                fp[cat] += 1
                if args.v:
                    print('FP', key, cat, claims[key][cat][:3])
            elif truth:
                fn[cat] += 1
                if args.m:
                    print('FN', key, cat, '|', label['note'][:90])
    print(f'{args.config}: TP {dict(tp)} FP {dict(fp)} FN {dict(fn)} over {len(labels)} labelled pairs')


if __name__ == '__main__':
    main()
