#!/usr/bin/env python3
"""Merge several runs (samples, keypoints, one Qwen config's answers, labels, frames if present) into a new run, for CPU re-derivation.

    python3 multiframe_experiment/tools/combine_runs.py --root frontend/public/multiframe-experiment/runs --out mf-new \\
        --config 'qwen35|observer|roster' mf-20261117-ep2pose mf-20261118-c160-170
    python3 multiframe_experiment/experiment.py rules  --run-id mf-new --root <root> --merge-with 'qwen35|observer|roster'
    python3 multiframe_experiment/experiment.py camera --run-id mf-new --root <root>   # optional, needs cv2 and frames: run it in the inference image
    python3 multiframe_experiment/report.py <root>/mf-new

Only the non-regenerable parts are merged (Qwen answers, labels); C and D are rebuilt by the `rules` stage. Later runs win on a repeated cut.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='qwen35|observer|roster', help='the Qwen config whose answers are kept')
    parser.add_argument('--only-labelled', action='store_true', help='keep only the cuts that have a hand label')
    args = parser.parse_args()
    root = Path(args.root)
    out = root / args.out
    if out.exists():
        raise SystemExit(f'{out} exists')
    (out / 'raw').mkdir(parents=True)
    (out / 'labels').mkdir()
    samples, pose, answers, config = {}, {'samples': {}}, {}, {}
    for name in args.runs:
        run = root / name
        for sample in json.loads((run / 'samples.json').read_text()):
            samples[sample['id']] = sample
        if (run / 'pose.json').exists():
            data = json.loads((run / 'pose.json').read_text())
            pose = {**data, 'samples': {**pose['samples'], **data['samples']}}
        if (run / 'raw' / 'answers-qwen35.jsonl').exists():
            for line in (run / 'raw' / 'answers-qwen35.jsonl').read_text().splitlines():
                record = json.loads(line)
                if record['config'] == args.config:
                    answers[record['key']] = line
        for label in (run / 'labels').glob('*.json') if (run / 'labels').exists() else []:
            shutil.copy(label, out / 'labels' / label.name)
        if (run / 'frames').exists():
            for folder in (run / 'frames').iterdir():
                shutil.copytree(folder, out / 'frames' / folder.name, dirs_exist_ok=True)
        config = json.loads((run / 'config.json').read_text())
    labelled = {p.stem for p in (out / 'labels').glob('*.json')}
    keep = [s for s in samples.values() if not args.only_labelled or s['id'] in labelled]
    ids = {s['id'] for s in keep}
    keep.sort(key=lambda s: s['cut_number'])
    (out / 'samples.json').write_text(json.dumps(keep, ensure_ascii=False))
    pose['samples'] = {k: v for k, v in pose['samples'].items() if k in ids}
    (out / 'pose.json').write_text(json.dumps(pose, ensure_ascii=False, default=list))
    (out / 'raw' / 'answers-qwen35.jsonl').write_text(''.join(line + '\n' for line in answers.values() if json.loads(line)['sample'] in ids))
    model, style, context = args.config.split('|')
    config.update({'cuts': ','.join(str(s['cut_number']) for s in keep), 'windows': sum(len(s['windows']) for s in keep if len(s['frames']) > 1), 'judge': False, 'pose': True,
                   'merge_with': [args.config], 'models': [model], 'styles': [style], 'contexts': [context], 'note': 'combined by tools/combine_runs.py from ' + ', '.join(args.runs)})
    (out / 'config.json').write_text(json.dumps(config))
    (out / 'progress.json').write_text(json.dumps({'run_id': args.out, 'status': 'done', 'stages': {}, 'started_at': '2026-11-18T00:00:00+00:00'}))
    for path in [out, *out.rglob('*')]:
        try:
            path.chmod(0o777 if path.is_dir() else 0o666)
        except OSError:
            pass
    print(f'{len(keep)} cuts, {sum(len(s["windows"]) for s in keep)} windows, {len(labelled & ids)} labelled cuts -> {out}')


if __name__ == '__main__':
    main()
