#!/usr/bin/env python3
"""Multi-frame change experiment: which prompt and context make Qwen3.5-27B / LFM3-Omni-2B describe the motion between
keyframes best, and at what cost?

    python experiment.py all --run-id mf-test --episode "Episode 2" --cuts 139-158      # everything
    python experiment.py prepare|infer|judge|report --run-id mf-test [--model lfm3]       # one stage
    python experiment.py all --dry --run-id dry-run                                       # fake models, no GPU
    python experiment.py labels-template --run-id mf-test                                 # ground-truth skeletons

Results land in ``<root>/<run-id>/`` (default root ``/report/runs`` in the container, which is
``frontend/public/multiframe-experiment/runs`` on the host): samples.json, frames/, raw/*.jsonl, progress.json,
results.json, report.html, report.md. A repeated run id resumes: finished calls are skipped.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for candidate in (os.environ.get('APP_ROOT'), '/app', str(HERE.parent / 'backend')):
    if candidate and (Path(candidate) / 'app' / '__init__.py').is_file():
        sys.path.append(candidate)
        break

import infer  # noqa: E402
import prompts  # noqa: E402
import report  # noqa: E402

log = logging.getLogger('mfx')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Run:
    def __init__(self, root, run_id):
        self.root, self.id = Path(root), run_id
        self.dir = self.root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        for path in (self.root, self.dir):
            try:
                os.chmod(path, 0o777)
            except OSError:
                pass
        self.lock = threading.Lock()

    def read(self, name, default=None):
        try:
            return json.loads((self.dir / name).read_text())
        except (OSError, ValueError):
            return default

    def write(self, name, data):
        with self.lock:
            temp = self.dir / (name + '.tmp')
            temp.write_text(json.dumps(data, ensure_ascii=False))
            os.chmod(temp, 0o666)
            os.replace(temp, self.dir / name)


class Progress:
    def __init__(self, run, dry):
        self.run, self.lock = run, threading.Lock()
        self.data = run.read('progress.json') or {'run_id': run.id, 'dry': dry, 'status': 'starting', 'started_at': now(), 'stages': {}}
        self.began = time.time()

    def reload(self):
        self.data = self.run.read('progress.json') or self.data

    def set(self, stage=None, status=None, done=None, total=None, message=None):
        with self.lock:
            self.data['updated_at'] = now()
            if status and stage is None:
                self.data['status'] = status
            if stage:
                item = self.data['stages'].setdefault(stage, {'status': 'pending', 'done': 0, 'total': 0})
                item.update({k: v for k, v in (('status', status), ('done', done), ('total', total), ('message', message)) if v is not None})
                if status == 'running' and 'started_at' not in item:
                    item['started_at'] = now()
                    item['_t0'] = time.time()
                if status in ('done', 'failed') and item.get('_t0'):
                    item['seconds'] = round(time.time() - item['_t0'], 1)
                self.data['current'] = stage
            try:
                started = datetime.fromisoformat(self.data['started_at']).timestamp()
                self.data['elapsed_seconds'] = round(time.time() - started)
            except (KeyError, ValueError):
                pass
            self.run.write('progress.json', self.data)


def rebuild(run, quiet=True, every=0.0, _last=[0.0]):  # noqa: B006 - deliberate throttle state
    if every and time.monotonic() - _last[0] < every:
        return
    _last[0] = time.monotonic()
    try:
        report.build(run.dir)
        report.build_index(run.root)
    except Exception:  # noqa: BLE001 - a report hiccup must never stop the run
        if not quiet:
            raise
        log.exception('report build failed')


# -- stages -----------------------------------------------------------------------------------------------------------------

def stage_prepare(args, run, progress):
    import measure
    import sampler
    progress.set('prepare', 'running', message='Reading cuts and keyframes')
    samples = run.read('samples.json')
    if not samples or args.resample:
        overrides = dict(args.kf or [])
        samples, meta = sampler.build(Path(args.data_dir) / 'app.sqlite3', args.data_dir, args.episode, args.cuts, run.dir, overrides, args.window, args.image_width, log=log.info)
        for number, sample in enumerate(samples, 1):
            sample['pairs'] = measure.measure_sample(sample, run.dir)
            progress.set('prepare', 'running', number, len(samples), f"Measured cut {sample['cut_number']}")
        run.write('samples.json', samples)
        run.write('config.json', {'created_at': now(), 'dry': args.dry, 'episode': meta['episode'], 'episode_id': meta['episode_id'], 'cuts': args.cuts,
                                  'missing_cuts': meta['missing'], 'kf_overrides': overrides, 'window_size': args.window, 'image_width': args.image_width,
                                  'models': args.models, 'styles': args.styles, 'contexts': args.contexts, 'judge': not args.no_judge,
                                  'grammar': not args.no_grammar, 'max_attempts': infer.MAX_ATTEMPTS, 'judge_parallel': args.judge_parallel,
                                  'windows': sum(len(s['windows']) for s in samples if len(s['frames']) > 1),
                                  'model_labels': {k: infer.MODELS[k]['label'] for k in args.models}})
    for path in [run.dir / 'frames', *(run.dir / 'frames').glob('*')]:
        try:
            os.chmod(path, 0o777)
        except OSError:
            pass
    progress.set('prepare', 'done', len(samples), len(samples), f'{len(samples)} cuts')
    rebuild(run)


def _server(model):
    from app import frame_model, visual_models
    vm = visual_models.MODELS[infer.MODELS[model]['visual_key']]
    return vm, lambda parallel: frame_model.frame_server(*map(str, vm.server_files()), startup_timeout=900, parallel=parallel,
                                                          context_per_slot=infer.MODELS[model]['context_per_slot'], extra_args=vm.server_args)


def stage_infer(args, run, progress):
    samples = run.read('samples.json', [])
    combos = [(style, context) for style in args.styles for context in args.contexts]
    for model in ([args.model] if args.model else args.models):
        stage = f'infer:{model}'
        store = infer.Store(run.dir, f'answers-{model}')
        progress.set(stage, 'running', message=f'Starting {infer.MODELS[model]["label"]}')

        def tick(done, total, message, stage=stage):
            progress.set(stage, 'running', done, total, message)
            rebuild(run, every=20)

        if args.dry:
            import fake_backend
            infer.run_model(store, samples, model, combos, fake_backend.FakeClient(), run.dir, not args.no_grammar, tick)
        else:
            _, server = _server(model)
            began = time.monotonic()
            with server(1) as url:
                startup = round(time.monotonic() - began, 1)
                log.info('%s server ready in %.1fs', model, startup)
                progress.data['stages'][stage]['server_startup_seconds'] = startup
                infer.run_model(store, samples, model, combos, infer.LlamaClient(url), run.dir, not args.no_grammar, tick)
        progress.set(stage, 'done', message=f'{len(store.records)} calls')
        rebuild(run)


def stage_judge(args, run, progress):
    import judge
    samples = run.read('samples.json', [])
    answers = report.load_answers(run.dir)
    store = infer.Store(run.dir, 'judge')
    progress.set('judge', 'running', message='Starting the Qwen3.5-27B judge')

    def tick(done, total, message):
        progress.set('judge', 'running', done, total, message)
        rebuild(run, every=20)

    if args.dry:
        import fake_backend
        judge.run_judge(store, samples, answers, fake_backend.FakeClient(), run.dir, 1, tick)
    else:
        _, server = _server('qwen35')
        with server(args.judge_parallel) as url:
            judge.run_judge(store, samples, answers, infer.LlamaClient(url, timeout=1800), run.dir, args.judge_parallel, tick)
    progress.set('judge', 'done', message=f'{len(store.records)} judge calls')
    rebuild(run)


def stage_report(args, run, progress):
    rebuild(run, quiet=False)
    print(run.dir / 'report.html')


def stage_labels(args, run, progress):
    """labels/<sample>.json skeletons: per pair, every reference flag false; a person or agent sets what really changes."""
    import schema
    folder = run.dir / 'labels'
    folder.mkdir(exist_ok=True)
    for sample in run.read('samples.json', []):
        target = folder / f"{sample['id']}.json"
        if target.exists():
            continue
        pairs = [{'from': p['from'], 'to': p['to'], 'start': p['start'], 'end': p['end'], 'camera': None,
                  'flags': {flag: False for flag in schema.REFERENCE_FLAGS}, 'note': ''} for p in sample.get('pairs') or []]
        target.write_text(json.dumps({'sample': sample['id'], 'cut_number': sample['cut_number'], 'frames': [f['path'] for f in sample['frames']], 'pairs': pairs}, indent=2))
    print(folder)


STAGES = {'prepare': stage_prepare, 'infer': stage_infer, 'judge': stage_judge, 'report': stage_report, 'labels-template': stage_labels}


def child_args(args, run):
    items = ['--run-id', run.id, '--root', str(run.root), '--data-dir', args.data_dir, '--models', *args.models, '--styles', *args.styles,
             '--contexts', *args.contexts, '--judge-parallel', str(args.judge_parallel)]
    for flag, value in (('--dry', args.dry), ('--no-grammar', args.no_grammar), ('--no-judge', args.no_judge)):
        if value:
            items.append(flag)
    return items


def stage_all(args, run, progress):
    progress.set(status='running')
    try:
        stage_prepare(args, run, progress)
    except Exception:
        log.exception('prepare failed')
        progress.set('prepare', 'failed')
        progress.set(status='failed')
        rebuild(run)
        return 1
    failed = []
    steps = [('infer', ['--model', model], f'infer:{model}') for model in args.models]
    if not args.no_judge:
        steps.append(('judge', [], 'judge'))
    for stage, extra, name in steps:
        # Each GPU stage is its own process so the model's memory is released completely before the next one.
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), stage, *child_args(args, run), *extra])
        progress.reload()
        if code:
            failed.append(name)
            progress.set(name, 'failed')
            log.error('stage %s ended with exit code %s', name, code)
    progress.set(status='partial' if failed else 'done', message=('failed: ' + ', '.join(failed)) if failed else None)
    rebuild(run)
    return 1 if failed else 0


def _kv(text):
    key, _, value = text.partition('=')
    if not key or not value:
        raise argparse.ArgumentTypeError('use key=value')
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['all', *STAGES])
    parser.add_argument('--run-id')
    parser.add_argument('--root', default=os.environ.get('MFX_RUNS_ROOT', '/report/runs'))
    parser.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '/data'))
    parser.add_argument('--episode', default='Episode 2', help='episode title or id')
    parser.add_argument('--cuts', default='139-158', help='UI cut numbers, e.g. "139-158" or "12,40-45"')
    parser.add_argument('--window', type=int, default=5, help='frames per model call (windows overlap by one frame)')
    parser.add_argument('--image-width', type=int, default=1024)
    parser.add_argument('--kf', type=_kv, action='append', help='keyframe selector override, e.g. --kf budget=10 --kf event_threshold=0.1 (recomputes on the CPU)')
    parser.add_argument('--models', nargs='+', default=['lfm3', 'qwen35'], choices=list(infer.MODELS))
    parser.add_argument('--model', choices=list(infer.MODELS), help='infer stage: only this model')
    parser.add_argument('--styles', nargs='+', default=list(prompts.STYLES), choices=list(prompts.STYLES))
    parser.add_argument('--contexts', nargs='+', default=list(prompts.CONTEXTS), choices=list(prompts.CONTEXTS))
    parser.add_argument('--judge-parallel', type=int, default=2)
    parser.add_argument('--no-judge', action='store_true')
    parser.add_argument('--no-grammar', action='store_true', help='json_object instead of the json_schema grammar')
    parser.add_argument('--resample', action='store_true', help='rebuild samples.json even if it exists')
    parser.add_argument('--dry', action='store_true', help='fake models, no GPU')
    args = parser.parse_args(argv)
    run_id = args.run_id or ('mf-' + datetime.now().strftime('%Y%m%d-%H%M') if args.stage == 'all' else None)
    if not run_id:
        parser.error('--run-id is required')
    run = Run(args.root, run_id)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(run.dir / 'log.txt')])
    try:
        os.chmod(run.dir / 'log.txt', 0o666)
    except OSError:
        pass
    progress = Progress(run, args.dry)
    try:
        if args.stage == 'all':
            return stage_all(args, run, progress)
        STAGES[args.stage](args, run, progress)
        return 0
    except Exception:
        log.exception('stage %s failed', args.stage)
        return 1


if __name__ == '__main__':
    sys.exit(main())
