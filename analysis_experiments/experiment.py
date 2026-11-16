#!/usr/bin/env python3
"""Pose-detail experiment: does the current vision stack produce a reliable per-character head/eye/body block?

    python experiment.py all --run-id pose-20261116        # everything, one process per model stage
    python experiment.py sample|frames|video|score|conclude --run-id ...
    python experiment.py all --dry --run-id dry-run        # fake models: the pipeline without a GPU

Results land in ``<root>/<run-id>/`` (default root ``/report/runs`` inside the container, which is
``frontend/public/analysis-experiments/runs`` on the host): ``samples.json``, ``thumbs/``, ``raw/*.jsonl``,
``conclusions.json``, ``progress.json`` and ``results.json`` (what the report page reads).
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

import conclude  # noqa: E402
import pose_schema as ps  # noqa: E402
import runners  # noqa: E402
import scoring  # noqa: E402

log = logging.getLogger('pose')
STAGES = ('frames', 'video', 'conclude')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Run:
    """The files of one run."""

    def __init__(self, root, run_id):
        self.root, self.id = Path(root), run_id
        self.dir = self.root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o777)
        self.write_lock = threading.Lock()

    def read(self, name, default=None):
        try:
            return json.loads((self.dir / name).read_text())
        except (OSError, ValueError):
            return default

    def write(self, name, data):
        with self.write_lock:
            target = self.dir / name
            temp = self.dir / (name + '.tmp')
            temp.write_text(json.dumps(data, ensure_ascii=False))
            os.chmod(temp, 0o644)
            os.replace(temp, target)


class Progress:
    def __init__(self, run, dry):
        self.run, self.lock = run, threading.Lock()
        self.data = run.read('progress.json') or {'run_id': run.id, 'dry': dry, 'status': 'starting', 'started_at': now(), 'stages': {}}

    def set(self, stage=None, status=None, done=None, total=None, message=None):
        with self.lock:
            self.data['updated_at'] = now()
            if status and stage is None:
                self.data['status'] = status
            if stage:
                item = self.data['stages'].setdefault(stage, {'status': 'pending', 'done': 0, 'total': 0})
                item.update({k: v for k, v in (('status', status), ('done', done), ('total', total), ('message', message)) if v is not None})
                self.data['current'] = stage
            self.run.write('progress.json', self.data)


def update_index(root):
    entries = []
    for path in Path(root).glob('*/progress.json'):
        try:
            data = json.loads(path.read_text())
            entries.append({'id': path.parent.name, 'status': data.get('status'), 'dry': data.get('dry', False), 'started_at': data.get('started_at'), 'updated_at': data.get('updated_at')})
        except (OSError, ValueError):
            continue
    entries.sort(key=lambda item: item.get('started_at') or '', reverse=True)
    temp = Path(root) / 'index.json.tmp'
    temp.write_text(json.dumps(entries))
    os.chmod(temp, 0o644)
    os.replace(temp, Path(root) / 'index.json')


# -- results.json -----------------------------------------------------------------------------------------------------------

def load_calls(run):
    calls = []
    for stage in ('frame', 'video'):
        calls.extend(runners.Store(run.dir, stage).records.values())
    return calls


def slim(sample):
    return {key: sample[key] for key in ('id', 'reason', 'episode', 'cut_index', 'start', 'end', 'duration', 'marked', 'background', 'stored')} | {'thumb': f'thumbs/{sample["id"]}.jpg'}


def build_results(run):
    """Score every recorded call and write ``results.json`` (cheap: stdlib only, a few hundred records)."""
    samples = run.read('samples.json', [])
    calls = load_calls(run)
    scored = scoring.score_all(samples, calls)
    conclusions = run.read('conclusions.json') or {}
    approaches = {}
    for approach in ps.APPROACHES:
        mine = [call for call in calls if call['approach'] == approach]
        grouped = {'frame': {}, 'video': {}}
        for call in mine:
            if call['stage'] == 'frame':
                grouped['frame'].setdefault(call['sample'], {})[call.get('variant', 'normal')] = call
            else:
                grouped['video'][call['sample']] = call
        block = scored['approaches'][approach]
        approaches[approach] = {'label': ps.APPROACH_LABELS[approach], 'note': conclude.APPROACH_NOTES[approach], 'final': block['final'], 'stage_scores': block['stage_scores'],
                                'summary': block['summary'], 'a2_choices': block['a2_choices'], 'calls': grouped, 'conclusion': (conclusions.get('runs') or {}).get(approach)}
    results = {'run_id': run.id, 'updated_at': now(), 'config': run.read('config.json', {}), 'progress': run.read('progress.json', {}), 'weights': scored['weights'], 'stage_weights': scored['stage_weights'],
               'samples': [slim(sample) for sample in samples], 'approaches': approaches, 'agreement_between': scored['agreement_between'], 'overall_conclusion': conclusions.get('overall'),
               'fields': list(ps.FIELD_ORDER)}
    run.write('results.json', results)
    return results


# -- stages -------------------------------------------------------------------------------------------------------------------

def stage_sample(args, run, progress):
    from app import media
    from sampler import build_samples
    progress.set('sample', 'running', message='Choosing cuts')
    samples = run.read('samples.json')
    if not samples or args.resample:
        samples, missing, total = build_samples(Path(args.data_dir) / 'app.sqlite3', args.data_dir)
        if args.limit:
            samples = samples[:args.limit]
        for label in missing:
            log.warning('no cut found for stratum: %s', label)
        log.info('%d qualifying cuts, %d samples chosen', total, len(samples))
        run.write('samples.json', samples)
        run.write('config.json', {'created_at': now(), 'dry': args.dry, 'approaches': {a: ps.APPROACH_LABELS[a] for a in args.approaches}, 'missing_strata': missing, 'qualifying_cuts': total,
                                  'models': {'frame': 'Qwen3.5-27B Q4_K_M (llama.cpp)', 'video': 'Qwen3-VL-8B (transformers, bf16)', 'text': 'Qwen3.6-35B-A3B Q4_K_M (llama.cpp)'},
                                  'frame_parallel': args.frame_parallel, 'mirror': not args.no_mirror, 'grammar': args.grammar, 'max_attempts': runners.MAX_ATTEMPTS})
    thumbs = run.dir / 'thumbs'
    thumbs.mkdir(exist_ok=True)
    os.chmod(thumbs, 0o777)
    for number, sample in enumerate(samples, 1):
        image = thumbs / f'{sample["id"]}.jpg'
        if not image.is_file():
            first, _ = media.cut_frame_times(sample['start'], sample['end'])
            media.extract_cut_frame(Path(sample['original']), image, first, sample['start'], sample['end'], width=1280)
            os.chmod(image, 0o644)
        progress.set('sample', 'running', number, len(samples), f'Frame of {sample["id"]}')
    progress.set('sample', 'done', len(samples), len(samples), f'{len(samples)} cuts')
    build_results(run)


def ticker(run, progress, stage):
    def tick(done, total, message):
        progress.set(stage, 'running', done, total, message)
        build_results(run)
    return tick


def stage_frames(args, run, progress):
    samples = run.read('samples.json', [])
    store = runners.Store(run.dir, 'frame')
    scratch = run.dir / 'scratch'
    scratch.mkdir(exist_ok=True)
    os.chmod(scratch, 0o777)
    progress.set('frames', 'running', len(store.records), None, 'Starting the frame model')
    tick = ticker(run, progress, 'frames')
    options = dict(mirror=not args.no_mirror, grammar=args.grammar, parallel=args.frame_parallel, progress=tick)
    if args.dry:
        import fake_backend
        runners.run_frames(store, samples, args.approaches, fake_backend.FakeFrameClient(), run.dir / 'thumbs', scratch, **options)
    else:
        from app import frame_model
        with frame_model.frame_server(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
                                      os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                                      os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'),
                                      startup_timeout=600, parallel=args.frame_parallel, context_per_slot=12288) as url:
            runners.run_frames(store, samples, args.approaches, runners.LlamaClient(url), run.dir / 'thumbs', scratch, **options)
    progress.set('frames', 'done', len(store.records), len(store.records))
    build_results(run)


def stage_video(args, run, progress):
    samples = run.read('samples.json', [])
    store = runners.Store(run.dir, 'video')
    progress.set('video', 'running', len(store.records), len(samples) * len(args.approaches), 'Loading the video model')
    tick = ticker(run, progress, 'video')
    model_path = os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b')
    if args.dry:
        import fake_backend
        runtime, fake = fake_backend.FakeRuntime(), fake_backend.FakeVideoBackend()
    else:
        from app import video_model
        runtime, fake = video_model.VideoRuntime(), None
    try:
        runners.run_video(store, samples, args.approaches, model_path, runtime, fake, progress=tick)
    finally:
        runtime.close()
    progress.set('video', 'done', len(store.records), len(store.records))
    build_results(run)


def stage_score(args, run, progress):
    build_results(run)


def stage_conclude(args, run, progress):
    samples = run.read('samples.json', [])
    calls = load_calls(run)
    scored = scoring.score_all(samples, calls)
    previous = run.read('conclusions.json')
    progress.set('conclude', 'running', 0, 4, 'Starting Qwen 3.6')

    def save(result):
        run.write('conclusions.json', result)
        progress.set('conclude', 'running', len(result['runs']) + (1 if result.get('overall') else 0), 4, 'Writing conclusions')
        build_results(run)

    if args.dry:
        import fake_backend
        result = conclude.run(lambda system, user: (fake_backend.fake_chat(system, user), 1.0, {}), scored, samples, calls, previous, save)
    else:
        import servers
        with servers.text_server() as url:
            client = runners.LlamaClient(url, timeout=1800)

            def chat(system, user):
                started = time.monotonic()
                reply = client.chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 6144, thinking=True, temperature=0.3)
                return reply['content'], time.monotonic() - started, reply['usage']
            result = conclude.run(chat, scored, samples, calls, previous, save)
    run.write('conclusions.json', result)
    ok = all(item.get('ok') for item in [*result['runs'].values(), result['overall'] or {}]) and len(result['runs']) == len(ps.APPROACHES)
    progress.set('conclude', 'done' if ok else 'failed', 4, 4)
    build_results(run)


STAGE_FUNCTIONS = {'sample': stage_sample, 'frames': stage_frames, 'video': stage_video, 'score': stage_score, 'conclude': stage_conclude}


def child_args(args, run):
    items = ['--run-id', run.id, '--root', str(run.root), '--data-dir', args.data_dir, '--frame-parallel', str(args.frame_parallel), '--approaches', *args.approaches]
    for flag, value in (('--dry', args.dry), ('--no-mirror', args.no_mirror), ('--grammar', args.grammar)):
        if value:
            items.append(flag)
    return items


def stage_all(args, run, progress):
    progress.set(status='running')
    update_index(run.root)
    failed = []
    try:
        stage_sample(args, run, progress)
    except Exception:
        log.exception('sampling failed')
        progress.set('sample', 'failed')
        progress.set(status='failed')
        update_index(run.root)
        return 1
    for stage in STAGES:
        update_index(run.root)
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), stage, *child_args(args, run)])
        if code:
            failed.append(stage)
            progress.data['stages'].setdefault(stage, {})['status'] = 'failed'
            log.error('stage %s ended with exit code %s', stage, code)
        progress.data = run.read('progress.json') or progress.data
        if code:
            progress.data['stages'].setdefault(stage, {})['status'] = 'failed'
    progress.data = run.read('progress.json') or progress.data
    progress.set(status='partial' if failed else 'done', message=('failed: ' + ', '.join(failed)) if failed else None)
    scratch = run.dir / 'scratch'
    for path in scratch.glob('*') if scratch.is_dir() else []:
        path.unlink(missing_ok=True)
    build_results(run)
    update_index(run.root)
    return 1 if failed else 0


def latest_run_id(root):
    try:
        return json.loads((Path(root) / 'index.json').read_text())[0]['id']
    except (OSError, ValueError, IndexError, KeyError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['all', *STAGE_FUNCTIONS])
    parser.add_argument('--run-id')
    parser.add_argument('--root', default=os.environ.get('POSE_RUNS_ROOT', '/report/runs'))
    parser.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '/data'))
    parser.add_argument('--dry', action='store_true', help='fake models, no GPU')
    parser.add_argument('--limit', type=int, help='use only the first N samples (quick checks)')
    parser.add_argument('--approaches', nargs='+', default=list(ps.APPROACHES), choices=ps.APPROACHES)
    parser.add_argument('--frame-parallel', type=int, default=1, help='frame calls at once (1 keeps timings honest)')
    parser.add_argument('--no-mirror', action='store_true', help='skip the flipped-frame probe')
    parser.add_argument('--grammar', action='store_true', help='constrain frame answers with a JSON-schema grammar (production style)')
    parser.add_argument('--resample', action='store_true', help='choose the cuts again even if samples.json exists')
    args = parser.parse_args(argv)
    run_id = args.run_id or latest_run_id(args.root) if args.stage != 'all' else args.run_id or 'pose-' + datetime.now().strftime('%Y%m%d-%H%M')
    if not run_id:
        parser.error('--run-id is required')
    run = Run(args.root, run_id)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(run.dir / 'log.txt')])
    progress = Progress(run, args.dry)
    try:
        if args.stage == 'all':
            return stage_all(args, run, progress)
        STAGE_FUNCTIONS[args.stage](args, run, progress)
        return 0
    except Exception:
        log.exception('stage %s failed', args.stage)
        if args.stage in ('frames', 'video', 'conclude'):
            progress.set(args.stage, 'failed')
        return 1
    finally:
        update_index(run.root)


if __name__ == '__main__':
    sys.exit(main())
