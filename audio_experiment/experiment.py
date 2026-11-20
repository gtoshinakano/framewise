#!/usr/bin/env python3
"""Audio experiment: turn an episode's audio into clean dialogue LINES with the right SPEAKER (see README.md).

  experiment.py phase <prepare|glossary|audio|qwen|second|lfm3|final> --run-id ID [options]   one phase (one process, one container)
  experiment.py all --dry --run-id dry-run --excerpt 20 230                                      every phase with fake models, CPU
  experiment.py rescore ID --labels labels.json                                                  scores again with spot-check labels (stdlib)
  experiment.py phase <music|voice|qwen2|final2> --run-id ID --base BASE_ID                       round 2 on top of a finished round-1 run
  experiment.py all2 --dry --run-id dry-run-r2 --base dry-run                                     every round-2 phase with fakes, CPU
Every phase is resumable: finished checkpoints are skipped.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PHASES = ['prepare', 'glossary', 'audio', 'qwen', 'second', 'lfm3', 'final']
PHASES2 = ['music', 'voice', 'qwen2', 'final2']
LABELS = Path(__file__).resolve().parent / 'labels'



@contextmanager
def real_client(parallel=3, startup=900, **kw):
    import qwen
    with qwen.text_server(parallel=parallel, startup_timeout=startup, **kw) as url:
        yield qwen.Client(url)


def fake_client(prep_holder):
    import fake_backend

    @contextmanager
    def factory(parallel=3, **kw):
        yield fake_backend.FakeClient()
    return factory


def make_clips(run, results):
    """Short wav clips of the sampled lines, cut with ffmpeg from the episode's audio.wav (never beside the original)."""
    done = 0
    for n, picks in results['samples'].items():
        out = run.path('clips', f'ep{n}')
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(out, 0o777)
            os.chmod(out.parent, 0o777)
        except OSError:
            pass
        source = common.episode_audio(int(n))
        for p in picks:
            target = out / f"{p['id']}.wav"
            if target.is_file():
                continue
            start = max(0., p['start'] - .3)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', f'{start:.3f}', '-t', f"{p['end'] - start + .3:.3f}", '-i', str(source),
                            '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)], check=False)
            try:
                target.chmod(0o666)
            except OSError:
                pass
            done += 1
    return done


def phase_final(ctx):
    import analysis
    import report
    run = ctx.run
    run.progress('final', 'running', 0, 3)
    results = analysis.compute(run)
    if results is None:
        run.note('final: nothing to analyse', 'warn')
        return
    results['characters'] = ctx.prep.get('characters', [])
    run.save(results, 'results.json')
    run.progress('final', 'running', 1, 3, 'audio clips')
    make_clips(run, results)
    run.set_status('finished' if not any(n['kind'] == 'fatal' for n in results['notes']) else 'finished-with-errors')
    results['progress'] = run.load('progress.json', default={})
    doc = report.build(results)
    (run.dir / 'report.html').write_text(doc.html(f'Audio experiment {run.id}'))
    (run.dir / 'report.md').write_text(doc.md())
    (run.dir / 'spotcheck.html').write_text(report.spotcheck(results))
    for name in ('report.html', 'report.md', 'spotcheck.html'):
        (run.dir / name).chmod(0o666)
    update_index(run.dir.parent)
    run.progress('final', 'done', 3, 3)


def update_index(root):
    rows = sorted((p.name for p in Path(root).iterdir() if (p / 'report.html').is_file()), reverse=True)
    items = ''.join(f'<li><a href="{r}/report.html">{r}</a> · <a href="{r}/spotcheck.html">spot check</a></li>' for r in rows)
    Path(root, 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Audio experiment runs</title><style>body{font:15px system-ui;margin:24px}</style><h1>Audio experiment runs</h1><ul>' + items + '</ul>')
    try:
        Path(root, 'index.html').chmod(0o666)
    except OSError:
        pass


def phase_final2(ctx):
    import round2_analysis
    run = ctx.run
    run.progress('final2', 'running', 0, 2)
    run.set_status('finished')
    timed = round2_analysis.load_labels(sorted(str(p) for p in LABELS.glob('*.json')), [], ctx.base)
    results = round2_analysis.write(run, ctx.base, timed)
    if results is None:
        run.note('final2: nothing to analyse', 'warn')
        return
    run.progress('final2', 'running', 1, 2, 'audio clips')
    make_clips(run, results)
    update_index(run.dir.parent)
    run.progress('final2', 'done', 2, 2)


def build_ctx(args, run):
    import stages
    if args.phase in PHASES2 or args.phase == 'all2':
        return build_ctx2(args, run)
    models = None
    factory = None
    prep_for_fake = None
    if args.phase in ('audio', 'second') or args.phase == 'all':
        if args.dry:
            import fake_backend
            prep = run.load('prep.json') or {}
            holder = {int(n): e for n, e in (prep.get('episodes') or {}).items()}
            owner = {}
            for sample in prep.get('samples', []):
                owner.setdefault(sample['episode'], []).extend((r['start'], r['end'], sample['character']) for r in sample['ranges'])
            holder['_sample_owner'] = owner
            models = fake_backend.FakeModels(holder)
        else:
            import backends
            models = backends.RealModels(device=os.environ.get('AEX_DEVICE') or None)
    if args.dry:
        factory = fake_client(None)
    else:
        factory = real_client
    return stages.Ctx(run, models=models, client_factory=factory, episodes=args.episodes, excerpt=args.excerpt, dry=args.dry)


def build_ctx2(args, run):
    import round2
    if not args.base:
        raise SystemExit('round 2 needs --base <round-1 run id>')
    base = common.Run(args.root, args.base)
    if not base.done('prep.json'):
        raise SystemExit(f'the base run {args.base} has no prep.json')
    models = None
    if args.phase in ('voice', 'all2'):
        if args.dry:
            import fake_backend
            prep = base.load('prep.json')
            holder = {int(n): e for n, e in prep['episodes'].items()}
            owner = {}
            for sample in prep.get('samples', []):
                owner.setdefault(sample['episode'], []).extend((r['start'], r['end'], sample['character']) for r in sample['ranges'])
            holder['_sample_owner'] = owner
            models = fake_backend.FakeModels(holder)
        else:
            import backends
            models = backends.RealModels(device=os.environ.get('AEX_DEVICE') or None)
    excerpt = args.excerpt or (base.load('prep.json') or {}).get('excerpt')
    return round2.Ctx2(run, base, models=models, client_factory=fake_client(None) if args.dry else real_client, episodes=args.episodes, excerpt=excerpt, dry=args.dry)


def run_phase(name, ctx, args):
    import stages
    run = ctx.run
    run.log(f'=== phase {name} ===')
    sampler = common.VramSampler()
    sampler.start()
    started = time.monotonic()
    try:
        if name == 'prepare':
            stages.stage_prepare(ctx, args.db)
        elif name == 'glossary':
            stages.stage_glossary(ctx)
        elif name == 'audio':
            stages.stage_audio(ctx)
        elif name == 'qwen':
            stages.stage_qwen(ctx)
        elif name == 'second':
            stages.stage_second(ctx)
        elif name == 'lfm3':
            stages.stage_lfm3(ctx)
        elif name == 'final':
            phase_final(ctx)
        elif name == 'music':
            import round2
            label = None
            if ctx.dry:
                import fake_backend
                label = fake_backend.music_labeller({str(n): ctx.base_ranges(n) for n in ctx.episodes})
            round2.stage_music(ctx, label)
        elif name == 'voice':
            import round2
            round2.stage_voice(ctx)
        elif name == 'qwen2':
            import round2
            round2.stage_qwen2(ctx)
        elif name == 'final2':
            phase_final2(ctx)
    finally:
        peak = sampler.finish()
        run.timing(f'phase.{name}', time.monotonic() - started, peak_vram_mib=peak)
        run.log(f'=== phase {name} took {time.monotonic() - started:.0f}s, peak VRAM {peak} MiB ===')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for cmd in ('phase', 'all', 'all2'):
        p = sub.add_parser(cmd)
        if cmd == 'phase':
            p.add_argument('phase', choices=PHASES + PHASES2)
        p.add_argument('--base', default=None, help='round 2: the finished round-1 run it builds on')
        p.add_argument('--run-id', required=True)
        p.add_argument('--root', default=os.environ.get('AEX_RUNS', '/report/runs'))
        p.add_argument('--db', default=os.environ.get('AEX_DB', '/priv/app.sqlite3'))
        p.add_argument('--dry', action='store_true')
        p.add_argument('--episodes', type=int, nargs='+', default=list(common.EVALUATED))
        p.add_argument('--excerpt', type=float, nargs=2, default=None)
        p.add_argument('--deadline', type=float, default=None, help='epoch seconds; work stops when it is reached')
    r = sub.add_parser('rescore')
    r.add_argument('run_id')
    r.add_argument('--root', default='frontend/public/audio-experiment/runs')
    r.add_argument('--labels', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    if args.cmd == 'rescore':
        import report
        sys.exit(report.main(['report', str(Path(args.root) / args.run_id), '--labels', args.labels]))
    run = common.Run(args.root, args.run_id, deadline=args.deadline, dry=args.dry)
    if args.cmd == 'all2':
        for name in PHASES2:
            args.phase = name
            ctx = build_ctx(args, run)
            run_phase(name, ctx, args)
        return
    if args.cmd == 'all':
        args.phase = 'all'
        for name in PHASES:
            args.phase = name
            ctx = build_ctx(args, run)
            run_phase(name, ctx, args)
        return
    ctx = build_ctx(args, run)
    run_phase(args.phase, ctx, args)


if __name__ == '__main__':
    main()
