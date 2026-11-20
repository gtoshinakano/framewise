"""The stages of the experiment. Each is resumable: it skips what its checkpoint files already hold.

phases (separate processes/containers, never two on the GPU):  prepare (CPU)  ->  glossary (Qwen)  ->  audio (ASR + diarization)
->  qwen (attribution)  ->  second (ASR again)  ->  lfm3 (probe)  ->  analyse (CPU).
"""
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

import common
import data
import glossary
import lines as L
import oped
import qwen
import voice_tables as VT
import voices

DIAR_SETTINGS = [  # (name, kwargs, masked audio?)  priority order; the first two are needed by everything after them
    ('D1', {}, True),
    ('D0', {}, False),
    ('thr0.45', {'threshold': .45}, True),
    ('thr0.75', {'threshold': .75}, True),
    ('thr0.9', {'threshold': .9}, True),
    ('hint3-8', {'min_speakers': 3, 'max_speakers': 8}, True),
    ('max6_thr0.75', {'max_speakers': 6, 'threshold': .75}, True),
]
ASR_VARIANTS = {  # name -> (context?, forced language)
    'A1': (False, None),
    'A2': (True, None),
    'A3': (True, 'Japanese'),
}
BASE_DIAR = 'D1'
BASE_ASR = 'A1'
Q_VARIANTS = {'Q1': {'subs': True, 'offset': 0}, 'Q2': {'subs': False, 'offset': 0}, 'Q1s': {'subs': True, 'offset': 20}, 'Q3': {'subs': True, 'offset': 0, 'profiles': True}}
REFS_PATH = ('emb', 'refs.json')


class Ctx:
    def __init__(self, run, models=None, client_factory=None, episodes=common.EVALUATED, excerpt=None, dry=False, log=None):
        self.run, self.models, self.client_factory = run, models, client_factory
        self.episodes, self.excerpt, self.dry = list(episodes), excerpt, dry
        self._prep = None
        self._wav = {}

    # ------------------------------------------------------------ shared data
    @property
    def prep(self):
        if self._prep is None:
            self._prep = self.run.load('prep.json')
            if not self._prep:
                raise RuntimeError('prep.json is missing: run the prepare stage first')
        return self._prep

    def ep(self, n):
        return self.prep['episodes'][str(n)]

    def wav(self, n):
        if n not in self._wav:
            samples, _ = oped.read_wav(common.episode_audio(n))
            self._wav = {n: samples}  # one episode at a time keeps memory low
        return self._wav[n]

    def mask(self, n):
        ranges = (self.run.load('oped.json') or {}).get('ranges', {}).get(str(n), [])
        return [(r['start'], r['end']) for r in ranges]

    def scope(self):
        return tuple(self.excerpt) if self.excerpt else None

    def in_scope(self, start, end):
        s = self.scope()
        return True if s is None else (end > s[0] and start < s[1])

    def out_of_time(self, reserve=0):
        return self.run.out_of_time(reserve)


# ====================================================================== prepare (CPU)

def stage_prepare(ctx, db_path):
    run = ctx.run
    if run.done('prep.json') and run.done('oped.json'):
        run.log('prepare: already done')
        return
    run.progress('prepare', 'running', 0, 3)
    conn = data.connect(db_path)
    samples = data.voice_samples(conn)
    characters = data.characters(conn)
    episodes = {}
    for n in ctx.episodes:
        e = data.prepare_episode(conn, n, samples)
        scope = ctx.scope()
        if scope:
            keep = lambda a, b: b > scope[0] and a < scope[1]  # noqa: E731
            e['baseline_rows'] = [r for r in e['baseline_rows'] if keep(r['start_time'], r['end_time'])]
            e['baseline_lines'] = [l for l in e['baseline_lines'] if keep(l['start_time'], l['end_time'])]
            e['subtitles'] = [c for c in e['subtitles'] if keep(c['start_time'], c['end_time'])]
            e['anchors'] = [a for a in e['anchors'] if keep(a['start'], a['end'])]
        episodes[str(n)] = e
    durations = {}
    for n in common.EPISODES:
        row = data.rows(conn, 'SELECT duration FROM episodes WHERE id=?', (common.EPISODES[n],))
        durations[str(n)] = row[0]['duration'] if row else None
    breakdown = {n: bool(e['breakdown']) for n, e in episodes.items()}
    run.save({'episodes': episodes, 'samples': samples, 'characters': characters, 'durations': durations, 'excerpt': ctx.excerpt, 'breakdown_available': breakdown}, 'prep.json')
    ctx._prep = None
    run.progress('prepare', 'running', 1, 3, 'repeated-audio search across 7 episodes')
    with run.timed('prepare.oped'):
        stage_oped(ctx, durations)
    run.progress('prepare', 'done', 3, 3)


def stage_oped(ctx, durations):
    run = ctx.run
    fps = {}
    for n in common.EPISODES:
        samples, rate = oped.read_wav(common.episode_audio(n))
        fps[n] = oped.fingerprint(samples, rate)
    pairs = {}
    for i in common.EPISODES:
        for j in common.EPISODES:
            if i < j:
                pairs[(i, j)] = oped.match_pair(fps[i], fps[j])
    consolidated = oped.consolidate(pairs, len(common.EPISODES))
    ranges = {}
    for n, items in consolidated.items():
        ranges[str(n)] = oped.label_ranges(items, durations.get(str(n)) or 1500.)
    run.save({'ranges': ranges, 'pairs': {f'{i}-{j}': [{'a': list(s['a']), 'b': list(s['b']), 'score': s['score']} for s in segs] for (i, j), segs in pairs.items()},
              'method': 'sub-fingerprints (32 bits / 0.1 s, bands 300-3000 Hz), half-word vote on offsets, verification along a drifting diagonal',
              'min_support': 2}, 'oped.json')


# ====================================================================== glossary (Qwen)

def stage_glossary(ctx):
    run = ctx.run
    names = sorted({c['title'] for c in ctx.prep['characters']} - set(glossary.VERIFIED))
    verified = {t: v['ja'] for t, v in glossary.VERIFIED.items()}
    result = {'verified': verified, 'proposed': {}, 'seconds': None}
    run.progress('glossary', 'running', 0, 1)
    try:
        with ctx.client_factory(parallel=1) as client:
            proposed = qwen.propose_glossary(client, names)
        result['seconds'] = proposed['seconds']
        for item in proposed['proposals']:
            result['proposed'][item['name']] = {'japanese': item['japanese'], 'reading': item['reading'], 'sure': bool(item['sure']), 'unverified': True}
    except Exception as exc:  # the experiment goes on with the verified names only
        run.note(f'glossary proposals failed ({exc}); the ASR context uses the verified names only', 'warn')
    used = list(verified.values()) + [p['japanese'] for p in result['proposed'].values() if p['japanese'] and p['sure']]
    result['asr_context'] = '登場人物：' + '、'.join(used)
    run.save(result, 'glossary.json')
    run.progress('glossary', 'done', 1, 1)


# ====================================================================== audio

def zero_mask(wav, mask, rate=16000):
    masked = wav.copy()
    for start, end in mask:
        masked[int(start * rate):int(end * rate)] = 0
    return masked


def episode_slice(wav, ctx):
    scope = ctx.scope()
    return wav


def build_refs(ctx):
    """Reference embeddings per sample range (needs the audio of the episodes that hold samples)."""
    run = ctx.run
    if run.done(*REFS_PATH):
        return
    samples = ctx.prep['samples']
    wavs = {}
    for number in sorted({s['episode'] for s in samples}):
        wavs[number] = oped.read_wav(common.episode_audio(number))[0]
    started = time.monotonic()
    refs = voices.build_references(samples, wavs, ctx.models.embed_span)
    run.timing('refs.embeddings', time.monotonic() - started, n=len(refs))
    np.savez_compressed(run.wpath('emb', 'refs.npz'), matrix=np.stack([r['vec'] for r in refs]) if refs else np.zeros((0, 256)))
    run.save([{k: v for k, v in r.items() if k != 'vec'} for r in refs], *REFS_PATH)


def load_refs(ctx):
    meta = ctx.run.load(*REFS_PATH, default=[])
    matrix = np.load(ctx.run.path('emb', 'refs.npz'))['matrix'] if meta else np.zeros((0, 256))
    for r, vec in zip(meta, matrix):
        r['vec'] = vec
    return meta


def diarize_setting(ctx, n, name, kwargs, masked, wav, mask):
    run = ctx.run
    path = (f'ep{n}', 'diar', f'{name}.json')
    if run.done(*path):
        return run.load(*path)
    audio = zero_mask(wav, mask) if masked and mask else wav
    ctx.models.bind(n)
    started = time.monotonic()
    result = ctx.models.diarize(audio, **kwargs)
    result['wall_seconds'] = round(time.monotonic() - started, 1)
    result['n_voices'] = len(result['labels'])
    run.save(result, *path)
    run.timing(f'ep{n}.diar.{name}', time.monotonic() - started, voices=result['n_voices'])
    return result


def speech_regions(diar, mask, dilate=.2):
    regions = [(t['start_time'] - dilate, t['end_time'] + dilate) for t in diar['turns']]
    return L.subtract_intervals(L.merge_intervals(regions), mask)


def plan_episode_windows(ctx, n, diar, wav):
    run = ctx.run
    path = (f'ep{n}', 'windows.json')
    if run.done(*path):
        return run.load(*path)
    mask = ctx.mask(n)
    rms, hop = oped.rms_levels(wav)
    regions = [r for r in speech_regions(diar, mask) if ctx.in_scope(*r)]
    scope = ctx.scope()
    if scope:
        regions = [(max(a, scope[0]), min(b, scope[1])) for a, b in regions if min(b, scope[1]) > max(a, scope[0])]
    windows = L.plan_windows(regions, rms, hop, mask=mask)
    duration = len(wav) / 16000.
    fixed, start = [], 0.
    while start < duration:                     # the app's way, for the time baseline re-run (episode 3 only)
        end = min(start + 25., duration)
        if ctx.in_scope(start, end):
            fixed.append((start, end))
        start = end
    saved = {'windows': [list(w) for w in windows], 'fixed': [list(w) for w in fixed], 'rms_hop': hop,
             'covered_seconds': round(sum(b - a for a, b in windows), 1), 'speech_seconds': round(sum(b - a for a, b in regions), 1)}
    run.save(saved, *path)
    return saved


def run_asr(ctx, n, variant, windows, wav, context, language):
    """Transcribe every window of one variant, one checkpoint line per window (resume skips the finished ones)."""
    run = ctx.run
    path = run.path(f'ep{n}', 'asr', f'{variant}.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o777)
    except OSError:
        pass
    have = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                have[record['i']] = record
            except (ValueError, KeyError):
                pass
    ctx.models.bind(n)
    started = time.monotonic()
    new = 0
    for i, (start, end) in enumerate(windows):
        if i in have:
            continue
        if ctx.out_of_time():
            run.note(f'ep{n} {variant}: stopped at window {i}/{len(windows)} (time budget)', 'skip')
            break
        t = time.monotonic()
        result = ctx.models.transcribe(wav, start, end, context=context, language=language)
        record = {'i': i, 'start': start, 'end': end, 'text': result['text'], 'language': result.get('language'), 'tokens': result['tokens'],
                  'seconds': round(time.monotonic() - t, 2), **({'error': result['error']} if result.get('error') else {}), **({'split': True} if result.get('split') else {})}
        with path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        have[i] = record
        new += 1
        run.progress(f'audio.ep{n}.{variant}', 'running', len(have), len(windows))
    if new:
        run.timing(f'ep{n}.asr.{variant}', sum(r['seconds'] for r in have.values()), windows=len(have), new=new)
    run.progress(f'audio.ep{n}.{variant}', 'done' if len(have) == len(windows) else 'partial', len(have), len(windows))
    return [have[i] for i in sorted(have)]


def turn_index(diar):
    return L.TurnIndex(diar['turns'])


def embed_lines(ctx, n, lines, wav):
    """Per-line embeddings of the base line set, plus cosines against every reference range."""
    run = ctx.run
    refs = load_refs(ctx)
    ctx.models.bind(n)
    ids, vectors, ok = [], [], {}
    cos = {}
    for line in lines:
        if line['end_time'] - line['start_time'] < .6:
            ok[line['id']] = False
            continue
        clip = wav[int(line['start_time'] * 16000):int(line['end_time'] * 16000)]
        vec = ctx.models.embed_span(n, line['start_time'], line['end_time'], clip)
        if vec is None:
            ok[line['id']] = False
            continue
        vec = voices.unit(vec)
        ids.append(line['id'])
        vectors.append(vec)
        ok[line['id']] = True
        cos[line['id']] = {r['range_id']: round(float(np.dot(vec, r['vec'])), 4) for r in refs}
    np.savez_compressed(run.wpath(f'ep{n}', 'emb_base.npz'), ids=np.array(ids), matrix=np.stack(vectors) if vectors else np.zeros((0, 256)))
    return {'ok': ok, 'line_ref_cos': cos}, refs


def cluster_tables(ctx, n, settings, refs):
    samples = {s['id']: s for s in ctx.prep['samples']}
    # the app compares a cluster centroid with the stored embedding of the whole sample
    out_range, out_sample = {}, {}
    for name in settings:
        diar = ctx.run.load(f'ep{n}', 'diar', f'{name}.json')
        if not diar:
            continue
        out_range[name], out_sample[name] = {}, {}
        for label, centroid in diar['centroids'].items():
            vec = voices.unit(centroid)
            out_range[name][label] = {r['range_id']: round(float(np.dot(vec, r['vec'])), 4) for r in refs}
            out_sample[name][label] = {sid: round(voices.cosine(centroid, s['embedding']), 4) for sid, s in samples.items()}
    return out_range, out_sample


def stage_audio(ctx):
    """Passes in value order: (1) base diarization, ASR A1, lines, embeddings; (2) other diarization settings; (3) A2/A3; (4) the app's own windows on episode 3."""
    run = ctx.run
    run.progress('audio', 'running', 0, 4)
    glossary_data = run.load('glossary.json', default={}) or {}
    context = glossary_data.get('asr_context') or ''
    ctx.models.bind(None)
    with run.timed('audio.refs'):
        build_refs(ctx)
    # ---- pass 1
    for n in ctx.episodes:
        if run.done(f'ep{n}', 'pass1.done'):
            continue
        wav = ctx.wav(n)
        mask = ctx.mask(n)
        diar = diarize_setting(ctx, n, *DIAR_SETTINGS[0], wav, mask)
        planned = plan_episode_windows(ctx, n, diar, wav)
        records = run_asr(ctx, n, BASE_ASR, [tuple(w) for w in planned['windows']], wav, '', None)
        lines = L.build_variant_lines(records, turn_index(diar))
        run.save(lines, f'ep{n}', 'lines', f'{BASE_ASR}.json')
        tables, refs = embed_lines(ctx, n, lines, wav)
        run.save(tables, f'ep{n}', 'voice_tables.json')
        c_range, c_sample = cluster_tables(ctx, n, [BASE_DIAR], refs)
        run.save({'cluster_ref_cos': c_range, 'cluster_sample_cos': c_sample}, f'ep{n}', 'cluster_tables.json')
        run.save({'at': time.time()}, f'ep{n}', 'pass1.done')
    run.progress('audio', 'running', 1, 4, 'pass 1 done (base diarization, ASR, embeddings)')
    # ---- pass 2: the other diarization settings
    for n in ctx.episodes:
        wav = ctx.wav(n)
        mask = ctx.mask(n)
        for setting in DIAR_SETTINGS[1:]:
            if ctx.out_of_time(reserve=60):
                run.note(f'ep{n}: diarization setting {setting[0]} skipped (time budget)', 'skip')
                continue
            diarize_setting(ctx, n, *setting, wav, mask)
        refs = load_refs(ctx)
        names = [s[0] for s in DIAR_SETTINGS]
        c_range, c_sample = cluster_tables(ctx, n, names, refs)
        run.save({'cluster_ref_cos': c_range, 'cluster_sample_cos': c_sample}, f'ep{n}', 'cluster_tables.json')
        # 1.b: loud stretches with little diarized speech (music-like), from the unmasked diarization
        d0 = run.load(f'ep{n}', 'diar', 'D0.json')
        if d0:
            rms, hop = oped.rms_levels(wav)
            speech = [(t['start_time'], t['end_time']) for t in d0['turns']]
            run.save({'music_like': [list(r) for r in oped.music_candidates(rms, hop, speech)]}, f'ep{n}', 'signals.json')
    run.progress('audio', 'running', 2, 4, 'pass 2 done (diarization settings)')
    # ---- pass 3: ASR with names as context
    for n in ctx.episodes:
        planned = run.load(f'ep{n}', 'windows.json')
        wav = ctx.wav(n)
        diar = run.load(f'ep{n}', 'diar', f'{BASE_DIAR}.json')
        for variant in ('A2', 'A3'):
            use_context, language = ASR_VARIANTS[variant]
            if ctx.out_of_time(reserve=120):
                run.note(f'ep{n} {variant} skipped (time budget)', 'skip')
                continue
            records = run_asr(ctx, n, variant, [tuple(w) for w in planned['windows']], wav, context if use_context else '', language)
            run.save(L.build_variant_lines(records, turn_index(diar)), f'ep{n}', 'lines', f'{variant}.json')
    run.progress('audio', 'running', 3, 4, 'pass 3 done (A2/A3)')
    # ---- pass 4: the current app pipeline on episode 3 (time baseline, repeatability of the stored transcript)
    n = 3
    if n in ctx.episodes and not ctx.out_of_time(reserve=300):
        planned = run.load(f'ep{n}', 'windows.json')
        wav = ctx.wav(n)
        d0 = run.load(f'ep{n}', 'diar', 'D0.json')
        records = run_asr(ctx, n, 'A0', [tuple(w) for w in planned['fixed']], wav, '', None)
        words = []
        index = turn_index(d0)
        for record in records:
            for token in record['tokens']:
                words.append({'start': token['start'], 'end': token['end'], 'text': token['text'], 'spk': index.speaker(token['start'], token['end'])})
        run.save({'words': words}, f'ep{n}', 'lines', 'A0_words.json')
        t = run.timings()
        asr_s = sum(r['seconds'] for r in records)
        run.timing('baseline_rerun.ep3', (t.get('ep3.diar.D0') or {}).get('seconds', 0) + asr_s,
                   diarization=(t.get('ep3.diar.D0') or {}).get('seconds'), asr=round(asr_s, 1), windows=len(records))
    run.progress('audio', 'done', 4, 4)
    ctx.models.release()


# ====================================================================== qwen

def cut_index_at(cuts, t):
    for cut in cuts:
        if cut['start_time'] <= t < cut['end_time']:
            return cut
    return None


def line_input(line, ep, tables, refs_meta, names, use_subs=True, extra=None):
    """What the model sees for one line (see qwen.SYSTEM)."""
    ok = tables['ok'].get(line['id'])
    voice = None
    if ok:
        exclude = VT.overlapping_ranges(refs_meta, ep['number'], line['start_time'], line['end_time'])
        sc = VT.scores(tables['line_ref_cos'].get(line['id'], {}), refs_meta, exclude)
        voice = [[c, round(v, 2)] for c, v in VT.ranked(sc)[:3]]
    cut = cut_index_at(ep['cuts'], line['start_time'])
    item = {'id': line['id'], 't': [round(line['start_time'], 1), round(line['end_time'], 1)], 'ja': line['text'], 'diar': line.get('diar')}
    if voice is not None:
        item['voice'] = voice
    if cut is not None:
        item['cut'] = cut['cut_index']
        shown = (ep.get('roster') or {}).get(cut['id'])
        if shown:
            item['on_screen'] = shown
    if use_subs:
        cues = L.subtitle_overlap(line, ep['subtitles'])
        if cues:
            item['pt'] = ' / '.join(c['text'].replace('\n', ' ') for c in cues)[:240]
    if extra:
        item.update(extra)
    return item


def episode_names(ctx, n):
    ep_id = common.EPISODES[n]
    return sorted({c['title'] for c in ctx.prep['characters'] if c['episode_id'] in (None, ep_id)})


def qwen_variant(ctx, client, n, variant, base_lines, tables, refs_meta, profile_text=None, only=None, system=None, name=None, extra=None):
    """Run one attribution variant for one episode, with a per-window checkpoint; returns the saved result."""
    run = ctx.run
    name = name or variant
    path = (f'ep{n}', 'qwen', f'{name}.json')
    if run.done(*path):
        return run.load(*path)
    cfg = Q_VARIANTS.get(variant, {'subs': True, 'offset': 0})
    ep = ctx.ep(n)
    ep['number'] = n
    names = episode_names(ctx, n)
    selected = [l for l in base_lines if only is None or l['id'] in only]
    items = [line_input(l, ep, tables, refs_meta, names, cfg['subs'], (extra or {}).get(l['id'])) for l in selected]
    cache_path = run.path(f'ep{n}', 'qwen', f'{name}.windows.jsonl')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cache_path.parent, 0o777)
    except OSError:
        pass
    cache = {}
    if cache_path.is_file():
        for row in cache_path.read_text().splitlines():
            try:
                record = json.loads(row)
                cache[f"{record['range'][0]}-{record['range'][1]}"] = record
            except (ValueError, KeyError):
                pass

    def save_record(record):
        with cache_path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    started = time.monotonic()
    deadline = ctx.run.deadline
    answers, records = qwen.attribute_lines(client, items, names, offset=cfg['offset'], profile_text=profile_text, key=f'{name}|{n}', system=system,
                                            parallel=int(os.environ.get('QWEN_PARALLEL', '3')), deadline=deadline, cache=cache, on_record=save_record,
                                            progress=lambda d, t: run.progress(f'qwen.ep{n}.{name}', 'running', d, t))
    seconds = time.monotonic() - started
    result = {'variant': variant, 'name': name, 'episode': n, 'answers': answers, 'windows': [{k: v for k, v in r.items() if k not in ('answers', 'ids')} for r in records],
              'lines': len(items), 'answered': len(answers), 'seconds': round(seconds, 1), 'profile': bool(profile_text), 'subs': cfg['subs'], 'offset': cfg['offset'],
              'tokens': {'prompt': sum((a.get('prompt_tokens') or 0) for r in records for a in r['attempts']),
                         'completion': sum((a.get('completion_tokens') or 0) for r in records for a in r['attempts'])}}
    run.save(result, *path)
    run.timing(f'ep{n}.qwen.{name}', seconds, windows=len(records), lines=len(items))
    run.progress(f'qwen.ep{n}.{name}', 'done', len(records), len(records))
    return result


def scored_lines(ctx, n, base_lines, answers):
    """Base lines carrying one variant's speaker, confidence and reason (for the enrolment, the cluster resolution and the report)."""
    out = []
    for line in base_lines:
        a = answers.get(line['id']) or {}
        out.append({**{k: line[k] for k in ('id', 'start_time', 'end_time', 'text', 'diar')}, 'speaker': a.get('speaker'), 'confidence': a.get('confidence'), 'reason': a.get('reason')})
    return out


def stage_qwen(ctx, plan=None):
    run = ctx.run
    refs_meta = ctx.run.load(*REFS_PATH, default=[])
    base = {}
    tables = {}
    for n in ctx.episodes:
        base[n] = run.load(f'ep{n}', 'lines', f'{BASE_ASR}.json', default=[])
        tables[n] = run.load(f'ep{n}', 'voice_tables.json', default={'ok': {}, 'line_ref_cos': {}})
    order = plan or [('Q1', n) for n in ctx.episodes] + [('R1', n) for n in ctx.episodes] + [('Q2', n) for n in ctx.episodes] + \
        [('Q3', 3), ('Q3', 2), ('Q1s', 3), ('Q1s', 2), ('Q1s', 1), ('Q3', 1)]
    run.progress('qwen', 'running', 0, len(order))
    with ctx.client_factory(parallel=int(os.environ.get('QWEN_PARALLEL', '3'))) as client:
        for index, (variant, n) in enumerate(order):
            if n not in ctx.episodes:
                continue
            if ctx.out_of_time(reserve=120):
                run.note(f'{variant} on episode {n} skipped (time budget)', 'skip')
                continue
            try:
                if variant == 'R1':
                    stage_r1(ctx, client, n, base[n], tables[n], refs_meta)
                elif variant == 'Q3':
                    profile = profile_for(ctx, n, base)
                    if not profile:
                        run.note(f'Q3 on episode {n} skipped: no confident lines of the other episodes to learn from', 'skip')
                        continue
                    qwen_variant(ctx, client, n, 'Q3', base[n], tables[n], refs_meta, profile_text=profile)
                else:
                    qwen_variant(ctx, client, n, variant, base[n], tables[n], refs_meta)
            except Exception as exc:
                run.note(f'{variant} on episode {n} failed: {exc}', 'warn')
            run.progress('qwen', 'running', index + 1, len(order))
    enrol_all(ctx, base)
    run.progress('qwen', 'done', len(order), len(order))


def profile_for(ctx, n, base):
    """Speech profiles of the characters learned from the OTHER evaluated episodes' confident Q1 lines (never from n itself)."""
    pool = []
    for other in ctx.episodes:
        if other == n:
            continue
        q = ctx.run.load(f'ep{other}', 'qwen', 'Q1.json')
        if q:
            pool.extend(scored_lines(ctx, other, base[other], q['answers']))
    rows = qwen.speech_profiles(pool)
    return '\n'.join(r['text'] for r in sorted(rows.values(), key=lambda r: -r['lines'])) if rows else None


def stage_r1(ctx, client, n, base_lines, tables, refs_meta):
    """Episode-level voice resolution: link clusters to characters from the lines' own evidence, then re-decide only the weak lines."""
    run = ctx.run
    q1 = run.load(f'ep{n}', 'qwen', 'Q1.json')
    if not q1:
        run.note(f'R1 on episode {n} skipped: no Q1', 'skip')
        return
    if run.done(f'ep{n}', 'qwen', 'R1.json'):
        return
    lines = []
    for line in base_lines:
        a = q1['answers'].get(line['id']) or {}
        sc = {}
        if tables['ok'].get(line['id']):
            exclude = VT.overlapping_ranges(refs_meta, n, line['start_time'], line['end_time'])
            sc = VT.scores(tables['line_ref_cos'].get(line['id'], {}), refs_meta, exclude)
        lines.append({'id': line['id'], 'diar': line.get('diar'), 'qwen': a.get('speaker'), 'qwen_confidence': a.get('confidence'), 'voice_scores': sc})
    links, merged = resolve_clusters_table(lines)
    weak = {l['id'] for l in lines if (l['qwen_confidence'] in (None, 'low')) or l['qwen'] in (None, 'unknown')}
    for l in lines:
        link = links.get(l['diar'])
        if link and link['share'] >= .8 and l['qwen'] not in (None, 'unknown', 'other') and l['qwen'] != link['character'] and l['qwen_confidence'] != 'high':
            weak.add(l['id'])
    extra = {l['id']: {'cluster_link': {'character': links[l['diar']]['character'], 'consistency': links[l['diar']]['share']}} for l in lines if l['id'] in weak and links.get(l['diar'])}
    system = qwen.SYSTEM + '\n7. Some lines carry cluster_link: the voice cluster of the line was linked to a character from the evidence of the whole episode (consistency is the share of evidence behind it). Treat it as strong evidence about the voice, but not about who speaks a line whose content shows that someone else is speaking.'
    redo = None
    if weak:
        redo = qwen_variant(ctx, client, n, 'Q1', base_lines, tables, refs_meta, only=weak, system=system, name='R1redo', extra=extra)
    final = dict(q1['answers'])
    if redo:
        for line_id, answer in redo['answers'].items():
            final[line_id] = {**answer, 'redecided': True}
    run.save({'name': 'R1', 'episode': n, 'answers': final, 'cluster_links': links, 'merged_clusters': merged, 'weak_lines': len(weak), 'redecided': len(redo['answers']) if redo else 0,
              'seconds': (redo or {}).get('seconds', 0)}, f'ep{n}', 'qwen', 'R1.json')


def resolve_clusters_table(lines):
    return voices.resolve_clusters(lines)


def enrol_all(ctx, base):
    """Self-enrolment: centroids of each character from the confident lines (Q1) of the other episodes, applied to every line of this one."""
    run = ctx.run
    cache = {}
    for n in ctx.episodes:
        npz = run.path(f'ep{n}', 'emb_base.npz')
        if npz.is_file():
            z = np.load(npz)
            cache[n] = dict(zip([str(i) for i in z['ids']], z['matrix']))
    for target in ctx.episodes:
        if run.done(f'ep{target}', 'enrol_scores.json'):
            continue
        pool = []
        for source in ctx.episodes:
            if source == target:
                continue
            q = run.load(f'ep{source}', 'qwen', 'Q1.json')
            if not q or source not in cache:
                continue
            for line in scored_lines(ctx, source, base[source], q['answers']):
                if line['id'] in cache[source]:
                    pool.append({**line, 'embedding': cache[source][line['id']]})
        centroids = voices.enrol(pool)
        if not centroids:
            continue
        scores = {line_id: {c: round(v, 3) for c, v in voices.enrolled_scores(vec, centroids).items()} for line_id, vec in cache.get(target, {}).items()}
        run.save({'characters': {c: v['lines'] for c, v in centroids.items()}, 'scores': scores, 'sources': [s for s in ctx.episodes if s != target]}, f'ep{target}', 'enrol_scores.json')


# ====================================================================== second opinion (audio models again)

def stage_second(ctx, limit=120):
    run = ctx.run
    refs = load_refs(ctx)
    glossary_data = run.load('glossary.json', default={}) or {}
    names = glossary_data.get('asr_context') or ''
    run.progress('second', 'running', 0, len(ctx.episodes))
    for index, n in enumerate(ctx.episodes):
        if run.done(f'ep{n}', 'second.json'):
            continue
        base = run.load(f'ep{n}', 'lines', f'{BASE_ASR}.json', default=[])
        final = (run.load(f'ep{n}', 'qwen', 'R1.json') or run.load(f'ep{n}', 'qwen', 'Q1.json') or {}).get('answers', {})
        weak = [l for l in base if (final.get(l['id']) or {}).get('confidence') in (None, 'low') or (final.get(l['id']) or {}).get('speaker') in (None, 'unknown')]
        weak = [l for l in weak if l['end_time'] - l['start_time'] >= .8][:limit]
        wav = ctx.wav(n)
        ctx.models.bind(n)
        by_id = {l['id']: i for i, l in enumerate(base)}
        results = {}
        started = time.monotonic()
        for line in weak:
            if ctx.out_of_time(reserve=60):
                run.note(f'second opinion on episode {n} stopped early (time budget)', 'skip')
                break
            i = by_id[line['id']]
            around = ' '.join(l['text'] for l in base[max(0, i - 2):i]) + ' … ' + ' '.join(l['text'] for l in base[i + 1:i + 2])
            result = ctx.models.transcribe(wav, max(0., line['start_time'] - .4), line['end_time'] + .4, context=(names + '。' + around)[:600], language='Japanese')
            text2 = ''.join(t['text'] for t in result['tokens']) or result['text']
            tight = None
            if result['tokens']:
                s, e = result['tokens'][0]['start'], result['tokens'][-1]['end']
                if e - s >= .6:
                    vec = ctx.models.embed_span(n, s, e, wav[int(s * 16000):int(e * 16000)])
                    if vec is not None:
                        vec = voices.unit(vec)
                        cos = {r['range_id']: float(np.dot(vec, r['vec'])) for r in refs}
                        exclude = VT.overlapping_ranges(refs, n, line['start_time'], line['end_time'])
                        sc = VT.scores(cos, refs, exclude)
                        who, top, gap = VT.decide(sc)
                        tight = {'scores': [[c, round(v, 3)] for c, v in VT.ranked(sc)[:3]], 'decision': who, 'top': round(top, 3), 'gap': round(gap, 3)}
            results[line['id']] = {'text2': text2, 'cer': round(L.cer(line['text'], text2), 3), 'tight_voice': tight}
        run.save({'episode': n, 'lines': results, 'considered': len(weak), 'seconds': round(time.monotonic() - started, 1)}, f'ep{n}', 'second.json')
        run.timing(f'ep{n}.second', time.monotonic() - started, lines=len(results))
        run.progress('second', 'running', index + 1, len(ctx.episodes))
    run.progress('second', 'done', len(ctx.episodes), len(ctx.episodes))
    ctx.models.release()


# ====================================================================== LFM3-Omni audio probe (1.c)

def wav_bytes(clip, rate=16000):
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(clip, -1, 1) * 32767).astype('<i2').tobytes())
    return buffer.getvalue()


def stage_lfm3(ctx, budget_minutes=20):
    """Does the bundled llama-server take audio with the LFM3-Omni projector? If so, label 10 s windows speech / singing / music."""
    import base64
    import urllib.request
    run = ctx.run
    started = time.monotonic()
    run.progress('lfm3', 'running', 0, 1)
    projector = os.environ.get('LFM3_OMNI_PROJECTOR_PATH', '/models/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf')
    result = {'audio': False, 'windows': []}
    try:
        with ctx.client_factory(parallel=1, model_env='LFM3_OMNI_MODEL_PATH', extra=['--mmproj', projector, '--reasoning', 'off'], reasoning_budget=False, ctx_per_slot=8192, startup=300) as client:
            props = client.props() if hasattr(client, 'props') else {}
            modalities = (props or {}).get('modalities') or {}
            result['modalities'] = modalities
            if not modalities.get('audio'):
                result['note'] = 'the bundled llama-server does not report audio input for this projector'
            else:
                wav = ctx.wav(3)
                def ask(clip):
                    body = {'messages': [{'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {'data': base64.b64encode(wav_bytes(clip)).decode(), 'format': 'wav'}},
                                                                    {'type': 'text', 'text': 'What is in this audio? Answer with exactly one word: speech, singing, music or silence.'}]}],
                            'max_tokens': 8, 'temperature': 0}
                    request = urllib.request.Request(client.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
                    with urllib.request.urlopen(request, timeout=120) as response:
                        return json.load(response)['choices'][0]['message']['content'].strip().lower()
                probes = [(30, 'dialogue'), (100, 'op')]
                answers = {}
                for start, kind in probes:
                    answers[kind] = ask(wav[int(start * 16000):int((start + 10) * 16000)])
                result['probe'] = answers
                result['audio'] = True
                ops = [(r['start'], r['end']) for r in (run.load('oped.json') or {}).get('ranges', {}).get('3', [])]
                spans = []
                for a, b in ops:
                    t = a
                    while t + 10 <= b + 5:
                        spans.append((t, 'op/ed'))
                        t += 10
                for t in range(180, 1400, 60):
                    spans.append((float(t), 'dialogue'))
                for t, kind in spans:
                    if time.monotonic() - started > budget_minutes * 60:
                        result['note'] = 'time box reached'
                        break
                    label = ask(wav[int(t * 16000):int((t + 10) * 16000)])
                    result['windows'].append({'start': t, 'kind': kind, 'label': label})
    except Exception as exc:
        result['error'] = str(exc)[:400]
        run.note(f'LFM3 audio probe: {str(exc)[:200]}', 'skip')
    result['seconds'] = round(time.monotonic() - started, 1)
    run.save(result, 'lfm3.json')
    run.progress('lfm3', 'done', 1, 1)
