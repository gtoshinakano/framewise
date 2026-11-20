"""Round 2 stages. They read a finished round-1 run (the BASE: prep.json, the OP/ED ranges, A1 lines, D1 diarization, reference and line
embeddings, Q1) read-only and write only into their own run folder. Line ids stay the base run's, so labels made on either run apply.

phases (one container each, never two on the GPU):  music (LFM3)  ->  voice (pyannote embeddings)  ->  qwen2 (Qwen3.6)  ->  final2 (CPU)
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import common
import lines as L
import names as N
import qwen
import round2_rules as R
import stages

MUSIC_WINDOW = 5.
MUSIC_QUESTION = 'What is in this audio? Answer with exactly one word: speech, singing, music or silence.'
LABELS = ('speech', 'singing', 'music', 'silence')

SYSTEM_OPEN = """You attribute lines of dialogue from one episode of a Japanese anime to the character who speaks them.

You see a stretch of the conversation. Lines marked "decided" already have their speaker from a clear match of the line's own voice with a recorded sample of that character; they are settled and you do not answer them, but they tell you who is talking around the open lines. You answer only the open lines listed at the end.

An open line comes with: its id and times in seconds, the Japanese text from a speech recognizer, a diarization label (a voice cluster; a decided line with the same label nearby is good evidence of the same voice, but one character can have several labels and one label can hold several characters), the characters whose recorded voice is closest to this line (weak: it did not decide the line, often because the line is very short), the characters recognised in the first frame of the cut when known (off-screen speech and inner monologue are normal), and a loose Brazilian Portuguese translation of the burned-in subtitles ("pt").

Rules.
1. Decide who speaks each open line from turn taking with the decided lines around it, the diarization label, who is addressed, first-person pronouns and sentence endings, answers to questions, and a line that continues the previous line of the same speaker. Who is on screen is weak evidence.
2. A name or honorific used to address someone names the listener, not the speaker.
3. The Portuguese translation is loose: use it for where sentences begin and end, whether two people speak inside one subtitle (a leading dash marks a second speaker) and rough meaning only.
4. speaker is one of the listed characters, "other" for a person who is not listed (announcer, passer-by, crowd), or "unknown" when the evidence does not decide. Prefer unknown to a guess.
5. confidence: high when two independent cues agree, medium when one clear cue supports it, low when the evidence is weak. reason is the single main cue.
6. text_fix is always an empty string and fix_reason none."""


class Ctx2(stages.Ctx):
    def __init__(self, run, base, **kw):
        super().__init__(run, **kw)
        self.base = base

    @property
    def prep(self):
        if self._prep is None:
            self._prep = self.base.load('prep.json')
            if not self._prep:
                raise RuntimeError(f'the base run {self.base.id} has no prep.json')
        return self._prep

    def base_ranges(self, n):
        return (self.base.load('oped.json') or {}).get('ranges', {}).get(str(n), [])

    def mask2(self, n):
        music = self.run.load('music.json') or {}
        ranges = (music.get('ranges') or {}).get(str(n))
        if ranges is None:
            ranges = self.base_ranges(n)
        return [(r['start'], r['end']) for r in ranges]

    def base_lines(self, n):
        return self.base.load(f'ep{n}', 'lines', 'A1.json', default=[])


def refs_meta(ctx):
    return ctx.base.load('emb', 'refs.json', default=[])


# ====================================================================== music map (LFM3)

def normalise_label(text):
    word = ''.join(c for c in (text or '').lower() if c.isalpha() or c == ' ').split()
    for candidate in word[:3]:
        if candidate in LABELS:
            return candidate
    return 'other'


def stage_music(ctx, label_fn=None):
    """LFM3 labels every 5 s of the evaluated episodes; the OP/ED ranges grow over the adjacent singing/music and new singing runs are added."""
    run = ctx.run
    run.progress('music', 'running', 0, len(ctx.episodes))
    started = time.monotonic()
    windows, out_ranges = {}, {}

    def work(label):
        for index, n in enumerate(ctx.episodes):
            path = run.wpath(f'ep{n}', 'music.jsonl')
            have = {}
            if path.is_file():
                for row in path.read_text().splitlines():
                    try:
                        item = json.loads(row)
                        have[item['start']] = item
                    except (ValueError, KeyError):
                        pass
            wav = ctx.wav(n)
            duration = len(wav) / common.SAMPLE_RATE
            t = 0.
            while t + 1. < duration:
                end = min(t + MUSIC_WINDOW, duration)
                if ctx.in_scope(t, end) and t not in have:
                    if ctx.out_of_time(reserve=60):
                        run.note(f'round 2 music map of episode {n} stopped at {t:.0f}s (time budget)', 'skip')
                        break
                    raw = label(n, t, end, wav[int(t * common.SAMPLE_RATE):int(end * common.SAMPLE_RATE)])
                    item = {'start': t, 'end': end, 'label': normalise_label(raw), 'raw': (raw or '')[:40]}
                    with path.open('a') as handle:
                        handle.write(json.dumps(item) + '\n')
                    have[t] = item
                t += MUSIC_WINDOW
            windows[n] = sorted(have.values(), key=lambda w: w['start'])
            out_ranges[str(n)] = R.extend_ranges(ctx.base_ranges(n), windows[n])
            run.progress('music', 'running', index + 1, len(ctx.episodes), f'episode {n}: {len(windows[n])} windows')

    if label_fn is not None:
        work(label_fn)
    else:
        projector = os.environ.get('LFM3_OMNI_PROJECTOR_PATH', '/models/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf')
        with ctx.client_factory(parallel=1, model_env='LFM3_OMNI_MODEL_PATH', extra=['--mmproj', projector, '--reasoning', 'off'],
                                reasoning_budget=False, ctx_per_slot=8192, startup=300) as client:
            work(lambda n, a, b, clip: client.audio_label(stages.wav_bytes(clip), MUSIC_QUESTION))
    run.save({'windows': {str(n): w for n, w in windows.items()}, 'ranges': out_ranges,
              'base_ranges': {str(n): ctx.base_ranges(n) for n in ctx.episodes}, 'window_seconds': MUSIC_WINDOW, 'question': MUSIC_QUESTION}, 'music.json')
    run.timing('round2.music', time.monotonic() - started, windows=sum(len(w) for w in windows.values()))
    run.progress('music', 'done', len(ctx.episodes), len(ctx.episodes))


# ====================================================================== voice for the lines the line's own clip does not decide

def stage_voice(ctx):
    """Embeds a padded clip and the diarization turn of every line its own clip does not decide (round 1: 22 % of lines, mostly < 0.8 s)."""
    import numpy as np
    import voices
    run = ctx.run
    meta = refs_meta(ctx)
    matrix = np.load(ctx.base.path('emb', 'refs.npz'))['matrix']
    run.progress('voice', 'running', 0, len(ctx.episodes))
    for index, n in enumerate(ctx.episodes):
        if run.done(f'ep{n}', 'voice2.json'):
            continue
        started = time.monotonic()
        base = sorted(ctx.base_lines(n), key=lambda l: l['start_time'])
        tables = ctx.base.load(f'ep{n}', 'voice_tables.json', default={'ok': {}, 'line_ref_cos': {}})
        turns = R.Turns(ctx.base.load(f'ep{n}', 'diar', 'D1.json')['turns'])
        wav = ctx.wav(n)
        ctx.models.bind(n)
        out, embedded = {}, 0
        for i, line in enumerate(base):
            if R.cascade(R.source_scores(line, tables, {}, meta, n, ('line',)))['speaker']:
                continue
            entry = {}
            spans = {'pad': R.pad_span(base, i)}
            if line.get('diar'):
                spans['turn'] = turns.stretch(line['start_time'], line['end_time'], line['diar'])
            for source, span in spans.items():
                if not span or span[1] - span[0] < R.MIN_CLIP:
                    continue
                clip = wav[int(span[0] * common.SAMPLE_RATE):int(span[1] * common.SAMPLE_RATE)]
                vec = ctx.models.embed_span(n, span[0], span[1], clip)
                if vec is None:
                    continue
                vec = voices.unit(vec)
                embedded += 1
                entry[source] = {'span': list(span), 'cos': {r['range_id']: round(float(np.dot(vec, voices.unit(row))), 4) for r, row in zip(meta, matrix)}}
            out[line['id']] = entry
        run.save(out, f'ep{n}', 'voice2.json')
        run.timing(f'ep{n}.round2.voice', time.monotonic() - started, lines=len(out), clips=embedded)
        run.progress('voice', 'running', index + 1, len(ctx.episodes))
    run.progress('voice', 'done', len(ctx.episodes), len(ctx.episodes))
    ctx.models.release()


# ====================================================================== Qwen: open lines only, and names

def open_items(ctx, n, kept, decided):
    ep = ctx.ep(n)
    ep['number'] = n
    shown = {}
    for line in kept:
        item = stages.line_input(line, ep, {'ok': {}}, [], [], True)
        d = decided.get(line['id']) or {}
        if d.get('speaker'):
            item = {k: item[k] for k in ('id', 't', 'ja', 'diar', 'pt', 'on_screen') if k in item}
            item['decided'] = d['speaker']
        elif d.get('evidence'):
            item['voice'] = [[c, round(v, 2)] for c, v in d['evidence']]
        shown[line['id']] = item
    return shown


def attribute_open(client, kept, shown, names, parallel=3, size=40, overlap=4, cache=None, on_record=None, deadline=None, progress=None, key=''):
    """Windows over the whole conversation; each asks only its open lines. A line answered by two windows keeps the answer of the window where it sits farther from an edge."""
    windows = qwen.make_windows(kept, size, overlap)
    records = [None] * len(windows)
    lock = threading.Lock()
    done = [0]

    def work(index):
        start, stop = windows[index]
        asked = [shown[l['id']] for l in kept[start:stop] if 'decided' not in shown[l['id']]]
        if not asked or (deadline and time.time() > deadline):
            return
        ids = [a['id'] for a in asked]
        cached = (cache or {}).get(f'{start}-{stop}')
        if cached and cached.get('asked') == ids:
            with lock:
                records[index] = cached
            return
        result = qwen.call_window(client, asked, names, key=f'{key}|{index}', system=SYSTEM_OPEN, shown=[shown[l['id']] for l in kept[start:stop]])
        result['range'] = [start, stop]
        result['asked'] = ids
        with lock:
            records[index] = result
            if on_record:
                on_record(result)
            done[0] += 1
            if progress:
                progress(done[0], len(windows))

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(work, range(len(windows))))
    position = {l['id']: i for i, l in enumerate(kept)}
    chosen = {}
    for index, result in enumerate(records):
        if not result:
            continue
        start, stop = windows[index]
        for line_id, answer in result['answers'].items():
            distance = min(position[line_id] - start, stop - 1 - position[line_id])
            if line_id not in chosen or distance >= chosen[line_id][0]:
                chosen[line_id] = (distance, answer)
    return {k: v[1] for k, v in chosen.items()}, [r for r in records if r]


def call_names(client, window, key=''):
    """One window of the name pass with the validator retry -> {fixes, attempts}."""
    messages = [{'role': 'system', 'content': N.SYSTEM}, {'role': 'user', 'content': N.user_message(window)}]
    attempts, fixes = [], {}
    for number in range(1, qwen.MAX_ATTEMPTS + 1):
        began = time.monotonic()
        try:
            reply = client.chat(messages, 4096, N.schema([l['id'] for l in window]), {'task': 'names', 'window': window, 'attempt': number, 'key': key})
        except OSError as exc:
            attempts.append({'n': number, 'seconds': round(time.monotonic() - began, 1), 'errors': [f'request failed: {exc}']})
            continue
        try:
            fixes, errors = N.validate(qwen.parse_json(reply['content']), window)
        except ValueError as exc:
            fixes, errors = {}, [f'the answer is not valid JSON ({exc})']
        usage = reply.get('usage') or {}
        attempts.append({'n': number, 'seconds': round(time.monotonic() - began, 1), 'prompt_tokens': usage.get('prompt_tokens'),
                         'completion_tokens': usage.get('completion_tokens'), 'errors': errors, 'kept': len(fixes)})
        if not errors:
            break
        if number < qwen.MAX_ATTEMPTS:
            messages = messages[:2] + [{'role': 'assistant', 'content': reply['content']},
                                       {'role': 'user', 'content': 'Some fixes were rejected:\n- ' + '\n- '.join(errors[:10]) + '\nAnswer again with the complete corrected JSON object.'}]
    return {'fixes': fixes, 'attempts': attempts}


def stage_qwen2(ctx):
    run = ctx.run
    refs = refs_meta(ctx)
    parallel = int(os.environ.get('QWEN_PARALLEL', '3'))
    total = 2 * len(ctx.episodes)
    step = 0
    run.progress('qwen2', 'running', 0, total)
    with ctx.client_factory(parallel=parallel) as client:
        for n in ctx.episodes:                                      # 1. open lines (the main question of round 2)
            step += 1
            if run.done(f'ep{n}', 'Q4.json'):
                continue
            if ctx.out_of_time(reserve=120):
                run.note(f'Q4 on episode {n} skipped (time budget)', 'skip')
                continue
            base = sorted(ctx.base_lines(n), key=lambda l: l['start_time'])
            kept, _ = R.split_by_mask(base, ctx.mask2(n))
            tables = ctx.base.load(f'ep{n}', 'voice_tables.json', default={'ok': {}, 'line_ref_cos': {}})
            decided = R.decisions(kept, tables, run.load(f'ep{n}', 'voice2.json', default={}), refs, n)
            shown = open_items(ctx, n, kept, decided)
            cache_path = run.wpath(f'ep{n}', 'Q4.windows.jsonl')
            cache = {}
            if cache_path.is_file():
                for row in cache_path.read_text().splitlines():
                    try:
                        record = json.loads(row)
                        cache[f"{record['range'][0]}-{record['range'][1]}"] = record
                    except (ValueError, KeyError):
                        pass

            def save(record, path=cache_path):
                with path.open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')

            started = time.monotonic()
            answers, records = attribute_open(client, kept, shown, stages.episode_names(ctx, n), parallel=parallel, cache=cache, on_record=save,
                                              deadline=run.deadline, key=f'Q4|{n}', progress=lambda d, t, n=n: run.progress(f'qwen2.ep{n}.Q4', 'running', d, t))
            asked = sum(1 for item in shown.values() if 'decided' not in item)
            run.save({'answers': answers, 'asked': asked, 'decided': len(shown) - asked, 'answered': len(answers), 'seconds': round(time.monotonic() - started, 1),
                      'windows': [{k: v for k, v in r.items() if k not in ('answers',)} for r in records],
                      'thresholds': {'fix_top': R.FIX_TOP, 'fix_gap': R.FIX_GAP, 'ref_guard': R.REF_GUARD}}, f'ep{n}', 'Q4.json')
            run.timing(f'ep{n}.round2.Q4', time.monotonic() - started, asked=asked, windows=len(records))
            run.progress('qwen2', 'running', step, total)
        for n in ctx.episodes:                                      # 2. names
            step += 1
            if run.done(f'ep{n}', 'names.json'):
                continue
            if ctx.out_of_time(reserve=60):
                run.note(f'name pass on episode {n} skipped (time budget)', 'skip')
                continue
            started = time.monotonic()
            base = sorted(ctx.base_lines(n), key=lambda l: l['start_time'])
            kept, _ = R.split_by_mask(base, ctx.mask2(n))
            aliased, alias_changes = {}, {}
            for line in kept:
                aliased[line['id']], changes = N.apply_aliases(line['text'])
                if changes:
                    alias_changes[line['id']] = changes
            items = [{'id': l['id'], 'ja': aliased[l['id']]} for l in kept]
            chunks = [items[i:i + 60] for i in range(0, len(items), 60)]
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                results = list(pool.map(lambda pair: call_names(client, pair[1], key=f'names|{n}|{pair[0]}'), enumerate(chunks)))
            fixes = {}
            for result in results:
                fixes.update(result['fixes'])
            run.save({'aliased': aliased, 'alias_changes': alias_changes, 'fixes': fixes, 'windows': [r['attempts'] for r in results],
                      'seconds': round(time.monotonic() - started, 1)}, f'ep{n}', 'names.json')
            run.timing(f'ep{n}.round2.names', time.monotonic() - started, lines=len(items), fixes=len(fixes), aliases=len(alias_changes))
            run.progress('qwen2', 'running', step, total)
    run.progress('qwen2', 'done', total, total)
