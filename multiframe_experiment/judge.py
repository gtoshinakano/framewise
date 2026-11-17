"""The blind vision judge (Qwen3.5-27B): a reference flag table per window and a true/false verdict per claim.

* reference: the frames only (plus the roster names) -> per pair, per visible character, which categories changed,
  and the camera move. Recall is measured against it per pair and category (character identity is ignored, so a
  config that says "girl with brown hair" is not punished for not knowing a name).
* verify: every judge-able claim of every config (and of the stored video analysis) for that window, deduplicated,
  shuffled and numbered without saying where it came from -> true / false / not_visible.

The judge shares weights with one of the candidates (Qwen3.5); the report states that bias.
"""
import hashlib
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

import infer
import prompts
import schema

log = logging.getLogger('mfx.judge')
BATCH = 40
MAX_TOKENS = 4096

REFERENCE_SYSTEM = '''You annotate keyframes of one anime shot for a quality check. The frames are labelled F<n> with their time in seconds from the start of the shot.
For every pair of consecutive frames, list each character visible in either frame and mark true only for what visibly changes between the two frames:
- walking: the character walks or runs (steps, travel across the ground)
- speaking: the mouth opens, closes or changes shape
- blink: the eyes close or half close in one frame and not the other
- head: the head turns, tilts, nods or changes angle
- eyes_gaze: the eyes look somewhere else
- arms_hands: an arm or hand changes position
- legs: a leg or foot changes position
- body_pose: the torso or overall pose changes (bending, leaning, sitting down, standing up, turning the body)
- position: the character moves to another place in the image
- expression: the facial expression changes
Mark false when it stays the same or you cannot see it. Also give the camera move between the two frames: static, pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, dolly_in, dolly_out, truck, orbit, shake or unclear.
Left and right mean the viewer's left and right. Ignore any burned-in subtitle text. Be strict: small but real changes count, invented ones do not.
Return JSON only: {"pairs": [{"from": "F1", "to": "F2", "camera": "...", "characters": [{"name": "...", "walking": false, ...}]}]}'''

VERIFY_SYSTEM = '''You check statements about consecutive keyframes of one anime shot. The frames are labelled F<n> with their time in seconds from the start of the shot.
Each statement says that something changes or happens between two of the frames. Some are written as animation directions ("raise her right hand..."): then judge whether doing that would reproduce what the frames show.
For each statement answer:
- "true" when the two frames clearly show it (the direction and rough size of the change are right; different wording is fine),
- "false" when the frames contradict it: the part does not change, it changes another way or in the other direction, the person is not there, or it describes something not in the picture,
- "not_visible" when the frames cannot show it (the part is hidden, cropped or too small to judge).
Left and right mean the viewer's left and right. Ignore any burned-in subtitle text. Judge every statement independently.
Return JSON only: {"verdicts": [{"id": 1, "verdict": "true|false|not_visible"}, ...]} with one entry per statement.'''


def reference_schema(window):
    names = schema.labels(window)
    flags = {flag: {'type': 'boolean'} for flag in schema.REFERENCE_FLAGS}
    person = {'type': 'object', 'properties': {'name': {'type': 'string'}, **flags}, 'required': ['name', *schema.REFERENCE_FLAGS]}
    pair = {'type': 'object', 'properties': {'from': {'enum': names[:-1]}, 'to': {'enum': names[1:]}, 'camera': {'enum': list(schema.CAMERA_MOVES)},
                                             'characters': {'type': 'array', 'items': person}}, 'required': ['from', 'to', 'camera', 'characters']}
    return {'type': 'object', 'properties': {'pairs': {'type': 'array', 'minItems': len(names) - 1, 'maxItems': len(names) - 1, 'items': pair}}, 'required': ['pairs']}


def verify_schema(count):
    item = {'type': 'object', 'properties': {'id': {'type': 'integer'}, 'verdict': {'enum': ['true', 'false', 'not_visible']}}, 'required': ['id', 'verdict']}
    return {'type': 'object', 'properties': {'verdicts': {'type': 'array', 'minItems': count, 'maxItems': count, 'items': item}}, 'required': ['verdicts']}


def _frames_content(sample, window, images, intro):
    content = [{'type': 'text', 'text': intro}]
    for frame, image in zip(window['frames'], images):
        content.append({'type': 'text', 'text': prompts.frame_heading(frame)})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}})
    return content


def _roster_note(sample):
    names = [p['name'] for p in sample.get('roster') or []]
    if not names:
        return 'Nobody was identified by name; describe characters by appearance (for example "girl with brown hair").'
    return 'Characters identified in the first frame: ' + ', '.join(f"{p['name']} ({p['side']})" for p in sample['roster']) + \
        '. Use these names; describe anyone else by appearance.'


def validate_reference(parsed, window):
    names = schema.labels(window)
    expected = list(zip(names, names[1:]))
    if not isinstance(parsed, dict) or not isinstance(parsed.get('pairs'), list):
        return None, ['return {"pairs": [...]}']
    pairs = parsed['pairs']
    if len(pairs) != len(expected):
        return None, [f'"pairs" must have exactly {len(expected)} entries: ' + ', '.join(f'{a}→{b}' for a, b in expected)]
    clean = []
    for pair, (a, b) in zip(pairs, expected):
        if not isinstance(pair, dict):
            return None, ['every pair must be an object']
        people = []
        for person in pair.get('characters') or []:
            if isinstance(person, dict):
                people.append({'name': str(person.get('name') or '?'), **{flag: bool(person.get(flag)) for flag in schema.REFERENCE_FLAGS}})
        camera = schema._norm(pair.get('camera'), schema.CAMERA_MOVES, schema.ALIASES['move']) or 'unclear'
        clean.append({'from': a, 'to': b, 'camera': camera, 'characters': people,
                      'categories': sorted({flag for person in people for flag in schema.REFERENCE_FLAGS if person[flag]})})
    return clean, []


def _ask(client, messages, grammar, validate, hint, attempts=2):
    tries = []
    for number in range(1, attempts + 1):
        started = time.monotonic()
        reply = client.chat(messages, MAX_TOKENS, grammar, {**hint, 'attempt': number})
        seconds = reply.get('simulated_seconds') or time.monotonic() - started
        try:
            result, errors = validate(schema.parse_json(reply['content']))
        except ValueError as exc:
            result, errors = None, [f'not valid JSON ({exc})']
        if reply.get('finish_reason') == 'length':
            result, errors = None, ['cut off by the length limit']
        usage = reply.get('usage') or {}
        tries.append({'n': number, 'seconds': round(seconds, 2), 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                      'errors': errors, 'raw': infer.clip(reply['content'])})
        if result is not None:
            return result, tries
        messages = [*messages, {'role': 'assistant', 'content': reply['content'] or '{}'}, {'role': 'user', 'content': schema.format_errors(errors)}]
    return None, tries


def reference_call(client, sample, window, images):
    intro = f"Shot length {sample['duration']:.2f} s. {_roster_note(sample)}"
    messages = [{'role': 'system', 'content': REFERENCE_SYSTEM}, {'role': 'user', 'content': _frames_content(sample, window, images, intro)}]
    result, tries = _ask(client, messages, reference_schema(window), lambda parsed: validate_reference(parsed, window),
                         {'task': 'reference', 'labels': schema.labels(window), 'roster': [p['name'] for p in sample.get('roster') or []]})
    return {'key': f"ref|{sample['id']}|w{window['index']}", 'kind': 'reference', 'sample': sample['id'], 'window': window['index'], 'ok': result is not None,
            'pairs': result, 'attempts': tries, 'seconds': round(sum(t['seconds'] for t in tries), 2), 'finished_at': time.time()}


def claim_text_key(claim):
    return f"{claim['from']}>{claim['to']}|{' '.join(claim['text'].lower().split())}"


def verify_call(client, sample, window, images, batch, batch_key):
    lines = [f"[{n}] {c['from']}→{c['to']} ({c['start']:.2f}–{c['end']:.2f} s): {c['text']}" for n, c in enumerate(batch, 1)]
    intro = f"Shot length {sample['duration']:.2f} s. {_roster_note(sample)}"
    content = _frames_content(sample, window, images, intro)
    content.append({'type': 'text', 'text': f'Statements ({len(batch)}):\n' + '\n'.join(lines)})

    def validate(parsed):
        if not isinstance(parsed, dict) or not isinstance(parsed.get('verdicts'), list):
            return None, ['return {"verdicts": [...]}']
        found = {}
        for item in parsed['verdicts']:
            if isinstance(item, dict) and isinstance(item.get('id'), int) and item.get('verdict') in ('true', 'false', 'not_visible'):
                found[item['id']] = item['verdict']
        missing = [n for n in range(1, len(batch) + 1) if n not in found]
        if missing:
            return None, [f'give a verdict for every statement; missing ids: {missing[:20]}']
        return found, []

    messages = [{'role': 'system', 'content': VERIFY_SYSTEM}, {'role': 'user', 'content': content}]
    result, tries = _ask(client, messages, verify_schema(len(batch)), validate, {'task': 'verify', 'n': len(batch), 'texts': [c['text'] for c in batch]})
    verdicts = {claim_text_key(c): result[n] for n, c in enumerate(batch, 1)} if result else {}
    return {'key': batch_key, 'kind': 'verify', 'sample': sample['id'], 'window': window['index'], 'ok': result is not None, 'verdicts': verdicts,
            'statements': [c['text'] for c in batch], 'attempts': tries, 'seconds': round(sum(t['seconds'] for t in tries), 2), 'finished_at': time.time()}


def collect_claims(samples, answers):
    """Judge-able claims per (sample, window), deduplicated by pair + text: {(sid, w): [claim, ...]}."""
    by_sample = {s['id']: s for s in samples}
    out = {}
    for record in answers.values():
        if not record.get('ok'):
            continue
        sample = by_sample.get(record['sample'])
        if not sample:
            continue
        window = sample['windows'][record['window']]
        for claim in schema.answer_claims(record['answer'], window, record['config'], sample['id']):
            if claim['judge']:
                out.setdefault((sample['id'], window['index']), {}).setdefault(claim_text_key(claim), claim)
    for sample in samples:
        view = (sample.get('baseline') or {}).get('view')
        if not view or len(sample['frames']) < 2:
            continue
        for window in sample['windows']:
            answer = schema.baseline_answer(view, window)
            for claim in schema.answer_claims(answer, window, infer.BASELINE_KEY, sample['id']):
                if claim['judge']:
                    out.setdefault((sample['id'], window['index']), {}).setdefault(claim_text_key(claim), claim)
    return {key: list(value.values()) for key, value in out.items()}


def known_verdicts(store):
    verdicts = {}
    for record in store.records.values():
        if record.get('kind') == 'verify' and record.get('ok'):
            for text_key, verdict in record['verdicts'].items():
                verdicts[(record['sample'], record['window'], text_key)] = verdict
    return verdicts


def run_judge(store, samples, answers, client, run_dir, parallel=1, progress=None, references=True):
    claims = collect_claims(samples, answers)
    verdicts = known_verdicts(store)
    jobs = []
    for sample in samples:
        if len(sample['frames']) < 2:
            continue
        for window in sample['windows']:
            if references and f"ref|{sample['id']}|w{window['index']}" not in store.records:
                jobs.append(('reference', sample, window, None, None))
            pending = [c for c in claims.get((sample['id'], window['index']), []) if (sample['id'], window['index'], claim_text_key(c)) not in verdicts]
            random.Random(f"{sample['id']}|{window['index']}").shuffle(pending)
            for start in range(0, len(pending), BATCH):
                batch = pending[start:start + BATCH]
                digest = hashlib.sha1('|'.join(sorted(claim_text_key(c) for c in batch)).encode()).hexdigest()[:10]
                jobs.append(('verify', sample, window, batch, f"verify|{sample['id']}|w{window['index']}|{digest}"))
    total, done = len(jobs), [0]
    log.info('judge: %d calls (%d claims pending)', total, sum(len(j[3]) for j in jobs if j[3]))
    images = {}

    def one(job):
        kind, sample, window, batch, key = job
        cache_key = (sample['id'], window['index'])
        if cache_key not in images:
            images[cache_key] = infer.encode_images(run_dir, window['frames'])
        try:
            record = reference_call(client, sample, window, images[cache_key]) if kind == 'reference' else verify_call(client, sample, window, images[cache_key], batch, key)
        except Exception as exc:  # noqa: BLE001 - one failed judge call must not stop the others
            log.warning('judge %s %s w%d failed: %s', kind, sample['id'], window['index'], exc)
            return
        store.add(record)
        done[0] += 1
        log.info('judge %s %s w%d: ok=%s %.1fs', kind, sample['id'], window['index'], record['ok'], record['seconds'])
        if progress:
            progress(done[0], total, f"{kind} {sample['id']} w{window['index']}")

    if parallel <= 1:
        for job in jobs:
            one(job)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for future in [pool.submit(one, job) for job in jobs]:
                future.result()

