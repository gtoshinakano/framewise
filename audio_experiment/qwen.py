"""Qwen3.6 as the text lane: the llama-server launcher, the speaker-attribution prompt, grammar and validator, and the window runner.

The model sees evidence per line and answers one speaker per line through a json_schema grammar (thinking on, capped budget, like the app).
A rejected answer goes back with the validator's sentences (3 attempts). Prompts describe rules, never example sentences.
"""
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import lines as L

MAX_ATTEMPTS = 3
CONFIDENCE = ['high', 'medium', 'low']
REASONS = ['voice', 'addressed', 'pronoun', 'on_screen', 'turn', 'subtitle', 'other']
FIX_REASONS = ['none', 'name', 'homophone']

SYSTEM = """You attribute lines of dialogue from one episode of a Japanese anime to the character who speaks them.

Every line comes with: its id and times in seconds, the Japanese text from a speech recognizer, a diarization label (a voice cluster; labels are unreliable, one character can have several labels and one label can hold several characters), the characters whose recorded voice is closest to the audio of this very line with similarity scores (a best score under 0.35, or under 0.06 of difference to the second, means weak voice evidence; missing means no voice evidence), the characters recognised in the first frame of the cut the line starts in when known (a speaker is often on screen but not always; off-screen speech and inner monologue are normal), and a loose Brazilian Portuguese translation of the burned-in subtitles ("pt", when present). The neighbouring lines give the conversation.

Rules.
1. Decide who speaks each line. Voice evidence and diarization labels are the main evidence. The dialogue is the second: who is addressed by name or honorific, first-person pronouns and sentence endings typical for a character, answers to questions, turn taking, and a line that continues the previous line of the same speaker. Who is on screen is weak evidence.
2. A name or honorific used to address someone names the listener, not the speaker.
3. The Portuguese translation is loose. Use it only for where sentences begin and end, whether two people speak inside one subtitle (a leading dash marks a second speaker) and rough meaning. Never change or correct the Japanese because of it.
4. speaker is one of the listed characters, "other" for a person who is not listed (announcer, passer-by, crowd), or "unknown" when the evidence does not decide. Do not guess to fill a gap: answer unknown with confidence low.
5. confidence: high when voice and context agree or one of them is decisive; medium when one clear piece of evidence supports it; low when the evidence is weak. reason is the single main cue.
6. text_fix stays an empty string unless the recognizer clearly misspelled a character name or wrote an obvious homophone that the context settles; then give the whole corrected line and set fix_reason to name or homophone. Never reword, add or remove sentences."""


# ------------------------------------------------------------------ server

@contextmanager
def text_server(parallel=3, ctx_per_slot=16384, startup_timeout=900, model_env='QWEN36_MODEL_PATH', extra=(), reasoning_budget=True):
    """Private llama-server for the standard Qwen3.6-35B-A3B file with the flags the app's LocalTextRuntime uses (no vision projector)."""
    runtime = Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'))
    model = Path(os.environ.get(model_env, '/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf'))
    if not runtime.is_file() or not model.is_file():
        raise RuntimeError(f'model or llama-server runtime is missing ({model})')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+t') as log:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        command = [str(runtime), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '--n-gpu-layers', '99',
                   '--ctx-size', str(ctx_per_slot * parallel), '--parallel', str(parallel), '--jinja', '--no-context-shift',
                   *(['--reasoning-budget', os.environ.get('QWEN36_THINKING_BUDGET', '2048')] if reasoning_budget else []), *extra]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError('llama-server failed to start: ' + log.read()[-1800:])
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('llama-server did not become ready in time')
                time.sleep(1)
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class Client:
    """llama-server chat endpoint with the json_schema grammar (a server that rejects it answers plain json_object)."""

    def __init__(self, url, timeout=1500, thinking=True):
        self.url, self.timeout, self.thinking = url, timeout, thinking

    def props(self):
        try:
            with urllib.request.urlopen(self.url + '/props', timeout=10) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.URLError):
            return {}

    def audio_label(self, wav_data, question):
        """One short audio clip (wav bytes) with a one-word question; for LFM3-Omni (round 1 probe, round 2 music map)."""
        import base64
        body = {'messages': [{'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {'data': base64.b64encode(wav_data).decode(), 'format': 'wav'}},
                                                          {'type': 'text', 'text': question}]}], 'max_tokens': 8, 'temperature': 0}
        return self._post(body)['choices'][0]['message']['content'].strip().lower()

    def _post(self, body):
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, messages, max_tokens, schema_=None, hint=None, temperature=.2):
        def body(fmt):
            return {'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'response_format': fmt,
                    'chat_template_kwargs': {'enable_thinking': self.thinking}}
        grammar = bool(schema_)
        try:
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True, 'schema': schema_}} if schema_ else {'type': 'json_object'}
            payload = self._post(body(fmt))
        except urllib.error.HTTPError as exc:
            if not schema_ or exc.code not in (400, 422, 500):
                raise
            grammar = False
            payload = self._post(body({'type': 'json_object'}))
        choice = payload['choices'][0]
        return {'content': choice['message'].get('content') or '', 'finish_reason': choice.get('finish_reason'), 'usage': payload.get('usage') or {},
                'timings': payload.get('timings') or {}, 'grammar': grammar}


# ------------------------------------------------------------------ attribution

def answer_schema(ids, names):
    speakers = list(names) + ['other', 'unknown']
    item = {'type': 'object', 'additionalProperties': False,
            'required': ['id', 'speaker', 'confidence', 'reason', 'text_fix', 'fix_reason'],
            'properties': {'id': {'type': 'string', 'enum': list(ids)}, 'speaker': {'type': 'string', 'enum': speakers},
                           'confidence': {'type': 'string', 'enum': CONFIDENCE}, 'reason': {'type': 'string', 'enum': REASONS},
                           'text_fix': {'type': 'string'}, 'fix_reason': {'type': 'string', 'enum': FIX_REASONS}}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['lines'],
            'properties': {'lines': {'type': 'array', 'items': item, 'minItems': len(ids), 'maxItems': len(ids)}}}


def parse_json(text):
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text[text.find('{'):]
    return json.loads(text)


def validate(answer, window, names):
    """-> (answers by id, errors as sentences for the model)."""
    errors, result = [], {}
    items = answer.get('lines') if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, ['the answer must be an object with a "lines" array']
    expected = [line['id'] for line in window]
    texts = {line['id']: line['ja'] for line in window}
    seen = [item.get('id') for item in items if isinstance(item, dict)]
    if seen != expected:
        missing = [i for i in expected if i not in seen]
        extra = [i for i in seen if i not in expected]
        errors.append(f'answer every line exactly once and in order; missing {missing[:6]}, unexpected {extra[:6]}' if (missing or extra) else 'the lines must appear in the order they were given')
    allowed = set(names) | {'other', 'unknown'}
    for item in items:
        if not isinstance(item, dict) or item.get('id') not in texts:
            continue
        lid = item['id']
        if item.get('speaker') not in allowed:
            errors.append(f'{lid}: speaker must be one of the listed characters, "other" or "unknown"')
        if item.get('confidence') not in CONFIDENCE:
            errors.append(f'{lid}: confidence must be high, medium or low')
        fix = (item.get('text_fix') or '').strip()
        if fix:
            if item.get('fix_reason') in (None, 'none', ''):
                errors.append(f'{lid}: text_fix needs fix_reason name or homophone, or leave text_fix empty')
            elif L.edit_distance(fix, texts[lid]) > max(3, int(.3 * len(texts[lid]))):
                errors.append(f'{lid}: text_fix changes too much of the line; fix only a name or an obvious homophone, or leave it empty')
        result[lid] = {'speaker': item.get('speaker'), 'confidence': item.get('confidence'), 'reason': item.get('reason'),
                       'text_fix': fix or None, 'fix_reason': item.get('fix_reason') if fix else None}
    return result, errors


def user_message(window, names, profile_text=None, before=(), shown=None):
    parts = ['Characters of this episode: ' + ', '.join(names) + '.']
    if profile_text:
        parts.append('How each character speaks (counted from other lines of this series):\n' + profile_text)
    if before:
        parts.append('Context, already decided, do not answer: ' + json.dumps(list(before), ensure_ascii=False))
    if shown is not None:   # round 2: the whole conversation is shown, only `window` is asked
        parts.append('Conversation (' + str(len(shown)) + ' lines; a line with "decided" already has its speaker and is not answered):\n' + json.dumps(shown, ensure_ascii=False))
        parts.append('Answer these ' + str(len(window)) + ' lines, in this order: ' + json.dumps([line['id'] for line in window]))
    else:
        parts.append('Lines to attribute (' + str(len(window)) + '):\n' + json.dumps(window, ensure_ascii=False))
    return '\n\n'.join(parts)


def call_window(client, window, names, profile_text=None, before=(), max_tokens=6144, key='', system=None, shown=None):
    """One window through the model with the validator retry; returns {answers, attempts, seconds, ...}.

    `shown` (round 2): the conversation the model reads, of which only `window` is answered.
    """
    schema_ = answer_schema([l['id'] for l in window], names)
    messages = [{'role': 'system', 'content': system or SYSTEM}, {'role': 'user', 'content': user_message(window, names, profile_text, before, shown)}]
    attempts, answers = [], {}
    started = time.monotonic()
    for number in range(1, MAX_ATTEMPTS + 1):
        began = time.monotonic()
        hint = {'task': 'speaker', 'window': window, 'names': names, 'attempt': number, 'key': key}
        try:
            reply = client.chat(messages, max_tokens, schema_, hint)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({'n': number, 'seconds': round(time.monotonic() - began, 1), 'errors': [f'request failed: {exc}']})
            continue
        raw = reply['content']
        usage, timings = reply.get('usage') or {}, reply.get('timings') or {}
        try:
            answer = parse_json(raw)
            answers, errors = validate(answer, window, names)
        except ValueError as exc:
            answers, errors = {}, [f'the answer is not valid JSON ({exc})']
        if reply.get('finish_reason') == 'length' and errors:
            errors.insert(0, 'the answer was cut off by the length limit')
        attempts.append({'n': number, 'seconds': round(time.monotonic() - began, 1), 'prompt_tokens': usage.get('prompt_tokens'),
                         'completion_tokens': usage.get('completion_tokens'), 'predicted_per_second': timings.get('predicted_per_second'),
                         'finish': reply.get('finish_reason'), 'grammar': reply.get('grammar'), 'errors': errors})
        if not errors:
            break
        echoed = raw if reply.get('finish_reason') != 'length' else (raw or '')[:1200] + ' …'
        messages = messages[:2] + [{'role': 'assistant', 'content': echoed},
                                   {'role': 'user', 'content': 'The answer was rejected:\n- ' + '\n- '.join(errors[:10]) + '\nAnswer again with the complete corrected JSON object.'}]
        answers = {}
    return {'answers': answers, 'attempts': attempts, 'seconds': round(time.monotonic() - started, 1), 'ok': bool(answers) and not attempts[-1]['errors']}


def make_windows(items, size=40, overlap=4, offset=0):
    """Index ranges [(start, stop)] over `items`; the first window can be shortened by `offset` (shifted-window runs)."""
    windows, start = [], 0
    first = size - offset if offset else size
    stop = min(len(items), first)
    while start < len(items):
        windows.append((start, stop))
        if stop >= len(items):
            break
        start = stop - overlap
        stop = min(len(items), start + size)
    return windows


def attribute_lines(client, items, names, size=40, overlap=4, offset=0, profile_text=None, parallel=3, progress=None, deadline=None, key='q', system=None, cache=None, on_record=None):
    """Run every window (parallel slots) and merge. `items` are the line dicts built by experiment.line_input().

    Returns (answers by line id, window records). A window that never validated leaves its lines unanswered.
    """
    windows = make_windows(items, size, overlap, offset)
    records = [None] * len(windows)
    lock = threading.Lock()
    done = [0]

    def work(index):
        start, stop = windows[index]
        if deadline and time.time() > deadline:
            return
        window = items[start:stop]
        cached = (cache or {}).get(f'{start}-{stop}')
        if cached and cached.get('ids') == {str(i): items[i]['id'] for i in range(start, stop)}:
            with lock:
                records[index] = cached
                done[0] += 1
            return
        before = [{'id': i['id'], 'ja': i['ja']} for i in items[max(0, start - 2):start]] if start else []
        result = call_window(client, window, names, profile_text, before, key=f'{key}|{index}', system=system)
        result['range'] = [start, stop]
        result['ids'] = {str(i): items[i]['id'] for i in range(start, stop)}
        with lock:
            records[index] = result
            if on_record:
                on_record(result)
            done[0] += 1
            if progress:
                progress(done[0], len(windows))

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(work, range(len(windows))))
    answers, stability = {}, {}
    chosen = {}
    for index, result in enumerate(records):
        if not result:
            continue
        for position, line_id in result['ids'].items():
            answer = result['answers'].get(line_id)
            if not answer:
                continue
            position = int(position)
            start, stop = windows[index]
            distance = min(position - start, stop - 1 - position)
            if line_id not in chosen or distance >= chosen[line_id][0]:
                chosen[line_id] = (distance, answer)
    answers = {line_id: value[1] for line_id, value in chosen.items()}
    return answers, [r for r in records if r]


# ------------------------------------------------------------------ glossary

GLOSSARY_SYSTEM = """You propose the Japanese writing of character names of an anime. For every character you get the English name used in the project. Give the most likely Japanese spelling in kanji and the kana reading. When you do not know the character or cannot be reasonably sure, answer with an empty spelling instead of inventing one. Characters named by a role (a child, a police officer) get an empty spelling."""


def glossary_schema(names):
    item = {'type': 'object', 'additionalProperties': False, 'required': ['name', 'japanese', 'reading', 'sure'],
            'properties': {'name': {'type': 'string', 'enum': list(names)}, 'japanese': {'type': 'string'}, 'reading': {'type': 'string'}, 'sure': {'type': 'boolean'}}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['characters'],
            'properties': {'characters': {'type': 'array', 'items': item, 'minItems': len(names), 'maxItems': len(names)}}}


def propose_glossary(client, names, series='Rascal Does Not Dream of Bunny Girl Senpai (Seishun Buta Yarou)'):
    messages = [{'role': 'system', 'content': GLOSSARY_SYSTEM},
                {'role': 'user', 'content': f'Series: {series}.\nCharacters: ' + json.dumps(list(names), ensure_ascii=False)}]
    started = time.monotonic()
    reply = client.chat(messages, 3000, glossary_schema(names), {'task': 'glossary', 'names': names})
    data = parse_json(reply['content'])
    return {'proposals': data.get('characters', []), 'seconds': round(time.monotonic() - started, 1), 'usage': reply.get('usage')}


# ------------------------------------------------------------------ speech profiles (Q3)

ENDINGS = ['わよ', 'のよ', 'かしら', 'だよ', 'だね', 'でしょ', 'わね', 'わ。', 'よ。', 'ね。', 'ぜ', 'ぞ', 'だ。', 'です', 'ます', 'っす', 'じゃん', 'のだ', 'だろ', 'ですか', 'ましょう']
ADDRESS = ['先輩', 'さん', 'くん', '君', 'ちゃん', 'お兄ちゃん', '兄ちゃん', '先生']


def speech_profiles(lines, min_lines=8):
    """Per character, from lines already attributed with high confidence: pronoun shares, frequent endings and address terms."""
    import glossary
    from collections import Counter, defaultdict
    per = defaultdict(lambda: {'n': 0, 'pronouns': Counter(), 'endings': Counter(), 'address': Counter()})
    for line in lines:
        speaker = line.get('speaker')
        if speaker in (None, 'unknown', 'other') or line.get('confidence') != 'high':
            continue
        row = per[speaker]
        row['n'] += 1
        text = line['text']
        for cls, spellings in glossary.PRONOUNS.items():
            if any(s in text for s in spellings):
                row['pronouns'][cls] += 1
        stripped = text.rstrip('」』）) ')
        for ending in ENDINGS:
            if stripped.endswith(ending) or stripped.rstrip('。！？!?…').endswith(ending.rstrip('。')):
                row['endings'][ending.rstrip('。')] += 1
        for term in ADDRESS:
            if term in text:
                row['address'][term] += 1
    rows = {}
    for speaker, row in per.items():
        if row['n'] < min_lines:
            continue
        pron = ', '.join(f'{k} {v}' for k, v in row['pronouns'].most_common(3)) or 'none seen'
        ends = ', '.join(k for k, _ in row['endings'].most_common(4)) or 'none seen'
        addr = ', '.join(k for k, _ in row['address'].most_common(3)) or 'none seen'
        rows[speaker] = {'lines': row['n'], 'text': f'{speaker} ({row["n"]} lines): first-person {pron}; frequent endings {ends}; address terms used {addr}'}
    return rows
