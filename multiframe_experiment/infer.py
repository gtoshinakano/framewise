"""Model calls: the llama-server client, per-call checkpoints, and the window loop with the validator retry.

Every finished call is one JSON line in ``raw/<name>.jsonl``; a restarted stage skips the keys it already has.
"""
import base64
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import prompts
import schema

log = logging.getLogger('mfx.infer')
MAX_ATTEMPTS = 3
MAX_TOKENS = 4096
RAW_LIMIT = 16000

# Experiment key -> app.visual_models key. Both run through the bundled llama-server.
MODELS = {
    'lfm3': {'visual_key': 'lfm3-omni-2b', 'label': 'LFM3-Omni-2B · BF16', 'context_per_slot': 16384},
    'qwen35': {'visual_key': 'qwen35-27b', 'label': 'Qwen3.5-27B · Q4_K_M', 'context_per_slot': 16384},
}
BASELINE_KEY = 'baseline|production|video'
# Setups without a model call of their own (computed on the CPU from keypoints and other answers).
PSEUDO = {'dwpose': 'DWPose whole-body + rules (CPU)', 'hybrid': 'Hybrid: DWPose body + Qwen3.5 rest'}
LABELS = {**{key: value['label'] for key, value in MODELS.items()}, **PSEUDO}


def config_key(model, style, context):
    return f'{model}|{style}|{context}'


class Store:
    """Append-only call records of one stage (thread-safe), resumable."""

    def __init__(self, run_dir, name):
        self.path = Path(run_dir) / 'raw' / f'{name}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o777)  # the container user differs from the host user
        except OSError:
            pass
        self.lock = threading.Lock()
        self.records = {}
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                try:
                    record = json.loads(line)
                    self.records[record['key']] = record
                except (ValueError, KeyError):
                    continue

    def add(self, record):
        with self.lock:
            with self.path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.records[record['key']] = record


def load_records(run_dir, name):
    return Store(run_dir, name).records if (Path(run_dir) / 'raw' / f'{name}.jsonl').is_file() else {}


def clip(text):
    text = text or ''
    return text if len(text) <= RAW_LIMIT else text[:RAW_LIMIT] + f'… [{len(text) - RAW_LIMIT} more characters]'


class LlamaClient:
    """One llama-server chat endpoint. ``chat`` returns content, finish reason, usage, timings and whether the grammar applied."""

    def __init__(self, url, timeout=900):
        self.url, self.timeout = url, timeout

    def _post(self, body):
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, messages, max_tokens, schema_=None, hint=None, temperature=0):
        def body(fmt):
            return {'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'response_format': fmt,
                    'chat_template_kwargs': {'enable_thinking': False}}
        grammar = bool(schema_)
        try:
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True, 'schema': schema_}} if schema_ else {'type': 'json_object'}
            payload = self._post(body(fmt))
        except urllib.error.HTTPError as exc:
            if not schema_ or exc.code not in (400, 422, 500):
                raise
            grammar = False
            payload = self._post(body({'type': 'json_object'}))  # a server that cannot build the grammar still answers JSON
        choice = payload['choices'][0]
        return {'content': choice['message'].get('content') or '', 'finish_reason': choice.get('finish_reason'), 'usage': payload.get('usage') or {},
                'timings': payload.get('timings') or {}, 'grammar': grammar}


def encode_images(run_dir, frames):
    return [base64.b64encode((Path(run_dir) / frame['path']).read_bytes()).decode('ascii') for frame in frames]


def call_window(client, model, style, context, sample, window, images, previous=None, use_grammar=True, max_attempts=MAX_ATTEMPTS):
    """One window through one config, validated and retried; returns the call record."""
    messages = prompts.build_messages(sample, window, style, context, images, previous)
    prompt = prompts.prompt_text(messages)
    grammar = schema.json_schema(window) if use_grammar else None
    attempts, answer = [], None
    for number in range(1, max_attempts + 1):
        started = time.monotonic()
        hint = {'task': 'answer', 'model': model, 'style': style, 'context': context, 'labels': schema.labels(window), 'roster': [p['name'] for p in sample.get('roster') or []],
                'attempt': number, 'key': f"{sample['id']}|{window['index']}|{style}|{context}"}
        try:
            reply = client.chat(messages, MAX_TOKENS, grammar, hint)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({'n': number, 'seconds': round(time.monotonic() - started, 2), 'errors': [f'request failed: {exc}'], 'raw': ''})
            continue
        seconds = reply.get('simulated_seconds') or time.monotonic() - started
        raw = reply['content']
        errors = []
        if reply.get('finish_reason') == 'length':
            errors = ['the answer was cut off by the length limit; return a complete, shorter JSON object (shorter descriptions)']
        else:
            try:
                answer, errors = schema.validate(schema.parse_json(raw), window)
            except ValueError as exc:
                answer, errors = None, [f'the answer is not valid JSON ({exc})']
        usage, timings = reply.get('usage') or {}, reply.get('timings') or {}
        attempts.append({'n': number, 'seconds': round(seconds, 2), 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                         'prompt_ms': timings.get('prompt_ms'), 'predicted_ms': timings.get('predicted_ms'), 'predicted_per_second': timings.get('predicted_per_second'),
                         'finish': reply.get('finish_reason'), 'grammar': reply.get('grammar'), 'errors': errors, 'raw': clip(raw)})
        if not errors:
            break
        answer = None
        # A cut-off answer is usually a repetition loop: resend only its start so the retry does not carry it (or blow the context).
        shown = raw if reply.get('finish_reason') != 'length' else (raw or '')[:1500] + ' …'
        messages = [*messages[:2], {'role': 'assistant', 'content': shown or '{}'}, {'role': 'user', 'content': schema.format_errors(errors)}]
    return {'key': f"{config_key(model, style, context)}|{sample['id']}|w{window['index']}", 'stage': 'answer', 'model': model, 'style': style, 'context': context,
            'config': config_key(model, style, context), 'sample': sample['id'], 'window': window['index'], 'labels': schema.labels(window),
            'ok': answer is not None, 'first_try_ok': bool(attempts) and not attempts[0]['errors'], 'attempts': attempts,
            'seconds': round(sum(a['seconds'] for a in attempts), 2), 'answer': answer, 'prompt': prompt,
            'error': None if answer is not None else 'invalid after %d attempts: %s' % (len(attempts), '; '.join((attempts[-1]['errors'] if attempts else ['no attempt'])[:3])),
            'finished_at': time.time()}


def run_model(store, samples, model, combos, client, run_dir, use_grammar=True, progress=None):
    """Every (sample, style, context, window) of one model not yet in the store. Windows of a config run in order (chained contexts)."""
    work = sum(len(s['windows']) for s in samples if len(s['frames']) > 1) * len(combos)
    done = sum(1 for key in store.records if key.startswith(model + '|'))
    log.info('%s: %d calls planned, %d already recorded', model, work, done)
    for sample in samples:
        if len(sample['frames']) < 2:
            continue
        cache = {}
        for style, context in combos:
            previous = None
            for window in sample['windows']:
                key = f"{config_key(model, style, context)}|{sample['id']}|w{window['index']}"
                if key in store.records:
                    previous = store.records[key].get('answer')
                    continue
                if window['index'] not in cache:
                    cache[window['index']] = encode_images(run_dir, window['frames'])
                chained = previous if prompts.CONTEXTS[context]['chained'] else None
                record = call_window(client, model, style, context, sample, window, cache[window['index']], chained, use_grammar)
                store.add(record)
                previous = record.get('answer')
                done += 1
                log.info('%s %s/%s %s w%d: ok=%s attempts=%d %.1fs', model, style, context, sample['id'], window['index'], record['ok'], len(record['attempts']), record['seconds'])
                if progress:
                    progress(done, work, f"{sample['id']} w{window['index']} {style}/{context}")
