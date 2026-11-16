"""The two model stages of the experiment, with the validator retry loop and per-call checkpoints.

* frames: Qwen3.5-27B through llama-server (the app's ``frame_model.frame_server``), the first frame of
  every sample, normal and mirrored, for every approach.
* video: Qwen3-VL-8B through the app's ``video_model.analyze_video`` (windows, roster, speech and
  earlier-window notes are the production ones); only the instruction and ``generate_window`` are replaced.

Every finished call is one JSON line in ``raw/<stage>.jsonl``; a restarted stage skips the keys it finds.
"""
import base64
import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pose_schema as ps
import prompts

log = logging.getLogger('pose.run')
MAX_ATTEMPTS = 3
FRAME_MAX_TOKENS = 4096
VIDEO_MAX_TOKENS = 4096
RAW_LIMIT = 12000


class Store:
    """Append-only call records of one stage (thread-safe), resumable."""

    def __init__(self, run_dir, stage):
        self.path = Path(run_dir) / 'raw' / f'{stage}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o777)  # the container's user differs from the host user's
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


def clip_raw(text):
    text = text or ''
    return text if len(text) <= RAW_LIMIT else text[:RAW_LIMIT] + f'… [{len(text) - RAW_LIMIT} more characters]'


# -- frame stage ------------------------------------------------------------------------------------------------------

class LlamaClient:
    """One llama-server chat endpoint. ``chat`` returns the reply text, finish reason and token usage."""

    def __init__(self, url, timeout=900):
        self.url, self.timeout = url, timeout

    def _post(self, body):
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, messages, max_tokens, schema=None, hint=None, thinking=False, temperature=0):
        def body(fmt):
            return {'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'response_format': fmt, 'chat_template_kwargs': {'enable_thinking': thinking}}
        try:
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'pose', 'strict': True, 'schema': schema}} if schema else {'type': 'json_object'}
            payload = self._post(body(fmt))
        except urllib.error.HTTPError as exc:
            if not schema or exc.code not in (400, 422, 500):
                raise
            payload = self._post(body({'type': 'json_object'}))  # a server that cannot build the grammar still answers in JSON
        choice = payload['choices'][0]
        return {'content': choice['message'].get('content') or '', 'finish_reason': choice.get('finish_reason'), 'usage': payload.get('usage') or {}}


def mirror_people(people):
    """The marked people as they stand on the flipped frame: boxes mirrored, handles renumbered left to right, plus new -> old handle."""
    flipped = [{**person, 'box': [1000 - person['box'][2], person['box'][1], 1000 - person['box'][0], person['box'][3]] if person.get('box') else None} for person in people]
    flipped.sort(key=lambda person: person['box'][0] if person.get('box') else 0)
    mapping = {}
    for number, person in enumerate(flipped, 1):
        mapping[f'B{number}'] = person['ref']
        person['ref'] = f'B{number}'
    return flipped, mapping


def make_mirror_image(source, target):
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        ImageOps.mirror(image.convert('RGB')).save(target, quality=92)


def strip_regions(subtitle, width, height):
    """The subtitle strip as a pixel box on the 1280-wide analysis frame (as ``single_cut_pipeline.strip_box``)."""
    if not subtitle or subtitle.get('top') is None:
        return []
    canvas_width = min(width or 1280, 1280)
    canvas_height = max(2, round((height or 720) * canvas_width / (width or 1280) / 2) * 2)
    return [[0, round(subtitle['top'] * canvas_height), canvas_width, round(subtitle['bottom'] * canvas_height)]]


def frame_call(client, sample, approach, variant, image_path, people, mirror_map=None, grammar=False, max_attempts=MAX_ATTEMPTS):
    """Ask for one frame, validate ``pose_detail`` and retry with the validator's complaints; returns the call record."""
    ids = [person['ref'] for person in people]
    image = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    regions = strip_regions(sample.get('subtitle'), sample.get('width'), sample.get('height'))
    text = prompts.frame_user_text(people, sample.get('hints'), regions, (sample.get('subtitle') or {}).get('language'))
    messages = [{'role': 'system', 'content': prompts.frame_system(approach)},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}}, {'type': 'text', 'text': text}]}]
    schema = None
    if grammar:
        from app.visual_contracts import frame_schema
        schema = frame_schema(ids)
        schema['properties']['pose_detail'] = ps.json_schema(approach, ids)
        schema['required'] = [*schema['required'], 'pose_detail']
    attempts, answer, last = [], None, None
    for number in range(1, max_attempts + 1):
        started = time.monotonic()
        reply = client.chat(messages, FRAME_MAX_TOKENS, schema, hint={'approach': approach, 'ids': ids, 'variant': variant, 'sample': sample['id'], 'attempt': number, 'mirror_map': mirror_map})
        seconds = reply.get('simulated_seconds') or time.monotonic() - started
        raw = reply['content']
        errors, parsed = [], None
        if reply.get('finish_reason') == 'length':
            errors = ['the answer was cut off by the length limit; return a complete, shorter JSON object (fewer words in the text fields)']
        else:
            try:
                parsed = ps.parse_json(raw)
                errors = ['the answer must be one JSON object'] if not isinstance(parsed, dict) else ps.validate_answer(parsed.get('pose_detail'), approach, ids)
            except ValueError as exc:
                errors = [str(exc)]
        usage = reply.get('usage') or {}
        attempts.append({'n': number, 'seconds': round(seconds, 2), 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                         'finish': reply.get('finish_reason'), 'errors': errors, 'raw': clip_raw(raw)})
        if isinstance(parsed, dict) and isinstance(parsed.get('pose_detail'), list):
            last = parsed['pose_detail']
        if not errors:
            answer = parsed['pose_detail']
            break
        messages = [*messages, {'role': 'assistant', 'content': raw}, {'role': 'user', 'content': prompts.frame_feedback(errors)}]
    return {'key': f'frame|{approach}|{sample["id"]}|{variant}', 'stage': 'frame', 'approach': approach, 'sample': sample['id'], 'variant': variant,
            'ok': answer is not None, 'first_try_ok': not attempts[0]['errors'], 'attempts': attempts, 'seconds': round(sum(a['seconds'] for a in attempts), 2),
            'answer': answer, 'last_answer': None if answer is not None else last, 'mirror_map': mirror_map, 'handles': ids,
            'error': None if answer is not None else 'invalid after ' + str(len(attempts)) + ' attempts: ' + '; '.join(attempts[-1]['errors'][:3]), 'finished_at': time.time()}


def run_frames(store, samples, approaches, client, thumbs, scratch, mirror=True, grammar=False, parallel=1, progress=None):
    """Every (sample, approach, variant) not yet in the store; sample-major so partial results are readable early."""
    work = []
    for sample in samples:
        people = [{'ref': p['handle'], 'name': p['name'], 'box': p['box']} for p in sample['marked']]
        flipped, mapping = mirror_people(people) if mirror and people else (people, {})
        variants = [('normal', people, None, Path(thumbs) / f'{sample["id"]}.jpg')]
        if mirror and people:  # nobody to compare in an empty frame
            variants.append(('mirror', flipped, mapping, Path(scratch) / f'{sample["id"]}-mirror.jpg'))
        for approach in approaches:
            for variant, who, mapping_, image in variants:
                if f'frame|{approach}|{sample["id"]}|{variant}' not in store.records:
                    work.append((sample, approach, variant, who, mapping_, image))
    total = len(store.records) + len(work)
    log.info('frames: %d calls to run, %d already recorded', len(work), len(store.records))
    lock = threading.Lock()

    def one(item):
        sample, approach, variant, who, mapping_, image = item
        if variant == 'mirror' and not image.is_file():
            make_mirror_image(Path(thumbs) / f'{sample["id"]}.jpg', image)
        record = frame_call(client, sample, approach, variant, image, who, mapping_, grammar)
        store.add(record)
        with lock:
            if progress:
                progress(len(store.records), total, f'{sample["id"]} {approach} {variant}')
        log.info('frame %s %s %s: ok=%s attempts=%d %.1fs', sample['id'], approach, variant, record['ok'], len(record['attempts']), record['seconds'])

    if parallel <= 1:
        for item in work:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for future in [pool.submit(one, item) for item in work]:
                future.result()


# -- video stage ------------------------------------------------------------------------------------------------------

def automatic_video_fps(duration):
    """Copy of ``main.automatic_video_fps`` (importing ``main`` needs FastAPI)."""
    return min(30.0, (30.0 if duration <= 6.0 else 56.0) / duration)


def video_config(sample):
    return {'video_fps': automatic_video_fps(max(0.000001, sample['end'] - sample['start'])), 'video_width': 640, 'video_height': 360, 'video_frame_budget': 60}


def _canonical(name, roster):
    from app.visual_contracts import _match_name
    match = _match_name(name, roster) if isinstance(name, str) else None
    return match['name'] if match else name


def make_generator(approach, windows, roster, fake=None, max_attempts=MAX_ATTEMPTS):
    """A ``video_model.generate_window`` replacement: same contract, but validating ``pose_detail`` and keeping every attempt."""
    from app import video_model

    def generate(processor, model, torch, clip, fps, prompt, duration, start, on_invalid_response=None, cancel_check=None):
        window = {'index': len(windows), 'start': start, 'duration': round(duration, 3), 'attempts': [], 'ok': False, 'answer': None, 'characters': []}
        windows.append(window)
        correction = ''
        for number in range(1, max_attempts + 1):
            began = time.monotonic()
            if fake is not None:
                raw, prompt_count, completion_count, simulated = fake.generate(approach, prompt, number, window['index'], [p['name'] for p in roster])
            else:
                inputs = video_model.video_inputs(processor, clip, fps, prompt + correction).to(model.device)
                with torch.inference_mode():
                    tokens = model.generate(**inputs, max_new_tokens=VIDEO_MAX_TOKENS, do_sample=False)
                raw = processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                prompt_count = int(inputs['input_ids'].shape[-1])
                completion_count = max(0, int(tokens.shape[-1]) - prompt_count)
                simulated = None
            seconds = simulated or time.monotonic() - began
            errors, output, names = [], None, []
            try:
                output = video_model.validate_video(raw, duration, start)
                if not isinstance(output, dict):
                    errors = ['the answer must be one JSON object']
                else:
                    people = output.get('characters')
                    names = [item.get('name') for item in people if isinstance(item, dict) and isinstance(item.get('name'), str)] if isinstance(people, list) else []
                    errors = [] if isinstance(people, list) else ['characters must be an array']
                    errors += ps.validate_video_answer(output.get('pose_detail'), approach, names)
            except ValueError as exc:
                errors = [f'the answer is not valid JSON ({exc})']
            if errors and completion_count >= VIDEO_MAX_TOKENS:
                errors.insert(0, f'the output reached the {VIDEO_MAX_TOKENS}-token limit; be more concise')
            window['attempts'].append({'n': number, 'seconds': round(seconds, 2), 'prompt_tokens': prompt_count, 'completion_tokens': completion_count,
                                       'finish': 'length' if completion_count >= VIDEO_MAX_TOKENS else 'stop', 'errors': errors, 'raw': clip_raw(raw)})
            if not errors:
                answer = output['pose_detail']
                for entry in answer:
                    entry['character'] = _canonical(entry.get('character'), roster)
                window.update(ok=True, answer=answer, characters=[_canonical(name, roster) for name in names], overview=output.get('overview'),
                              camera=output.get('camera'))
                usage = {'prompt_tokens': prompt_count, 'completion_tokens': completion_count, 'total_tokens': prompt_count + completion_count}
                return output, usage, number
            correction = prompts.video_correction(errors)
        raise RuntimeError(f'Invalid pose_detail for {start:.2f}s after {max_attempts} attempts: ' + '; '.join(window['attempts'][-1]['errors'][:3]))

    return generate


def video_call(runtime, model_path, sample, approach, scratch, fake=None):
    """One cut through the production windowing with the approach's prompt; returns the call record."""
    from app import media, video_model
    from app.visual_contracts import video_roster
    roster = video_roster([{'name': p['name'], 'character_context_id': p['character_context_id'], 'box': p['box']} for p in sample['marked']], None, sample['initial'])
    windows = []
    proxy = Path(sample['proxy'])
    cut = {'start_time': sample['start'], 'end_time': sample['end'], 'first_frame_time': sample['first_frame_time'], 'last_frame_time': sample['last_frame_time']}
    saved = (video_model.VIDEO_INSTRUCTION, video_model.generate_window)
    video_model.VIDEO_INSTRUCTION = prompts.video_instruction(approach)
    video_model.generate_window = make_generator(approach, windows, roster, fake)
    error = None
    began = time.monotonic()
    try:
        video_model.analyze_video(proxy if proxy.is_file() else Path(sample['original']), cut, video_config(sample), model_path, scratch,
                                  subtitle_strip=sample.get('subtitle'), source_size=(sample.get('width'), sample.get('height')), context_hints=sample.get('hints'),
                                  runtime=runtime, roster=roster, initial=sample['initial'], transcript=sample.get('transcript'))
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
    finally:
        video_model.VIDEO_INSTRUCTION, video_model.generate_window = saved
    attempts = [attempt for window in windows for attempt in window['attempts']]
    seconds = sum(attempt['seconds'] for attempt in attempts) if fake is not None else time.monotonic() - began
    return {'key': f'video|{approach}|{sample["id"]}', 'stage': 'video', 'approach': approach, 'sample': sample['id'], 'ok': bool(windows) and all(w['ok'] for w in windows) and error is None,
            'first_try_ok': bool(windows) and all(w['ok'] and len(w['attempts']) == 1 for w in windows), 'windows': windows, 'seconds': round(seconds, 2),
            'roster': [p['name'] for p in roster], 'error': error, 'finished_at': time.time()}


def run_video(store, samples, approaches, model_path, runtime, fake=None, progress=None):
    total = len(samples) * len(approaches)
    with tempfile.TemporaryDirectory(prefix='pose-video-') as scratch:
        for sample in samples:
            for approach in approaches:
                key = f'video|{approach}|{sample["id"]}'
                if key in store.records:
                    continue
                record = video_call(runtime, model_path, sample, approach, scratch, fake)
                store.add(record)
                if progress:
                    progress(len(store.records), total, f'{sample["id"]} {approach}')
                log.info('video %s %s: ok=%s windows=%d %.1fs', sample['id'], approach, record['ok'], len(record['windows']), record['seconds'])
