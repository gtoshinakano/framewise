"""Isolated llama.cpp server for a vision model of ``visual_models`` and the still-image calls made to it.

The model a job really used is the registry's ``VisualModel`` (its ``model_id``/``revision`` are saved with every result)."""
import base64
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .visual_contracts import FRAME_INSTRUCTION, frame_schema, person_context, resolve_frame, series_note, subtitle_note

FRAME_MAX_TOKENS = 3072  # detailed outfits make answers longer; the cap only stops a runaway


@contextmanager
def frame_server(runtime, model, projector, startup_timeout=300, cancel_check=None, parallel=1, context_per_slot=8192, extra_args=()):
    """A private llama-server for one multimodal GGUF model (any model of ``visual_models`` with that runtime)."""
    runtime, model, projector = map(Path, (runtime, model, projector))
    if not all(path.is_file() for path in (runtime, model, projector)):
        raise RuntimeError(f'Vision model {model.name} or llama-server runtime is missing')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+t') as log:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        process = subprocess.Popen([str(runtime), '-m', str(model), '--mmproj', str(projector), '--host', '127.0.0.1', '--port', str(port),
                                    '--n-gpu-layers', '99', '--ctx-size', str(context_per_slot * parallel), '--parallel', str(parallel), '--jinja', *extra_args],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + startup_timeout
            while True:
                if cancel_check:
                    cancel_check()
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError('Frame model server failed to start: ' + log.read()[-1800:])
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('Frame model server did not become ready in time')
                time.sleep(1)
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@contextmanager
def model_server(model, cancel_check=None, parallel=1, context_per_slot=8192, extra_args=None):
    """The llama-server URL for a stage that sends pictures to ``model`` (a ``visual_models.VisualModel``).

    The resident Qwen 3.6 is the server that already answers text and chat, so it is used as it is: nothing is
    stopped or started, and ``parallel``/``context_per_slot`` only describe the stage (its slots are those of
    ``local_text_runtime.server_slots``). Any other model gets its own private server for the stage, as before; the caller
    has then stopped the resident one (``needs_device`` of the job kind).
    """
    if model.resident:
        from . import local_text_runtime
        runtime = local_text_runtime.runtime
        if runtime is None:
            raise RuntimeError('Local text model worker is unavailable')
        with runtime.visual_session(cancel_check, model.variants) as url:
            yield url
        return
    with frame_server(*model.server_files(), cancel_check=cancel_check, parallel=parallel, context_per_slot=context_per_slot,
                      extra_args=model.private_args() if extra_args is None else extra_args) as url:
        yield url


def workers(model, wanted, what='A visual stage'):
    """How many requests a stage may send at once: its own setting, and for the resident server never more than its slots."""
    if not model.resident:
        return wanted
    from . import local_text_runtime
    return local_text_runtime.clamp_parallel(wanted, what)


def thinking_kwargs(enabled=False):
    """The per-request switch of the model's reasoning (Qwen and Inkling read different variables, see ``local_text_runtime``).

    Pictures are described without it unless a call asks for it."""
    from . import local_text_runtime
    return local_text_runtime.thinking_kwargs(enabled)


def _chat_body(messages, schema, max_tokens, thinking=False):
    response_format = {'type': 'json_schema', 'json_schema': {'name': 'frame', 'strict': True, 'schema': schema}} if schema else {'type': 'json_object'}
    return {'messages': messages, 'temperature': 0, 'max_tokens': max_tokens, 'response_format': response_format, 'chat_template_kwargs': thinking_kwargs(thinking)}


def describe_frame(server_url, image_path, subtitle_regions=None, character_context=None, cancel_check=None, usage_out=None, subtitle_language=None,
                   people=(), frame_label='first frame', thinking=False):
    """Return a validated frame description from a still, never audio.

    ``people`` are the user-marked characters of this frame (``ref``, ``name``, ``box``, ids); the model
    only sees their short handles and the stored result carries the real records.
    """
    people = list(people)
    image = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    parts = [f'This image is the {frame_label} of the cut.', person_context(people, frame_label)]
    note = series_note(character_context)
    if note:
        parts.append(note)
    parts.append(subtitle_note(subtitle_language, subtitle_regions, 'image pixels').strip())
    # The fixed instruction goes first so the server can reuse its cached prefix between cuts.
    messages = [{'role': 'system', 'content': FRAME_INSTRUCTION},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}},
                                             {'type': 'text', 'text': '\n\n'.join(part for part in parts if part)}]}]
    schema = frame_schema([person['ref'] for person in people])
    try:
        from .inference_worker import request_json
        def ask(body):
            request = urllib.request.Request(server_url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
            return request_json(request, cancel_check, timeout=600)
        try:
            payload = ask(_chat_body(messages, schema, FRAME_MAX_TOKENS, thinking))
        except urllib.error.HTTPError as exc:
            if exc.code not in (400, 422, 500):
                raise
            payload = ask(_chat_body(messages, None, FRAME_MAX_TOKENS, thinking))  # a server that cannot build the grammar still answers in plain JSON
        if usage_out is not None and isinstance(payload.get('usage'), dict):
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if isinstance(payload['usage'].get(key), int):
                    usage_out[key] = payload['usage'][key]
        choice = payload['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise ValueError('Response was truncated')
        return resolve_frame(choice['message']['content'], people)
    except (OSError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise RuntimeError(f'Frame description failed: {exc}') from exc


def match_character(server_url, target_image, references, cancel_check=None):
    """Compare a detected person with labeled crops; confidence is model-reported."""
    if not references:
        return None
    content = [{'type': 'text', 'text': 'Target person to identify:'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(target_image).decode('ascii')}}]
    for ref in references:
        content.append({'type': 'text', 'text': 'Reference for character ID ' + ref['character_context_id'] + ' (' + ref['title'] + '):'})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(ref['image']).decode('ascii')}})
    candidates = list({ref['character_context_id']: ref['title'] for ref in references}.items())
    content.append({'type': 'text', 'text': 'Compare the first image (target) against the labeled reference images above. Candidate character IDs and names: ' + json.dumps(candidates, ensure_ascii=False) + '. Return only JSON with exactly character_id, confidence, reason. character_id must be a candidate ID from this list or null; confidence must be high, medium, or low. Use null and low when uncertain. High requires strong distinctive visible similarity; similar clothing alone is not enough. Do not infer identity from names, text, or narrative.'})
    request = urllib.request.Request(server_url + '/v1/chat/completions', data=json.dumps({'messages': [{'role': 'user', 'content': content}],
        'temperature': 0, 'max_tokens': 512, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': thinking_kwargs()}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        from .inference_worker import request_json
        choice = request_json(request, cancel_check, timeout=600)['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise ValueError('Response was truncated')
        result = json.loads(choice['message']['content'])
        if not isinstance(result, dict) or set(result) != {'character_id', 'confidence', 'reason'} or result['confidence'] not in ('high', 'medium', 'low') or not isinstance(result['reason'], str) or result['character_id'] not in {None, *(ref['character_context_id'] for ref in references)}:
            raise ValueError('Invalid character match response')
        return result
    except (OSError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise RuntimeError(f'Character comparison failed: {exc}') from exc
