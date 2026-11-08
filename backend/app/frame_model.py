"""Isolated llama.cpp server for the quantized Qwen3.5 frame model."""
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

from .visual_contracts import FRAME_INSTRUCTION, subtitle_note, validate_frame

MODEL_ID = 'Advantech-EIOT/Qwen3.5-27B-GGUF:Q4_K_M'
MODEL_REVISION = '4d92249e905871588b9c64fddb8ea6498507fa90'


@contextmanager
def frame_server(runtime, model, projector, startup_timeout=300, cancel_check=None, parallel=1, context_per_slot=8192):
    runtime, model, projector = map(Path, (runtime, model, projector))
    if not all(path.is_file() for path in (runtime, model, projector)):
        raise RuntimeError('Qwen3.5 frame model or llama-server runtime is missing')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+t') as log:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        process = subprocess.Popen([str(runtime), '-m', str(model), '--mmproj', str(projector), '--host', '127.0.0.1', '--port', str(port),
                                    '--n-gpu-layers', '99', '--ctx-size', str(context_per_slot * parallel), '--parallel', str(parallel), '--jinja'],
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


def describe_frame(server_url, image_path, subtitle_regions=None, character_context=None, cancel_check=None, reference_images=None, usage_out=None, subtitle_language=None):
    """Return validated structured observations from a still, never audio."""
    image = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    prompt = FRAME_INSTRUCTION + subtitle_note(subtitle_language, subtitle_regions, 'image pixels')
    if character_context:
        prompt += '\nContext and character detections: ' + json.dumps(character_context, ensure_ascii=False) + '. Named boxes marked labels_confirmed_by_queue are user-verified identity assignments: use the supplied name for the corresponding person in the target_frame. Other labels are unverified hints. Boxes apply only to their named frame; do not infer unseen people or actions.'
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}}, {'type': 'text', 'text': prompt}]
    if reference_images:
        content.append({'type': 'text', 'text': 'The first image is the target frame. The following labeled crops are reference examples only; do not describe them as part of the target scene. Identity is uncertain unless visually supported.'})
        for ref in reference_images:
            content.append({'type': 'text', 'text': 'Reference character ' + ref['title'] + ' (ID ' + ref['character_context_id'] + '):'})
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(ref['image']).decode('ascii')}})
    request = urllib.request.Request(server_url + '/v1/chat/completions', data=json.dumps({
        'messages': [{'role': 'user', 'content': content}],
        'temperature': 0, 'max_tokens': 4096, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}
    }).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        from .inference_worker import request_json
        payload = request_json(request, cancel_check, timeout=600)
        if usage_out is not None and isinstance(payload.get('usage'), dict):
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if isinstance(payload['usage'].get(key), int):
                    usage_out[key] = payload['usage'][key]
        choice = payload['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise ValueError('Response was truncated')
        return validate_frame(choice['message']['content'])
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
        'temperature': 0, 'max_tokens': 512, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
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
