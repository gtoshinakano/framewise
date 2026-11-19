"""Transcribe one cut's audio with an OpenRouter model that accepts sound.

The app's own transcript is local speech-to-text plus diarization, so its words,
its speaker labels and its character links can all be wrong. This module sends
the same audio to a second model on another machine, which hears only that clip,
so Cut Assistance can compare three readings of the same speech: the transcript,
the burned-in subtitles and a fresh transcription of the audio with its own split
of the voices. Only the words and the speaker labels come from here: a model
guesses at timing, so the app's own timings stay the truth.
"""
import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from .. import db

log = logging.getLogger('framewise.audio_transcribe')

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
# A free NVIDIA model with audio input; the user can name another in the Text engine dialog.
DEFAULT_AUDIO_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'
MAX_SECONDS = 60.0    # past this the request body grows and one answer stops being reliable
MIN_SECONDS = 0.5     # shorter than this is a click, not speech
REQUEST_SECONDS = 180
ATTEMPTS = 3
MAX_ANSWER_TOKENS = 4096

SYSTEM = (
    'You transcribe speech for a video production team. You hear one short clip from an animated scene and write down '
    'exactly what is said. Work only from the audio.\n'
    '- Write the words verbatim in the language spoken. Never translate, never paraphrase, never add or drop words.\n'
    '- Give one segment per utterance, in the order they occur. Keep each segment to one speaker turn.\n'
    '- Label every distinct voice S1, S2, S3, and keep one label for one voice through the whole clip. A new voice '
    'starts a new segment even when the words run on; when two voices overlap, give a segment for each.\n'
    '- Give start and end in seconds from the beginning of the clip, in order and inside its length. The times are '
    'your estimate, so round to 0.1 second; the words matter more than the times.\n'
    '- Do not transcribe music, sound effects or ambient noise, never read text that appears on screen, and do not '
    'describe the picture.\n'
    '- When a word is unclear, write the most likely reading in the spoken language; never invent words to fill a gap.\n'
    '- If the clip has no speech, return an empty segments list.\n'
    'Answer with one JSON object and nothing else: '
    '{"segments":[{"start":0.0,"end":2.4,"speaker":"S1","text":"..."}]}'
)

JSON_FENCE = re.compile(r'^```[a-zA-Z]*\s*|\s*```$')


class TranscriptionError(RuntimeError):
    """The audio or the remote model could not answer; the message says why, in words the model can act on."""


def settings_model(conn=None):
    """The audio model the user chose, or the app's default when the row is older than this setting."""
    row = db.one('SELECT openrouter_audio_model FROM text_generation_settings WHERE id=1', conn=conn)
    model = (row['openrouter_audio_model'] if row else '').strip()
    return model or DEFAULT_AUDIO_MODEL


def audio_source(project_id, episode_id, original_ext):
    """The file to transcribe: the prepared audio track, else the original upload, or None."""
    from .. import media

    path = media.episode_dir(project_id, episode_id)
    prepared = path / 'audio.wav'
    if prepared.is_file():
        return prepared
    original = path / f'original{original_ext or ""}'
    return original if original.is_file() else None


def clip(source, target, start, end):
    """Cut the window as 16 kHz mono WAV, with the same encoder flags as the app's own transcription."""
    from ..audio_worker import clip_audio

    clip_audio(source, target, start, end)  # absolute episode seconds; lazy import keeps the tool module light
    return target


def transcribe(model, clip_path, seconds):
    """Send one clip to the model and return (segments, speakers)."""
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        raise TranscriptionError('OPENROUTER_API_KEY is not configured for the inference worker, so the audio cannot be sent to OpenRouter')
    with clip_path.open('rb') as handle:
        audio = base64.b64encode(handle.read()).decode()
    payload = {'model': model, 'max_tokens': MAX_ANSWER_TOKENS, 'temperature': 0.0,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': [{'type': 'text', 'text': f'The clip is {seconds:.1f} seconds long. Transcribe it.'},
                                                         {'type': 'input_audio', 'input_audio': {'data': audio, 'format': 'wav'}}]}],
               'response_format': {'type': 'json_object'}}
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'X-Title': 'Framewise Cut Assistance'}
    answer, last_error = '', 'the audio request failed'
    for attempt in range(ATTEMPTS):
        try:
            answer = _answer(payload, headers)
            break
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code == 400 and payload.get('response_format') and 'response_format' in detail:
                # This route does not support the JSON format; the prompt still asks for one object.
                log.warning('audio model %s rejected response_format (HTTP %s); asking without it', model, exc.code)
                payload.pop('response_format')
                continue
            last_error = f'OpenRouter refused the audio (HTTP {exc.code}){detail}'
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == ATTEMPTS - 1:
                raise TranscriptionError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = f'OpenRouter could not answer the audio request: {str(exc) or type(exc).__name__}'
            if attempt == ATTEMPTS - 1:
                raise TranscriptionError(last_error) from exc
        time.sleep(2 ** (attempt + 1))
    segments = answer_segments(answer, seconds)
    log.info('audio transcribed by %s: %s segments in %s s of audio', model, len(segments), round(seconds, 1))
    return segments, sorted({item['speaker'] for item in segments if item['speaker']})


def _answer(payload, headers):
    request = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=REQUEST_SECONDS) as response:
        result = json.load(response)
    try:
        content = result['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise TranscriptionError('the audio model answered without a message') from exc
    if not isinstance(content, str) or not content.strip():
        raise TranscriptionError('the audio model answered with nothing')
    return content


def _error_detail(exc):
    try:
        message = json.loads(exc.read()).get('error', {}).get('message', '')
    except (ValueError, TypeError, AttributeError):
        message = ''
    return f': {message[:300]}' if message else ''


def answer_segments(text, duration):
    """The model's answer as ordered segments with times clamped to the clip; a lenient read of a free model."""
    body = JSON_FENCE.sub('', text.strip())
    first, last = body.find('{'), body.rfind('}')
    if first < 0 or last <= first:
        raise TranscriptionError('the audio model did not return the JSON object it was asked for')
    try:
        data = json.loads(body[first:last + 1])
    except ValueError as exc:
        raise TranscriptionError(f'the audio model returned JSON that could not be read: {exc}') from exc
    rows = data.get('segments') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise TranscriptionError('the audio model answered without a segments list')
    segments = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        text = item.get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        start, end = _time(item.get('start'), duration), _time(item.get('end'), duration)
        if start is not None and end is not None and end < start:
            end = min(duration, start + 0.1)
        segments.append({'start': start, 'end': end,
                         'speaker': str(item.get('speaker') or '').strip(), 'text': text.strip()})
    return segments


def _time(value, duration):
    """One timestamp the model invented, clamped into the clip, or None when it gave no usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(duration, float(value)))
