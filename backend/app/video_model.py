"""Video motion analysis of one cut, in bounded windows, on a switchable model.

``analyze_video`` owns everything that does not depend on the model (windows, frame budget, saved
windows, the prompt notes, linking names and merging windows into one direction). The model is a
*runtime* with ``load(path)``, ``generate(...)``, ``close()``, ``model_id``, ``revision`` and
``hears_audio``: ``VideoRuntime`` runs Qwen3-VL through transformers, ``LlamaVideoRuntime`` runs any
GGUF model that reads video through the bundled llama-server (LFM3-Omni). ``runtime_for`` picks one
for a ``visual_models.VisualModel``.
"""
import base64
import gc
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import media
from .visual_contracts import AUDIO_NOTE, VIDEO_CONTRACT, VIDEO_INSTRUCTION, cut_direction, earlier_note, link_video, roster_note, series_note, speech_note, subtitle_note, validate_video

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
VIDEO_TOKEN_LIMITS = (2048, 4096)
BOUNDARY_POLICY = 'ui-frame-range-v1'
RETRY_INSTRUCTION = ('\nThe previous response could not be parsed as complete JSON. '
    'Start again from the video and return one complete JSON object. '
    'Be concise: combine repeated or sustained actions, omit redundant events, '
    'and report only distinct changes supported by the frames. '
    'Do not repeat the same event to fill time. Finish the object within the output limit.')


class VideoRuntime:
    """Lazily loaded Qwen3-VL weights that several cuts can share.

    `analyze_video` creates a private runtime when none is passed, so a single
    cut still loads and frees the model by itself. A batch passes one runtime
    for every cut and calls `close()` once at the end.
    """
    model_id, revision, hears_audio = MODEL_ID, MODEL_REVISION, False

    def __init__(self):
        self.torch = self.processor = self.model = None

    def generate(self, clip, fps, prompt, duration, start, on_invalid_response=None, cancel_check=None):
        return generate_window(self.processor, self.model, self.torch, clip, fps, prompt, duration, start, on_invalid_response, cancel_check)

    def load(self, model_path):
        if self.model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            index_path = Path(model_path) / 'model.safetensors.index.json'
            try:
                index = json.loads(index_path.read_text())
                shards = set(index['weight_map'].values())
                if not shards or any(Path(name).name != name or not (Path(model_path) / name).is_file() for name in shards):
                    raise ValueError('Model shards are missing')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError('Qwen3-VL 8B model is missing or incomplete') from exc
            if not torch.cuda.is_available():
                raise RuntimeError('Qwen3-VL video analysis requires a compatible GPU')
            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
        return self.torch, self.processor, self.model

    def close(self):
        if self.model is not None:
            self.model = self.processor = None
            gc.collect()
            self.torch.cuda.empty_cache()


def video_inputs(processor, clip, fps, prompt):
    """Decode the prepared clip with PyAV, preserving frame timestamps."""
    import av
    import numpy as np
    from transformers.video_utils import VideoMetadata
    with av.open(str(clip)) as source:
        frames = [frame.to_ndarray(format='rgb24') for frame in source.decode(video=0)]
    if len(frames) < 2:
        raise RuntimeError('Prepared video has fewer than two frames')
    metadata = VideoMetadata(total_num_frames=len(frames), fps=fps, frames_indices=list(range(len(frames))))
    messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], videos=[np.stack(frames)], video_metadata=[metadata], do_sample_frames=False, return_tensors='pt')


def generate_window(processor, model, torch, clip, fps, prompt, duration, start, on_invalid_response=None, cancel_check=None):
    """Retry malformed output once with a concise prompt; keep every rejection."""
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    usage_available = True
    last_error = ''
    for attempt, limit in enumerate(VIDEO_TOKEN_LIMITS):
        if cancel_check:
            cancel_check()
        inputs = video_inputs(processor, clip, fps, prompt + (RETRY_INSTRUCTION if attempt else '')).to(model.device)
        options = {}
        if cancel_check:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class CheckCancellation(StoppingCriteria):
                next_check = 0.0
                def __call__(self, input_ids, scores, **kwargs):
                    now = time.monotonic()
                    if now >= self.next_check:
                        self.next_check = now + 0.5
                        cancel_check()
                    return False
            options['stopping_criteria'] = StoppingCriteriaList([CheckCancellation()])
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=limit, do_sample=False, **options)
        if cancel_check:
            cancel_check()
        raw = processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            prompt_count = int(inputs['input_ids'].shape[-1])
            completion_count = max(0, int(tokens.shape[-1]) - prompt_count)
            usage['prompt_tokens'] += prompt_count
            usage['completion_tokens'] += completion_count
            usage['total_tokens'] += prompt_count + completion_count
        except (KeyError, TypeError, AttributeError, IndexError, ValueError):
            completion_count = None
            usage_available = False
        try:
            output = validate_video(raw, duration, start)
            return output, usage if usage_available else {}, attempt + 1
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if completion_count is not None and completion_count >= limit:
                last_error = f'Output reached the {limit}-token generation limit; {last_error}'
            if on_invalid_response:
                on_invalid_response(last_error, raw)
    raise RuntimeError(f'Invalid Qwen3-VL video response for {start:.3f}–{start + duration:.3f}s after {len(VIDEO_TOKEN_LIMITS)} attempts: {last_error}')


# One window holds at most 32 frames plus its soundtrack; a 32K slot leaves ample room for the prompt and answer.
LLAMA_VIDEO_CONTEXT = 32768


class LlamaVideoRuntime:
    """A GGUF video model behind a private llama-server, started once for every cut of a stage.

    Each window is sent as an mp4 data URL (with its soundtrack when the model hears audio). The server
    resamples it to the model's own video rate (``sample_fps``; LFM3 reads 2 fps whatever the clip, the
    request or ``--video-fps`` says, verified on build 12110), so the saved window records that rate.
    The answer is validated and retried exactly like the Qwen3-VL runtime.
    """

    def __init__(self, model):
        self.spec = model
        self.model_id, self.revision, self.hears_audio = model.model_id, model.revision, model.hears_audio
        self.sample_fps = model.video_fps
        self.server = self.url = None

    def load(self, _model_path=None):
        if self.url is None:
            from . import frame_model
            self.server = frame_model.frame_server(*self.spec.server_files(), parallel=1, context_per_slot=LLAMA_VIDEO_CONTEXT,
                                                   extra_args=self.spec.server_args)
            self.url = self.server.__enter__()
        return self.url

    def close(self):
        if self.server is not None:
            server, self.server, self.url = self.server, None, None
            server.__exit__(None, None, None)

    def body(self, clip, prompt, limit):
        video = base64.b64encode(Path(clip).read_bytes()).decode('ascii')
        return {'messages': [{'role': 'user', 'content': [
                    {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + video}},
                    {'type': 'text', 'text': prompt}]}],
                'temperature': 0, 'max_tokens': limit, 'response_format': {'type': 'json_object'},
                'chat_template_kwargs': {'enable_thinking': False}}

    def generate(self, clip, fps, prompt, duration, start, on_invalid_response=None, cancel_check=None):
        from .inference_worker import request_json
        url = self.load()
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        last_error = ''
        for attempt, limit in enumerate(VIDEO_TOKEN_LIMITS):
            if cancel_check:
                cancel_check()
            request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(self.body(clip, prompt + (RETRY_INSTRUCTION if attempt else ''), limit)).encode(),
                                             headers={'Content-Type': 'application/json'}, method='POST')
            try:
                payload = request_json(request, cancel_check, timeout=600)
                choice = payload['choices'][0]
                raw = choice['message']['content'] or ''
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f'{self.spec.label} rejected the video request: HTTP {exc.code} {exc.read()[:500].decode("utf-8", "replace")}') from exc
            except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(f'{self.spec.label} video request failed: {exc}') from exc
            for key in usage:
                if isinstance((payload.get('usage') or {}).get(key), int):
                    usage[key] += payload['usage'][key]
            try:
                return validate_video(raw, duration, start), usage, attempt + 1
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if choice.get('finish_reason') == 'length':
                    last_error = f'Output reached the {limit}-token generation limit; {last_error}'
                if on_invalid_response:
                    on_invalid_response(last_error, raw)
        raise RuntimeError(f'Invalid {self.spec.label} video response for {start:.3f}–{start + duration:.3f}s after {len(VIDEO_TOKEN_LIMITS)} attempts: {last_error}')


def runtime_for(model):
    """The video runtime of a ``visual_models.VisualModel``; Qwen3-VL keeps its transformers runtime."""
    if model is None or model.runtime == 'transformers':
        return VideoRuntime()
    if model.runtime == 'llama-server':
        return LlamaVideoRuntime(model)
    raise RuntimeError(f'No video runtime for {model.label}')


def analyze_video(original, cut, config, model_path, scratch, on_window=None, subtitle_strip=None, source_size=None, load_window=None, save_window=None, context_hints=None, on_invalid_response=None, cancel_check=None, runtime=None,
                  roster=None, initial=None, transcript=None):
    """Process every cut window, recording both clip-relative and episode times.

    Each model invocation is bounded to limit video tokens. Windows partition the
    cut; events spanning a window boundary may be split. Clip files are transient and timing metadata is retained.

    ``roster`` (visual_contracts.video_roster) names the characters to follow, ``initial`` is the cut's
    first-frame result and ``transcript`` its spoken segments; each window is also told what the window
    before it reported. The windows are merged into one ``direction`` for the whole cut.
    """
    owns_runtime = runtime is None
    runtime = runtime or VideoRuntime()
    hears_audio = getattr(runtime, 'hears_audio', False) is True
    windows = []
    start, cut_end = float(cut['start_time']), float(cut['end_time'])
    first, last = cut.get('first_frame_time'), cut.get('last_frame_time')
    if first is not None and last is not None:
        first, last = float(first), float(last)
        # Frame times come from ffprobe presentation timestamps, which can sit a few microseconds
        # (never more than a millisecond) outside the stored cut boundary; clamp instead of failing.
        if not all(math.isfinite(value) for value in (first, last)) or not start - .001 <= first <= last < cut_end + .001:
            raise ValueError('Cut preview frame timestamps are invalid; refresh the cut frames before analysis')
        start, cut_end = max(start, first), min(cut_end, last + .000002)
    fps = config['video_fps']
    frame_budget = config.get('video_frame_budget')
    window_seconds = min(max(8.0, 2.0 / fps), 32.0 / fps) if frame_budget else min(8.0, 32.0 / fps)
    window_count = max(1, math.ceil((cut_end - start) / window_seconds - 1e-12))
    if frame_budget and window_count * 2 > frame_budget:
        raise ValueError('Frame budget is too small for the requested video frame rate')
    remaining_frames = frame_budget
    window_index = 0
    try:
        while window_index < window_count:
            if cancel_check:
                cancel_check()
            end = cut_end if window_index == window_count - 1 else min(cut_end, start + window_seconds)
            cached = load_window(window_index, start, end) if load_window else None
            if cached is not None and (cached.get('source', {}).get('boundary_policy') != BOUNDARY_POLICY
                                       or cached['source'].get('contract') != VIDEO_CONTRACT  # saved under an older prompt
                                       or cached['source'].get('source_file') != Path(original).name):
                cached = None
            if cached is not None:
                if remaining_frames is not None:
                    remaining_frames -= cached['source']['frame_count']
                windows.append(cached)
                start, window_index = end, window_index + 1
                continue
            if on_window:
                on_window(start, end)
            runtime.load(model_path)
            clip = Path(scratch) / 'video-window.mp4'
            sound = {'keep_audio': True} if hears_audio else {}
            if remaining_frames is None:
                metadata = media.prepare_analysis_video(Path(original), clip, start, end, config['video_width'], config['video_height'], fps, **sound)
            else:
                limit = min(32, remaining_frames - 2 * (window_count - window_index - 1))
                metadata = media.prepare_analysis_video(Path(original), clip, start, end, config['video_width'], config['video_height'], fps, frame_limit=limit, **sound)
                frame_probe = media.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(clip)])
                count = int(json.loads(frame_probe.stdout)['streams'][0]['nb_read_frames'])
                if count < 2 or count > limit:
                    raise RuntimeError('Prepared video frame count is outside its allocated budget')
                metadata['frame_count'] = count
                remaining_frames -= count
            metadata['boundary_policy'] = BOUNDARY_POLICY
            metadata['source_file'] = Path(original).name
            metadata['contract'] = VIDEO_CONTRACT
            if isinstance(getattr(runtime, 'sample_fps', None), (int, float)):
                metadata['model_sample_fps'] = runtime.sample_fps  # the rate the model really read, not the clip's
            boxes = []
            if subtitle_strip and subtitle_strip.get('top') is not None and source_size and all(source_size):
                width, height = source_size
                scale = min(config['video_width'] / width, config['video_height'] / height)
                offset_x = (config['video_width'] - width * scale) / 2
                offset_y = (config['video_height'] - height * scale) / 2
                boxes = [[round(offset_x), round(subtitle_strip['top'] * height * scale + offset_y),
                          round(width * scale + offset_x), round(subtitle_strip['bottom'] * height * scale + offset_y)]]
            final = end >= cut_end - 0.000001
            part = 'the whole cut, from its first to its last frame.' if window_count == 1 else (
                f'part {window_index + 1} of {window_count} of the cut, ' + ('the final part: its last frames are the end of the cut.' if final
                                                                         else 'an intermediate part: the cut continues after it, so its end is not a hard cut.'))
            prompt = VIDEO_INSTRUCTION + f'\n\nClip: {end-start:.2f} seconds, {part} Ignore any duplicated tail frame and give no times beyond that duration.'
            if 'single_sample_time' in metadata:
                prompt += ' It holds only one distinct image: infer no motion or change.'
            prompt += '\n\n' + roster_note(roster or [], initial if window_index == 0 else None)
            if transcript is not None:
                prompt += '\n' + speech_note(transcript, start, end, roster)
            if windows:
                prompt += '\n\n' + earlier_note(windows[-1])
            notes = series_note(context_hints)
            if notes:
                prompt += '\n' + notes
            prompt += subtitle_note((subtitle_strip or {}).get('language'), boxes, 'video canvas pixels')
            if metadata.get('has_audio'):
                prompt += '\n\n' + AUDIO_NOTE
            reject = (lambda error, raw: on_invalid_response(window_index, start, end, error, raw)) if on_invalid_response else None
            output, usage, attempts = runtime.generate(clip, config['video_fps'], prompt, end - start, start, reject, cancel_check)
            result = {'source': metadata, 'analysis': link_video(output, roster), 'generation_attempts': attempts}
            if attempts > 1:
                result['retry_instruction'] = RETRY_INSTRUCTION.strip()
            if usage:
                result['usage'] = usage
            if save_window:
                save_window(window_index, result)
            windows.append(result)
            clip.unlink(missing_ok=True)
            start, window_index = end, window_index + 1
        usage = {key: sum(window.get('usage', {}).get(key, 0) for window in windows) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')} if windows and all('usage' in window for window in windows) else {}
        return {'model': runtime.model_id, 'revision': runtime.revision, 'contract': VIDEO_CONTRACT, 'direction': cut_direction(windows, roster), 'windows': windows, 'usage': usage,
                'limitations': 'Window boundaries may split events. Model-provided event times are unverified and may overlap, be incomplete, or fall outside the source window.'}
    finally:
        if owns_runtime:
            runtime.close()
