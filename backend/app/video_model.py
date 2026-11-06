"""Local Qwen3-VL video inference for bounded windows of a single cut."""
import gc
import json
import math
import time
from pathlib import Path

from . import media
from .visual_contracts import VIDEO_INSTRUCTION, validate_video

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
VIDEO_TOKEN_LIMITS = (2048, 4096)
RETRY_INSTRUCTION = ('\nThe previous response could not be parsed as complete JSON. '
    'Start again from the video and return one complete JSON object. '
    'Be concise: combine repeated or sustained actions, omit redundant events, '
    'and report only distinct changes supported by the frames. '
    'Do not repeat the same event to fill time. Finish the object within the output limit.')


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


def analyze_video(original, cut, config, model_path, scratch, on_window=None, subtitle_samples=None, load_window=None, save_window=None, context_hints=None, on_invalid_response=None, cancel_check=None):
    """Process every cut window, recording both clip-relative and episode times.

    Each model invocation is bounded to limit video tokens. Windows partition the
    cut; events spanning a window boundary may be split. No audio is sent to the
    visual model. Clip files are transient and timing metadata is retained.
    """
    model = processor = torch = None
    windows = []
    start, cut_end = float(cut['start_time']), float(cut['end_time'])
    fps = config['video_fps']
    frame_budget = config.get('video_frame_budget')
    window_seconds = min(max(8.0, 2.0 / fps), 32.0 / fps) if frame_budget else min(8.0, 32.0 / fps)
    window_count = max(1, math.ceil((cut_end - start - 0.000001) / window_seconds))
    if frame_budget and window_count * 2 > frame_budget:
        raise ValueError('Frame budget is too small for the requested video frame rate')
    remaining_frames = frame_budget
    window_index = 0
    try:
        while start < cut_end - 0.000001:
            if cancel_check:
                cancel_check()
            end = min(cut_end, start + window_seconds)
            cached = load_window(window_index, start, end) if load_window else None
            if cached is not None:
                if remaining_frames is not None:
                    remaining_frames -= cached['source']['frame_count']
                windows.append(cached)
                start, window_index = end, window_index + 1
                continue
            if on_window:
                on_window(start, end)
            if model is None:
                import torch as torch_module
                from transformers import AutoModelForImageTextToText, AutoProcessor
                torch = torch_module
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
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
            clip = Path(scratch) / 'video-window.mp4'
            if remaining_frames is None:
                metadata = media.prepare_analysis_video(Path(original), clip, start, end, config['video_width'], config['video_height'], fps)
            else:
                limit = min(32, remaining_frames - 2 * (window_count - window_index - 1))
                metadata = media.prepare_analysis_video(Path(original), clip, start, end, config['video_width'], config['video_height'], fps, frame_limit=limit)
                frame_probe = media.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(clip)])
                count = int(json.loads(frame_probe.stdout)['streams'][0]['nb_read_frames'])
                if count < 2 or count > limit:
                    raise RuntimeError('Prepared video frame count is outside its allocated budget')
                metadata['frame_count'] = count
                remaining_frames -= count
            regions = []
            for sample in subtitle_samples or []:
                if not start <= sample['time'] < end or not sample.get('regions'):
                    continue
                width, height = sample['image_size']
                scale = min(config['video_width'] / width, config['video_height'] / height)
                offset_x = (config['video_width'] - width * scale) / 2
                offset_y = (config['video_height'] - height * scale) / 2
                boxes = [[round(box[0] * scale + offset_x), round(box[1] * scale + offset_y), round(box[2] * scale + offset_x), round(box[3] * scale + offset_y)] for box in sample['regions']]
                regions.append({'seconds_from_clip_start': round(sample['time'] - start, 3), 'boxes': boxes})
            prompt = VIDEO_INSTRUCTION + f'\nThe real source interval lasts {end-start:.6f} seconds; ignore any duplicated tail frame and do not give times beyond that duration.'
            if end >= cut_end - 0.000001:
                prompt += ' This is the final analysis window of the current cut; check its final frames for an accidental hard cut to the next shot as instructed.'
            else:
                prompt += ' This is an intermediate analysis window, not the end of the current cut; its end is not evidence of a hard cut.'
            if 'single_sample_time' in metadata:
                prompt += ' This window contains only one distinct sampled image; do not infer any motion or change.'
            prompt += '\nPredicted subtitle regions in video canvas pixels at sampled times (may be wrong): ' + json.dumps(regions, ensure_ascii=False)
            if context_hints:
                prompt += '\nContext and character detections: ' + json.dumps(context_hints, ensure_ascii=False)
            reject = (lambda error, raw: on_invalid_response(window_index, start, end, error, raw)) if on_invalid_response else None
            output, usage, attempts = generate_window(processor, model, torch, clip, config['video_fps'], prompt, end - start, start, reject, cancel_check)
            result = {'source': metadata, 'analysis': output, 'generation_attempts': attempts}
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
        return {'model': MODEL_ID, 'revision': MODEL_REVISION, 'windows': windows, 'usage': usage,
                'limitations': 'Visual analysis uses a silent video; window boundaries may split events. Model-provided event times are unverified and may overlap, be incomplete, or fall outside the source window.'}
    finally:
        if model is not None:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
