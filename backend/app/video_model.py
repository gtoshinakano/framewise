"""Local Qwen3-VL video inference for bounded windows of a single cut."""
import gc
import json
from pathlib import Path

from . import media
from .visual_contracts import VIDEO_INSTRUCTION, validate_video

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


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


def analyze_video(original, cut, config, model_path, scratch, on_window=None, subtitle_samples=None, load_window=None, save_window=None):
    """Process every cut window, recording both clip-relative and episode times.

    Each model invocation is bounded to limit video tokens. Windows partition the
    cut; events spanning a window boundary may be split. No audio is sent to the
    visual model. Clip files are transient and timing metadata is retained.
    """
    model = processor = torch = None
    windows = []
    start, cut_end = float(cut['start_time']), float(cut['end_time'])
    window_seconds = min(8.0, 32.0 / config['video_fps'])
    window_index = 0
    try:
        while start < cut_end - 0.000001:
            end = min(cut_end, start + window_seconds)
            cached = load_window(window_index, start, end) if load_window else None
            if cached is not None:
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
            metadata = media.prepare_analysis_video(Path(original), clip, start, end, config['video_width'], config['video_height'], config['video_fps'])
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
            if 'single_sample_time' in metadata:
                prompt += ' This window contains only one distinct sampled image; do not infer any motion or change.'
            prompt += '\nPredicted subtitle regions in video canvas pixels at sampled times (may be wrong): ' + json.dumps(regions, ensure_ascii=False)
            inputs = video_inputs(processor, clip, config['video_fps'], prompt).to(model.device)
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            raw = processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            try:
                output = validate_video(raw, end - start, start)
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f'Invalid Qwen3-VL video response for {start:.3f}–{end:.3f}s: {exc}') from exc
            result = {'source': metadata, 'analysis': output}
            if save_window:
                save_window(window_index, result)
            windows.append(result)
            clip.unlink(missing_ok=True)
            start, window_index = end, window_index + 1
        return {'model': MODEL_ID, 'revision': MODEL_REVISION, 'windows': windows,
                'limitations': 'Visual analysis uses a silent video and window boundaries may split events.'}
    finally:
        if model is not None:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
