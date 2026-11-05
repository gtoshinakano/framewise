"""Checkpointed visual analysis for a single cut.

The worker only calls this for jobs explicitly marked single_cut_v2. Legacy and
batch jobs retain their original snapshots and execution path.
"""
import gc
import json
import os
import tempfile
from pathlib import Path
from . import db, frame_model, media, structured_results, subtitle_model, video_model, worker
from .visual_contracts import FRAME_INSTRUCTION, VIDEO_INSTRUCTION


def release_gpu_models():
    from . import inference_worker
    inference_worker._model = None
    inference_worker._caption_model = None
    inference_worker._caption_processor = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass



def ensure_models_available():
    required = [Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server')),
                Path(os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf')),
                Path(os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'))]
    video_dir = Path(os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'))
    try:
        index = json.loads((video_dir / 'model.safetensors.index.json').read_text())
        shards = set(index['weight_map'].values())
        if not shards or any(Path(name).name != name or not (video_dir / name).is_file() for name in shards):
            raise ValueError('Missing model shard')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('Qwen3-VL 8B video model download is incomplete') from exc
    if any(not path.is_file() for path in required):
        raise RuntimeError('Qwen3.5 frame model or llama-server runtime is not available')

def run(job, owner, episode, item):
    from . import inference_worker
    cut, cut_id, config, context = item['context']['cut'], item['cut_id'], item['config'], item['context']
    original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
    row = db.one('SELECT * FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job['id'], cut_id))
    if row and row['completed']:
        return
    ensure_models_available()
    checkpoint = json.loads(row['evidence_json']) if row else {}
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Single-cut checkpoint is invalid')
    base_ids = item.get('base_result_ids', {})

    def save(result_type, data, model_id, prompt, extra=None):
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            new_id = structured_results.save_generated(conn, cut_id, job['id'], owner, result_type, data, model_id, prompt,
                {**config, **(extra or {})}, {'cut': cut, **({'frame_time': data['frame_time']} if 'frame_time' in data else {})},
                expected_current=checkpoint.get(result_type, base_ids.get(result_type)))
            checkpoint[result_type] = new_id
            conn.execute('INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,?) ON CONFLICT(job_id,cut_id) DO UPDATE SET evidence_json=excluded.evidence_json',
                         (job['id'], cut_id, json.dumps(checkpoint)))

    def saved(result_type):
        result_id = checkpoint.get(result_type)
        if not result_id:
            return None
        row = db.one('SELECT content FROM analysis_results WHERE id=? AND cut_id=?', (result_id, cut_id))
        if not row:
            raise RuntimeError('A completed stage result was deleted while the job was running')
        return json.loads(row['content'])

    release_gpu_models()
    try:
        with tempfile.TemporaryDirectory(prefix='framewise-cut-analysis-') as scratch:
            subtitles = saved('subtitle_ocr')
            if subtitles is None:
                with worker.analysis_stage(job['id'], owner, cut_id, 'subtitle_ocr', 'Subtitle detection and OCR'):
                    model = inference_worker.get_model()
                    try:
                        subtitles = subtitle_model.inspect_subtitles(original, cut, config, bool(context['transcript_segments']), scratch, model,
                            on_frame=lambda i, n: worker.stage(job['id'], owner, episode['id'], f'Subtitle frames {i}/{n}'))
                    finally:
                        del model
                        release_gpu_models()
                    save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, 'Detect subtitles; OCR only when audio transcript exists.', {'model_revision': inference_worker.MODEL_REVISION})

            def frame_stage(position):
                kind = 'initial_frame' if position == 'first' else 'last_frame'
                if saved(kind) is not None:
                    return
                label = 'Initial frame' if position == 'first' else 'Last frame'
                with worker.analysis_stage(job['id'], owner, cut_id, kind, label):
                    worker.stage(job['id'], owner, episode['id'], f'Analyzing {label.lower()}')
                    image = Path(scratch) / 'boundary.jpg'
                    first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
                    actual = media.extract_cut_frame(original, image, first if position == 'first' else last, cut['start_time'], cut['end_time'], width=1280, last=position == 'last')
                    closest = min(subtitles['samples'], key=lambda value: abs(value['time'] - actual), default=None)
                    regions = closest['regions'] if closest and abs(closest['time'] - actual) < .1 else []
                    with frame_model.frame_server(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
                        os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                        os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'), cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner)) as server:
                        characters = [{'id': entry['id'], 'title': entry['title'], 'description': entry['content']} for entry in context['context_items'] if entry['kind'] == 'character']
                        data = frame_model.describe_frame(server, image, regions, characters, cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner))
                        if position == 'first' and subtitles.get('events') and not subtitles.get('language_model'):
                            languages = frame_model.classify_subtitle_languages(server, subtitles['events'], cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner))
                            for event, language in zip(subtitles['events'], languages):
                                event['language'] = language
                            subtitles['language_model'] = frame_model.MODEL_ID
                            subtitles['language_note'] = 'Model-estimated language; unknown means uncertain.'
                            save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, 'Detect subtitles; OCR only when audio transcript exists.', {'model_revision': inference_worker.MODEL_REVISION, 'language_model': frame_model.MODEL_ID, 'language_revision': frame_model.MODEL_REVISION})
                    data['frame_time'] = actual
                    data['subtitle_regions'] = regions
                    save(kind, data, frame_model.MODEL_ID, FRAME_INSTRUCTION, {'model_revision': frame_model.MODEL_REVISION, 'image_width': 1280})

            frame_stage('first')
            if saved('video_analysis') is None:
                with worker.analysis_stage(job['id'], owner, cut_id, 'video_analysis', 'Video analysis'):
                    def load_window(index, start, end):
                        row = db.one('SELECT * FROM analysis_video_windows WHERE job_id=? AND cut_id=? AND window_index=?', (job['id'], cut_id, index))
                        if not row:
                            return None
                        if row['start_time'] != start or row['end_time'] != end or row['model_revision'] != video_model.MODEL_REVISION:
                            raise RuntimeError('A saved video window no longer matches this run')
                        data = json.loads(row['content_json'])
                        if not isinstance(data, dict) or not isinstance(data.get('source'), dict) or not isinstance(data.get('analysis'), dict):
                            raise RuntimeError('A saved video window is unreadable')
                        return data
                    def save_window(index, window):
                        with db.connect() as conn:
                            worker.require_lease(conn, job['id'], owner)
                            conn.execute('INSERT INTO analysis_video_windows(job_id,cut_id,window_index,start_time,end_time,model_revision,content_json) VALUES (?,?,?,?,?,?,?)',
                                (job['id'], cut_id, index, window['source']['source_start'], window['source']['source_end'], video_model.MODEL_REVISION, json.dumps(window, ensure_ascii=False)))
                    data = video_model.analyze_video(original, cut, config, os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'), scratch,
                        on_window=lambda start, end: worker.stage(job['id'], owner, episode['id'], f'Analyzing video {start:.1f}–{end:.1f}s'), subtitle_samples=subtitles['samples'], load_window=load_window, save_window=save_window)
                    save('video_analysis', data, video_model.MODEL_ID, VIDEO_INSTRUCTION, {'model_revision': video_model.MODEL_REVISION})
            frame_stage('last')
            with db.connect() as conn:
                worker.require_lease(conn, job['id'], owner)
                conn.execute('UPDATE analysis_job_cuts SET completed=1 WHERE job_id=? AND cut_id=?', (job['id'], cut_id))
    finally:
        release_gpu_models()
