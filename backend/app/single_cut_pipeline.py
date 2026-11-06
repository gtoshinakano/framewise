"""Checkpointed visual analysis for a single cut.

The worker only calls this for jobs explicitly marked single_cut_v2. Legacy and
batch jobs retain their original snapshots and execution path.
"""
import gc
import json
import time
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



def ensure_models_available(stages=None):
    stages = set(stages or ('subtitle_ocr', 'initial_frame', 'last_frame', 'video_analysis'))
    required = [Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server')),
                Path(os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf')),
                Path(os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'))]
    if 'video_analysis' in stages:
        video_dir = Path(os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'))
        try:
            index = json.loads((video_dir / 'model.safetensors.index.json').read_text())
            shards = set(index['weight_map'].values())
            if not shards or any(Path(name).name != name or not (video_dir / name).is_file() for name in shards):
                raise ValueError('Missing model shard')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError('Qwen3-VL 8B video model download is incomplete') from exc
    if stages.intersection(('initial_frame', 'last_frame')) and any(not path.is_file() for path in required):
        raise RuntimeError('Qwen3.5 frame model or llama-server runtime is not available')

def run(job, owner, episode, item):
    from . import inference_worker
    cut, cut_id, config, context = item['context']['cut'], item['cut_id'], item['config'], item['context']
    payload = json.loads(job['payload_json'])
    stages = set(payload.get('stages') or ('subtitle_ocr', 'initial_frame', 'last_frame', 'video_analysis'))
    original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
    row = db.one('SELECT * FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job['id'], cut_id))
    if row and row['completed']:
        return
    checkpoint = json.loads(row['evidence_json']) if row else {}
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Single-cut checkpoint is invalid')
    base_ids = item.get('base_result_ids', {})
    hints = {'selected_context': [{'id': entry['id'], 'kind': entry['kind'], 'title': entry['title'],
                                  'description': entry['content'][:500]} for entry in context['context_items']],
             'additional_context': context.get('additional_context', '')[:2000],
             'character_detections': context.get('character_detections'),
             'detection_note': 'Boxes are pixel coordinates in the named boundary frame only. When labels_confirmed_by_queue is true, the user verified the named boxes when queuing this cut; use those names for the corresponding visible people. Otherwise labels are unverified hints. Labels alone do not establish presence, actions, or identity across the entire cut.',
             'reference_ids': [ref['id'] for ref in item.get('character_references', [])]}

    def check_cut(conn):
        current = conn.execute('SELECT active,start_time,end_time FROM cuts WHERE id=? AND episode_id=?', (cut_id, episode['id'])).fetchone()
        if not current or not current['active'] or current['start_time'] != cut['start_time'] or current['end_time'] != cut['end_time']:
            raise RuntimeError('The cut changed while analysis was queued or running; start a new analysis')

    with db.connect() as conn:
        worker.require_lease(conn, job['id'], owner)
        check_cut(conn)
    ensure_models_available(stages)

    def save(result_type, data, model_id, prompt, extra=None):
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            worker.require_lease(conn, job['id'], owner)
            check_cut(conn)
            new_id = structured_results.save_generated(conn, cut_id, job['id'], owner, result_type, data, model_id, prompt,
                {**config, 'provider': 'local', **(extra or {})}, {'cut': cut, 'model_context_hints': hints, **({'subtitle_source_result_id': item['subtitle_input']['result_id']} if item.get('subtitle_input') and result_type != 'subtitle_ocr' else {}), **({'frame_time': data['frame_time']} if 'frame_time' in data else {})},
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
            if 'subtitle_ocr' in stages and subtitles is None:
                with worker.analysis_stage(job['id'], owner, cut_id, 'subtitle_ocr', 'Subtitle detection and OCR'):
                    stage_started = time.monotonic()
                    model = inference_worker.get_model()
                    try:
                        subtitles = subtitle_model.inspect_subtitles(original, cut, config, bool(context['transcript_segments']), scratch, model,
                            on_frame=lambda i, n: worker.stage(job['id'], owner, episode['id'], f'Subtitle frames {i}/{n}'),
                            cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner))
                    finally:
                        del model
                        release_gpu_models()
                    save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, 'Detect subtitles; OCR only when audio transcript exists.', {'model_revision': inference_worker.MODEL_REVISION, 'generation_seconds': time.monotonic() - stage_started})
            if subtitles is None:
                subtitles = item.get('subtitle_input', {}).get('content') or {'samples': [], 'events': []}

            def frame_stage(position, server):
                kind = 'initial_frame' if position == 'first' else 'last_frame'
                if saved(kind) is not None:
                    return
                label = 'Initial frame' if position == 'first' else 'Last frame'
                with worker.analysis_stage(job['id'], owner, cut_id, kind, label):
                    stage_started = time.monotonic()
                    usage = {}
                    worker.stage(job['id'], owner, episode['id'], f'Analyzing {label.lower()}')
                    image = Path(scratch) / 'boundary.jpg'
                    first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
                    actual = media.extract_cut_frame(original, image, first if position == 'first' else last, cut['start_time'], cut['end_time'], width=1280, last=position == 'last')
                    closest = min(subtitles['samples'], key=lambda value: abs(value['time'] - actual), default=None)
                    regions = closest['regions'] if closest and abs(closest['time'] - actual) < .1 else []
                    frame_hints = {**hints, 'target_frame': position}
                    data = frame_model.describe_frame(server, image, regions, frame_hints, cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner), reference_images=refs, usage_out=usage)
                    if position == 'first' and 'subtitle_ocr' in stages and subtitles.get('events') and not subtitles.get('language_model'):
                        languages = frame_model.classify_subtitle_languages(server, subtitles['events'], cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner))
                        for event, language in zip(subtitles['events'], languages):
                            event['language'] = language
                        subtitles['language_model'] = frame_model.MODEL_ID
                        subtitles['language_note'] = 'Model-estimated language; unknown means uncertain.'
                        save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, 'Detect subtitles; OCR only when audio transcript exists.', {'model_revision': inference_worker.MODEL_REVISION, 'language_model': frame_model.MODEL_ID, 'language_revision': frame_model.MODEL_REVISION})
                    data['frame_time'] = actual
                    data['subtitle_regions'] = regions
                    save(kind, data, frame_model.MODEL_ID, FRAME_INSTRUCTION, {'model_revision': frame_model.MODEL_REVISION, 'image_width': 1280, 'generation_seconds': time.monotonic() - stage_started, 'usage': usage})

            if any(kind in stages and saved(kind) is None for kind in ('initial_frame', 'last_frame')):
                from . import character_detection
                refs = []
                for index, ref in enumerate(item.get('character_references', [])):
                    source = media.episode_dir(ref['project_id'], ref['episode_id']) / f"original{ref['original_ext']}"
                    refs.append({**ref, 'image': character_detection.crop_frame(source, ref['frame'], ref['cut_start'], ref['cut_end'], ref['box'], scratch, f'analysis-ref-{index}')})
                with frame_model.frame_server(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
                    os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                    os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'), cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner)) as server:
                    if 'initial_frame' in stages:
                        frame_stage('first', server)
                    if 'last_frame' in stages:
                        frame_stage('last', server)
            if 'video_analysis' in stages and saved('video_analysis') is None:
                with worker.analysis_stage(job['id'], owner, cut_id, 'video_analysis', 'Video analysis'):
                    stage_started = time.monotonic()
                    def load_window(index, start, end):
                        row = db.one('SELECT * FROM analysis_video_windows WHERE job_id=? AND cut_id=? AND window_index=?', (job['id'], cut_id, index))
                        if not row:
                            return None
                        if row['start_time'] != start or row['end_time'] != end or row['model_revision'] != video_model.MODEL_REVISION:
                            raise RuntimeError('A saved video window no longer matches this run')
                        data = json.loads(row['content_json'])
                        if not isinstance(data, dict) or not isinstance(data.get('source'), dict) or 'analysis' not in data:
                            raise RuntimeError('A saved video window is unreadable')
                        return data
                    def save_window(index, window):
                        with db.connect() as conn:
                            conn.execute('BEGIN IMMEDIATE')
                            worker.require_lease(conn, job['id'], owner)
                            check_cut(conn)
                            conn.execute('INSERT INTO analysis_video_windows(job_id,cut_id,window_index,start_time,end_time,model_revision,content_json) VALUES (?,?,?,?,?,?,?)',
                                (job['id'], cut_id, index, window['source']['source_start'], window['source']['source_end'], video_model.MODEL_REVISION, json.dumps(window, ensure_ascii=False)))
                    def save_invalid_response(index, start, end, error, response):
                        with db.connect() as conn:
                            worker.require_lease(conn, job['id'], owner)
                            conn.execute('INSERT INTO rejected_video_responses(job_id,cut_id,window_index,start_time,end_time,error,response) VALUES (?,?,?,?,?,?,?)',
                                         (job['id'], cut_id, index, start, end, error, response))
                    data = video_model.analyze_video(original, cut, config, os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'), scratch,
                        on_window=lambda start, end: worker.stage(job['id'], owner, episode['id'], f'Analyzing video {start:.1f}–{end:.1f}s'), subtitle_samples=subtitles['samples'], load_window=load_window, save_window=save_window, context_hints=hints, on_invalid_response=save_invalid_response, cancel_check=lambda: inference_worker.check_summary_lease(job['id'], owner))
                    save('video_analysis', data, video_model.MODEL_ID, VIDEO_INSTRUCTION, {'model_revision': video_model.MODEL_REVISION, 'generation_seconds': time.monotonic() - stage_started, 'usage': data.get('usage', {})})
            with db.connect() as conn:
                worker.require_lease(conn, job['id'], owner)
                conn.execute('UPDATE analysis_job_cuts SET completed=1 WHERE job_id=? AND cut_id=?', (job['id'], cut_id))
    finally:
        release_gpu_models()
