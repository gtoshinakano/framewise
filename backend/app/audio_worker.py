"""Dedicated local audio inference worker; all jobs use the shared lease discipline."""
import json
import logging
import math
import os
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

from . import db, job_inputs, media, worker
from .audio_matching import assign_words, match_voices, overlap_index, reconcile_speakers, restore_aligned_text

log = logging.getLogger(__name__)
DIARIZATION = 'pyannote/speaker-diarization-community-1'
ASR = 'Qwen/Qwen3-ASR-1.7B'
ALIGNER = 'Qwen/Qwen3-ForcedAligner-0.6B'
# A re-run diarizes the selected audio plus this much on each side, so the
# speakers have enough speech to be told apart and lined up with the rest.
CONTEXT_SECONDS = 20.
MIN_RANGE_SECONDS = .5


def diarize(audio):
    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(DIARIZATION, token=os.environ.get('HF_TOKEN') or None)
    if pipeline is None:
        raise RuntimeError('Community-1 could not be loaded. Accept its Hugging Face conditions and configure HF_TOKEN.')
    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda'))
    # Some ROCm/MIOpen builds fail to compile PyanNet's instance-normalization
    # kernel on gfx1201. PyTorch's native path keeps inference on the GPU.
    with torch.backends.cudnn.flags(enabled=False):
        return pipeline(str(audio))


def clip_audio(source, target, start, end):
    media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', str(start),
               '-i', str(source), '-t', str(end - start), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)])


def join_clips(source, target, ranges, temp):
    with wave.open(str(target), 'wb') as combined:
        combined.setnchannels(1)
        combined.setsampwidth(2)
        combined.setframerate(16000)
        for index, item in enumerate(ranges):
            clip = temp / f'clip-{index}.wav'
            clip_audio(source, clip, item['start_time'], item['end_time'])
            with wave.open(str(clip), 'rb') as part:
                combined.writeframes(part.readframes(part.getnframes()))


class AlignmentError(ValueError):
    """The model returned speech that cannot be reliably timed in this window."""


def aligned_words(result, start, end):
    words = []
    duration = end - start
    for item in result.time_stamps or []:
        if not item.text.strip():
            continue
        try:
            relative_start, relative_end = float(item.start_time), float(item.end_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlignmentError('non-numeric timestamps') from exc
        # Small rounding excursions can be clipped; larger ones often mean the
        # model generated speech beyond the supplied audio and must be retried.
        if (not math.isfinite(relative_start) or not math.isfinite(relative_end)
                or relative_end < relative_start or relative_start < -.08 or relative_end > duration + .08):
            raise AlignmentError('timestamps outside the audio window or in reverse order')
        word_start = start + max(0., min(duration, relative_start))
        word_end = start + max(0., min(duration, relative_end))
        if word_end == word_start:
            word_start = max(start, min(word_start, end - .01))
            word_end = min(end, word_start + .01)
        words.append({'start_time': word_start, 'end_time': word_end, 'text': item.text.strip()})
    if result.text.strip() and not words:
        raise AlignmentError(f'no aligned timestamps ({result.language})')
    restored = restore_aligned_text(result.text, words)
    if restored is None:
        # Preserve the original text as one editable span when the aligner's
        # tokenization cannot be mapped back to the ASR transcription.
        restored = [{'start_time': min(word['start_time'] for word in words),
                     'end_time': max(word['end_time'] for word in words), 'text': result.text}]
    return restored


def transcribe_window(model, audio, sample, start, end, turns, depth=0):
    clip_audio(audio, sample, start, end)
    result = model.transcribe(audio=str(sample), language=None, return_time_stamps=True)[0]
    try:
        return assign_words(aligned_words(result, start, end), turns)
    except AlignmentError as exc:
        if depth >= 3:
            raise RuntimeError(f'Could not align speech at {start:.2f}–{end:.2f}s after retrying smaller excerpts: {exc}') from exc
        midpoint = (start + end) / 2
        log.warning('Retrying speech alignment at %.2f–%.2fs in smaller excerpts: %s', start, end, exc)
        return (transcribe_window(model, audio, sample, start, midpoint, turns, depth + 1)
                + transcribe_window(model, audio, sample, midpoint, end, turns, depth + 1))


def load_asr():
    import torch
    from qwen_asr import Qwen3ASRModel
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    return Qwen3ASRModel.from_pretrained(ASR, dtype=dtype, device_map=device, max_inference_batch_size=1,
        max_new_tokens=512, forced_aligner=ALIGNER, forced_aligner_kwargs={'dtype': dtype, 'device_map': device})


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    episode = db.one('SELECT id,project_id,original_ext,duration,has_audio FROM episodes WHERE id=?', (job['episode_id'],), conn=conn)
    path = media.episode_dir(episode['project_id'], episode['id'])
    upload_id = payload.get('upload_id')
    if upload_id:
        source = path / 'voice-uploads' / str(uuid.UUID(upload_id))
    else:
        source = path / 'audio.wav'
        if not source.is_file():
            source = path / f"original{episode['original_ext']}"
    try:
        stat = source.stat()
        source_identity = [source.name, stat.st_size, stat.st_mtime_ns]
    except OSError:
        source_identity = None
    values = {'episode': episode, 'source': source_identity}
    if job['pipeline'] == 'voice_sample':
        values['character'] = db.one("SELECT id FROM context_items WHERE id=? AND project_id=? AND kind='character'",
                                     (payload['character_context_id'], episode['project_id']), conn=conn)
        return values
    values['segments'] = db.all_rows("""SELECT id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,deleted
        FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL ORDER BY start_time,id""", (episode['id'],), conn=conn)
    values['references'] = db.all_rows("""SELECT v.id,v.character_context_id AS character_id,v.embedding_json
        FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
        WHERE c.project_id=? AND v.model_id=? ORDER BY v.id""", (episode['project_id'], DIARIZATION), conn=conn)
    return values


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    from .structured_results import invalidate_transcript
    if job['pipeline'] == 'transcribe':
        span = json.loads(job['payload_json']).get('range') or {}
        invalidate_transcript(conn, job['episode_id'], span.get('start_time', 0), span.get('end_time'))


def voice_references(snapshot):
    return [{'character_id': r['character_id'], 'embedding': json.loads(r['embedding_json'])} for r in snapshot.data['references']]


def transcript_provenance(snapshot, manual_ids=None):
    return {'fingerprint': snapshot.stamp, 'source': snapshot.data['source'],
            'segment_ids': [row['id'] for row in snapshot.data['segments']],
            'voice_reference_ids': [row['id'] for row in snapshot.data['references']],
            'replaceable_manual_ids': manual_ids}


def protected_deletions(rows):
    return [(row['start_time'], row['end_time']) for row in rows if row['deleted'] and row['origin'] != 'model']


def invalidate_subtitles_if_timing_changed(conn, episode_id, previously):
    from .subtitle_store import invalidate_probes
    old = sorted((row['start_time'], row['end_time']) for row in previously)
    new = [tuple(row) for row in conn.execute("""SELECT start_time,end_time FROM transcript_segments WHERE episode_id=?
        AND superseded_by IS NULL AND deleted=0 ORDER BY start_time,end_time""", (episode_id,))]
    if old != new:
        invalidate_probes(conn, episode_id)


def speech_windows(start, end, size=25.):
    """Split [start, end] into excerpts of at most ``size`` seconds without a tiny tail."""
    windows = []
    while start < end - 1e-6:
        stop = end if end - start < size + 1. else start + size
        windows.append((start, stop))
        start = stop
    return windows


def transcribe_range(snapshot, payload, audio, temp):
    """Redo transcription and speaker identification for one selected interval only.

    The rest of the transcript is untouched. Speakers come from diarizing the
    interval with some surrounding audio, then take the labels the neighbouring
    speech already uses so they stay consistent with the full-episode run.
    """
    job_id, owner, episode = snapshot.job['id'], snapshot.owner, snapshot.data['episode']
    duration = float(episode['duration'])
    start = max(0., float(payload['range']['start_time']))
    if float(payload['range']['end_time']) > duration + .2:
        raise RuntimeError('The selected audio ends after the video.')
    end = min(duration, float(payload['range']['end_time']))
    if end - start < MIN_RANGE_SECONDS:
        raise RuntimeError('The selected audio is too short to transcribe.')
    replace_manual = payload.get('replace_manual') is True
    rows = snapshot.data['segments']
    current = [row for row in rows if not row['deleted']]
    previous = json.loads(snapshot.job['input_json'] or '{}')
    manual_ids = set(previous.get('replaceable_manual_ids') or [])
    if replace_manual and 'replaceable_manual_ids' not in previous:
        manual_ids = {row['id'] for row in current if row['origin'] != 'model'}
        with snapshot.publish(invalidate=False) as conn:
            conn.execute('UPDATE jobs SET input_json=? WHERE id=?',
                         (json.dumps(transcript_provenance(snapshot, sorted(manual_ids))), job_id))
    low, high = max(0., start - CONTEXT_SECONDS), min(duration, end + CONTEXT_SECONDS)
    excerpt = temp / 'excerpt.wav'
    worker.stage(job_id, owner, episode['id'], f'Identifying voices at {start:.1f}–{end:.1f}s')
    clip_audio(audio, excerpt, low, high)
    output = diarize(excerpt)
    labels = output.speaker_diarization.labels()
    embeddings = output.speaker_embeddings
    speakers = {label: embeddings[i].tolist() for i, label in enumerate(labels)} if embeddings is not None else {}
    matches = match_voices(speakers, voice_references(snapshot))
    turns = [{'start_time': low + turn.start, 'end_time': low + turn.end, 'speaker': speaker} for turn, speaker in output.exclusive_speaker_diarization]
    worker.stage(job_id, owner, episode['id'], 'Loading transcription and alignment models')
    model = load_asr()
    windows = speech_windows(start, end)
    words = []
    for index, (window_start, window_end) in enumerate(windows):
        worker.stage(job_id, owner, episode['id'], f'Transcribing speech {index + 1}/{len(windows)} at {start:.1f}–{end:.1f}s')
        words.extend(transcribe_window(model, audio, temp / 'sample.wav', window_start, window_end, turns))
    model_id = ASR + ' + ' + ALIGNER + ' + ' + DIARIZATION
    provenance = json.dumps(transcript_provenance(snapshot, sorted(manual_ids)))

    def preserve_manual_base(conn):
        conn.execute('UPDATE jobs SET input_json=? WHERE id=?', (provenance, job_id))

    with snapshot.publish(on_changed=preserve_manual_base) as conn:
        names = reconcile_speakers(turns, current, matches, (start, end))
        replaced, kept, protected = 0, 0, protected_deletions(rows)
        for old in current:
            # A segment belongs to the interval by its midpoint, so a word cut by
            # the boundary is neither lost nor duplicated.
            inside = start <= (old['start_time'] + old['end_time']) / 2 < end
            if inside and (old['id'] in manual_ids or (old['origin'] == 'model' and (replace_manual or (old['model_id'] or '').startswith(ASR)))):
                tombstone = str(uuid.uuid4())
                conn.execute("""INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,deleted,origin,model_id,job_id,parent_id)
                    VALUES (?,?,?,?,?,?,?,1,'model',?,?,?)""", (tombstone, episode['id'], old['start_time'], old['end_time'], old['text'], old['speaker_label'], old['character_context_id'], model_id, job_id, old['id']))
                conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (tombstone, old['id']))
                replaced += 1
            else:
                kept += inside
                if old['start_time'] < end and old['end_time'] > start:
                    protected.append((old['start_time'], old['end_time']))
        overlaps_kept = overlap_index(protected)
        added = 0
        for word in words:
            if overlaps_kept(word['start_time'], word['end_time']):
                continue
            name = names.get(word['speaker'], {'label': '', 'character_id': None})
            conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id) VALUES (?,?,?,?,?,?,?,'model',?,?)",
                (str(uuid.uuid4()), episode['id'], word['start_time'], word['end_time'], word['text'], name['label'], name['character_id'], model_id, job_id))
            added += 1
        note = f', {kept} edited by hand kept' if kept else ''
        invalidate_subtitles_if_timing_changed(conn, episode['id'], current)
        preserve_manual_base(conn)
        finish(conn, job_id, episode['id'], f'Complete · {added} new, {replaced} replaced{note}')


def process(job_id, owner):
    job = db.one('SELECT * FROM jobs WHERE id=? AND status=\'running\' AND lease_owner=?', (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    terminal = False
    uploaded = None
    try:
        with db.connect() as conn:
            conn.execute('BEGIN')
            snapshot = job_inputs.JobInputs(conn, job, owner)
        episode = snapshot.data['episode']
        path = media.episode_dir(episode['project_id'], episode['id'])
        uploaded = path / 'voice-uploads' / str(uuid.UUID(payload['upload_id'])) if payload.get('upload_id') else None
        if not uploaded and episode['has_audio'] == 0:
            with snapshot.publish(invalidate=False) as conn:
                finish(conn, job_id, episode['id'], 'Skipped · this video has no audio track')
            return
        if job['pipeline'] == 'voice_sample' and not snapshot.data['character']:
            with snapshot.publish(invalidate=False) as conn:
                finish(conn, job_id, episode['id'], 'Skipped · character no longer exists')
            terminal = True
            return
        audio = path / 'audio.wav'
        with tempfile.TemporaryDirectory(dir=path) as temp:
            temp = Path(temp)
            if not uploaded and not audio.is_file():
                worker.stage(job_id, owner, episode['id'], 'Extracting audio')
                audio = temp / 'audio.wav'
                media.extract_audio(path / f"original{episode['original_ext']}", audio)
            worker.stage(job_id, owner, episode['id'], 'Identifying voices')
            sample = temp / 'sample.wav'
            if job['pipeline'] == 'voice_sample':
                ranges = payload.get('ranges') or ([{'start_time': payload['start_time'], 'end_time': payload['end_time']}] if not uploaded else [])
                if ranges and episode['duration'] and max(item['end_time'] for item in ranges) > episode['duration']:
                    raise RuntimeError('The selected audio ends after the video.')
                if uploaded:
                    worker.stage(job_id, owner, episode['id'], 'Preparing uploaded voice')
                    media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(uploaded),
                               '-map', '0:a:0', '-vn', '-t', '61', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(sample)])
                    with wave.open(str(sample), 'rb') as wav:
                        duration = wav.getnframes() / wav.getframerate()
                    if not 2 <= duration <= 60:
                        raise RuntimeError('Uploaded voice audio must be 2 to 60 seconds long.')
                elif len(ranges) == 1:
                    clip_audio(audio, sample, ranges[0]['start_time'], ranges[0]['end_time'])
                else:
                    join_clips(audio, sample, ranges, Path(temp))
                output = diarize(sample)
                labels = output.speaker_diarization.labels()
                if len(labels) != 1 or output.speaker_embeddings is None:
                    raise RuntimeError('Select a sample containing one clear speaker; none or multiple speakers were detected.')
                vector = output.speaker_embeddings[0].tolist()
                if not vector or any(not math.isfinite(x) for x in vector) or not any(vector):
                    raise RuntimeError('Could not extract a usable voice embedding from this sample.')
                reference = media.voice_reference_path(episode['project_id'], job_id)
                saved = False
                try:
                    with snapshot.publish() as conn:
                        character = snapshot.data['character']
                        conn.execute('INSERT INTO character_voice_samples (id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json,source_filename) VALUES (?,?,?,?,?,?,?,?,?)',
                                     (job_id, character['id'], episode['id'], min((r['start_time'] for r in ranges), default=0), max((r['end_time'] for r in ranges), default=duration if uploaded else 0), json.dumps(vector), DIARIZATION, json.dumps(ranges), payload.get('source_filename')))
                        reference.parent.mkdir(parents=True, exist_ok=True)
                        sample.replace(reference)
                        saved = True
                        conn.execute('UPDATE jobs SET input_json=? WHERE id=?', (json.dumps({
                            'fingerprint': snapshot.stamp, 'source': snapshot.data['source'], 'character_id': character['id'],
                        }), job_id))
                        finish(conn, job_id, episode['id'])
                except Exception:
                    if saved:
                        reference.unlink(missing_ok=True)
                    raise
                terminal = True
                return
            if payload.get('range'):
                transcribe_range(snapshot, payload, audio, temp)
                return
            output = diarize(audio)
            labels = output.speaker_diarization.labels()
            embeddings = output.speaker_embeddings
            speakers = {label: embeddings[i].tolist() for i, label in enumerate(labels)} if embeddings is not None else {}
            matches = match_voices(speakers, voice_references(snapshot))
            turns = [{'start_time': turn.start, 'end_time': turn.end, 'speaker': speaker} for turn, speaker in output.exclusive_speaker_diarization]
            worker.stage(job_id, owner, episode['id'], 'Loading transcription and alignment models')
            model = load_asr()
            segments = []
            # Cover the entire recording even where diarization missed speech.
            # Short bounded windows also keep forced-alignment inference manageable.
            windows = []
            start = 0.
            while start < episode['duration']:
                end = min(start + 25., episode['duration'])
                windows.append((start, end))
                start = end
            for index, (start, end) in enumerate(windows):
                worker.stage(job_id, owner, episode['id'], f'Transcribing speech {index + 1}/{len(windows)}')
                segments.extend(transcribe_window(model, audio, sample, start, end, turns))
            with snapshot.publish() as conn:
                # Replace untouched output from this pipeline. Preserve human
                # revisions and other human-created segments.
                rows = snapshot.data['segments']
                current = [row for row in rows if not row['deleted']]
                protected = protected_deletions(rows)
                for old in current:
                    if old['origin'] == 'model' and (old['model_id'] or '').startswith(ASR):
                        tombstone = str(uuid.uuid4())
                        conn.execute("""INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,deleted,origin,model_id,job_id,parent_id)
                            VALUES (?,?,?,?,?,?,?,1,'model',?,?,?)""", (tombstone, episode['id'], old['start_time'], old['end_time'], old['text'], old['speaker_label'], old['character_context_id'], old['model_id'], old['job_id'], old['id']))
                        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (tombstone, old['id']))
                    else:
                        protected.append((old['start_time'], old['end_time']))
                overlaps_reviewed = overlap_index(protected)
                for word in segments:
                    if overlaps_reviewed(word['start_time'], word['end_time']):
                        continue
                    speaker = word['speaker'] or ''
                    character_id = matches.get(speaker, {}).get('character_id')
                    conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id) VALUES (?,?,?,?,?,?,?,'model',?,?)",
                        (str(uuid.uuid4()), episode['id'], word['start_time'], word['end_time'], word['text'], speaker, character_id, ASR + ' + ' + ALIGNER + ' + ' + DIARIZATION, job_id))
                invalidate_subtitles_if_timing_changed(conn, episode['id'], current)
                conn.execute('UPDATE jobs SET input_json=? WHERE id=?', (json.dumps(transcript_provenance(snapshot)), job_id))
                finish(conn, job_id, episode['id'])
    except worker.LeaseLost:
        # Cancelled while it ran (worker.after_cancel already removed an uploaded voice), or taken over:
        # nothing more is written for it, and an upload a new owner may still need stays.
        log.info('Audio job %s stopped before publishing', job_id)
    except Exception as exc:
        log.exception('Audio job failed: %s', job_id)
        terminal = worker.fail_job(job_id, owner, job['episode_id'], exc)
    finally:
        if uploaded and terminal:
            uploaded.unlink(missing_ok=True)


def finish(conn, job_id, episode_id, stage='Complete'):
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job_id))
    worker.sync_episode_status(conn, episode_id)


def release_gpu():
    """Hand the GPU back after a job.

    The models are locals of the job, but PyTorch's caching allocator keeps their memory in this
    long-lived process, which left the text model unable to load until the worker was restarted.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    while True:
        try:
            claimed = worker.claim_job(('transcribe', 'voice_sample'))
            if not claimed:
                time.sleep(2)
                continue
            job_id, owner = claimed
            done = threading.Event()
            beat = threading.Thread(target=worker.heartbeat, args=(job_id, owner, done), daemon=True)
            beat.start()
            try:
                process(job_id, owner)
            finally:
                done.set()
                beat.join(timeout=1)
                release_gpu()
        except Exception:
            log.exception('Audio worker error')
            time.sleep(2)


if __name__ == '__main__':
    main()
