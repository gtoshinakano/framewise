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

from . import db, media, worker
from .audio_matching import assign_words, match_voices, overlap_index, restore_aligned_text

log = logging.getLogger(__name__)
DIARIZATION = 'pyannote/speaker-diarization-community-1'
ASR = 'Qwen/Qwen3-ASR-1.7B'
ALIGNER = 'Qwen/Qwen3-ForcedAligner-0.6B'


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


def process(job_id, owner):
    job = db.one('SELECT * FROM jobs WHERE id=? AND status=\'running\' AND lease_owner=?', (job_id, owner))
    if not job:
        return
    episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
    if not episode:
        return
    payload = json.loads(job['payload_json'])
    path = media.episode_dir(episode['project_id'], episode['id'])
    uploaded = path / 'voice-uploads' / str(uuid.UUID(payload['upload_id'])) if payload.get('upload_id') else None
    terminal = False
    try:
        audio = path / 'audio.wav'
        if not uploaded and not audio.is_file():
            worker.stage(job_id, owner, episode['id'], 'Extracting audio')
            media.extract_audio(path / f"original{episode['original_ext']}", audio)
        worker.stage(job_id, owner, episode['id'], 'Identifying voices')
        with tempfile.TemporaryDirectory(dir=path) as temp:
            sample = Path(temp) / 'sample.wav'
            if job['pipeline'] == 'voice_sample':
                ranges = payload.get('ranges') or ([{'start_time': payload['start_time'], 'end_time': payload['end_time']}] if not uploaded else [])
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
                    with db.connect() as conn:
                        worker.require_lease(conn, job_id, owner)
                        character = conn.execute("SELECT id FROM context_items WHERE id=? AND project_id=? AND kind='character'", (payload['character_context_id'], episode['project_id'])).fetchone()
                        if not character:
                            raise RuntimeError('Character no longer exists in this project')
                        conn.execute('INSERT INTO character_voice_samples (id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json,source_filename) VALUES (?,?,?,?,?,?,?,?,?)',
                                     (job_id, character['id'], episode['id'], min((r['start_time'] for r in ranges), default=0), max((r['end_time'] for r in ranges), default=duration if uploaded else 0), json.dumps(vector), DIARIZATION, json.dumps(ranges), payload.get('source_filename')))
                        reference.parent.mkdir(parents=True, exist_ok=True)
                        sample.replace(reference)
                        saved = True
                        finish(conn, job_id, episode['id'])
                except Exception:
                    if saved:
                        reference.unlink(missing_ok=True)
                    raise
                terminal = True
                return
            output = diarize(audio)
            labels = output.speaker_diarization.labels()
            embeddings = output.speaker_embeddings
            speakers = {label: embeddings[i].tolist() for i, label in enumerate(labels)} if embeddings is not None else {}
            refs = db.all_rows('SELECT v.character_context_id AS character_id,v.embedding_json FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE c.project_id=? AND v.model_id=?', (episode['project_id'], DIARIZATION))
            matches = match_voices(speakers, [{'character_id': r['character_id'], 'embedding': json.loads(r['embedding_json'])} for r in refs])
            turns = [{'start_time': turn.start, 'end_time': turn.end, 'speaker': speaker} for turn, speaker in output.exclusive_speaker_diarization]
            worker.stage(job_id, owner, episode['id'], 'Loading transcription and alignment models')
            import torch
            from qwen_asr import Qwen3ASRModel
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            dtype = torch.bfloat16 if device != 'cpu' else torch.float32
            model = Qwen3ASRModel.from_pretrained(ASR, dtype=dtype, device_map=device, max_inference_batch_size=1,
                max_new_tokens=512, forced_aligner=ALIGNER, forced_aligner_kwargs={'dtype': dtype, 'device_map': device})
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
            with db.connect() as conn:
                worker.require_lease(conn, job_id, owner)
                # Replace untouched output from this pipeline. Preserve human
                # revisions and other human-created segments.
                current = conn.execute("SELECT * FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0", (episode['id'],)).fetchall()
                protected = []
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
                finish(conn, job_id, episode['id'])
    except Exception as exc:
        log.exception('Audio job failed: %s', job_id)
        with db.connect() as conn:
            changed = conn.execute("UPDATE jobs SET status='failed',stage='Failed',error=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running' AND lease_owner=?", (str(exc)[-1800:], job_id, owner)).rowcount
            if changed:
                worker.sync_episode_status(conn, episode['id'], str(exc)[-1800:])
                terminal = True
    finally:
        if uploaded and terminal:
            uploaded.unlink(missing_ok=True)


def finish(conn, job_id, episode_id):
    conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
    worker.sync_episode_status(conn, episode_id)


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
        except Exception:
            log.exception('Audio worker error')
            time.sleep(2)


if __name__ == '__main__':
    main()
