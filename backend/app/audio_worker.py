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

from . import audio_lines, audio_ranges, line_voices, name_glossary, transcript_names, db, gpu_device, job_inputs, job_kinds, media, worker
from .audio_matching import assign_words, overlap_index, reconcile_speakers, restore_aligned_text

log = logging.getLogger(__name__)
DIARIZATION = 'pyannote/speaker-diarization-community-1'
ASR = 'Qwen/Qwen3-ASR-1.7B'
ALIGNER = 'Qwen/Qwen3-ForcedAligner-0.6B'
# A re-run diarizes the selected audio plus this much on each side, so the
# speakers have enough speech to be told apart and lined up with the rest.
CONTEXT_SECONDS = 20.
MIN_RANGE_SECONDS = .5


def load_diarizer():
    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(DIARIZATION, token=os.environ.get('HF_TOKEN') or None)
    if pipeline is None:
        raise RuntimeError('Community-1 could not be loaded. Accept its Hugging Face conditions and configure HF_TOKEN.')
    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda'))
    pipeline.instantiate({'segmentation': {'min_duration_off': 0.},
                          'clustering': {'threshold': .6, 'Fa': .07, 'Fb': .8}})
    return pipeline


RATE = 16000


def diarize(pipeline, samples):
    import numpy as np
    import torch
    # Some ROCm/MIOpen builds fail to compile PyanNet's instance-normalization
    # kernel on gfx1201. PyTorch's native path keeps inference on the GPU.
    with torch.backends.cudnn.flags(enabled=False):
        return pipeline({'waveform': torch.from_numpy(np.ascontiguousarray(samples))[None],
                         'sample_rate': RATE, 'uri': 'episode'})


def embed(pipeline, samples):
    import numpy as np
    import torch
    if len(samples) < .3 * RATE:
        return None
    waveform = torch.from_numpy(np.ascontiguousarray(samples))[None, None, :]
    with torch.backends.cudnn.flags(enabled=False):
        result = np.asarray(pipeline._embedding(waveform))[0]
    return result if np.isfinite(result).all() else None


def read_audio(path):
    import numpy as np
    # Prepare and voice sample normalization produce mono, 16 kHz PCM.
    with wave.open(str(path), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
            raise RuntimeError('Expected 16 kHz mono audio. Run Prepare again.')
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768



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
    if isinstance(audio, Path):
        clip_audio(audio, sample, start, end)
        waveform = str(sample)
    else:
        waveform = (audio[max(0, int(start * RATE)):int(end * RATE)], RATE)
    result = model.transcribe(audio=waveform, language=None, return_time_stamps=True)[0]
    try:
        words = aligned_words(result, start, end)
        return words if turns is None else assign_words(words, turns)
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
    values['segments'] = db.all_rows("""SELECT id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,deleted,words_json,speaker_source,confidence,evidence_json
        FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL ORDER BY start_time,id""", (episode['id'],), conn=conn)
    values['references'] = db.all_rows("""SELECT v.id,v.character_context_id AS character_id,v.episode_id,v.start_time,v.end_time,v.ranges_json,v.source_filename,v.range_embeddings_json
        FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
        WHERE c.project_id=? AND v.model_id=? ORDER BY v.id""", (episode['project_id'], DIARIZATION), conn=conn)
    values['glossary'] = name_glossary.rows(conn, episode['project_id'], episode['id'])
    values['ranges'] = audio_ranges.rows(conn, episode['id'], user_only=not payload.get('range'))
    if not payload.get('range'):
        values['project_audio'] = audio_ranges.audio_files(conn, episode['project_id'])
    return values


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    from .structured_results import invalidate_transcript
    if job['pipeline'] == 'transcribe':
        span = json.loads(job['payload_json']).get('range') or {}
        invalidate_transcript(conn, job['episode_id'], span.get('start_time', 0), span.get('end_time'))


def voice_references(snapshot, pipeline):
    import numpy as np
    project = snapshot.data['episode']['project_id']
    cached_audio = {}
    refs = []
    for row in snapshot.data['references']:
        if row['range_embeddings_json'] is not None:
            vectors = json.loads(row['range_embeddings_json'])
        elif row['source_filename']:
            path = media.voice_reference_path(project, row['id'])
            if not path.is_file():
                continue
            samples = read_audio(path)
            vectors = line_voices.range_embeddings(samples, [{'start_time': 0, 'end_time': len(samples) / RATE}],
                                                   lambda clip: embed(pipeline, clip))
        else:
            episode_id = row['episode_id']
            if not episode_id:
                continue
            if episode_id not in cached_audio:
                path = media.episode_dir(project, episode_id) / 'audio.wav'
                cached_audio[episode_id] = read_audio(path) if path.is_file() else None
            samples = cached_audio[episode_id]
            if samples is None:
                continue
            ranges = json.loads(row['ranges_json'] or 'null') or [row]
            vectors = line_voices.range_embeddings(samples, ranges, lambda clip: embed(pipeline, clip))
        for index, item in enumerate(vectors):
            vector = np.asarray(item['embedding'], dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm and np.isfinite(norm):
                refs.append({'character': row['character_id'], 'range_id': f"{row['id']}:{index}", 'unit': vector / norm})
    return refs


def transcript_provenance(snapshot, manual_ids=None):
    return {'fingerprint': snapshot.stamp, 'source': snapshot.data['source'],
            'segment_ids': [row['id'] for row in snapshot.data['segments']],
            'voice_reference_ids': [row['id'] for row in snapshot.data['references']],
            'replaceable_manual_ids': manual_ids}


def protected_deletions(rows):
    return [(row['start_time'], row['end_time']) for row in rows if row['deleted'] and row['origin'] != 'model']


def invalidate_subtitles_if_timing_changed(conn, episode_id, previously):
    from .subtitle_store import invalidate_probes
    from .transcript_parts import timings
    current = conn.execute("""SELECT start_time,end_time,text,words_json FROM transcript_segments WHERE episode_id=?
        AND superseded_by IS NULL AND deleted=0 ORDER BY start_time,end_time""", (episode_id,))
    old = sorted(span for row in previously for span in timings(row))
    new = sorted(span for row in current for span in timings(row))
    if old != new:
        invalidate_probes(conn, episode_id)


def transcribe_episode(snapshot, payload, audio, temp):
    job_id, owner, episode = snapshot.job['id'], snapshot.owner, snapshot.data['episode']
    episode_id = episode['id']
    duration = float(episode['duration'])
    span = payload.get('range')
    start, end = 0., duration
    if span:
        start = max(0., float(span['start_time']))
        end = min(duration, float(span['end_time']))
        if float(span['end_time']) > duration + .2:
            raise RuntimeError('The selected audio ends after the video.')
        if end - start < MIN_RANGE_SECONDS:
            raise RuntimeError('The selected audio is too short to transcribe.')
    rows = snapshot.data['segments']
    current = [row for row in rows if not row['deleted']]
    previous = json.loads(snapshot.job['input_json'] or '{}')
    replace_manual = bool(span and payload.get('replace_manual') is True)
    manual_ids = set(previous.get('replaceable_manual_ids') or [])
    if replace_manual and 'replaceable_manual_ids' not in previous:
        manual_ids = {row['id'] for row in current if row['origin'] != 'model'}
        with snapshot.publish(invalidate=False) as conn:
            conn.execute('UPDATE jobs SET input_json=? WHERE id=?',
                         (json.dumps(transcript_provenance(snapshot, sorted(manual_ids))), job_id))

    ranges = [row for row in snapshot.data['ranges'] if not row['suppressed']]
    automatic = []
    if not span:
        worker.stage(job_id, owner, episode_id, 'Finding repeated audio')
        automatic = audio_ranges.detect(episode, snapshot.data['project_audio'], snapshot.data['ranges'])
        ranges += automatic
    mask = audio_lines.merge_intervals((row['start_time'], row['end_time']) for row in ranges)
    samples = read_audio(audio)
    for left, right in mask:
        samples[round(left * RATE):round(right * RATE)] = 0
    low, high = (max(0., start - CONTEXT_SECONDS), min(duration, end + CONTEXT_SECONDS)) if span else (0., duration)
    worker.stage(job_id, owner, episode_id, 'Identifying voices')
    pipeline = load_diarizer()
    output = diarize(pipeline, samples[round(low * RATE):round(high * RATE)])
    turns = [{'start_time': low + turn.start, 'end_time': low + turn.end, 'speaker': speaker}
             for turn, speaker in output.exclusive_speaker_diarization]
    del output
    from .audio_fingerprint import rms_levels
    rms, hop = rms_levels(samples)
    end = min(end, len(samples) / RATE)
    regions = [(max(start, turn['start_time'] - .2), min(end, turn['end_time'] + .2)) for turn in turns]
    excluded = mask + [(0., start), (end, duration + 1.)]
    regions = audio_lines.subtract_intervals(audio_lines.merge_intervals(regions), excluded)
    windows = audio_lines.plan_windows(regions, rms, hop, excluded)
    lines = []
    if windows:
        worker.stage(job_id, owner, episode_id, 'Loading transcription and alignment models')
        model = load_asr()
        index = audio_lines.TurnIndex(turns)
        for number, (left, right) in enumerate(windows, 1):
            worker.stage(job_id, owner, episode_id, f'Transcribing speech {number}/{len(windows)}')
            words = transcribe_window(model, samples, temp / 'sample.wav', left, right, None)
            tokens = [{'start': word['start_time'], 'end': word['end_time'], 'text': word['text']} for word in words]
            lines.extend(audio_lines.build_lines(tokens, index))
        del model
    lines, _ = line_voices.split_by_mask(lines, mask)
    lines.sort(key=lambda row: (row['start_time'], row['end_time']))
    worker.stage(job_id, owner, episode_id, f'Matching voices on {len(lines)} lines')
    refs = voice_references(snapshot, pipeline) if lines else []
    for number, line in enumerate(lines):
        left, right = line['start_time'], line['end_time']
        scores = {}
        if refs and right - left >= line_voices.MIN_CLIP:
            scores['line'] = line_voices.score_clip(embed(pipeline, samples[int(left * RATE):int(right * RATE)]), refs)
        decision = line_voices.cascade(scores)
        if refs and not decision['speaker']:
            a, b = line_voices.pad_span(lines, number)
            if b - a >= line_voices.MIN_CLIP and (a, b) != (left, right):
                scores['pad'] = line_voices.score_clip(embed(pipeline, samples[max(0, int(a * RATE)):int(b * RATE)]), refs)
                decision = line_voices.cascade(scores)
        line['decision'] = decision

    names = reconcile_speakers(turns, current, {}, (start, end)) if span else {}
    model_id = f'{ASR} + {ALIGNER} + {DIARIZATION}'
    provenance = {**transcript_provenance(snapshot, sorted(manual_ids)), 'model_ids': [ASR, ALIGNER, DIARIZATION],
                  'ranges': mask, 'voice_threshold': line_voices.FIX_TOP, 'voice_gap': line_voices.FIX_GAP,
                  'pad_seconds': line_voices.PAD_REACH, 'minimum_clip_seconds': line_voices.MIN_CLIP}

    def record(conn):
        conn.execute('UPDATE jobs SET input_json=? WHERE id=?', (json.dumps(provenance), job_id))

    with snapshot.publish(on_changed=record) as conn:
        if not span:
            audio_ranges.replace_fingerprints(conn, episode_id, automatic)
            conn.execute('UPDATE episodes SET transcript_stale_reason=NULL WHERE id=?', (episode_id,))
        replaced, kept = 0, 0
        protected = protected_deletions(rows)
        for old in current:
            inside = not span or start <= (old['start_time'] + old['end_time']) / 2 < end
            can_replace = old['id'] in manual_ids or old['origin'] == 'model' and (old['model_id'] or '').startswith(ASR)
            if inside and can_replace:
                tombstone = str(uuid.uuid4())
                conn.execute("""INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,
                    character_context_id,deleted,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json)
                    VALUES (?,?,?,?,?,?,?,1,'model',?,?,?,?,?,?,?)""",
                    (tombstone, episode_id, old['start_time'], old['end_time'], old['text'], old['speaker_label'],
                     old['character_context_id'], old['model_id'], job_id, old['id'], old['words_json'], old['speaker_source'],
                     old['confidence'], old['evidence_json']))
                conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (tombstone, old['id']))
                replaced += 1
            else:
                kept += inside and old['origin'] != 'model'
                protected.append((old['start_time'], old['end_time']))
        overlaps = overlap_index(protected)
        added = 0
        open_lines = False
        rules = transcript_names.alias_rules(snapshot.data['glossary'])
        for line in lines:
            if overlaps(line['start_time'], line['end_time']):
                continue
            speaker = line['diar'] or ''
            label = names.get(speaker, {}).get('label', speaker)
            decision = line['decision']
            open_lines |= not bool(decision['speaker'])
            text, aliases = transcript_names.apply_aliases(line['text'], rules)
            source = {'line': 'voice', 'pad': 'voice_padded'}.get(decision['source'])
            confidence = {'voice': 'high', 'voice_padded': 'medium'}.get(source)
            conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,
                origin,model_id,job_id,words_json,speaker_source,confidence,evidence_json) VALUES (?,?,?,?,?,?,?,'model',?,?,?,?,?,?)""",
                (str(uuid.uuid4()), episode_id, line['start_time'], line['end_time'], text, label, decision['speaker'], model_id, job_id,
                 json.dumps([[w['s'], w['e'], w['t']] for w in line['words']], ensure_ascii=False), source, confidence,
                 json.dumps(dict(decision, fingerprint=snapshot.stamp, aliases=aliases))))
            added += 1
        if open_lines or snapshot.data['glossary']:
            from .speaker_attribution import insert
            insert(conn, episode_id, span=span, dependencies=(job_id,))
        invalidate_subtitles_if_timing_changed(conn, episode_id, current)
        record(conn)
        note = f', {kept} edited by hand kept' if kept else ''
        finish(conn, job_id, episode_id, f'Complete · {added} new, {replaced} replaced{note}')


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
                pipeline = load_diarizer()
                joined = read_audio(sample)
                output = diarize(pipeline, joined)
                labels = output.speaker_diarization.labels()
                if len(labels) != 1 or output.speaker_embeddings is None:
                    raise RuntimeError('Select a sample containing one clear speaker; none or multiple speakers were detected.')
                vector = output.speaker_embeddings[0].tolist()
                if not vector or any(not math.isfinite(x) for x in vector) or not any(vector):
                    raise RuntimeError('Could not extract a usable voice embedding from this sample.')
                source_samples = joined if uploaded else read_audio(audio)
                embed_ranges = [{'start_time': 0, 'end_time': duration}] if uploaded else ranges
                embeddings = line_voices.range_embeddings(source_samples, embed_ranges, lambda clip: embed(pipeline, clip))
                reference = media.voice_reference_path(episode['project_id'], job_id)
                saved = False
                try:
                    with snapshot.publish() as conn:
                        character = snapshot.data['character']
                        conn.execute('INSERT INTO character_voice_samples (id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json,source_filename,range_embeddings_json) VALUES (?,?,?,?,?,?,?,?,?,?)',
                                     (job_id, character['id'], episode['id'], min((r['start_time'] for r in ranges), default=0), max((r['end_time'] for r in ranges), default=duration if uploaded else 0), json.dumps(vector), DIARIZATION, json.dumps(ranges), payload.get('source_filename'), json.dumps(embeddings)))
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
            transcribe_episode(snapshot, payload, audio, temp)
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
            claimed = worker.claim_job(job_kinds.pipelines_for('audio'))
            if not claimed:
                time.sleep(2)
                continue
            job_id, owner = claimed
            done = threading.Event()
            beat = threading.Thread(target=worker.heartbeat, args=(job_id, owner, done), daemon=True)
            beat.start()
            try:
                def check():
                    with db.connect() as conn:
                        worker.require_lease(conn, job_id, owner)

                with gpu_device.hold(check):
                    try:
                        pipeline = db.one('SELECT pipeline FROM jobs WHERE id=?', (job_id,))['pipeline']
                        kind = job_kinds.KINDS[pipeline]
                        if kind.handler:
                            kind.run(job_id, owner)
                        else:
                            process(job_id, owner)
                    finally:
                        release_gpu()
            except worker.LeaseLost:
                pass
            finally:
                done.set()
                beat.join(timeout=1)
        except Exception:
            log.exception('Audio worker error')
            time.sleep(2)


if __name__ == '__main__':
    main()
