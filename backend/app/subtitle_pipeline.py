"""The ``subtitles`` job: extract the burned-in subtitles of a whole episode.

Stages, each loading its model once for the entire episode:

1. locate  - WeVisOne looks for the subtitle strip in a few frames, preferably while people speak
2. scan    - CPU only: decode the strip once and find when its text changes (subtitle_scan)
3. read    - the vision model's llama-server (``visual_models.general()``, Qwen3.6 since 2026-11-18) reads one image per distinct
             subtitle, several at a time

Readings are checkpointed one by one, so a restarted or retried run reads only what is still missing. The result
is a list of time-ranged lines on the episode; cuts see the ones that overlap them (subtitle_store).
"""
import hashlib
import io
import json
import logging
import os
import time
import uuid

from . import db, frame_model, job_inputs, media, subtitle_reader, subtitle_scan, subtitle_store, visual_models, worker

log = logging.getLogger('framewise.subtitles')

PROBE_COUNT = 16
DETECTOR_QUERY = ['subtitles']
PROGRESS_SECONDS = 1.5


def gather(conn, job):
    episode = db.one('SELECT id,project_id,original_ext,duration,width,height FROM episodes WHERE id=?',
                     (job['episode_id'],), conn=conn)
    folder = media.episode_dir(episode['project_id'], episode['id'])
    identities = {}
    for name in (f"original{episode['original_ext']}", 'proxy.mp4'):
        try:
            stat = (folder / name).stat()
            identities[name] = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            pass
    speech = db.all_rows("""SELECT id,start_time,end_time,text,words_json FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL
        AND deleted=0 ORDER BY start_time,end_time,id""", (job['episode_id'],), conn=conn)
    return {'episode': episode, 'media': identities, 'speech': speech}


def inputs(conn, job):
    values = job.get('gathered') or gather(conn, job)
    # The detector uses speech times. Transcript spelling and speakers do not affect the strip.
    return job_inputs.fingerprint({**values, 'speech': speech_spans(values['speech'])})


def speech_spans(rows):
    # Probe the aligned word times, without depending on their spelling.
    from .transcript_parts import timings
    return [span for row in rows for span in timings(row)]


def invalidate(conn, job):
    from .speaker_attribution import mark_stale
    mark_stale(conn, job['episode_id'], 'Episode subtitles changed')
    conn.execute("""UPDATE analysis_results SET stale_reason='Episode subtitles changed' WHERE superseded_by IS NULL
        AND cut_id IN (SELECT id FROM cuts WHERE episode_id=? AND active=1)
        AND (json_extract(settings_json,'$.structured_type') IN ('initial_frame','keyframe_analysis','performance_beats','consistency_summary')
             OR title='Cut summary')""", (job['episode_id'],))


def read_parallel():
    try:
        return max(1, min(8, int(os.environ.get('SUBTITLE_READ_PARALLEL', '4'))))
    except ValueError:
        return 4


class Run:
    def __init__(self, snapshot, payload):
        from . import inference_worker
        self.snapshot = snapshot
        self.job, self.owner, self.episode, self.payload = snapshot.job, snapshot.owner, snapshot.data['episode'], payload
        self.language = subtitle_reader.resolve_language(payload.get('language'))
        folder = media.episode_dir(self.episode['project_id'], self.episode['id'])
        self.original = folder / f"original{self.episode['original_ext']}"
        self.proxy = folder / 'proxy.mp4'
        self.id = None
        self.band = None
        self.scan_diagnostics = {}
        self.timings = {}
        self._last_progress = 0.
        self.cancel_check = lambda: inference_worker.check_summary_lease(self.job['id'], self.owner)

    def publish(self, *, invalidate=False):
        return self.snapshot.publish(invalidate=invalidate, on_changed=self.interrupt)

    def interrupt(self, conn):
        if self.id:
            conn.execute("UPDATE subtitle_runs SET status='interrupted',error='Subtitle inputs changed' WHERE id=? AND status='running'",
                         (self.id,))

    # -- progress ----------------------------------------------------------------------------------

    def progress(self, message, force=False):
        now = time.monotonic()
        if force or now - self._last_progress >= PROGRESS_SECONDS:
            self._last_progress = now
            worker.stage(self.job['id'], self.owner, self.episode['id'], message)

    # -- run record --------------------------------------------------------------------------------

    def open(self):
        """Continue this episode's interrupted or failed run for the same language and prompt, or start a new one."""
        prompt = subtitle_reader.system_prompt(self.language)
        model = visual_models.general()
        with self.publish() as conn:
            # Another model's readings are not continued: a run records one reader.
            reusable = conn.execute("""SELECT * FROM subtitle_runs WHERE episode_id=? AND status IN ('running','failed','interrupted')
                AND language=? AND prompt=? AND model_id=? AND model_revision=? AND input_fingerprint=? ORDER BY rowid DESC LIMIT 1""",
                (self.episode['id'], self.language['name'], prompt, model.model_id, model.revision, self.snapshot.stamp)).fetchone()
            conn.execute("UPDATE subtitle_runs SET status='interrupted',error=COALESCE(error,'Interrupted') WHERE episode_id=? AND status='running'",
                         (self.episode['id'],))
            provenance = {'fingerprint': self.snapshot.stamp, 'media': self.snapshot.data['media'],
                          'transcript_segment_ids': [row['id'] for row in self.snapshot.data['speech']]}
            conn.execute('UPDATE jobs SET input_json=? WHERE id=?', (json.dumps(provenance), self.job['id']))
            if reusable:
                self.id = reusable['id']
                self.band = json.loads(reusable['band_json']) if reusable['band_json'] else None
                conn.execute("UPDATE subtitle_runs SET status='running',job_id=?,error=NULL WHERE id=?", (self.job['id'], self.id))
            else:
                self.id = str(uuid.uuid4())
                conn.execute('''INSERT INTO subtitle_runs(id,episode_id,job_id,language,language_code,status,model_id,model_revision,prompt,settings_json,input_fingerprint)
                                VALUES (?,?,?,?,?,'running',?,?,?,?,?)''',
                             (self.id, self.episode['id'], self.job['id'], self.language['name'], self.language['code'], model.model_id, model.revision, prompt,
                              json.dumps({'detector': 'tencent/WeVisOne-VL-4B', 'read_parallel': read_parallel(), 'scan_width': subtitle_scan.SCAN_WIDTH,
                                          'language_input': self.payload.get('language'), 'inputs': provenance}), self.snapshot.stamp))

    def update(self, **values):
        with self.publish() as conn:
            columns = ','.join(f'{key}=?' for key in values)
            conn.execute(f'UPDATE subtitle_runs SET {columns} WHERE id=?', (*values.values(), self.id))

    # -- stage 1: locate the strip -----------------------------------------------------------------

    def speech(self):
        return speech_spans(self.snapshot.data['speech'])

    def locate(self):
        from . import inference_worker, single_cut_pipeline
        started = time.monotonic()
        times = subtitle_scan.choose_probe_times(self.episode['duration'], self.speech(), PROBE_COUNT)
        if not times:
            raise RuntimeError('The episode has no usable duration; prepare it again')
        from PIL import Image
        from wevisone_vl.visualization import parse_predictions
        model = inference_worker.get_model()
        probes, shares = [], []
        try:
            for index, moment in enumerate(times):
                self.cancel_check()
                self.progress(f'Locating the subtitle strip {index + 1}/{len(times)}', force=index == 0)
                jpeg = subtitle_reader.extract_frame(self.original, moment)
                with Image.open(io.BytesIO(jpeg)) as image:
                    image = image.convert('RGB')
                    detection = model.predict(image, task='detection', query=DETECTOR_QUERY, max_new_tokens=512, temperature=0.0, decode_matting=False)
                    regions = parse_predictions(detection.response, image.width, image.height)
                    boxes = [entry['coords'] for values in regions.values() for entry in values if entry.get('type') == 'box']
                    probes.append({'time': moment, 'width': image.width, 'height': image.height, 'boxes': boxes})
                    shares.append(subtitle_scan.outline_share_of(image, boxes) if boxes else None)
        finally:
            model = None
            single_cut_pipeline.release_gpu_models()
        self.band = subtitle_scan.band_from_probes(probes)
        self.band['probe_boxes'] = [len(probe['boxes']) for probe in probes]
        # Burned-in subtitles are usually outlined. When the detector's own frames show that, the scan can
        # ignore the thin bright structure of still backgrounds, which otherwise looks like text.
        self.band['outlined'], self.band['outline_share'] = subtitle_scan.outlined_style(shares)
        self.timings['locate_seconds'] = round(time.monotonic() - started, 1)
        self.update(band_json=json.dumps(self.band))

    # -- stage 2: scan -----------------------------------------------------------------------------

    def scan(self):
        started = time.monotonic()
        source = self.proxy if 'proxy.mp4' in self.snapshot.data['media'] else self.original
        info = media.probe(source)
        self.progress('Scanning for subtitle changes', force=True)
        result = subtitle_scan.scan_video(source, info['width'], info['height'], self.episode['duration'], self.band, self.cancel_check,
                                          lambda share: self.progress(f'Scanning for subtitle changes {share:.0%}'), outlined=bool(self.band.get('outlined')))
        self.timings['scan_seconds'] = round(time.monotonic() - started, 1)
        events = result['events']
        digest = hashlib.sha1(json.dumps([[event['start_time'], event['end_time'], event['rep_time']] for event in events] + [self.band['top'], self.band['bottom'], bool(self.band.get('outlined'))]).encode()).hexdigest()
        with self.publish() as conn:
            known = conn.execute('SELECT scan_hash FROM subtitle_runs WHERE id=?', (self.id,)).fetchone()['scan_hash']
            if known and known != digest:  # a different strip or video than the readings were made for
                conn.execute('DELETE FROM subtitle_readings WHERE run_id=?', (self.id,))
            conn.execute('UPDATE subtitle_runs SET scan_hash=? WHERE id=?', (digest, self.id))
        self.scan_diagnostics = result['diagnostics']
        return events

    # -- stage 3: read -----------------------------------------------------------------------------

    def read(self, events):
        from . import single_cut_pipeline
        with db.connect() as conn:
            done = {row['event_index'] for row in conn.execute("SELECT event_index FROM subtitle_readings WHERE run_id=? AND outcome!='error'", (self.id,))}
        todo = [dict(event, index=index) for index, event in enumerate(events) if index not in done]
        self.timings['reused_readings'] = len(done)
        if not todo:
            return
        single_cut_pipeline.ensure_models_available(('initial_frame',))
        single_cut_pipeline.release_gpu_models()
        started = time.monotonic()
        finished = 0
        workers = read_parallel()

        def save(event, outcome, text, legible, detail):
            nonlocal finished
            with self.publish() as conn:
                conn.execute('''INSERT INTO subtitle_readings(run_id,event_index,start_time,end_time,rep_time,outcome,text,legible,detail) VALUES (?,?,?,?,?,?,?,?,?)
                                ON CONFLICT(run_id,event_index) DO UPDATE SET start_time=excluded.start_time,end_time=excluded.end_time,rep_time=excluded.rep_time,
                                outcome=excluded.outcome,text=excluded.text,legible=excluded.legible,detail=excluded.detail''',
                             (self.id, event['index'], event['start_time'], event['end_time'], event['rep_time'], outcome, text, legible, detail))
            finished += 1
            self.progress(f'Reading subtitles {len(done) + finished}/{len(events)}', force=finished == 1)

        model = visual_models.general()
        visual_models.require(model.key)
        workers = frame_model.workers(model, workers, 'Subtitle readers')
        with frame_model.model_server(model, cancel_check=self.cancel_check, parallel=workers, context_per_slot=4096) as server:
            subtitle_reader.read_events(server, self.original, self.band, self.language, todo, save, workers=workers, cancel_check=self.cancel_check)
        self.timings['read_seconds'] = round(time.monotonic() - started, 1)

    # -- finish ------------------------------------------------------------------------------------

    def finish(self, events):
        with db.connect() as conn:
            rows = subtitle_store.reading_rows(self.id, conn)
        errors = [row for row in rows if row['outcome'] == 'error']
        unread = []
        if errors and self.payload.get('accept_unread') and len(errors) <= unread_allowed(len(events)):
            unread, errors = [{'time': row['rep_time'], 'detail': (row['detail'] or '')[:200]} for row in errors], []
        if errors:
            raise RuntimeError(f"{len(errors)} of {len(events)} subtitle images could not be read ({errors[0]['detail']}). The others were saved; run the extraction again to read only the missing ones.")
        readings = [{'start_time': row['start_time'], 'end_time': row['end_time'], 'text': row['text'], 'legible': row['legible']} for row in rows if row['outcome'] == 'text']
        lines = subtitle_reader.merge_readings(readings)
        rejected = [{'time': row['rep_time'], 'reason': row['detail']} for row in rows if row['outcome'] == 'rejected']
        diagnostics = {**self.scan_diagnostics, 'events': len(events), 'readings': {outcome: sum(1 for row in rows if row['outcome'] == outcome) for outcome in ('text', 'empty', 'rejected')},
                       'lines': len(lines), 'rejected_examples': rejected[:20], 'timings': self.timings,
                       **({'warning': 'No subtitle text was found. Check the language and the detected strip.'} if not lines else {}),
                       **({'unread': unread} if unread else {})}
        with self.publish(invalidate=True) as conn:
            conn.execute('DELETE FROM subtitle_lines WHERE run_id=?', (self.id,))
            conn.executemany('INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence,readings) VALUES (?,?,?,?,?,?,?,?,?)',
                             [(str(uuid.uuid4()), self.id, self.episode['id'], line['line_index'], line['start_time'], line['end_time'], line['text'], line['confidence'], line['readings']) for line in lines])
            conn.execute("UPDATE subtitle_runs SET status='complete',error=NULL,diagnostics_json=?,finished_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(diagnostics, ensure_ascii=False), self.id))
            conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         (f'Complete · {len(lines)} subtitle lines', self.job['id']))
            worker.sync_episode_status(conn, self.episode['id'])
        return len(lines)


UNREAD_SHARE = 0.01  # a run queued with accept_unread completes while at most this share (or UNREAD_MIN images) stays unreadable
UNREAD_MIN = 2


def unread_allowed(events):
    return max(UNREAD_MIN, int(events * UNREAD_SHARE))


def validate(episode, original):
    if not episode.get('duration') or not episode.get('width') or not episode.get('height'):
        raise RuntimeError('Prepare the video before extracting subtitles')
    if not original.is_file():
        raise RuntimeError('The original video is missing')


def process(job_id, owner):
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    run = None
    try:
        with db.connect() as conn:
            conn.execute('BEGIN')
            snapshot = job_inputs.JobInputs(conn, job, owner)
        run = Run(snapshot, json.loads(job['payload_json']))
        episode = run.episode
        validate(episode, run.original)
        run.open()
        if not run.band:
            run.locate()
        events = run.scan()
        run.read(events)
        lines = run.finish(events)
        log.info('Subtitles for episode %s: %s lines from %s scan events', episode['id'], lines, len(events))
    except inference_worker.SummaryCancelled:
        log.info('Subtitle extraction %s was cancelled', job_id)
        _mark_failed(run, 'Cancelled')
    except worker.LeaseLost:
        log.warning('Subtitle extraction %s lost its lease', job_id)
    except Exception as exc:
        log.exception('Subtitle extraction failed for job %s', job_id)
        _mark_failed(run, str(exc)[-1800:])
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)


def _mark_failed(run, message):
    if run is not None and run.id:
        try:
            with db.connect() as conn:
                conn.execute("UPDATE subtitle_runs SET status='failed',error=? WHERE id=? AND status='running'", (message, run.id))
        except Exception:
            log.exception('Could not record the failed subtitle run')
