"""CPU shot keyframes. Indices are visual change indices, not movement percentages."""
import hashlib
import json
import math
import queue
import re
import subprocess
import threading
from collections import deque

import cv2
import numpy as np

VERSION = 'keyframes-v2'
DEFAULTS = {'longest_edge': 640, 'grid': 4, 'normalization': 12.0, 'event_threshold': .15,
            'novelty_threshold': .12, 'state_novelty_threshold': .06, 'budget': 15, 'mask_top': .8, 'mask_bottom': 1.0,
            'blur': 3, 'ssim_low': .10, 'ssim_high': .22, 'ssim_threshold': .995,
            'global_weight': .35, 'regional_weight': .35, 'accumulated_weight': .30}


def config(overrides):
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
    c = {**DEFAULTS, **overrides}
    for k, v in c.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'Invalid {k}')
    for k, lo, hi in [('longest_edge', 64, 1920), ('grid', 1, 16), ('budget', 1, 15), ('blur', 1, 9)]:
        if int(c[k]) != c[k] or not lo <= c[k] <= hi:
            raise ValueError(f'Invalid {k}')
        c[k] = int(c[k])
    if c['blur'] % 2 != 1 or c['normalization'] <= 0:
        raise ValueError('Blur must be odd and normalization positive')
    if not 0 <= c['mask_top'] < c['mask_bottom'] <= 1 or (c['mask_top'] == 0 and c['mask_bottom'] == 1):
        raise ValueError('Invalid subtitle mask')
    for k in ('event_threshold', 'novelty_threshold', 'state_novelty_threshold', 'ssim_low', 'ssim_high', 'ssim_threshold'):
        if not 0 <= c[k] <= 1: raise ValueError(f'Invalid {k}')
    if c['state_novelty_threshold'] <= 0: raise ValueError('State novelty threshold must be positive')
    if c['ssim_low'] > c['ssim_high']: raise ValueError('Invalid SSIM band')
    weights = [c[k] for k in ('global_weight', 'regional_weight', 'accumulated_weight')]
    if min(weights) < 0 or not math.isclose(sum(weights), 1, abs_tol=1e-6): raise ValueError('Weights must be nonnegative and sum to 1')
    return c


def cache_key(source, cut, settings, mask):
    stat = source.stat()
    value = [VERSION, str(source.resolve()), stat.st_size, stat.st_mtime_ns, cut['id'], cut['start_time'], cut['end_time'], settings, mask]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def mask_for(episode_id, settings, conn=None):
    from . import subtitle_store
    run = subtitle_store.current_run(episode_id, conn)
    band = (run or {}).get('band') or {}
    top, bottom = band.get('top'), band.get('bottom')
    if isinstance(top, (int, float)) and isinstance(bottom, (int, float)) and 0 <= top < bottom <= 1 and (top > 0 or bottom < 1):
        return {'top': top, 'bottom': bottom, 'source': 'detected', 'subtitle_run_id': run['id']}
    return {'top': settings['mask_top'], 'bottom': settings['mask_bottom'], 'source': 'fallback', 'subtitle_run_id': None}


def latest_job(conn, episode_id, cut_id):
    """The newest ``keyframes`` job that covers the cut (any status), or None."""
    return conn.execute('''SELECT j.id,j.status,j.error,j.payload_json FROM jobs j WHERE j.episode_id=? AND j.pipeline='keyframes'
                           AND (json_extract(j.payload_json,'$.all_cuts')=1
                           OR EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.cut_ids') WHERE value=?)
                           OR EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.cuts') WHERE json_extract(value,'$.id')=?))
                           ORDER BY j.rowid DESC LIMIT 1''', (episode_id, cut_id, cut_id)).fetchone()


def current_selection(cut_id, conn=None):
    """``(cut, source path, key, result)`` of the cut's latest selection; key and result are None when it is missing or stale.

    The cut is None when it does not exist or is not active. The key is the ``keyframe_results`` cache key: the
    settings of the cut's latest selection job with today's subtitle mask, so extracting subtitles makes it stale.
    """
    from . import db, media
    if conn is None:
        with db.connect() as conn:
            conn.execute('BEGIN')
            return current_selection(cut_id, conn)
    cut = db.one('''SELECT c.*,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id
                    WHERE c.id=? AND c.active=1''', (cut_id,), conn=conn)
    if not cut:
        return None, None, None, None
    source = media.episode_dir(cut['project_id'], cut['episode_id']) / f"original{cut['original_ext']}"
    job = latest_job(conn, cut['episode_id'], cut_id)
    if not job:
        return cut, source, None, None
    settings = json.loads(job['payload_json'])['keyframe_settings']
    try:
        key = cache_key(source, cut, settings, mask_for(cut['episode_id'], settings, conn))
    except OSError:
        return cut, source, None, None
    row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,), conn=conn)
    return (cut, source, key, json.loads(row['result_json'])) if row else (cut, source, None, None)


def ensure_selection(conn, episode_id, cut_ids):
    """Queue keyframe selection for the cuts without a current one, inside the caller's transaction.

    The keyframe analysis needs the selection; the visual analysis job that runs it waits for the jobs returned
    here (``payload.depends_on``). A pending job that already covers a cut is reused. A new job uses the
    settings of the cut's latest selection, else the defaults, so a selection the user tuned is not reset.
    Returns the ids of the keyframes jobs to wait for.
    """
    from . import media, worker
    episode = conn.execute('SELECT project_id,original_ext FROM episodes WHERE id=?', (episode_id,)).fetchone()
    source = media.episode_dir(episode['project_id'], episode_id) / f"original{episode['original_ext']}"
    if not source.is_file():
        return []  # nothing to select from; the keyframe stage reports the missing selection itself
    waits, groups = [], {}
    for cut_id in cut_ids:
        cut = conn.execute('SELECT id,start_time,end_time FROM cuts WHERE id=? AND episode_id=? AND active=1', (cut_id, episode_id)).fetchone()
        if not cut:
            continue
        job = latest_job(conn, episode_id, cut_id)
        if job and job['status'] in ('queued', 'running'):
            waits.append(job['id'])
            continue
        settings = json.loads(job['payload_json'])['keyframe_settings'] if job else config({})
        band = mask_for(episode_id, settings, conn)
        try:
            key = cache_key(source, cut, settings, band)
        except OSError:
            key = None
        if key and conn.execute('SELECT 1 FROM keyframe_results WHERE cache_key=?', (key,)).fetchone():
            continue
        group = groups.setdefault(json.dumps(settings, sort_keys=True), {'keyframe_settings': settings, 'cut_ids': []})
        group['cut_ids'].append(cut_id)
    for payload in groups.values():
        waits.append(worker.enqueue_in(conn, episode_id, 'keyframes', payload))
    return list(dict.fromkeys(waits))


def decoded(source, width, height, edge):
    """Stream grayscale images and FFmpeg presentation timestamps; never drop or duplicate frames."""
    scale = min(1, edge / max(width, height))
    w, h = max(2, round(width * scale)), max(2, round(height * scale))
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', f'scale={w}:{h},format=gray,showinfo', '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    stamps = queue.Queue()
    tail = deque(maxlen=20)
    def read_log():
        tb = None
        for raw in proc.stderr:
            line = raw.decode(errors='replace'); tail.append(line)
            if 'Parsed_showinfo' not in line: continue
            match = re.search(r'config in time_base:\s*(\d+)/(\d+)', line)
            if match: tb = int(match[1]) / int(match[2])
            match = re.search(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)', line)
            if match: stamps.put((int(match[1]), int(match[2]) * tb if tb is not None else None))
        stamps.put(None)
    thread = threading.Thread(target=read_log, daemon=True); thread.start()
    try:
        n = 0; last = -math.inf
        while True:
            data = bytearray()
            while len(data) < w * h:
                part = proc.stdout.read(w * h - len(data))
                if not part: break
                data.extend(part)
            if not data: break
            if len(data) != w * h: raise RuntimeError('Incomplete decoded frame')
            stamp = stamps.get()
            if stamp is None or stamp[0] != n or stamp[1] is None or stamp[1] < last:
                raise RuntimeError('Missing or non-monotonic decoded presentation timestamp')
            last = stamp[1]
            yield n, last, np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            n += 1
        if proc.wait() != 0: raise RuntimeError(''.join(tail)[-1800:])
        if stamps.get() is not None: raise RuntimeError('Unmatched decoded presentation timestamp')
    finally:
        if proc.poll() is None: proc.kill()
        proc.wait(); proc.stdout.close(); thread.join(timeout=2); proc.stderr.close()


def compare(a, b, mask, c):
    diff = cv2.absdiff(a, b).astype(np.float32)
    valid = mask
    global_score = min(1., float(diff[valid].mean()) / c['normalization'])
    regions = []
    for ys in np.array_split(np.arange(diff.shape[0]), c['grid']):
        for xs in np.array_split(np.arange(diff.shape[1]), c['grid']):
            region = diff[np.ix_(ys, xs)]; good = valid[np.ix_(ys, xs)]
            if good.any(): regions.append(min(1., float(region[good].mean()) / c['normalization']))
    regional = max(float(np.percentile(regions, 90)), .7 * max(regions))
    return global_score, regional, sum(x >= .15 for x in regions) / len(regions)


def state_distance(a, b, mask, c):
    """Unclipped visual difference on a 0–1 grayscale scale, including regional changes."""
    raw_config = {**c, 'normalization': 255.0}
    global_change, regional_change, _ = compare(a, b, mask, raw_config)
    return max(global_change, regional_change)


def frame_budget(duration, configured):
    """Five frames per started five-second window, capped at 15 and the configured limit."""
    return min(configured, 15, 5 * math.ceil(duration / 5.0))


def frame_window(item):
    return int(item['relative_timestamp_seconds'] // 5.0)


def similarity(a, b, mask):
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    mx, my = x.mean(), y.mean(); vx, vy = x.var(), y.var()
    return float(((2*mx*my + 6.5025) * (2*((x-mx)*(y-my)).mean() + 58.5225)) / ((mx*mx + my*my + 6.5025) * (vx+vy+58.5225)))


class Shot:
    def __init__(self, cut, settings, band):
        self.cut, self.c, self.band = cut, config(settings), band
        self.states = []; self.state_dropped = 0; self.events = []; self.event_count = 0; self.count = 0; self.prev = None; self.ref = None; self.active_event = None
        self.peak_g = self.peak_r = self.peak_a = self.peak_cov = 0.

    def add(self, ordinal, t, frame):
        c = self.c
        frame = frame.copy()
        frame[int(self.band['top'] * frame.shape[0]):math.ceil(self.band['bottom'] * frame.shape[0]), :] = 0
        frame = cv2.GaussianBlur(frame, (c['blur'], c['blur']), 0) if c['blur'] > 1 else frame
        if self.prev is None:
            self.mask = np.ones(frame.shape, dtype=bool)
            self.mask[int(self.band['top'] * frame.shape[0]):math.ceil(self.band['bottom'] * frame.shape[0]), :] = False
            if not self.mask.any(): raise RuntimeError('Subtitle mask leaves no usable pixels')
            self.first = self.ref = self.state_ref = frame
            g = r = a = cov = 0.
        else:
            g, r, cov = compare(frame, self.prev, self.mask, c)
            ag, ar, _ = compare(frame, self.first, self.mask, c)
            rg, rr, _ = compare(frame, self.ref, self.mask, c)
            a = max(.5*(ag+ar), .5*(rg+rr))
        score = c['global_weight']*g + c['regional_weight']*r + c['accumulated_weight']*a
        item = {'source_frame_index': ordinal, 'frame_index': self.count, 'timestamp_seconds': t,
                'relative_timestamp_seconds': t-self.cut['start_time'], 'global_change_score': 100*g,
                'regional_change_score': 100*r, 'accumulated_change_score': 100*a, 'change_score': 100*score, '_image': frame}
        if self.prev is not None and state_distance(frame, self.state_ref, self.mask, c) >= c['state_novelty_threshold']:
            self._record_state(item)
            self.state_ref = frame
        self.count += 1
        self.peak_g = max(self.peak_g, g); self.peak_r = max(self.peak_r, r); self.peak_a = max(self.peak_a, a); self.peak_cov = max(self.peak_cov, cov)
        strength = max(g, r, .5*(rg+rr) if self.prev is not None else 0)
        meaningful = strength >= c['event_threshold']
        if meaningful and c['ssim_low'] <= strength <= c['ssim_high'] and r <= 1.5*g and similarity(frame, self.prev, self.mask) >= c['ssim_threshold']: meaningful = False
        if meaningful:
            if self.active_event is None:
                self.active_event = {'peak': item, 'before': self.last, 'strength': strength}
                self.ref = frame
            elif strength > self.active_event['strength']:
                self.active_event.update(peak=item, strength=strength)
        elif self.active_event:
            self.active_event['after'] = item; self._record_event(self.active_event); self.ref = self.active_event['peak']['_image']; self.active_event = None
        self.prev = frame; self.last = item
        if self.count == 1: self.initial = item

    def _record_state(self, item):
        # Retain a bounded, time-spread set, including the latest state. Event peaks
        # are retained separately, so compacting these never removes transient evidence.
        self.states.append(item)
        if len(self.states) > 64:
            index = min(range(1, len(self.states) - 1),
                        key=lambda i: (self.states[i + 1]['timestamp_seconds'] - self.states[i - 1]['timestamp_seconds'], i))
            self.states.pop(index)
            self.state_dropped += 1

    def _record_event(self, event):
        self.event_count += 1
        self.events.append(event)
        if len(self.events) > 64:
            weakest = min(range(len(self.events)), key=lambda i: (self.events[i]['strength'], -i))
            self.events.pop(weakest)

    def finish(self):
        if not self.count: raise RuntimeError('No decoded frames within cut')
        if self.active_event: self.active_event['after'] = self.last; self._record_event(self.active_event); self.active_event = None
        c = self.c
        budget = frame_budget(self.cut['end_time'] - self.cut['start_time'], c['budget'])
        selected = [(self.initial, 'initial_state')]
        remaining = [(item, 'visual_state', -1) for item in self.states]
        for i, ev in enumerate(self.events):
            remaining.append((ev['peak'], 'localized_change_peak' if ev['peak']['regional_change_score'] > ev['peak']['global_change_score'] else 'global_change_peak', i))
            if ev['after']: remaining.append((ev['after'], 'event_after', i))
            if ev['before']: remaining.append((ev['before'], 'event_before', i))
        remaining.append((self.last, 'final_state', -1))
        threshold = c['state_novelty_threshold']
        distances = {}
        def distance(a, b):
            key = tuple(sorted((a['frame_index'], b['frame_index'])))
            if key not in distances:
                distances[key] = state_distance(a['_image'], b['_image'], self.mask, c)
            return distances[key]

        while len(selected) < budget and remaining:
            scored = []
            for item, reason, event in remaining:
                if any(item['frame_index'] == x['frame_index'] for x, _ in selected): continue
                if sum(frame_window(x) == frame_window(item) for x, _ in selected) >= 5: continue
                novelty = min(distance(item, x) for x, _ in selected)
                # A distinctive local event may affect a small part of the image.
                peak = event >= 0 and reason.endswith('peak')
                local_novelty = (min(max(compare(item['_image'], x['_image'], self.mask, c)[:2]) for x, _ in selected)
                                 if peak and novelty < threshold else 0)
                is_return = bool(
                    reason == 'final_state' and self.events and len(selected) > 1
                    and item['frame_index'] > self.events[-1]['peak']['frame_index']
                    and distance(item, self.initial) < threshold
                    and any(distance(x, self.initial) >= threshold for x, _ in selected[1:])
                    and not any(x['frame_index'] > self.events[-1]['peak']['frame_index']
                                and distance(item, x) < threshold for x, _ in selected[1:])
                )
                if novelty < threshold and not (peak and local_novelty >= c['novelty_threshold']) and not (is_return and len(selected) > 1):
                    continue
                temporal = min(abs(item['timestamp_seconds'] - x['timestamp_seconds']) for x, _ in selected) / max(.001, self.cut['end_time'] - self.cut['start_time'])
                # A peak is useful, but a saturated early peak must not crowd out
                # distinct stages of a continuous camera move.
                bonus = .02 if peak else 0
                scored.append((novelty + bonus + .06*temporal, -item['frame_index'], item, reason))
            if not scored: break
            _, _, item, reason = max(scored, key=lambda x: (x[0], x[1]))
            selected.append((item, reason))
            remaining = [x for x in remaining if x[0]['frame_index'] != item['frame_index']]

        def represented(item):
            return any(distance(item, chosen) < threshold for chosen, _ in selected)

        uncovered_states = sum(not represented(item) for item in self.states)
        covered_events = 0
        for event in self.events:
            before = event['before']['frame_index'] if event['before'] else event['peak']['frame_index']
            after = event['after']['frame_index'] if event['after'] else event['peak']['frame_index']
            if any(before <= item['frame_index'] <= after and
                   distance(item, event['peak']) < threshold
                   for item, _ in selected):
                covered_events += 1
        uncovered_events = self.event_count - covered_events
        limit_reached = len(selected) >= budget or any(sum(frame_window(x) == frame_window(item) for x, _ in selected) >= 5 for item in self.states if not represented(item))
        coverage_limited = bool(uncovered_events or (limit_reached and (uncovered_states or self.state_dropped)))
        shot_score = 100*min(1, .25*self.peak_g + .25*self.peak_r + .25*self.peak_a + .15*min(self.event_count/5, 1) + .10*self.peak_cov)
        frames = [{**{k:v for k,v in x.items() if k != '_image'}, 'reason': reason} for x, reason in sorted(selected, key=lambda y:y[0]['frame_index'])]
        return {'shot_id': self.cut['id'], 'start_timestamp_seconds': self.cut['start_time'], 'end_timestamp_seconds': self.cut['end_time'], 'duration_seconds': self.cut['end_time']-self.cut['start_time'], 'total_frames': self.count, 'shot_change_score': shot_score, 'change_level': 'low' if shot_score < 25 else 'moderate' if shot_score < 60 else 'high', 'coverage_limited': coverage_limited, 'detected_events': self.event_count, 'uncovered_events': uncovered_events, 'subtitle_mask': self.band, 'selected_frame_count': len(frames), 'selected_frames': frames}


def gather(conn, job):
    from . import db, media
    payload = json.loads(job['payload_json'])
    cut = db.one("""SELECT c.id,c.start_time,c.end_time,e.project_id,e.original_ext
        FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=? AND c.episode_id=? AND c.active=1""",
                 (job['unit_id'], job['episode_id']), conn=conn)
    if not cut:
        return {'skip': 'Cut archived or removed'}
    source = media.episode_dir(cut.pop('project_id'), job['episode_id']) / f"original{cut.pop('original_ext')}"
    settings = payload['keyframe_settings']
    band = mask_for(job['episode_id'], settings, conn)
    try:
        key = cache_key(source, cut, settings, band)
    except OSError:
        return {'skip': 'Original video unavailable'}
    return {'cut': cut, 'mask': band, 'cache_key': key}


def inputs(conn, job):
    from . import job_inputs
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    conn.execute("""UPDATE analysis_results SET stale_reason='Target frame selection changed'
        WHERE cut_id=? AND superseded_by IS NULL AND
        (json_extract(settings_json,'$.structured_type') IN ('keyframe_analysis','performance_beats','consistency_summary')
         OR title='Cut summary')""", (job['unit_id'],))


def process(job, owner, episode, payload, source):
    from . import db, job_inputs, media, target_reviews, worker
    settings = payload['keyframe_settings']
    with db.connect() as conn:
        conn.execute('BEGIN')
        active = conn.execute('SELECT id FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time', (episode['id'],)).fetchall()
        requested = None if payload.get('all_cuts') else set(payload.get('cut_ids', [cut['id'] for cut in payload.get('cuts', [])]))
        units = [job_inputs.JobInputs(conn, job, owner, row['id']) for row in active
                 if requested is None or row['id'] in requested]
        missing = [unit for unit in units if not unit.data.get('skip') and not conn.execute(
            'SELECT 1 FROM keyframe_results WHERE cache_key=?', (unit.data['cache_key'],)).fetchone()]
    if not missing:
        if not units:
            return 'Skipped · no active cuts'
        if all(unit.data.get('skip') for unit in units):
            return f"Skipped · {units[0].data['skip']}"
        return 'Complete'
    info = media.probe(source)
    worker.stage(job['id'], owner, episode['id'], 'Selecting keyframes')
    index, shot = 0, None

    def save(current, unit):
        result = current.finish()
        cut_id, key = unit.job['unit_id'], unit.data['cache_key']
        with unit.publish() as conn:
            conn.execute('INSERT INTO keyframe_results(cache_key,episode_id,cut_id,result_json) VALUES (?,?,?,?)',
                         (key, episode['id'], cut_id, json.dumps({**result, 'input_fingerprint': unit.stamp, 'input_settings': settings})))
            # Preserve the reviews of frames that no longer belong to the current selection.
            target_reviews.archive_selection(conn, cut_id, key)
            conn.execute('DELETE FROM target_detection_runs WHERE cut_id=? AND selection_key!=?', (cut_id, key))

    for ordinal, timestamp, frame in decoded(source, info['width'], info['height'], settings['longest_edge']):
        if ordinal % 128 == 0:
            with db.connect() as conn:
                worker.require_lease(conn, job['id'], owner)
        while index < len(missing) and timestamp >= missing[index].data['cut']['end_time']:
            if shot is None:
                raise RuntimeError('No decoded frames within cut')
            save(shot, missing[index])
            index += 1
            shot = None
        if index == len(missing):
            break
        unit = missing[index]
        cut = unit.data['cut']
        if timestamp >= cut['start_time']:
            if shot is None:
                shot = Shot(cut, settings, unit.data['mask'])
            shot.add(ordinal, timestamp, frame)
    if index < len(missing) and shot is not None:
        save(shot, missing[index])
        index += 1
    if index < len(missing):
        raise RuntimeError('Video ended before all cuts were decoded')
    return 'Complete'
