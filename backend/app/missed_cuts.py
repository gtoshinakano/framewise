"""Hard cuts the episode-wide detection missed (docs/autopilot/HARNESS.md, milestone M2).

The episode-wide detection splits where ffmpeg's ``scene`` score passes one threshold (0.32 by default). Cuts between
similar shots score lower, so the user used to compare each cut's first and last frame and re-detect inside it with a
lower threshold (``POST /api/cuts/{id}/detect-hardcuts``). This module does that check on the CPU:

1. **Scores** (``scene_scores``): one ffmpeg pass over the proxy gives every frame's scene score (about 3 s for a 25-minute
   episode), cached under ``DATA_DIR/cache/scene_scores`` by the proxy's identity.
2. **Candidates** (``candidates``): local peaks inside a cut, away from its edges by ``EDGE`` like the detection itself,
   with their *isolation*: the peak divided by the strongest other score within ``NEIGHBOURS`` frames. A cut is one
   lone spike; camera motion, effects and fast action raise many frames in a row.
3. **Pictures** (``picture_features``): the frames on each side of the peak. A good ORB + RANSAC fit between them
   (``keyframe_camera.camera_estimate``) means the same shot moved (pan, zoom, shake), not a new one; a near-white or
   near-black frame at the peak is a flash or a fade; the grey-level histogram correlation says how alike they look.
4. **Verdict** (``verdict``): ``apply`` (split it), ``grey`` (a second reviewer looks at a two-frame strip) or ``reject``,
   with thresholds measured by the replay (``autopilot/replay/missed_cuts.py``).
5. **Split** (``split_at``): exactly at the peak frame, like the editor's Split, never through a lower detection
   threshold: a threshold splits at every frame above it, so one strong motion frame elsewhere in the cut would block
   (or wrongly add) splits. It is a ``cut_history`` edit (the editor's Undo works) and an agent decision of kind
   ``missed_cuts``; ``undo_split`` merges the two pieces back while they are exactly as the split left them.

The job (``process``, pipeline ``missed_cuts``, CPU, media lane) stores one ``missed_cut_checks`` row per cut, splits the
``apply`` peaks in its publish transaction, and opens one ``missed_cut`` escalation per grey peak, for Claude first
(``reviewer='claude'``, with a strip image at ``/api/cuts/{id}/missed-cut-strip``) when the profile allows it. Cuts with
analysis switched off are skipped (the opening and ending, which the user leaves alone); a cut that belongs to a sequence
is never split by the agent: its peaks go to a person. ``answer`` applies Claude's or the person's answer.
"""
import hashlib
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from . import agent_log, cut_history, db, job_inputs, worker

EDGE = 0.15            # media.detect_cuts_* never split this close to a cut's edges
FLOOR = 0.10           # lowest scene score worth a look (the user's re-detections went down to 0.11)
NEIGHBOURS = 12        # frames on each side that a peak must stand out from (half a second at 24 fps)
WIDTH = 640            # frames are compared at this width
SLIVER = 0.5           # a split leaving a piece shorter than this is asked about, never applied
VERSION = 2            # of the scoring; a check made by another version is redone

# Verdict thresholds (tuned with autopilot/replay/missed_cuts.py; see docs/autopilot/M2.md).
APPLY = {'score': 0.15, 'isolation': 2.5, 'correlation': 0.97}
GREY = {'score': 0.10, 'isolation': 1.5}
CONTINUOUS = {'inliers': 40, 'ratio': 0.5}   # a camera fit this good means the same shot
FLASH_LUMA = (20, 235)


def _identity(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{VERSION}'.encode()).hexdigest()[:24]


def cache_dir() -> Path:
    return db.DATA_DIR / 'cache' / 'scene_scores'


def scene_scores(proxy: Path, timeout: float = 1800) -> tuple[list[float], list[float]]:
    """``(times, scores)`` of every frame of the proxy, from ffmpeg's ``scene`` measure (the one the detection uses)."""
    cache = cache_dir() / f'{_identity(proxy)}.json'
    if cache.is_file():
        try:
            data = json.loads(cache.read_text())
            return data['t'], data['s']
        except (ValueError, KeyError):
            pass
    with tempfile.TemporaryDirectory(prefix='framewise-scene-') as temp:
        out = Path(temp) / 'scores.txt'
        result = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-loglevel', 'error', '-i', str(proxy), '-an',
                                 '-vf', f"select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={out}", '-f', 'null', '-'],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-1800:])
        times, scores = parse_scores(out.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix('.partial')
    partial.write_text(json.dumps({'t': times, 's': scores}))
    partial.replace(cache)
    return times, scores


def parse_scores(text: str) -> tuple[list[float], list[float]]:
    """The metadata filter's ``pts_time:`` and ``lavfi.scene_score=`` lines as two parallel lists."""
    times, scores, pending = [], [], None
    for line in text.splitlines():
        if (match := re.search(r'pts_time:(-?[\d.]+)', line)):
            pending = float(match.group(1))
        elif (match := re.search(r'lavfi\.scene_score=([\d.]+)', line)) and pending is not None:
            times.append(pending)
            scores.append(float(match.group(1)))
            pending = None
    return times, scores


# --- candidates ----------------------------------------------------------------------------------------------------

def isolation(scores: list[float], index: int, span: int = NEIGHBOURS) -> float:
    """The peak divided by the strongest other score within ``span`` frames (its direct neighbours left out)."""
    others = [scores[j] for j in range(max(0, index - span), min(len(scores), index + span + 1)) if abs(j - index) > 1]
    return round(scores[index] / max(max(others, default=0.0), 0.01), 3)


def candidates(times: list[float], scores: list[float], start: float, end: float, floor: float = FLOOR) -> list[dict]:
    """Local peaks of at least ``floor`` strictly inside a cut (``EDGE`` from both edges), strongest first."""
    found = []
    for index in range(1, len(scores) - 1):
        moment = times[index]
        if moment <= start + EDGE:
            continue
        if moment >= end - EDGE:
            break
        score = scores[index]
        if score < floor or score < scores[index - 1] or score < scores[index + 1]:
            continue
        found.append({'index': index, 'time': round(moment, 6), 'score': round(score, 4), 'isolation': isolation(scores, index)})
    return sorted(found, key=lambda item: -item['score'])


# --- pictures ------------------------------------------------------------------------------------------------------

def _dimensions(proxy: Path) -> tuple[int, int]:
    from . import media
    out = media.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(proxy)], timeout=60).stdout
    width, height = (int(value) for value in out.strip().split(',')[:2])
    return width, height


def frames_near(proxy: Path, at: float, span: float = 0.25, width: int = WIDTH, size: tuple[int, int] | None = None):
    """``[(time, bgr ndarray)]`` of the frames within ``span`` seconds of ``at``, ``width`` pixels wide."""
    import numpy as np
    from . import media
    source_w, source_h = size or _dimensions(proxy)
    height = max(2, round(source_h * width / source_w / 2) * 2)
    start = max(0.0, at - span)
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-loglevel', 'info', '-ss', f'{start:.6f}', '-i', str(proxy),
                             '-t', f'{2 * span:.6f}', '-an', '-vf', f'scale={width}:{height},format=bgr24,showinfo', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1800:])
    stamps = list(media.showinfo_times(result.stderr.decode(errors='replace').splitlines()))
    frame_bytes = width * height * 3
    count = min(len(stamps), len(result.stdout) // frame_bytes)
    pixels = np.frombuffer(result.stdout[:count * frame_bytes], dtype=np.uint8).reshape(count, height, width, 3)
    return [(start + stamps[i], pixels[i]) for i in range(count)]


def nearest(frames, moment):
    return min(frames, key=lambda item: abs(item[0] - moment)) if frames else None


def peak_pair(frames, after_time: float, reach: float = 2.5 / 24):
    """The adjacent frames that differ most within ``reach`` seconds of the peak.

    A seek gives timestamps that can be a frame off the whole-file pass (the proxy's start offset), so the pair is found
    from the pixels instead of trusted from the clock."""
    import numpy as np
    best, best_diff = None, -1.0
    for k in range(1, len(frames)):
        if abs(frames[k][0] - after_time) > reach:
            continue
        a, b = frames[k - 1][1][::4, ::4].astype(np.int16), frames[k][1][::4, ::4].astype(np.int16)
        diff = float(np.abs(a - b).mean())
        if diff > best_diff:
            best, best_diff = (frames[k - 1], frames[k]), diff
    return best


def sharpness(gray) -> float:
    """Variance of the Laplacian: line art is sharp; a motion-blurred or defocused frame is not."""
    import cv2
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def picture_features(frames, before_time: float, after_time: float) -> dict:
    """Camera fit, histogram correlation and flash checks between the frames on each side of a peak."""
    import cv2
    from . import keyframe_camera
    pair = peak_pair(frames, after_time)
    if pair is None:
        return {'error': 'no frames around the peak'}
    before, after = pair
    before_time, after_time = before[0], after[0]
    a, b = keyframe_camera.gray(before[1]), keyframe_camera.gray(after[1])
    camera = keyframe_camera.camera_estimate(a, b, keyframe_camera.band_mask(a.shape, None))
    hist_a = cv2.calcHist([a], [0], None, [32], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [32], [0, 256])
    correlation = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    lumas = [(moment, float(frame.mean())) for moment, frame in frames]
    close = [luma for moment, luma in lumas if abs(moment - after_time) <= 2.5 / 24 or abs(moment - before_time) <= 1.5 / 24]
    edges = [lumas[0][1], lumas[-1][1]]
    dark, bright = FLASH_LUMA
    flash = any(luma < dark or luma > bright for luma in close) and not all(luma < dark or luma > bright for luma in edges)
    # Recorded for tuning only: a blur gate sent more real cuts to review than it caught false ones (docs/autopilot/M2.md).
    sharp = [sharpness(a), sharpness(b)]
    return {'camera': camera, 'correlation': round(correlation, 3), 'flash': flash, 'sharpness': sharp,
            'luma': [round(float(a.mean()), 1), round(float(b.mean()), 1)]}


STRIP_TILE = 360


def strip_jpeg(proxy: Path, after_time: float, size=None) -> bytes:
    """What a reviewer looks at: two frames before and two after the peak, with a red bar where the cut would be, and below
    them eight frames over a second on each side for context (dark close-ups are easy to misread from four frames)."""
    import cv2
    import numpy as np
    frames = frames_near(proxy, after_time, span=1.0, width=STRIP_TILE, size=size)
    pair = peak_pair(frames, after_time)
    if pair is None:
        raise ValueError('no frames around the peak')
    k = next(i for i, item in enumerate(frames) if item[0] == pair[1][0])
    chosen = [frames[i][1] for i in range(max(0, k - 2), min(len(frames), k + 2))]
    split = k - max(0, k - 2)
    height = chosen[0].shape[0]
    bar = np.zeros((height, 8, 3), dtype=np.uint8)
    bar[:, :] = (0, 0, 255)
    gap = np.full((height, 3, 3), 40, dtype=np.uint8)
    parts = []
    for i, frame in enumerate(chosen):
        if i:
            parts.append(bar if i == split else gap)
        parts.append(frame)
    top = np.hstack(parts)
    step = max(1, len(frames) // 8)
    context = [frames[i] for i in range(0, len(frames), step)][:8]
    tile_w = top.shape[1] // max(1, len(context))
    tile_h = max(2, round(height * tile_w / STRIP_TILE))
    first_after = next((i for i, (moment, _) in enumerate(context) if moment >= pair[1][0]), None)
    tiles = []
    for i, (moment, frame) in enumerate(context):
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f'{moment - pair[1][0]:+.2f}s', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if i == first_after:
            tile[:, :3] = (0, 0, 255)  # the first context frame after the peak
        tiles.append(tile)
    bottom = np.hstack(tiles)
    bottom = np.pad(bottom, ((0, 0), (0, top.shape[1] - bottom.shape[1]), (0, 0)))
    ok, encoded = cv2.imencode('.jpg', np.vstack([top, np.full((4, top.shape[1], 3), 40, dtype=np.uint8), bottom]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('could not encode the strip')
    return encoded.tobytes()


def continuous(camera: dict) -> bool:
    """The camera fit says both frames are the same shot."""
    return (camera.get('move') not in (None, 'unclear') and camera.get('inliers', 0) >= CONTINUOUS['inliers']
            and camera.get('ratio', 0) >= CONTINUOUS['ratio'])


def verdict(candidate: dict, picture: dict | None) -> tuple[str, str]:
    """``(apply|grey|reject, why)`` for one peak."""
    score, alone = candidate['score'], candidate['isolation']
    if score < GREY['score'] or alone < GREY['isolation']:
        return 'reject', f'score {score:.2f}, {alone:.1f}x its neighbours: motion or noise, not a cut'
    if picture is None or picture.get('error'):
        return 'grey', (picture or {}).get('error') or 'no picture check'
    camera = picture['camera']
    if continuous(camera):
        return 'reject', f"the frames on both sides are the same shot ({camera['move']}, {camera['inliers']} matching points)"
    if picture['flash']:
        return 'grey', 'a flash or fade at the peak'
    if score >= APPLY['score'] and alone >= APPLY['isolation'] and picture['correlation'] < APPLY['correlation']:
        return 'apply', f'score {score:.2f}, {alone:.1f}x its neighbours, no camera fit across it'
    return 'grey', f'score {score:.2f}, {alone:.1f}x its neighbours, correlation {picture["correlation"]:.2f}'


def check_cut(proxy: Path, times, scores, start: float, end: float, size=None, skip=()) -> dict:
    """Everything the checker says about one cut: its peaks with features and verdicts, where to split, what to ask.

    ``skip`` holds times already asked about (a reviewer said no cut, or the question is open)."""
    peaks = [peak for peak in candidates(times, scores, start, end) if not any(abs(peak['time'] - moment) <= 0.06 for moment in skip)]
    for peak in peaks:
        if peak['score'] < GREY['score'] or peak['isolation'] < GREY['isolation']:
            peak['verdict'], peak['why'] = verdict(peak, None)
            continue
        index = peak['index']
        try:
            frames = frames_near(proxy, times[index], size=size)
            peak['picture'] = picture_features(frames, times[index - 1], times[index])
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            peak['picture'] = {'error': str(exc)[-300:]}
        peak['verdict'], peak['why'] = verdict(peak, peak['picture'])
    def edge_score(moment):
        nearby = [score for when, score in zip(times, scores) if abs(when - moment) <= 1.5 / 24]
        return max(nearby, default=0.0)

    split = []
    for peak in sorted((peak for peak in peaks if peak['verdict'] == 'apply'), key=lambda item: -item['score']):
        edge = start if peak['time'] - start < end - peak['time'] else end
        if abs(peak['time'] - edge) < SLIVER:
            # A piece this short is a flash cut, or the cut's own boundary is a few frames off the real one.
            peak['misplaced_edge'] = edge_score(edge) < FLOOR
            peak['verdict'], peak['why'] = 'grey', (
                f"it is {abs(peak['time'] - edge):.2f}s from the cut's boundary at {edge:.3f}s, where the picture does not change: "
                'that boundary is probably misplaced (move it here in the editor rather than split)' if peak['misplaced_edge']
                else f"it would leave a piece of {abs(peak['time'] - edge):.2f}s: a flash cut or a double detection")
            continue
        if all(abs(peak['time'] - other['time']) > EDGE for other in split):
            split.append(peak)
    grey = [peak for peak in peaks if peak['verdict'] == 'grey' and all(abs(peak['time'] - other['time']) > EDGE for other in split)]
    outcome = 'apply' if split else 'grey' if grey else 'clean'

    def slim(peak):
        return {key: value for key, value in peak.items() if key != 'index'} | {'frame': peak['index']}
    return {'verdict': outcome, 'split_at': sorted(peak['time'] for peak in split), 'grey': sorted(peak['time'] for peak in grey),
            'peaks': [slim(peak) for peak in peaks if peak['verdict'] != 'reject' or peak['score'] >= GREY['score']][:8], 'version': VERSION}


# --- the job: check, split, ask --------------------------------------------------------------------------------------

KIND = 'missed_cuts'          # agent_decisions.kind and the Autopilot profile's decision kind
ESCALATION = 'missed_cut'     # agent_escalations.kind, one per grey peak
TOLERANCE = 1e-6


def proxy_path(conn, episode_id) -> Path:
    from . import media
    episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    return media.episode_dir(episode['project_id'], episode_id) / 'proxy.mp4'


def sequenced_cuts(conn, episode_id) -> set[str]:
    """Cuts that belong to a sequence (the user's work: the agent never changes their boundaries)."""
    found = set()
    for row in conn.execute('SELECT first_cut_id,last_cut_id,cuts_json FROM sequences WHERE episode_id=?', (episode_id,)):
        found |= {row['first_cut_id'], row['last_cut_id']}
        try:
            found |= {item.get('id') if isinstance(item, dict) else item for item in json.loads(row['cuts_json'] or '[]')}
        except ValueError:
            pass
    found.discard(None)
    return found


def asked_times(conn, episode_id) -> list[float]:
    """Peaks already put to a reviewer (open, answered or dismissed): never asked again."""
    return sorted(row[0] for row in conn.execute("""SELECT json_extract(evidence_json,'$.time') FROM agent_escalations
        WHERE episode_id=? AND kind=? AND json_extract(evidence_json,'$.time') IS NOT NULL""", (episode_id, ESCALATION)))


def policy_for(conn, episode_id, run_id=None) -> dict:
    """``{mode, claude}`` from the run's profile snapshot, else the project's profile."""
    from . import autopilot_profile
    profile = None
    if run_id:
        row = conn.execute('SELECT policy_json,episode_id FROM agent_runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row['episode_id'] != episode_id:
            raise ValueError('The agent run belongs to another episode')
        profile = json.loads(row['policy_json'] or '{}') or None
    if not profile or KIND not in (profile.get('modes') or {}):
        project = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
        profile, _ = autopilot_profile.load(conn, project['project_id'])
    claude = profile.get('claude') or {}
    return {'mode': profile['modes'][KIND], 'claude': bool(claude.get('enabled', True))}


def queue(conn, episode_id, *, first_index=0, last_index=None, cut_ids=None, redo=False, run_id=None):
    """Queue a check of the episode's cuts (or some) inside the caller's transaction. Returns the job id, or None when the
    profile switched the check off."""
    policy = policy_for(conn, episode_id, run_id)
    if policy['mode'] == 'off':
        return None
    payload = {'policy': policy, 'first_index': first_index, 'last_index': last_index, 'redo': bool(redo), 'run_id': run_id}
    if cut_ids:
        payload['cut_ids'] = list(dict.fromkeys(cut_ids))
    waits = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts'))
    if waits:
        payload['depends_on'] = waits
    job_id = worker.insert_job(conn, episode_id, 'missed_cuts', payload, priority_class='batch')
    worker.sync_episode_status(conn, episode_id)
    return job_id


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    episode_id = job['episode_id']
    proxy = proxy_path(conn, episode_id)
    wanted = set(payload['cut_ids']) if payload.get('cut_ids') else None
    last = payload.get('last_index')
    cuts = [[row['id'], row['cut_index'], row['start_time'], row['end_time'], row['analysis_enabled']] for row in conn.execute(
        """SELECT id,cut_index,start_time,end_time,analysis_enabled FROM cuts WHERE episode_id=? AND active=1 AND cut_index>=?
           AND (? IS NULL OR cut_index<=?) ORDER BY cut_index""", (episode_id, payload.get('first_index') or 0, last, last))
        if wanted is None or row['id'] in wanted]
    return {'proxy': _identity(proxy) if proxy.is_file() else None, 'cuts': cuts,
            'sequenced': sorted(sequenced_cuts(conn, episode_id) & {cut[0] for cut in cuts})}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """Splits mark their own derivatives stale (``cut_previews.after_timeline_edit``)."""


def _covering(conn, episode_id, moment):
    return conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND start_time<? AND end_time>? ORDER BY cut_index LIMIT 1',
                        (episode_id, moment, moment)).fetchone()


def split_at(conn, episode_id, moment, *, tier, run_id=None, evidence=None, label_prefix='Autopilot split'):
    """Split the active cut covering ``moment`` exactly there, as a cut_history edit and an agent decision.

    Returns ``(decision_id, changed_ids)``, or ``(None, [])`` when no cut has the moment far enough from its edges."""
    cut = _covering(conn, episode_id, moment)
    if cut is None or not cut['start_time'] + EDGE < moment < cut['end_time'] - EDGE:
        return None, []
    before = cut_history.snapshot(conn, episode_id)
    for row in conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND cut_index>? ORDER BY cut_index DESC',
                            (episode_id, cut['cut_index'])).fetchall():
        conn.execute('UPDATE cuts SET cut_index=? WHERE id=?', (row['cut_index'] + 1, row['id']))
    new_id = str(uuid.uuid4())
    conn.execute('UPDATE cuts SET end_time=? WHERE id=?', (moment, cut['id']))
    conn.execute('INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,analysis_enabled,created_at) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                 (new_id, episode_id, cut['cut_index'] + 1, moment, cut['end_time'], cut['analysis_enabled']))
    number = cut['cut_index'] + 1
    cut_history.record(conn, episode_id, 'split', f"{label_prefix}: cut {number} at {cut_history.timecode(moment)} (a missed hard cut)",
                       cut['id'], before, {new_id})
    left = {'id': cut['id'], 'start_time': cut['start_time'], 'end_time': moment}
    right = {'id': new_id, 'start_time': moment, 'end_time': cut['end_time']}
    decision = agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=cut['id'], kind=KIND, tier=tier, target_table='cuts',
                                target_id=cut['id'], before={'id': cut['id'], 'start_time': cut['start_time'], 'end_time': cut['end_time']},
                                after={'left': left, 'right': right, 'created': new_id}, evidence=evidence or {},
                                summary=f"Split cut {number} at {cut_history.timecode(moment)}: a hard cut the detection missed")
    return decision, [cut['id'], new_id]


def _question(number, moment, peak):
    picture = peak.get('picture') or {}
    return (f"Is there a hard cut in cut {number} at {cut_history.timecode(moment)}? Scene score {peak['score']:.2f}, "
            f"{peak['isolation']:.1f}x its neighbours; {peak.get('why') or 'the checker is not sure'}. "
            f"The strip shows two frames before the red bar and two after it"
            + (f" (correlation {picture['correlation']:.2f})" if picture.get('correlation') is not None else '') + '.')


def escalate_peak(conn, episode_id, cut, peak, *, reviewer, run_id=None, propose_split=False, reason=''):
    moment = peak['time']
    number = cut['cut_index'] + 1
    evidence = {'cut_id': cut['id'], 'time': moment, 'score': peak['score'], 'isolation': peak['isolation'], 'why': peak.get('why'),
                'picture': peak.get('picture'), 'strip': f"/api/cuts/{cut['id']}/missed-cut-strip?time={moment:.6f}", 'reason': reason}
    return agent_log.escalate(conn, episode_id=episode_id, kind=ESCALATION, group_key=f'peak:{moment:.3f}',
                              question=_question(number, moment, peak) + (f' {reason}' if reason else ''), cut_ids=[cut['id']],
                              options=['Split here', 'No cut'], proposal={'answer': 'split'} if propose_split else None,
                              evidence=evidence, run_id=run_id, reviewer=reviewer)


def publish(conn, job, episode, data, results):
    """Save the checks, apply the splits, ask about the rest. Returns counts and the cuts whose previews change."""
    payload = json.loads(job['payload_json'])
    policy, run_id = payload.get('policy') or {'mode': 'apply', 'claude': True}, payload.get('run_id')
    sequenced = set(data['sequenced'])
    counts = {'checked': len(results), 'split': 0, 'asked_claude': 0, 'asked_person': 0}
    changed = set()
    proxy_key = data['proxy']
    for cut_id, start, end, result in results:
        conn.execute("""INSERT INTO missed_cut_checks(cut_id,episode_id,cut_start,cut_end,version,proxy_key,verdict,result_json,job_id)
            VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(cut_id) DO UPDATE SET cut_start=excluded.cut_start,cut_end=excluded.cut_end,
            version=excluded.version,proxy_key=excluded.proxy_key,verdict=excluded.verdict,result_json=excluded.result_json,
            job_id=excluded.job_id,checked_at=CURRENT_TIMESTAMP""",
                     (cut_id, episode['id'], start, end, VERSION, proxy_key, result['verdict'], json.dumps(result), job['id']))
        peaks = {peak['time']: peak for peak in result['peaks']}
        in_sequence = cut_id in sequenced
        for moment in result['split_at']:
            peak = peaks.get(moment) or {'time': moment, 'score': 0, 'isolation': 0}
            cut = _covering(conn, episode['id'], moment)
            if cut is None:
                continue
            if in_sequence or policy['mode'] == 'ask':
                reason = 'This cut belongs to a sequence, so a person decides.' if in_sequence else 'The Autopilot profile asks before splitting.'
                escalate_peak(conn, episode['id'], cut, peak, reviewer='human', run_id=run_id, propose_split=True, reason=reason)
                counts['asked_person'] += 1
                continue
            evidence = {key: peak.get(key) for key in ('time', 'score', 'isolation', 'why', 'picture')}
            decision, ids = split_at(conn, episode['id'], moment, tier='rules', run_id=run_id, evidence=evidence)
            if decision:
                changed.update(ids)
                counts['split'] += 1
        for moment in result['grey']:
            cut = _covering(conn, episode['id'], moment)
            if cut is None:
                continue
            misplaced = (peaks.get(moment) or {}).get('misplaced_edge')
            claude = policy.get('claude') and not in_sequence and not misplaced
            escalate_peak(conn, episode['id'], cut, peaks.get(moment) or {'time': moment, 'score': 0, 'isolation': 0},
                          reviewer='claude' if claude else 'human', run_id=run_id,
                          reason='' if not in_sequence else 'This cut belongs to a sequence, so a person decides.')
            counts['asked_claude' if claude else 'asked_person'] += 1
    return counts, changed


def process(job_id, owner):
    from . import cut_previews
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    with db.connect() as conn:
        conn.execute('BEGIN')
        snapshot = job_inputs.JobInputs(conn, job, owner)
        episode = dict(conn.execute('SELECT * FROM episodes WHERE id=?', (job['episode_id'],)).fetchone())
        proxy = proxy_path(conn, job['episode_id'])
        checked = {row['cut_id']: row for row in conn.execute('SELECT * FROM missed_cut_checks WHERE episode_id=?', (job['episode_id'],))}
        skip = asked_times(conn, job['episode_id'])
    data = snapshot.data
    if not data['proxy']:
        raise RuntimeError('Preview video is missing; run Prepare before checking for missed cuts')
    worker.stage(job_id, owner, job['episode_id'], 'Scoring frames for missed cuts')
    times, scores = scene_scores(proxy)
    size = _dimensions(proxy)
    results, todo = [], []
    for cut_id, index, start, end, enabled in data['cuts']:
        if not enabled:
            continue
        row = checked.get(cut_id)
        if (not payload.get('redo') and row and row['version'] == VERSION and row['proxy_key'] == data['proxy']
                and abs(row['cut_start'] - start) <= TOLERANCE and abs(row['cut_end'] - end) <= TOLERANCE):
            continue
        todo.append((cut_id, index, start, end))
    for position, (cut_id, index, start, end) in enumerate(todo):
        if position % 25 == 0:
            worker.stage(job_id, owner, job['episode_id'], f'Checking cuts for missed hard cuts ({position}/{len(todo)})')
        results.append((cut_id, start, end, check_cut(proxy, times, scores, start, end, size=size, skip=skip)))
    with snapshot.publish(invalidate=False) as conn:
        if cut_history.busy(conn, job['episode_id']):
            raise RuntimeError('Media preparation or full hard-cut detection is pending; check again after it')
        counts, changed = publish(conn, job, episode, data, results)
        cancelled = cut_previews.after_timeline_edit(conn, episode, changed) if changed else []
        stage = (f"Complete · {counts['split']} split, {counts['asked_claude']} for Claude, {counts['asked_person']} for a person"
                 f" · {len(results)} checked, {len(data['cuts']) - len(results)} unchanged or off")
        payload['result'] = counts
        conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,payload_json=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (stage, json.dumps(payload), job_id))
        worker.sync_episode_status(conn, job['episode_id'])
    worker.after_cancel(cancelled)
    worker.wake_event.set()


# --- answers and undo ------------------------------------------------------------------------------------------------

ANSWERS = ('split', 'no_cut', 'unsure')


class Refused(Exception):
    """The answer cannot be applied (the message says why)."""


def answer(conn, escalation_id, reply, tier, run_id=None, note=None):
    """Apply Claude's or the person's answer to one missed-cut question. Returns ``{status, decision_id?, changed}``."""
    if reply not in ANSWERS:
        raise Refused(f"answer must be one of {', '.join(ANSWERS)}")
    if tier not in ('claude', 'human'):
        raise Refused('tier must be claude or human')
    row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
    if row is None:
        raise KeyError(escalation_id)
    if row['kind'] != ESCALATION:
        raise Refused('This is not a missed-cut question')
    if row['status'] != 'open':
        raise Refused('This question is already closed')
    if tier == 'claude' and row['reviewer'] != 'claude':
        raise Refused('This question waits for a person')
    evidence = json.loads(row['evidence_json'] or '{}')
    episode_id, moment = row['episode_id'], float(evidence['time'])
    resolution = {'answer': reply, 'by': tier, **({'note': str(note)[:1000]} if note else {})}
    if reply == 'unsure':
        if tier == 'claude':
            agent_log.hand_to_person(conn, escalation_id, note)
            return {'status': 'handed_to_person', 'changed': []}
        return {'status': 'open', 'changed': []}
    if reply == 'no_cut':
        agent_log.close_escalation(conn, escalation_id, 'resolved', resolution)
        return {'status': 'resolved', 'changed': []}
    cut = _covering(conn, episode_id, moment)
    if cut is None or not cut['start_time'] + EDGE < moment < cut['end_time'] - EDGE:
        agent_log.close_escalation(conn, escalation_id, 'resolved', {**resolution, 'applied': 'already a boundary'})
        return {'status': 'resolved', 'changed': []}
    if tier == 'claude' and cut['id'] in sequenced_cuts(conn, episode_id):
        raise Refused('This cut belongs to a sequence: a person decides')
    if cut_history.busy(conn, episode_id):
        raise Refused('Wait for media preparation or full hard-cut detection to finish')
    decision, changed = split_at(conn, episode_id, moment, tier=tier, run_id=run_id or row['run_id'],
                                 evidence={key: evidence.get(key) for key in ('time', 'score', 'isolation', 'why', 'picture')} | {'escalation_id': escalation_id},
                                 label_prefix='Split after review')
    agent_log.close_escalation(conn, escalation_id, 'resolved', {**resolution, 'decision_id': decision})
    return {'status': 'resolved', 'decision_id': decision, 'changed': changed}


def undo_split(conn, decision):
    """Merge the two pieces of an agent split back, while both are exactly as the split left them."""
    from . import cut_previews
    after = json.loads(decision['after_json'] or '{}')
    left, right = after.get('left'), after.get('right')
    if not left or not right:
        raise agent_log.UndoConflict('This decision did not split a cut')
    episode_id = decision['episode_id']
    if cut_history.busy(conn, episode_id):
        raise agent_log.UndoConflict('Wait for media preparation or full hard-cut detection to finish')
    rows = {row['id']: row for row in conn.execute('SELECT * FROM cuts WHERE id IN (?,?)', (left['id'], right['id']))}
    for piece in (left, right):
        row = rows.get(piece['id'])
        if (row is None or not row['active'] or abs(row['start_time'] - piece['start_time']) > TOLERANCE
                or abs(row['end_time'] - piece['end_time']) > TOLERANCE):
            raise agent_log.UndoConflict('The cuts changed since the split; undo would overwrite that')
    if rows[right['id']]['cut_index'] != rows[left['id']]['cut_index'] + 1:
        raise agent_log.UndoConflict('The two pieces are no longer next to each other')
    before = cut_history.snapshot(conn, episode_id)
    removed_index = rows[right['id']]['cut_index']
    conn.execute('UPDATE cuts SET end_time=? WHERE id=?', (right['end_time'], left['id']))
    if right['id'] == after.get('created') and not cut_history.has_history(conn, right['id']):
        conn.execute('DELETE FROM cuts WHERE id=?', (right['id'],))
    else:
        lowest = conn.execute('SELECT MIN(cut_index) FROM cuts WHERE episode_id=?', (episode_id,)).fetchone()[0]
        conn.execute('UPDATE cuts SET active=0,cut_index=? WHERE id=?', (min(-1, lowest - 1), right['id']))
    for row in conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND cut_index>? ORDER BY cut_index',
                            (episode_id, removed_index)).fetchall():
        conn.execute('UPDATE cuts SET cut_index=? WHERE id=?', (row['cut_index'] - 1, row['id']))
    number = rows[left['id']]['cut_index'] + 1
    cut_history.record(conn, episode_id, 'merge', f'Undid the autopilot split of cut {number}', left['id'], before)
    episode = conn.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    cut_previews.after_timeline_edit(conn, episode, {left['id']}, {right['id']})


agent_log.UNDO_KINDS[KIND] = undo_split
