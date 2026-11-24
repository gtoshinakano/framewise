"""Run Multiple: server-side runs of an episode's batch stages (docs/autopilot/HARNESS.md D37).

A run is one ``batch_runs`` row for one stage, started by the person from the Run Multiple dialog or ``scripts/fw``:

* **prepare**, once per episode: media, hard cuts, the missed-cut check, transcription and speakers, the burned-in subtitles
  (optional, off unless asked for), line splits, voices, first-frame boxes and recognition, target frames, the cast
  resolver. Only this run's voice resolver may open questions, so an unknown voice is asked once.
* **prepare_cuts**: boxes, target frames and the cast resolver for the cuts whose boundaries or first-frame boxes changed
  after Prepare (``stage_targets`` ``prepare_cuts``).
* **evaluate**, after Prepare: visual analysis, performance beats and the consistency review, each on its target cuts
  (``stage_targets`` ``evaluate``: missing or stale, inputs settled, no open inbox item). Cuts that wait for the person are
  left out and listed, so a partial evaluation always runs on the valid cuts only.

Steps run one after another. A step's ``plan`` reads the database when the step starts (``will_run`` with its cuts, or
``done``/``off``/``skipped`` with a reason); ``start`` queues its jobs through the same helpers the editor's buttons use
(identical requests reuse the active job, so a repeated start is harmless); the step is over when its jobs finished and no
job of its ``waits`` pipelines is queued or running for the episode (transcription queues speaker attribution, a visual job
queues beats). Failed jobs that ``triage`` has a fix rule for are retried once. ``stats`` then reads what the step did and
found, with a one-line ``summary``. A failure of media, hard cuts or transcription ends the run (``failed``: Prepare can be
started again and skips what is done); other failures are listed and the run goes on.

``tick`` advances every running run; the API process calls it every few seconds and right after a run starts, so a run
survives a closed tab and an API restart. ``preview`` shows the plans before anything runs, from the same functions, so the
dialog says exactly what will run. Nothing here is an approval status.
"""
import json
import logging
import threading
import time

from . import agent_log, autopilot_profile, cut_readiness, db, media, runs, stage_targets, worker

log = logging.getLogger(__name__)

STAGES = ('prepare', 'prepare_cuts', 'evaluate')
AGENT = 'run-multiple'
DONE_STATES = ('done', 'skipped', 'failed', 'cancelled')   # a step that is over
REQUIRED = {'media': 'prepare', 'hardcuts': 'hardcuts', 'transcribe': 'transcribe'}   # step -> the job whose failure ends the run
VISUAL_CHUNK = 16   # cuts per visual job: beats of a finished chunk start while the next chunk runs; a failure costs one chunk
FIXABLE = ('requeue', 'rerun', 'dismiss', 'prerequisite')
MAX_MOVES = 24
TICK_SECONDS = 5
LISTED = 12
_LOCK = threading.Lock()


class Refused(Exception):
    """The run cannot start or stop now (the message says why)."""


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())  # UTC, like SQLite's CURRENT_TIMESTAMP


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _numbers_text(numbers):
    numbers = sorted(numbers)
    shown = ', '.join(str(number) for number in numbers[:LISTED])
    return shown + (f' and {len(numbers) - LISTED} more' if len(numbers) > LISTED else '')


def _cuts_text(numbers):
    """"cut 4" or "cuts 2, 5 and 3 more"."""
    return f"cut{'s' if len(set(numbers)) != 1 else ''} {_numbers_text(numbers)}"


def _plural(count, word, plural=None):
    return f"{count} {word if count == 1 else plural or word + 's'}"


def _clock(seconds):
    minutes, rest = divmod(int(round(seconds or 0)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{rest:02d}' if hours else f'{minutes}:{rest:02d}'


# --- plans ------------------------------------------------------------------------------------------------------------

def will_run(detail, cuts=None):
    return {'state': 'will_run', 'detail': detail, **({'cuts': cuts} if cuts is not None else {})}


def settled(state, detail):
    return {'state': state, 'detail': detail}


def _cut_items(cuts):
    return [{'cut': cut['cut_index'] + 1, 'cut_id': cut['id']} for cut in cuts]


def _stage_cuts(conn, ctx):
    options = ctx['options']
    return stage_targets.cut_rows(conn, ctx['episode_id'], options.get('from_cut'), options.get('to_cut'))


def _job_done(conn, episode_id, pipeline):
    return bool(conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline=? AND status='done' LIMIT 1", (episode_id, pipeline)).fetchone())


def _mode(ctx, kind):
    modes = ctx['profile'].get('modes') or {}
    return modes.get(kind, 'off' if kind in autopilot_profile.OFF_BY_DEFAULT else 'apply')


def _full_hardcuts(conn, episode_id):
    return [row['id'] for row in conn.execute("""SELECT id FROM jobs WHERE episode_id=? AND pipeline='hardcuts' AND status IN ('queued','running')
        AND json_extract(payload_json,'$.cut_id') IS NULL ORDER BY rowid""", (episode_id,))]


def _full_transcription(conn, episode_id, statuses):
    marks = ','.join('?' for _ in statuses)
    return [row['id'] for row in conn.execute(f"""SELECT id FROM jobs WHERE episode_id=? AND pipeline='transcribe' AND status IN ({marks})
        AND json_extract(payload_json,'$.range') IS NULL ORDER BY rowid""", (episode_id, *statuses))]


def plan_media(conn, ctx):
    episode = ctx['episode']
    if worker.pending_jobs(conn, ctx['episode_id'], ('prepare',)):
        return will_run('preparing the video now')
    if episode['duration'] and (media.episode_dir(episode['project_id'], episode['id']) / 'proxy.mp4').is_file():
        return settled('done', 'the video is prepared')
    return will_run('the preview video, the audio track and the waveform')


def plan_hardcuts(conn, ctx):
    if _full_hardcuts(conn, ctx['episode_id']):
        return will_run('detecting the hard cuts now')
    count = conn.execute('SELECT COUNT(*) FROM cuts WHERE episode_id=? AND active=1', (ctx['episode_id'],)).fetchone()[0]
    if count:
        return settled('done', f'{_plural(count, "cut")} already detected; detection runs once')
    return will_run('finds the hard cuts of the whole episode, once')


def _resolver_plan(conn, ctx, pipeline, kind, doing):
    if _mode(ctx, kind) == 'off':
        return settled('off', 'off in the Autopilot profile')
    if worker.pending_jobs(conn, ctx['episode_id'], (pipeline,)):
        return will_run(f'{doing} (already queued)')
    if _job_done(conn, ctx['episode_id'], pipeline):
        return settled('done', 'ran once already; it never runs again')
    return will_run(doing)


def plan_missed_cuts(conn, ctx):
    return _resolver_plan(conn, ctx, 'missed_cuts', 'missed_cuts',
                          'splits the hard cuts the detection missed where it is sure; doubtful places are only counted')


def plan_transcribe(conn, ctx):
    if ctx['episode']['has_audio'] == 0:
        return settled('skipped', 'the video has no audio')
    if _full_transcription(conn, ctx['episode_id'], ('queued', 'running')):
        return will_run('transcribing now; speaker attribution follows')
    if _full_transcription(conn, ctx['episode_id'], ('done',)):
        return settled('done', 'transcribed once already; it never runs again')
    return will_run('lines, voices and timings of the whole episode; speaker attribution follows')


def plan_subtitles(conn, ctx):
    from . import subtitle_store
    language = (ctx['options'].get('subtitles') or {}).get('language')
    if not language:
        return settled('off', 'off: burned-in subtitles are optional and nothing waits for them')
    run = subtitle_store.current_run(ctx['episode_id'], conn)
    if run and run['language'] == language:
        return settled('done', f'{language} subtitles already read')
    return will_run(f'reads the burned-in {language} subtitles (supporting material only)')


def plan_line_splits(conn, ctx):
    return _resolver_plan(conn, ctx, 'line_splits', 'line_splits', 'splits lines that run across a hard cut at a pause; never asks')


def plan_voices(conn, ctx):
    return _resolver_plan(conn, ctx, 'voice_resolver', 'voices',
                          'gives unknown voices to characters when two signals agree; asks you once about the rest')


def plan_boxes(conn, ctx):
    cuts = _stage_cuts(conn, ctx)
    if not cuts:
        return will_run('every analysed cut, once the cuts are detected')
    missing = [cut for cut in cuts if cut['analysis_enabled'] and not cut['no_characters'] and cut['detection_phase'] != 'complete']
    if not missing:
        return settled('done' if ctx['stage'] == 'prepare' else 'skipped',
                       'every cut has its boxes' if ctx['stage'] == 'prepare' else 'no cut changed since Prepare')
    return will_run(f'first-frame boxes and recognition on {_plural(len(missing), "cut")}', _cut_items(missing))


def plan_target_frames(conn, ctx):
    cuts = _stage_cuts(conn, ctx)
    if not cuts:
        return will_run('every cut with people, after the boxes')
    found = stage_targets.target_frames(conn, cuts)
    boxless = [cut for cut in cuts if cut['analysis_enabled'] and not cut['no_characters'] and cut['detection_phase'] != 'complete']
    if not found and boxless and not ctx['agent_run_id']:  # a preview: the boxes step comes first
        return will_run(f'every cut with people, after the boxes ({_plural(len(boxless), "cut")} have none yet)')
    if not found:
        return settled('done' if ctx['stage'] == 'prepare' else 'skipped',
                       'every cut has current target frames' if ctx['stage'] == 'prepare' else 'no cut needs its target frames again')
    return will_run(f'people on the target frames of {_plural(len(found), "cut")}', [{'cut': item['cut'], 'cut_id': item['cut_id']} for item in found])


def plan_cast(conn, ctx):
    if _mode(ctx, 'box_names') == 'off' and _mode(ctx, 'target_frames') == 'off':
        return settled('off', 'off in the Autopilot profile')
    doing = 'names boxes from the gallery, groups unknown people, settles flagged target frames'
    if ctx['stage'] == 'prepare_cuts':
        earlier = [step for step in ctx['steps'] if step['step'] in ('boxes', 'target_frames')]
        if not any(step.get('state') in ('will_run', 'starting', 'running', 'done') and step.get('cuts') for step in earlier):
            return settled('skipped', 'no cut changed since Prepare')
        return will_run(doing)
    return _resolver_plan(conn, ctx, 'character_decisions', 'box_names', doing)


def _evaluate_targets(conn, ctx):
    """``stage_targets`` evaluate section, read once per context (a preview plans three steps from it)."""
    if '_evaluate' not in ctx:
        options = ctx['options']
        ctx['_evaluate'] = stage_targets.targets(conn, ctx['episode_id'], 'evaluate', options.get('from_cut'), options.get('to_cut'))['evaluate']
    return ctx['_evaluate']


def _evaluate_plan(step):
    def plan(conn, ctx):
        only = ctx['options'].get('only')
        if only and only != step:
            return settled('skipped', f'only {only} was asked for')
        found = _evaluate_targets(conn, ctx)[step]
        cuts = [{'cut': item['cut'], 'cut_id': item['cut_id']} for item in found['targets']]
        held = found['held']
        waiting = f'; {_plural(len(held), "cut")} wait for you' if held else ''
        if not cuts:
            return {**settled('skipped', f'no cut needs it now{waiting}'), 'held': held}
        stale = sum(item['why'] == 'stale' for item in found['targets'])
        return {**will_run(f"{_plural(len(cuts), 'cut')}" + (f' ({stale} out of date)' if stale else '') + waiting, cuts), 'held': held}
    return plan


# --- starting a step ------------------------------------------------------------------------------------------------

def _write():
    """A write transaction of its own (the queue helpers that take a connection)."""
    class Write:
        def __enter__(self):
            self.context = db.connect()
            self.conn = self.context.__enter__()
            self.conn.execute('BEGIN IMMEDIATE')
            return self.conn

        def __exit__(self, *exc):
            return self.context.__exit__(*exc)
    return Write()


def start_media(ctx, step):
    with _write() as conn:
        return worker.pending_jobs(conn, ctx['episode_id'], ('prepare',)) or [worker.enqueue_in(conn, ctx['episode_id'], 'prepare')]


def start_hardcuts(ctx, step):
    with _write() as conn:
        pending = _full_hardcuts(conn, ctx['episode_id'])
        if pending:
            return pending
        if conn.execute('SELECT 1 FROM cuts WHERE episode_id=? AND active=1 LIMIT 1', (ctx['episode_id'],)).fetchone():
            return []
        return [worker.enqueue_in(conn, ctx['episode_id'], 'hardcuts')]


def start_missed_cuts(ctx, step):
    from . import missed_cuts
    with _write() as conn:
        job_id = missed_cuts.queue(conn, ctx['episode_id'], run_id=ctx['agent_run_id'])
    return [job_id] if job_id else []


def start_transcribe(ctx, step):
    with _write() as conn:
        pending = _full_transcription(conn, ctx['episode_id'], ('queued', 'running'))
        return pending or [worker.enqueue_in(conn, ctx['episode_id'], 'transcribe')]


def start_subtitles(ctx, step):
    from . import main
    with _write() as conn:
        return [main.queue_subtitles(conn, ctx['episode_id'], ctx['options']['subtitles']['language'])]


def start_line_splits(ctx, step):
    from . import line_splits
    with _write() as conn:
        job_id = line_splits.queue(conn, ctx['episode_id'], run_id=ctx['agent_run_id'])
    return [job_id] if job_id else []


def start_voices(ctx, step):
    from . import voice_resolver
    with _write() as conn:
        # Prepare's run is the only one that may ask: an unknown voice is asked once (HARNESS.md D37)
        job_id = voice_resolver.queue(conn, ctx['episode_id'], run_id=ctx['agent_run_id'], ask=ctx['stage'] == 'prepare')
    return [job_id] if job_id else []


def start_boxes(ctx, step):
    from . import anime_bulk
    with _write() as conn:
        job_ids = anime_bulk.enqueue(conn, ctx['episode_id'], 'full')
        if job_ids:
            worker.sync_episode_status(conn, ctx['episode_id'])
    return job_ids


def start_target_frames(ctx, step):
    from .target_characters_api import queue_targets
    ids = [item['cut_id'] for item in step.get('cuts') or []]
    if not ids:
        return []
    with _write() as conn:
        marks = ','.join('?' for _ in ids)
        cuts = [dict(row) for row in conn.execute(f"""SELECT id,cut_index,analysis_enabled,no_characters FROM cuts WHERE id IN ({marks}) AND active=1
            ORDER BY cut_index""", ids)]
        result = queue_targets(conn, ctx['episode_id'], cuts, only_missing=True)
        if result['job_ids']:
            worker.sync_episode_status(conn, ctx['episode_id'])
    return result['job_ids']


def start_cast(ctx, step):
    from . import character_decisions
    with _write() as conn:
        job_id = character_decisions.queue(conn, ctx['episode_id'], run_id=ctx['agent_run_id'])
    return [job_id] if job_id else []


def runs_of(numbers):
    """[(first, last)] of consecutive cut numbers."""
    spans = []
    for number in sorted(set(numbers)):
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return [tuple(span) for span in spans]


def start_visual(ctx, step):
    from . import main
    beats = 'off' if ctx['options'].get('only') == 'visual' else 'when_ready'
    ids = []
    for first, last in runs_of(item['cut'] for item in step.get('cuts') or []):
        for start in range(first, last + 1, VISUAL_CHUNK):
            # Only the stages that are missing or stale run; saving a cut's keyframes queues its beats only when the cut is
            # ready (when_ready), so beats overlap the later chunks.
            result = main.queue_episode_visual_analysis(ctx['episode_id'], from_cut=start, to_cut=min(last, start + VISUAL_CHUNK - 1),
                                                        only_missing=True, beats=beats)
            ids += result['job_ids']
    return ids


def _per_cut(queue):
    def start(ctx, step):
        from fastapi import HTTPException
        ids = []
        for item in step.get('cuts') or []:
            try:
                result = queue(item['cut_id'])
            except HTTPException as exc:
                step.setdefault('errors', []).append({'cut': item['cut'], 'cut_id': item['cut_id'], 'error': str(exc.detail)})
                continue
            ids.append(result['job_id'])
        return ids
    return start


def start_beats(ctx, step):
    from . import main
    return _per_cut(main.queue_performance_beats)(ctx, step)


def start_consistency(ctx, step):
    from . import main
    return _per_cut(main.queue_consistency_summary)(ctx, step)


# --- what a step did and found -----------------------------------------------------------------------------------------

def _result(jobs, pipeline):
    """The result counts of the newest finished job of ``pipeline`` among the step's jobs."""
    for job in sorted(jobs, key=lambda item: item['rowid'], reverse=True):
        if job['pipeline'] == pipeline and job['status'] == 'done':
            return (_loads(job['payload_json'], {}).get('result') or {})
    return {}


def _numbers(conn, episode_id):
    return {row['id']: row['cut_index'] + 1 for row in conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1', (episode_id,))}


def _cut_rows(conn, items):
    ids = [item['cut_id'] for item in items]
    if not ids:
        return []
    marks = ','.join('?' for _ in ids)
    return [dict(row) for row in conn.execute(f"""SELECT id,episode_id,cut_index,start_time,end_time,no_characters FROM cuts
        WHERE id IN ({marks}) AND active=1 ORDER BY cut_index""", ids)]


def stats_media(conn, ctx, step, jobs):
    episode = dict(conn.execute('SELECT duration,width,height,has_audio FROM episodes WHERE id=?', (ctx['episode_id'],)).fetchone())
    size = f", {episode['width']}×{episode['height']}" if episode['width'] else ''
    audio = 'with audio' if episode['has_audio'] else 'without audio'
    return episode, f"Video prepared: {_clock(episode['duration'])}{size}, {audio}"


def stats_hardcuts(conn, ctx, step, jobs):
    count = conn.execute('SELECT COUNT(*) FROM cuts WHERE episode_id=? AND active=1', (ctx['episode_id'],)).fetchone()[0]
    return {'cuts': count}, f'{_plural(count, "cut")} detected'


def stats_missed_cuts(conn, ctx, step, jobs):
    result = _result(jobs, 'missed_cuts')
    numbers = _numbers(conn, ctx['episode_id'])
    split = sorted(numbers[cut_id] for cut_id in result.get('split_cuts') or [] if cut_id in numbers)
    stats = {'checked': result.get('checked', 0), 'split': result.get('split', 0), 'doubtful': result.get('doubtful', 0),
             'in_sequence': result.get('in_sequence', 0), 'cuts': split}
    text = f"Checked {_plural(stats['checked'], 'cut')}: split {stats['split']}" + (f" ({_cuts_text(split)})" if split else '')
    if stats['doubtful']:
        text += f"; {_plural(stats['doubtful'], 'doubtful place')} left as it is"
    return stats, text


def stats_transcribe(conn, ctx, step, jobs):
    episode_id = ctx['episode_id']
    row = conn.execute("""SELECT COUNT(*) AS lines, COUNT(DISTINCT speaker_label) AS voices, SUM(character_context_id IS NOT NULL) AS named
        FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0""", (episode_id,)).fetchone()
    ranges = conn.execute('SELECT kind FROM episode_audio_ranges WHERE episode_id=? AND suppressed=0', (episode_id,)).fetchall()
    stats = {'lines': row['lines'] or 0, 'voices': row['voices'] or 0, 'with_speaker': row['named'] or 0,
             'excluded': sorted({item['kind'] for item in ranges})}
    text = f"{_plural(stats['lines'], 'line')}, {_plural(stats['voices'], 'voice')}; {stats['with_speaker']} with a speaker"
    if stats['excluded']:
        text += f"; {', '.join(stats['excluded'])} left out"
    return stats, text


def stats_subtitles(conn, ctx, step, jobs):
    from . import subtitle_store
    run = subtitle_store.current_run(ctx['episode_id'], conn)
    if not run:
        return {'lines': 0}, 'No subtitles were read'
    lines = conn.execute('SELECT COUNT(*) FROM subtitle_lines WHERE run_id=?', (run['id'],)).fetchone()[0]
    unread = ((run.get('diagnostics') or {}).get('unread')) or 0
    return {'lines': lines, 'unread': unread}, f"{_plural(lines, 'subtitle line')} read" + (f'; {unread} images unread' if unread else '')


def stats_line_splits(conn, ctx, step, jobs):
    result = _result(jobs, 'line_splits')
    stats = {key: result.get(key, 0) for key in ('split', 'trim', 'leave', 'in_sequence', 'by_person')}
    text = f"Split {_plural(stats['split'], 'line')}, trimmed {stats['trim']}; {stats['leave']} left across a cut (no pause near it)"
    return stats, text


def stats_voices(conn, ctx, step, jobs):
    result = _result(jobs, 'voice_resolver')
    stats = {'lines_given': result.get('lines_applied', 0), 'asked_you': result.get('asked_human', 0) + result.get('asked_person', 0),
             'asked_claude': result.get('asked_claude', 0), 'left_unasked': result.get('left_unasked', 0)}
    text = f"Gave a speaker to {_plural(stats['lines_given'], 'line')}"
    asked = [part for part in (f"{_plural(stats['asked_you'], 'voice')} for you" if stats['asked_you'] else '',
                               f"{stats['asked_claude']} for Claude" if stats['asked_claude'] else '') if part]
    if asked:
        text += f"; asked about {' and '.join(asked)} in the Inbox"
    if stats['left_unasked']:
        text += f"; {_plural(stats['left_unasked'], 'voice')} left as they are"
    return stats, text


def stats_boxes(conn, ctx, step, jobs):
    cuts = _cut_rows(conn, step.get('cuts') or [])
    detected, boxes, named, unnamed, without = 0, 0, 0, 0, []
    for cut in cuts:
        run = conn.execute("""SELECT id,phase FROM character_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=?
            ORDER BY rowid DESC LIMIT 1""", (cut['id'], cut['start_time'], cut['end_time'])).fetchone()
        if not run or run['phase'] != 'complete':
            without.append(cut['cut_index'] + 1)
            continue
        detected += 1
        for row in conn.execute("SELECT character_context_id,label_source FROM character_detections WHERE run_id=? AND position='first'", (run['id'],)):
            boxes += 1
            if row['character_context_id']:
                named += 1
            elif row['label_source'] != 'manual':
                unnamed += 1
    stats = {'cuts': detected, 'boxes': boxes, 'named': named, 'unnamed': unnamed, 'not_detected': without}
    text = f"{_plural(boxes, 'box', 'boxes')} on {_plural(detected, 'cut')}: {named} named" + (f', {unnamed} without a name' if unnamed else '')
    if without:
        text += f"; not detected on {_cuts_text(without)}"
    return stats, text


def stats_target_frames(conn, ctx, step, jobs):
    cuts = _cut_rows(conn, step.get('cuts') or [])
    done, flagged, missing = 0, [], []
    for cut in cuts:
        if not conn.execute('SELECT 1 FROM target_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=? LIMIT 1',
                            (cut['id'], cut['start_time'], cut['end_time'])).fetchone():
            missing.append(cut['cut_index'] + 1)
            continue
        done += 1
        if cut_readiness.target_review(conn, cut):
            flagged.append(cut['cut_index'] + 1)
    stats = {'cuts': done, 'to_check': flagged, 'not_detected': missing}
    text = f"Target frames on {_plural(done, 'cut')}" + (f"; {len(flagged)} to check ({_cuts_text(flagged)})" if flagged else '')
    if missing:
        text += f"; not detected on {_cuts_text(missing)}"
    return stats, text


def stats_cast(conn, ctx, step, jobs):
    result = _result(jobs, 'character_decisions')
    stats = {'named': result.get('boxes_name', 0), 'unboxed': result.get('boxes_remove', 0) + result.get('boxes_background', 0),
             'frames_settled': result.get('frames_reviewed', 0),
             'asked': result.get('asked_labels', 0) + result.get('asked_people', 0) + result.get('asked_target_cuts', 0)}
    text = (f"Named {_plural(stats['named'], 'box', 'boxes')}, unboxed {stats['unboxed']}, settled {_plural(stats['frames_settled'], 'target frame')}"
            + (f"; {stats['asked']} questions in the Inbox" if stats['asked'] else ''))
    return stats, text


def _evaluate_stats(check, noun):
    def stats(conn, ctx, step, jobs):
        cuts = _cut_rows(conn, step.get('cuts') or [])
        done, failed, extra = [], [], {}
        for cut in cuts:
            ok, info = check(conn, cut)
            (done if ok else failed).append(cut['cut_index'] + 1)
            for key, value in (info or {}).items():
                extra.setdefault(key, []).append(value)
        result = {'done': len(done), 'not_done': failed}
        text = f"{noun} for {_plural(len(done), 'cut')}"
        if failed:
            text += f"; not done for {_cuts_text(failed)}"
        for key, values in extra.items():
            result[key] = values
        if extra.get('questions_for_you'):
            text += f"; {_plural(len(extra['questions_for_you']), 'cut')} with questions for you (Results tab)"
        return result, text
    return stats


def _visual_done(conn, cut):
    reasons, keyframes, video = cut_readiness._visual(conn, cut)
    info = {'beats_written': cut['cut_index'] + 1} if keyframes and not cut_readiness._beats(conn, cut, keyframes, video) else None
    return not reasons, info


def _beats_done(conn, cut):
    _, keyframes, video = cut_readiness._visual(conn, cut)
    return bool(keyframes) and not cut_readiness._beats(conn, cut, keyframes, video), None


def _consistency_done(conn, cut):
    from . import review_decisions
    row = conn.execute("""SELECT content,stale_reason FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
        AND json_extract(settings_json,'$.structured_type')='consistency_summary' ORDER BY rowid DESC LIMIT 1""", (cut['id'],)).fetchone()
    if not row or row['stale_reason']:
        return False, None
    questions = review_decisions.asked([item for item in _loads(row['content'], {}).get('questions') or [] if isinstance(item, dict)])
    return True, ({'questions_for_you': cut['cut_index'] + 1} if questions else None)


# --- the steps --------------------------------------------------------------------------------------------------------

class Step:
    def __init__(self, name, label, plan, start, waits, stats):
        self.name, self.label, self.plan, self.start, self.waits, self.stats = name, label, plan, start, waits, stats


STEPS = {step.name: step for step in (
    Step('media', 'Prepare the video', plan_media, start_media, ('prepare',), stats_media),
    Step('hardcuts', 'Detect hard cuts', plan_hardcuts, start_hardcuts, ('hardcuts',), stats_hardcuts),
    Step('missed_cuts', 'Check for missed hard cuts', plan_missed_cuts, start_missed_cuts, ('missed_cuts',), stats_missed_cuts),
    Step('transcribe', 'Transcribe and attribute speakers', plan_transcribe, start_transcribe, ('transcribe', 'speaker_attribution'), stats_transcribe),
    Step('subtitles', 'Read burned-in subtitles', plan_subtitles, start_subtitles, ('subtitles',), stats_subtitles),
    Step('line_splits', 'Split lines at hard cuts', plan_line_splits, start_line_splits, ('line_splits',), stats_line_splits),
    Step('voices', 'Give voices to characters', plan_voices, start_voices, ('voice_resolver', 'speaker_attribution'), stats_voices),
    Step('boxes', 'Box and recognize characters', plan_boxes, start_boxes, ('character_detect',), stats_boxes),
    Step('target_frames', 'Find people on the target frames', plan_target_frames, start_target_frames, ('character_detect', 'keyframes'), stats_target_frames),
    Step('cast', 'Settle names and target frames', plan_cast, start_cast, ('character_decisions',), stats_cast),
    Step('visual', 'Visual analysis', _evaluate_plan('visual'), start_visual, ('analyze', 'keyframes', 'performance_beats'),
         _evaluate_stats(_visual_done, 'First frame and keyframes analysed')),
    Step('beats', 'Performance beats', _evaluate_plan('beats'), start_beats, ('performance_beats',),
         _evaluate_stats(_beats_done, 'Beats written')),
    Step('consistency', 'Consistency review', _evaluate_plan('consistency'), start_consistency, ('consistency_summary',),
         _evaluate_stats(_consistency_done, 'Reviewed')),
)}
ORDER = {'prepare': ('media', 'hardcuts', 'missed_cuts', 'transcribe', 'subtitles', 'line_splits', 'voices', 'boxes', 'target_frames', 'cast'),
         'prepare_cuts': ('boxes', 'target_frames', 'cast'),
         'evaluate': ('visual', 'beats', 'consistency')}


# --- runs -------------------------------------------------------------------------------------------------------------

def _context(conn, episode_id, stage, options, agent_run_id=None, steps=None):
    episode = conn.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    profile = None
    if agent_run_id:
        row = conn.execute('SELECT policy_json FROM agent_runs WHERE id=?', (agent_run_id,)).fetchone()
        profile = _loads(row['policy_json'], {}) if row else None
    if not profile or not profile.get('modes'):
        profile, _ = autopilot_profile.load(conn, episode['project_id'])
    return {'episode_id': episode_id, 'episode': dict(episode), 'stage': stage, 'options': options or {}, 'agent_run_id': agent_run_id,
            'profile': profile, 'steps': steps or []}


def availability(conn, episode_id, stage):
    """(available, reason) of starting ``stage`` now."""
    running = conn.execute("SELECT stage FROM batch_runs WHERE episode_id=? AND status='running'", (episode_id,)).fetchone()
    if running:
        return False, f"A {running['stage'].replace('_', ' ')} run is going on; wait for it or cancel it"
    prepared = stage_targets.prepare_run(conn, episode_id)
    if stage == 'prepare' and prepared:
        return False, f"Prepare ran on {(prepared['finished_at'] or '')[:16]}; it runs once"
    if stage != 'prepare' and not prepared:
        return False, 'Run Prepare first'
    return True, ''


def normalize_options(stage, options):
    options = dict(options or {})
    clean = {}
    if stage == 'prepare' and (options.get('subtitles') or {}).get('language'):
        from . import subtitle_reader
        try:
            clean['subtitles'] = {'language': subtitle_reader.resolve_language(options['subtitles']['language'])['name']}
        except ValueError as exc:
            raise Refused(str(exc)) from exc
    if stage == 'evaluate':
        for key in ('from_cut', 'to_cut'):
            if options.get(key) is not None:
                value = int(options[key])
                if value < 1:
                    raise Refused(f'{key} starts at 1')
                clean[key] = value
        if clean.get('to_cut') is not None and clean['to_cut'] < clean.get('from_cut', 1):
            raise Refused('to_cut must not be before from_cut')
        if options.get('only'):
            if options['only'] not in ORDER['evaluate']:
                raise Refused(f"only must be one of {', '.join(ORDER['evaluate'])}")
            clean['only'] = options['only']
    return clean


def preview(conn, episode_id, stage, options=None):
    """What a run of ``stage`` would do now, step by step, with the same plans the run uses. Read-only."""
    if stage not in STAGES:
        raise Refused(f"stage must be one of {', '.join(STAGES)}")
    options = normalize_options(stage, options)
    available, reason = availability(conn, episode_id, stage)
    ctx = _context(conn, episode_id, stage, options)
    steps = []
    for name in ORDER[stage]:
        plan = STEPS[name].plan(conn, ctx)
        steps.append({'step': name, 'label': STEPS[name].label, **plan})
        ctx['steps'] = steps
    out = {'stage': stage, 'available': available, 'reason': reason, 'options': options, 'steps': steps,
           'prepared': stage_targets.prepare_run(conn, episode_id)}
    if stage == 'evaluate':
        cuts, held = {}, {}
        for step in steps:
            for item in step.get('cuts') or []:
                cuts.setdefault(item['cut'], item['cut_id'])
            for item in step.get('held') or []:
                held.setdefault(item['cut'], {**item, 'step': step['step']})
        out['cuts'] = len(cuts)
        out['held'] = [held[number] for number in sorted(held)]
    return out


def start(conn, episode_id, stage, options=None):
    """Create a run inside the caller's write transaction; ``tick``/``advance`` then carry it out. Returns its id."""
    if stage not in STAGES:
        raise Refused(f"stage must be one of {', '.join(STAGES)}")
    episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    if not episode:
        raise KeyError(episode_id)
    options = normalize_options(stage, options)
    available, reason = availability(conn, episode_id, stage)
    if not available:
        raise Refused(reason)
    profile, _ = autopilot_profile.load(conn, episode['project_id'])
    # Each waiting step shows what it will do; its real plan is read again when it starts
    planned = {step['step']: step.get('detail') for step in preview(conn, episode_id, stage, options)['steps']}
    agent_run_id = agent_log.start_run(conn, episode_id, AGENT, f'{AGENT}:{stage}', profile)
    steps = [{'step': name, 'label': STEPS[name].label, 'state': 'pending', 'detail': planned.get(name) or ''} for name in ORDER[stage]]
    run_id = agent_run_id  # one id names the run and its provenance
    conn.execute('INSERT INTO batch_runs(id,episode_id,stage,agent_run_id,options_json,steps_json) VALUES (?,?,?,?,?,?)',
                 (run_id, episode_id, stage, agent_run_id, json.dumps(options), json.dumps(steps)))
    return run_id


def _save(conn, run_id, steps, **fields):
    sets = ', '.join(f'{key}=?' for key in fields)
    conn.execute(f"UPDATE batch_runs SET steps_json=?,updated_at=CURRENT_TIMESTAMP{', ' + sets if sets else ''} WHERE id=?",
                 (json.dumps(steps), *fields.values(), run_id))


def _step_jobs(conn, episode_id, step, waits, payloads=True):
    """The jobs a step queued, and the jobs of its pipelines that were queued after it started (follow-ups)."""
    columns = 'rowid,id,pipeline,status,dismissed_at' + (',payload_json' if payloads else '')
    found = {}
    own = list(step.get('job_ids') or [])
    if own:
        marks = ','.join('?' for _ in own)
        for row in conn.execute(f'SELECT {columns} FROM jobs WHERE id IN ({marks})', own):
            found[row['id']] = dict(row)
    if waits and step.get('since_rowid') is not None:
        marks = ','.join('?' for _ in waits)
        for row in conn.execute(f'SELECT {columns} FROM jobs WHERE rowid>? AND episode_id=? AND pipeline IN ({marks})',
                                (step['since_rowid'], episode_id, *waits)):
            found.setdefault(row['id'], dict(row))
    return list(found.values())


def _busy(conn, episode_id, jobs, waits):
    if any(job['status'] in ('queued', 'running') for job in jobs):
        return True
    if not waits:
        return False
    marks = ','.join('?' for _ in waits)
    return bool(conn.execute(f"SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ({marks}) AND status IN ('queued','running') LIMIT 1",
                             (episode_id, *waits)).fetchone())


def _triage(conn, ctx, failed):
    """Apply the fix rules that exist to the step's failed jobs (once per step). True when something was queued again."""
    from . import diagnose, triage
    mode = triage.mode_for(conn, ctx['episode_id'], ctx['agent_run_id'])
    plan = diagnose.triage(conn, ctx['episode_id'], mode, set(failed))
    fixable = [item['job_id'] for item in plan['items'] if item['action'] in FIXABLE]
    if not fixable:
        return False
    outcome = triage.apply(conn, ctx['episode_id'], ctx['agent_run_id'], fixable)
    return any(item.get('job_ids') for item in outcome['applied'])


def _finish(conn, row, steps, status, error=None):
    episode_id = row['episode_id']
    summary = {'inbox': stage_targets.inbox_counts(conn, episode_id), 'failures': len(runs.attention_rows(conn, episode_id))}
    if row['stage'] == 'evaluate':
        options = _loads(row['options_json'], {})
        evaluate = stage_targets.targets(conn, episode_id, 'evaluate', options.get('from_cut'), options.get('to_cut'))['evaluate']
        held = {}
        for name in ORDER['evaluate']:
            for item in evaluate[name]['held']:
                held.setdefault(item['cut'], {**item, 'step': name})
        summary['held'] = [held[number] for number in sorted(held)]
    _save(conn, row['id'], steps, status=status, error=error, summary_json=json.dumps(summary), finished_at=_now())
    if row['agent_run_id']:
        agent_log.finish_run(conn, row['agent_run_id'], {'stage': row['stage'], 'status': status,
                                                         'steps': {step['step']: step['state'] for step in steps}})


def _move(run_id):
    """One step forward for one run. Returns True when it may go on at once (a step was skipped, finished or started)."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM batch_runs WHERE id=?', (run_id,)).fetchone()
        if not row or row['status'] != 'running':
            return False
        steps = _loads(row['steps_json'], [])
        index = next((i for i, step in enumerate(steps) if step['state'] not in DONE_STATES), None)
        if index is None:
            _finish(conn, row, steps, 'finished')
            return False
        step, spec = steps[index], STEPS[steps[index]['step']]
        ctx = _context(conn, row['episode_id'], row['stage'], _loads(row['options_json'], {}), row['agent_run_id'], steps)
        if step['state'] == 'running':
            jobs = _step_jobs(conn, row['episode_id'], step, spec.waits)
            if _busy(conn, row['episode_id'], jobs, spec.waits):
                return False
            failed = [job['id'] for job in jobs if job['status'] == 'failed' and not job['dismissed_at']]
            if failed and not step.get('retried'):
                step['retried'] = True
                requeued = _triage(conn, ctx, failed)
                _save(conn, run_id, steps)
                return not requeued
            stats, summary = spec.stats(conn, ctx, step, jobs)
            counts = {state: sum(job['status'] == state for job in jobs) for state in ('done', 'failed', 'cancelled')}
            step.update(stats=stats, summary=summary, jobs=counts, failed_jobs=failed[:50], finished_at=_now())
            if step['step'] in REQUIRED and any(job['status'] == 'failed' and job['pipeline'] == REQUIRED[step['step']] for job in jobs):
                step['state'] = 'failed'
                for later in steps[index + 1:]:
                    later.update(state='skipped', detail=f"{step['label']} failed", skip='blocked')
                _finish(conn, row, steps, 'failed', f"{step['label']} failed; fix it and start the run again")
                return False
            step['state'] = 'done'
            _save(conn, run_id, steps)
            return True
        if step['state'] == 'pending':
            plan = spec.plan(conn, ctx)
            if plan['state'] != 'will_run':
                step.update(state='skipped', skip=plan['state'], detail=plan['detail'], finished_at=_now())
                _save(conn, run_id, steps)
                return True
            since = conn.execute('SELECT COALESCE(MAX(rowid),0) FROM jobs').fetchone()[0]
            step.update(state='starting', detail=plan['detail'], since_rowid=since, started_at=_now(),
                        **({'cuts': plan['cuts']} if plan.get('cuts') is not None else {}))
            _save(conn, run_id, steps)
    # 'starting': queue outside the transaction (the helpers commit their own); after a crash this runs again safely,
    # because identical requests reuse the active job.
    try:
        job_ids = spec.start(ctx, step)
        error = None
    except Exception as exc:  # noqa: BLE001 - the run records it and goes on
        log.exception('Run Multiple %s could not start %s', run_id, step['step'])
        job_ids, error = [], getattr(exc, 'detail', None) or str(exc)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM batch_runs WHERE id=?', (run_id,)).fetchone()
        if not row or row['status'] != 'running':
            return False
        steps = _loads(row['steps_json'], [])
        current = steps[index]
        if current['state'] != 'starting':
            return False
        current.update(state='running', job_ids=list(dict.fromkeys(job_ids)),
                       **({'errors': step['errors']} if step.get('errors') else {}), **({'start_error': error} if error else {}))
        _save(conn, run_id, steps)
    worker.wake_event.set()
    return True


def advance(run_id):
    """Carry one run as far as it can go now (skipped steps chain, a started step then waits for its jobs)."""
    with _LOCK:
        for _ in range(MAX_MOVES):
            if not _move(run_id):
                return


def tick():
    """Advance every running run; the API process calls this every few seconds."""
    with db.connect() as conn:
        ids = [row['id'] for row in conn.execute("SELECT id FROM batch_runs WHERE status='running' ORDER BY rowid")]
    for run_id in ids:
        try:
            advance(run_id)
        except Exception:  # noqa: BLE001 - one broken run never stops the others
            log.exception('Run Multiple %s could not advance', run_id)


def cancel(conn, run_id):
    """Stop a running run inside the caller's write transaction: its current step's jobs are cancelled. Returns the
    cancelled job rows for ``worker.after_cancel``."""
    row = conn.execute('SELECT * FROM batch_runs WHERE id=?', (run_id,)).fetchone()
    if not row:
        raise KeyError(run_id)
    if row['status'] != 'running':
        raise Refused('This run is not running')
    steps = _loads(row['steps_json'], [])
    cancelled = []
    for step in steps:
        if step['state'] in ('starting', 'running'):
            jobs = _step_jobs(conn, row['episode_id'], step, STEPS[step['step']].waits, payloads=False)
            cancelled += worker.cancel_jobs(conn, [job for job in jobs if job['status'] in ('queued', 'running')], 'Run Multiple was cancelled')
            step.update(state='cancelled', finished_at=_now())
        elif step['state'] == 'pending':
            step.update(state='skipped', skip='cancelled', detail='the run was cancelled')
    _finish(conn, row, steps, 'cancelled')
    return cancelled


def view(conn, row):
    """A run as the dialog and ``fw`` show it: its steps, the running step's job progress, the summary."""
    steps = _loads(row['steps_json'], [])
    for step in steps:
        if step['state'] == 'running':
            jobs = _step_jobs(conn, row['episode_id'], step, STEPS[step['step']].waits, payloads=False)
            step['progress'] = {'total': len(jobs), 'finished': sum(job['status'] in ('done', 'failed', 'cancelled') for job in jobs),
                                'failed': sum(job['status'] == 'failed' for job in jobs)}
    return {'id': row['id'], 'episode_id': row['episode_id'], 'stage': row['stage'], 'status': row['status'],
            'options': _loads(row['options_json'], {}), 'steps': steps, 'summary': _loads(row['summary_json'], {}),
            'error': row['error'], 'created_at': row['created_at'], 'updated_at': row['updated_at'], 'finished_at': row['finished_at']}


def runs_of_episode(conn, episode_id, limit=20):
    return [view(conn, row) for row in conn.execute('SELECT * FROM batch_runs WHERE episode_id=? ORDER BY rowid DESC LIMIT ?', (episode_id, limit))]
