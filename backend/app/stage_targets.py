"""The target cuts of each autopilot stage, read for the stage commands and their dry runs (docs/autopilot/HARNESS.md D31).

Read-only and free of FastAPI. An episode goes through three stages, each started by the person (``scripts/fw``):

* **prepare** runs once: media, hard cuts, the missed-cut check, transcription and speakers, subtitles, line splits,
  voices, first-frame boxes and recognition, target frames on every cut, the cast resolver. It is done once an agent run
  with purpose ``prepare`` finished; after that it never runs again and only ``prepare_cuts`` and ``lines`` touch its work.
* **prepare_cuts**: cuts whose boundaries changed after Prepare (no first-frame detection of their current boundaries) and
  cuts whose Target frames run is missing or stale (selection changed, frames added, first-frame boxes changed).
* **lines**: current lines across a hard cut that the line splitter would still plan (``line_splits.open_crossings``).
* **evaluate**: visual analysis, beats and the consistency review, each for its *targets*: a result missing or stale, and
  the cut's inputs settled with no open inbox item (``cut_readiness``, ``progress.held`` of ``review_report``).
* **results**: "Your answers" (the M4 agent) and Cut summaries, only when the profile turns them on.

Each step gives ``targets`` ([{cut, cut_id, why: missing|stale, reason}]) and ``held`` ([{cut, cut_id, reasons}]: cuts that
need the step but wait for the person or for Claude). Nothing is queued or saved here.
"""
from . import agent_log, autopilot_profile, line_splits, review_report, target_characters

STAGES = ('prepare', 'prepare_cuts', 'lines', 'evaluate', 'results')
# Stale result titles by the evaluate/results step that makes them again; titles the flow no longer makes (last frame,
# subtitle OCR) are left alone.
STALE_TITLES = {'visual': ('Initial frame description', 'Keyframe analysis', 'Video analysis'), 'beats': ('Performance beats',),
                'consistency': ('Consistency summary',), 'summary': ('Cut summary',)}
STALE_ORDER = ('visual', 'beats', 'consistency', 'summary')
# The resolvers of Prepare, with the profile kind that switches each one off
PREPARE_RESOLVERS = (('missed_cuts', 'missed_cuts'), ('line_splits', 'line_splits'), ('voice_resolver', 'voices'),
                     ('character_decisions', 'box_names'))
LISTED_REASONS = 3


def prepare_run(conn, episode_id):
    """The finished Prepare run of the episode, or None."""
    row = conn.execute("""SELECT id,started_at,finished_at,summary_json FROM agent_runs WHERE episode_id=? AND purpose='prepare'
        AND status='finished' ORDER BY rowid DESC LIMIT 1""", (episode_id,)).fetchone()
    return {'id': row['id'], 'started_at': row['started_at'], 'finished_at': row['finished_at']} if row else None


def _cuts(conn, episode_id, from_cut=None, to_cut=None):
    first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut is not None else None)
    return [dict(row) for row in conn.execute("""SELECT c.id,c.episode_id,c.cut_index,c.start_time,c.end_time,c.analysis_enabled,c.no_characters,
        (SELECT r.phase FROM character_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time
         ORDER BY r.rowid DESC LIMIT 1) AS detection_phase,
        EXISTS(SELECT 1 FROM target_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time) AS has_target_run
        FROM cuts c WHERE c.episode_id=? AND c.active=1 AND c.cut_index>=? AND (? IS NULL OR c.cut_index<=?) ORDER BY c.cut_index""",
        (episode_id, first, last, last))]


def _item(cut, why, reason=None):
    return {'cut': cut['cut_index'] + 1, 'cut_id': cut['id'], 'why': why, **({'reason': reason} if reason else {})}


def _job_done(conn, episode_id, pipeline):
    return bool(conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline=? AND status='done' LIMIT 1", (episode_id, pipeline)).fetchone())


def _mode(profile, kind):
    return (profile.get('modes') or {}).get(kind, 'off' if kind in autopilot_profile.OFF_BY_DEFAULT else 'apply')


def target_frames(conn, cuts):
    """Cuts whose Target frames should be detected (again): every analysed cut with people possible and a finished
    first-frame detection of its boundaries, whose run is missing or stale."""
    found = []
    for cut in cuts:
        if not cut['analysis_enabled'] or cut['no_characters'] or cut['detection_phase'] != 'complete':
            continue
        if not cut['has_target_run']:
            found.append(_item(cut, 'missing'))
            continue
        run = (target_characters.state(cut['id'], conn) or {}).get('run') or {}
        if run.get('stale'):
            found.append(_item(cut, 'stale', run.get('stale_reason')))
    return found


def _prepare(conn, episode_id, report, cuts, profile):
    setup = report['setup']
    enabled = [cut for cut in cuts if cut['analysis_enabled']]
    boxes = [_item(cut, 'missing') for cut in enabled if not cut['no_characters'] and cut['detection_phase'] != 'complete']
    targets = target_frames(conn, cuts)
    steps = [{'step': 'media', 'done': setup['media']}, {'step': 'hardcuts', 'done': setup['cuts']}]
    resolvers = dict(PREPARE_RESOLVERS)
    steps.append({'step': 'missed_cuts', 'done': _job_done(conn, episode_id, 'missed_cuts'), 'mode': _mode(profile, resolvers['missed_cuts'])})
    steps += [{'step': 'transcribe', 'done': setup['transcript']}, {'step': 'subtitles', 'done': setup['subtitles']}]
    for pipeline, kind in PREPARE_RESOLVERS[1:3]:
        steps.append({'step': pipeline, 'done': _job_done(conn, episode_id, pipeline), 'mode': _mode(profile, kind)})
    steps.append({'step': 'boxes', 'done': setup['boxes'] and not boxes, 'targets': boxes})
    steps.append({'step': 'target_frames', 'done': not targets, 'targets': targets})
    steps.append({'step': 'character_decisions', 'done': _job_done(conn, episode_id, 'character_decisions'), 'mode': _mode(profile, 'box_names')})
    return {'done': prepare_run(conn, episode_id), 'steps': steps}


def _held(report, step, numbers, by_number):
    reasons = {item['cut']: item['reasons'] for item in report['progress']['held'].get(step) or []}
    return [{'cut': number, 'cut_id': by_number[number]['id'], 'reasons': reasons[number][:LISTED_REASONS]}
            for number in sorted(numbers) if number in reasons and number in by_number]


def _stale(report):
    """{step: {cut number: reason}} of stale results, each cut under its earliest stale step."""
    earliest = {}
    for item in report['checks']['stale_results']:
        for rank, step in enumerate(STALE_ORDER):
            if item['title'] in STALE_TITLES[step] and rank < earliest.get(item['cut'], (len(STALE_ORDER), None))[0]:
                earliest[item['cut']] = (rank, item['reason'])
    stale = {step: {} for step in STALE_ORDER}
    for number, (rank, reason) in earliest.items():
        stale[STALE_ORDER[rank]][number] = reason
    return stale


def _split(report, step, missing, stale, by_number):
    """(targets, held) of one gated step: the cuts that need it, split by whether ``progress.held`` lets them run."""
    held_numbers = {item['cut'] for item in report['progress']['held'].get(step) or []}
    needed = {number: ('missing', None) for number in missing}
    needed.update({number: ('stale', reason) for number, reason in stale.items() if number not in needed})
    targets = [_item(by_number[number], *needed[number]) for number in sorted(needed) if number in by_number and number not in held_numbers]
    return targets, _held(report, step, set(needed) & held_numbers, by_number)


def _evaluate(report, by_number):
    progress, stale = report['progress'], _stale(report)
    visual_missing = set(progress['visual_missing'])
    visual, visual_held = _split(report, 'visual', visual_missing, stale['visual'], by_number)
    upstream = visual_missing | set(stale['visual'])
    beats_missing = set(progress['beats_missing']) - upstream
    beats, beats_held = _split(report, 'beats', beats_missing, {k: v for k, v in stale['beats'].items() if k not in upstream}, by_number)
    upstream |= beats_missing | set(stale['beats'])
    consistency, consistency_held = _split(report, 'consistency', set(progress['consistency_missing']) - upstream,
                                           {k: v for k, v in stale['consistency'].items() if k not in upstream}, by_number)
    return {'visual': {'targets': visual, 'held': visual_held}, 'beats': {'targets': beats, 'held': beats_held},
            'consistency': {'targets': consistency, 'held': consistency_held}}, upstream | set(progress['consistency_missing']) | set(stale['consistency'])


def _results(report, by_number, upstream, profile):
    stale = _stale(report)
    held_numbers = {item['cut'] for item in report['progress']['held'].get('consistency') or []}
    blocked = upstream | held_numbers
    decisions = {item['cut'] for item in report['gates']['decisions']}
    answers = [_item(by_number[number], 'missing', 'review questions without an answer') for number in sorted(decisions - blocked) if number in by_number]
    wanted = {number: ('missing', None) for number in report['progress']['summary_missing']}
    wanted.update({number: ('stale', reason) for number, reason in stale['summary'].items() if number not in wanted})
    summaries, waiting = [], []
    for number in sorted(wanted):
        if number not in by_number:
            continue
        if number in blocked or number in decisions:
            reasons = ['review questions without an answer'] if number in decisions else ['an earlier step is not current']
            waiting.append({'cut': number, 'cut_id': by_number[number]['id'], 'reasons': reasons})
        else:
            summaries.append(_item(by_number[number], *wanted[number]))
    return {'answers': {'targets': answers, 'held': [], 'mode': _mode(profile, 'review_answers')},
            'summary': {'targets': summaries, 'held': waiting, 'mode': _mode(profile, 'cut_summaries')}}


def inbox_counts(conn, episode_id):
    """Open inbox items by kind and reviewer."""
    counts = {}
    for item in agent_log.escalations(conn, episode_id, 'open', limit=5000):
        entry = counts.setdefault(item['kind'], {'human': 0, 'claude': 0})
        entry[item['reviewer']] += 1
    return counts


def targets(conn, episode_id, stage=None, from_cut=None, to_cut=None):
    """{stage: …} for one stage or all of them (``stage`` None)."""
    if stage is not None and stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    report = review_report.report(conn, episode_id, from_cut, to_cut)
    profile = autopilot_profile.profile_for(conn, episode_id)
    cuts = _cuts(conn, episode_id, from_cut, to_cut)
    by_number = {cut['cut_index'] + 1: cut for cut in cuts}
    wanted = [stage] if stage else list(STAGES)
    out = {'episode': report['episode'], 'range': report['range'], 'inbox': inbox_counts(conn, episode_id), 'failures': len(report['failures'])}
    if 'prepare' in wanted:
        out['prepare'] = _prepare(conn, episode_id, report, cuts, profile)
    if 'prepare_cuts' in wanted:
        enabled = [cut for cut in cuts if cut['analysis_enabled'] and not cut['no_characters']]
        out['prepare_cuts'] = {'boxes': {'targets': [_item(cut, 'missing') for cut in enabled if cut['detection_phase'] != 'complete']},
                               'target_frames': {'targets': target_frames(conn, cuts)}}
    if 'lines' in wanted:
        crossing = []
        for row, hard_cut in line_splits.open_crossings(conn, episode_id):
            cut = next((item for item in cuts if item['start_time'] <= hard_cut < item['end_time']), None)
            if cut:
                crossing.append({'cut': cut['cut_index'] + 1, 'cut_id': cut['id'], 'line_id': row['id'], 'start_time': row['start_time'],
                                 'end_time': row['end_time'], 'text': row['text'][:80], 'hard_cut': hard_cut})
        out['lines'] = {'targets': crossing, 'mode': _mode(profile, 'line_splits')}
    if 'evaluate' in wanted or 'results' in wanted:
        evaluate, upstream = _evaluate(report, by_number)
        if 'evaluate' in wanted:
            out['evaluate'] = evaluate
        if 'results' in wanted:
            out['results'] = _results(report, by_number, upstream, profile)
    return out
