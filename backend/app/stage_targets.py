"""The target cuts of each stage, read by Run Multiple (``batch_runs``) and ``scripts/fw`` (docs/autopilot/HARNESS.md D31, D37).

Read-only and free of FastAPI. An episode goes through these stages, each started by the person:

* **prepare** runs once: media, hard cuts, the missed-cut check, transcription and speakers, subtitles (optional), line
  splits, voices, first-frame boxes and recognition, target frames on every cut, the cast resolver. Its steps are planned by
  ``batch_runs`` (this module reports them). It is done once a Run Multiple Prepare finished (or, before D37, an agent run
  with purpose ``prepare``); after that it never runs again and only ``prepare_cuts`` touches its work.
* **prepare_cuts**: cuts whose boundaries changed after Prepare (no first-frame detection of their current boundaries) and
  cuts whose Target frames run is missing or stale (selection changed, frames added, first-frame boxes changed).
* **evaluate**: visual analysis, beats and the consistency review, each for its *targets*: a result missing or stale, and
  the cut's inputs settled with no open inbox item (``cut_readiness``, ``progress.held`` of ``review_report``).
* **results**: "Your answers" (the M4 agent) and Cut summaries, only when the profile turns them on (``fw results``).

Each step gives ``targets`` ([{cut, cut_id, why: missing|stale, reason}]) and ``held`` ([{cut, cut_id, reasons}]: cuts that
need the step but wait for the person or for Claude). Nothing is queued or saved here.
"""
from . import agent_log, autopilot_profile, review_report, target_characters

STAGES = ('prepare', 'prepare_cuts', 'evaluate', 'results')
# Stale result titles by the evaluate/results step that makes them again; titles the flow no longer makes (last frame,
# subtitle OCR) are left alone.
STALE_TITLES = {'visual': ('Initial frame description', 'Keyframe analysis', 'Video analysis'), 'beats': ('Performance beats',),
                'consistency': ('Consistency summary',), 'summary': ('Cut summary',)}
STALE_ORDER = ('visual', 'beats', 'consistency', 'summary')
LISTED_REASONS = 3


def prepare_run(conn, episode_id):
    """The finished Prepare of the episode: a Run Multiple run, else (before D37) an ``fw`` agent run with purpose ``prepare``.
    None when Prepare never finished (a failed or cancelled run does not count)."""
    row = conn.execute("""SELECT id,created_at AS started_at,finished_at FROM batch_runs WHERE episode_id=? AND stage='prepare'
        AND status='finished' ORDER BY rowid DESC LIMIT 1""", (episode_id,)).fetchone()
    row = row or conn.execute("""SELECT id,started_at,finished_at FROM agent_runs WHERE episode_id=? AND purpose='prepare'
        AND status='finished' ORDER BY rowid DESC LIMIT 1""", (episode_id,)).fetchone()
    return {'id': row['id'], 'started_at': row['started_at'], 'finished_at': row['finished_at']} if row else None


def cut_rows(conn, episode_id, from_cut=None, to_cut=None):
    first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut is not None else None)
    return [dict(row) for row in conn.execute("""SELECT c.id,c.episode_id,c.cut_index,c.start_time,c.end_time,c.analysis_enabled,c.no_characters,
        (SELECT r.phase FROM character_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time
         ORDER BY r.rowid DESC LIMIT 1) AS detection_phase,
        EXISTS(SELECT 1 FROM target_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time) AS has_target_run
        FROM cuts c WHERE c.episode_id=? AND c.active=1 AND c.cut_index>=? AND (? IS NULL OR c.cut_index<=?) ORDER BY c.cut_index""",
        (episode_id, first, last, last))]


def _item(cut, why, reason=None):
    return {'cut': cut['cut_index'] + 1, 'cut_id': cut['id'], 'why': why, **({'reason': reason} if reason else {})}


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


def _prepare(conn, episode_id):
    """Prepare's steps as Run Multiple plans them (``batch_runs.preview``), and whether it is done."""
    from . import batch_runs
    plan = batch_runs.preview(conn, episode_id, 'prepare')
    return {'done': plan['prepared'], 'steps': [{key: step.get(key) for key in ('step', 'label', 'state', 'detail', 'cuts') if key in step}
                                                for step in plan['steps']]}


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
    cuts = cut_rows(conn, episode_id, from_cut, to_cut)
    by_number = {cut['cut_index'] + 1: cut for cut in cuts}
    wanted = [stage] if stage else list(STAGES)
    out = {'episode': report['episode'], 'range': report['range'], 'inbox': inbox_counts(conn, episode_id), 'failures': len(report['failures'])}
    if 'prepare' in wanted:
        out['prepare'] = _prepare(conn, episode_id)
    if 'prepare_cuts' in wanted:
        enabled = [cut for cut in cuts if cut['analysis_enabled'] and not cut['no_characters']]
        out['prepare_cuts'] = {'boxes': {'targets': [_item(cut, 'missing') for cut in enabled if cut['detection_phase'] != 'complete']},
                               'target_frames': {'targets': target_frames(conn, cuts)}}
    if 'evaluate' in wanted or 'results' in wanted:
        evaluate, upstream = _evaluate(report, by_number)
        if 'evaluate' in wanted:
            out['evaluate'] = evaluate
        if 'results' in wanted:
            out['results'] = _results(report, by_number, upstream, profile)
    return out
