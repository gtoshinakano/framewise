"""What each cut of one episode still needs, for the editor's filters, the analysis dialog and the range inspector.

Read-only. It regroups what ``review_report.report`` already finds per episode (missing steps, gates, checks) by cut, adds
the open autopilot escalations and which agents applied something on the cut. ``people`` is what waits for the person,
``claude`` what waits for Claude (the second reviewer). A person's item that an open Claude escalation covers on the same
cut is left to Claude. Nothing here is stored and nothing is an approval status.
"""
from fastapi import APIRouter, HTTPException

from . import agent_log, cut_readiness, db, review_report

router = APIRouter(prefix='/api')

# The editor's order of an episode's flow (frontend/src/cutNeedsView.ts NEED_ORDER). Missed cuts and lines across a hard
# cut are never a need, and a voice is one only while its question is open (HARNESS.md D37).
PEOPLE = ('voice', 'box_names', 'target_frames', 'decisions', 'picture_check', 'failed_run')
# Escalation kind -> the person's need it stands for.
ESCALATION_NEED = {'voice': 'voice', 'box_label': 'box_names', 'new_person': 'box_names', 'target_frame': 'target_frames',
                   'review_question': 'decisions', 'failed_run': 'failed_run'}
STEPS = (('boxes', 'boxes_missing'), ('target', 'target_missing'), ('first_frame', 'first_frame_missing'), ('keyframes', 'keyframes_missing'),
         ('beats', 'beats_missing'), ('consistency', 'consistency_missing'), ('summary', 'summary_missing'))


def needs(conn, episode_id):
    report = review_report.report(conn, episode_id)
    cuts = [dict(row) for row in conn.execute('SELECT id,cut_index,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index',
                                              (episode_id,))]
    by_number = {cut['cut_index'] + 1: cut['id'] for cut in cuts}
    entries = {cut['id']: {'steps': [], 'people': {}, 'claude': [], 'answered_by': [], 'analysis_off': False, 'stale': []} for cut in cuts}

    def person(cut_id, key, count=1, escalation_id=None):
        if cut_id not in entries:
            return
        item = entries[cut_id]['people'].setdefault(key, {'key': key, 'count': 0, 'escalation_ids': []})
        item['count'] += count
        if escalation_id and escalation_id not in item['escalation_ids']:
            item['escalation_ids'].append(escalation_id)

    progress = report['progress']
    for step, field in STEPS:
        for number in progress[field]:
            if number in by_number:
                entries[by_number[number]]['steps'].append(step)
    for number in progress['analysis_off']:
        if number in by_number:
            entries[by_number[number]]['analysis_off'] = True
    gates, checks = report['gates'], report['checks']
    for item in checks['stale_results']:  # outputs an input changed after they were made: the next Evaluate re-runs them
        if item['cut_id'] in entries:
            entries[item['cut_id']]['stale'].append({'title': item['title'], 'reason': item['reason']})
    for item in gates['first_frame']:
        person(item['cut_id'], 'box_names', item['unlabeled_boxes'])
    for item in gates['target_frames']:
        person(item['cut_id'], 'target_frames', len(item['frames']) + item['unnamed_people'])
    for item in gates['decisions']:
        person(item['cut_id'], 'decisions', item['questions'])
    for item in checks['picture_contradictions']:
        person(item['cut_id'], 'picture_check', len(item['checks']))
    off = {cut_id for cut_id, entry in entries.items() if entry['analysis_off']}
    analysed = [cut for cut in cuts if cut['id'] not in off]

    claude_keys = {}
    for item in agent_log.escalations(conn, episode_id, 'open', limit=5000):
        key = ESCALATION_NEED.get(item['kind'])
        for cut_id in cut_readiness.escalation_cuts(conn, item, analysed):
            if cut_id not in entries or cut_id in off:  # a cut left out of the analysis needs nothing
                continue
            if item['reviewer'] == 'claude':
                entries[cut_id]['claude'].append({'kind': item['kind'], 'escalation_id': item['id']})
                if key:
                    claude_keys.setdefault(cut_id, set()).add(key)
            elif key:
                # An escalation restates a gate the report already counted (unlabeled boxes, open questions): link it, count it once.
                person(cut_id, key, 0 if key in entries[cut_id]['people'] else 1, item['id'])
    for cut_id, keys in claude_keys.items():
        for key in keys:
            entries[cut_id]['people'].pop(key, None)

    for row in conn.execute("""SELECT cut_id,GROUP_CONCAT(DISTINCT tier) AS tiers FROM agent_decisions WHERE episode_id=? AND undone_at IS NULL
                               AND cut_id IS NOT NULL GROUP BY cut_id""", (episode_id,)):
        if row['cut_id'] in entries:
            entries[row['cut_id']]['answered_by'] = sorted((row['tiers'] or '').split(','))
    for entry in entries.values():
        entry['people'] = [entry['people'][key] for key in PEOPLE if key in entry['people']]
    return {'cuts': entries}


@router.get('/episodes/{episode_id}/cut-needs', summary='Per cut: missing steps, what waits for the person, what waits for Claude')
def episode_cut_needs(episode_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        return needs(conn, episode_id)
