"""What the user tells the autopilot once per project (docs/autopilot/HARNESS.md, decision D14).

The profile is small on purpose and every field is optional; the defaults are the harness's decisions:

- ``modes``: per kind of decision, ``apply`` (the agent applies it and logs it), ``ask`` (it opens an escalation
  with its proposal) or ``off`` (it leaves that work alone).
- ``claude``: whether Claude may be asked as the second reviewer, and how many items one run may send it.
- ``topics``: per consistency-question topic (``consistency_summary.TOPICS``), a standing rule written by the user and
  what to do when the evidence does not settle a question: follow the evidence, take the recommended option, or ask.
- ``characters``: per character, whether it always matters (box it even as a fragment), is background (never box
  it), or is decided by the relevance rule (the default, left out of the stored profile).
- ``look_alikes``: pairs of characters the agent must never tell apart on its own.
- ``notes``: anything else the agent should know when it decides for this project.

Jobs that use it snapshot it in their payload, like ``visual_models.snapshot``; a run stores it in ``agent_runs.policy_json``.
"""
import json

from . import consistency_summary

VERSION = 1
MODES = ('apply', 'ask', 'off')
# The kinds of decision the autopilot makes, in the order of an episode's flow (HARNESS.md section 7).
DECISIONS = {
    'missed_cuts': 'Split cuts the hard-cut detection missed',
    'line_splits': 'Split lines that run across a hard cut, at a pause',
    'voices': 'Assign unknown voices to characters',
    'box_names': 'Box and name the characters that matter',
    'target_frames': 'Settle flagged target frames',
    'review_answers': 'Answer the consistency review questions',
    'stale_reruns': 'Re-run stale results up to the Cut summary',
}
TOPICS = consistency_summary.TOPICS
WHEN_UNSURE = ('evidence', 'recommended', 'ask')
RELEVANCE = ('auto', 'always', 'background')
MAX_RULE = 600
MAX_NOTES = 4000
MAX_CLAUDE_ITEMS = 500
DEFAULT_CLAUDE_ITEMS = 40


def defaults():
    return {
        'version': VERSION,
        'modes': {key: 'apply' for key in DECISIONS},
        'claude': {'enabled': True, 'max_items_per_run': DEFAULT_CLAUDE_ITEMS},
        'topics': {topic: {'rule': '', 'when_unsure': 'evidence'} for topic in TOPICS},
        'characters': {},
        'look_alikes': [],
        'notes': '',
    }


class ProfileError(ValueError):
    pass


def _text(value, limit, name):
    text = str(value or '').strip()
    if len(text) > limit:
        raise ProfileError(f'{name} is longer than {limit} characters')
    return text


def normalize(raw, character_ids):
    """A complete, valid profile from what the client sent. Unknown keys are dropped; bad values are errors."""
    raw = raw if isinstance(raw, dict) else {}
    profile = defaults()
    for key, mode in (raw.get('modes') or {}).items():
        if key in DECISIONS:
            if mode not in MODES:
                raise ProfileError(f'{DECISIONS[key]}: choose one of {", ".join(MODES)}')
            profile['modes'][key] = mode
    claude = raw.get('claude') or {}
    if 'enabled' in claude:
        profile['claude']['enabled'] = bool(claude['enabled'])
    if 'max_items_per_run' in claude:
        try:
            items = int(claude['max_items_per_run'])
        except (TypeError, ValueError):
            raise ProfileError('Claude items per run must be a whole number') from None
        if not 0 <= items <= MAX_CLAUDE_ITEMS:
            raise ProfileError(f'Claude items per run must be between 0 and {MAX_CLAUDE_ITEMS}')
        profile['claude']['max_items_per_run'] = items
    for topic, entry in (raw.get('topics') or {}).items():
        if topic not in TOPICS or not isinstance(entry, dict):
            continue
        when = entry.get('when_unsure', 'evidence')
        if when not in WHEN_UNSURE:
            raise ProfileError(f'{topic}: "when unsure" must be one of {", ".join(WHEN_UNSURE)}')
        profile['topics'][topic] = {'rule': _text(entry.get('rule'), MAX_RULE, f'The {topic} rule'), 'when_unsure': when}
    known = set(character_ids)
    for character_id, entry in (raw.get('characters') or {}).items():
        relevance = (entry or {}).get('relevance', 'auto') if isinstance(entry, dict) else entry
        if relevance not in RELEVANCE:
            raise ProfileError('Character relevance must be auto, always or background')
        if character_id in known and relevance != 'auto':
            profile['characters'][character_id] = {'relevance': relevance}
    pairs = set()
    for pair in raw.get('look_alikes') or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ProfileError('A look-alike pair names exactly two characters')
        first, second = (str(item) for item in pair)
        if first == second:
            raise ProfileError('A look-alike pair needs two different characters')
        if first in known and second in known:
            pairs.add(tuple(sorted((first, second))))
    profile['look_alikes'] = [list(pair) for pair in sorted(pairs)]
    profile['notes'] = _text(raw.get('notes'), MAX_NOTES, 'Notes')
    return profile


def character_ids(conn, project_id):
    return [row[0] for row in conn.execute("SELECT id FROM context_items WHERE project_id=? AND kind='character'", (project_id,))]


def load(conn, project_id):
    """The project's profile, completed with defaults; characters that no longer exist are dropped."""
    row = conn.execute('SELECT profile_json,updated_at FROM autopilot_profiles WHERE project_id=?', (project_id,)).fetchone()
    stored = json.loads(row['profile_json']) if row else {}
    profile = normalize(stored, character_ids(conn, project_id))
    return profile, (row['updated_at'] if row else None)


def save(conn, project_id, raw):
    profile = normalize(raw, character_ids(conn, project_id))
    conn.execute("""INSERT INTO autopilot_profiles(project_id,profile_json) VALUES (?,?)
        ON CONFLICT(project_id) DO UPDATE SET profile_json=excluded.profile_json,updated_at=CURRENT_TIMESTAMP""",
                 (project_id, json.dumps(profile, ensure_ascii=False, sort_keys=True)))
    return profile


def _answered(answers, question):
    """The option ids of ``question`` whose text appears in the saved answers (the format of ``consistencyView.setAnswer``)."""
    text = answers.casefold()
    return [option.get('id') for option in question.get('options') or [] if str(option.get('text') or '').strip() and str(option['text']).strip().casefold() in text]


def topic_stats(conn, project_id):
    """Per topic, from the current consistency reviews of the project: questions asked, answered, and answered with the recommended option."""
    stats = {topic: {'asked': 0, 'answered': 0, 'recommended': 0} for topic in TOPICS}
    rows = conn.execute("""SELECT r.content,b.content AS answers FROM analysis_results r
        JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id
        LEFT JOIN cut_brief_instructions b ON b.cut_id=r.cut_id
        WHERE e.project_id=? AND r.title='Consistency summary' AND r.superseded_by IS NULL AND c.active=1""", (project_id,)).fetchall()
    for row in rows:
        try:
            content = json.loads(row['content'])
        except (TypeError, ValueError):
            continue
        for question in (content.get('questions') if isinstance(content, dict) else None) or []:
            topic = question.get('topic') if question.get('topic') in stats else 'other'
            entry = stats[topic]
            entry['asked'] += 1
            picked = _answered(row['answers'] or '', question)
            if picked:
                entry['answered'] += 1
                entry['recommended'] += question.get('recommended') in picked
    return stats
