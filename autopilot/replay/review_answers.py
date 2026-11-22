"""Replay of the review-answer resolver (M4) against the user's own answers to the consistency review questions.

Ground truth: the current consistency reviews whose questions the user answered (``cut_brief_instructions``, read with
the UI's own rule, ``review_decisions.selection``), leaving out answers the agent wrote. Each question is scored:

* **recommended**: the reviewer's recommended option, by topic and by whether its ``why`` cites a picture. This is
  what applying "take the recommended option" would score, and the stand-in trust before shadow runs exist.
* **decider** (once shadow runs exist: ``fw review-decisions run EP --shadow``): the local decider's choice with the
  user's answers hidden; overall, at high confidence, and when it agrees with the reviewer at high confidence (the
  only case the resolver applies alone, ``TRUST`` = 90%).
* **resolver**: what the resolver would do today with each question (apply / Claude / person) under the project's
  profile and trust table, and how often what it would apply matches the user.

The disagreements are listed for a person to read: they are where standing rules come from.

    autopilot/replay/run.sh review_answers [--project Seishun]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PICTURE = re.compile(r'line strip|picture|\bF\d+\b', re.I)


def basis(question):
    return 'picture' if PICTURE.search(question.get('why') or '') else 'text'


def rate(right, total):
    return f'{right}/{total} ({right / total:.0%})' if total else '—'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3')
    parser.add_argument('--project', default='')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import autopilot_profile, review_decisions as rd
    import sqlite3
    with sqlite3.connect(args.db) as copy:  # a copy made before M4 was deployed has no decisions table yet (the copy only)
        copy.execute('CREATE TABLE IF NOT EXISTS review_decisions (id TEXT PRIMARY KEY, cut_id TEXT, episode_id TEXT, result_id TEXT, job_id TEXT, run_id TEXT, '
                     "shadow INTEGER DEFAULT 0, decisions_json TEXT DEFAULT '{}', outcomes_json TEXT DEFAULT '{}', model_id TEXT, provenance_json TEXT DEFAULT '{}', created_at TEXT)")
    conn = common.open_copy(Path(args.db))
    projects = [row['id'] for row in conn.execute('SELECT id FROM projects WHERE title LIKE ?', (f'%{args.project}%',))]
    rows = conn.execute(f"""SELECT r.id AS result_id,r.cut_id,r.content,b.content AS answers,c.cut_index,e.title AS episode,e.project_id
        FROM analysis_results r JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id JOIN cut_brief_instructions b ON b.cut_id=r.cut_id
        WHERE r.title='Consistency summary' AND r.superseded_by IS NULL AND c.active=1 AND e.project_id IN ({','.join('?' for _ in projects) or 'NULL'})""",
                        projects).fetchall()
    held = rd.agent_held(conn, [row['cut_id'] for row in rows])
    trust = {project: rd.trust_table(conn, project) for project in projects}
    profiles = {project: autopilot_profile.load(conn, project)[0] for project in projects}

    rec = collections.defaultdict(lambda: [0, 0])          # (topic, basis) -> [right, total]
    decider = collections.defaultdict(lambda: [0, 0])      # (topic, slice) -> [right, total]
    resolver = collections.defaultdict(lambda: collections.Counter())
    free, disagreements, shadowed, answered = collections.Counter(), [], 0, 0
    for row in rows:
        if row['cut_id'] in held:
            continue
        try:
            content = json.loads(row['content'])
        except (TypeError, ValueError):
            continue
        questions = content.get('questions') if isinstance(content, dict) else None
        if not questions or not any(rd.selection(row['answers'], question, questions) for question in questions):
            continue
        answered += 1
        shadow = conn.execute('SELECT decisions_json FROM review_decisions WHERE cut_id=? AND result_id=? AND shadow=1 ORDER BY rowid DESC LIMIT 1',
                              (row['cut_id'], row['result_id'])).fetchone()
        decided = json.loads(shadow['decisions_json']) if shadow else {}
        shadowed += bool(shadow)
        profile = profiles[row['project_id']]
        for question in questions:
            picked = rd.selection(row['answers'], question, questions)
            topic = question.get('topic') or 'other'
            if picked is None:
                continue
            if picked['option'] is None:
                free[topic] += 1
                continue
            user = picked['option']
            if question.get('recommended'):
                entry = rec[(topic, basis(question))]
                entry[0] += user == question['recommended']
                entry[1] += 1
                if user != question['recommended']:
                    disagreements.append({'episode': row['episode'], 'cut': row['cut_index'] + 1, 'topic': topic, 'character': question.get('character'),
                                          'question': question['question'], 'recommended': rd.option_text(question, question['recommended']),
                                          'user': rd.option_text(question, user), 'why': question.get('why')})
            decision = decided.get(question['id'])
            if decision and decision.get('choice'):
                for name in ('all', *(('high',) if decision['confidence'] == 'high' else ()),
                             *(('agrees+high',) if decision['confidence'] == 'high' and decision['choice'] == question.get('recommended') else ())):
                    entry = decider[(topic, name)]
                    entry[0] += decision['choice'] == user
                    entry[1] += 1
            # The resolver today: with a shadow decision when there is one, else as if the decider echoed the reviewer.
            stand_in = decision or ({'choice': question.get('recommended'), 'confidence': 'high', 'kind': 'fact', 'rule': False} if question.get('recommended') else None)
            outcome = rd.resolve(question, stand_in, (profile.get('topics') or {}).get(topic), (trust[row['project_id']].get(topic) or {}).get('rate'),
                                 profile['modes'].get(rd.KIND, 'apply'))
            resolver[topic][outcome['action']] += 1
            if outcome['action'] == 'apply':
                resolver[topic]['applied_right'] += outcome['option'] == user

    topics = sorted({topic for topic, _ in rec} | set(resolver) | set(free))
    lines = ['## Ground truth', '', f"{answered} cuts where the user answered a current review's questions; "
             f"{shadowed} of them have a shadow decision.", '',
             '## The reviewer\'s recommended option against the user', '', '| topic | cites a picture | text only | user wrote own text |', '|---|---|---|---|']
    for topic in topics:
        lines.append(f"| {topic} | {rate(*rec[(topic, 'picture')])} | {rate(*rec[(topic, 'text')])} | {free[topic]} |")
    total = [sum(entry[0] for entry in rec.values()), sum(entry[1] for entry in rec.values())]
    lines += ['', f"All: {rate(*total)}. The resolver applies an answer alone only at {rd.TRUST:.0%} measured agreement (D12).", '']
    if decider:
        lines += ['## The local decider (shadow runs) against the user', '', '| topic | all | high confidence | agrees with the reviewer, high |', '|---|---|---|---|']
        for topic in topics:
            lines.append(f"| {topic} | {rate(*decider[(topic, 'all')])} | {rate(*decider[(topic, 'high')])} | {rate(*decider[(topic, 'agrees+high')])} |")
        lines.append('')
    else:
        lines += ['## The local decider', '', 'No shadow decisions yet: run `fw review-decisions run EP --shadow` on an episode the user answered '
                  '(Episode 2), wait for it, copy the database again (`FRESH=1`) and re-run this replay.', '']
    lines += ['## What the resolver would do today (profile and trust as stored)', '', '| topic | apply | right of applied | Claude | person |', '|---|---|---|---|---|']
    for topic in topics:
        counts = resolver[topic]
        lines.append(f"| {topic} | {counts['apply']} | {rate(counts['applied_right'], counts['apply'])} | {counts['claude']} | {counts['human']} |")
    lines += ['', '## Trust per topic (as the resolver reads it)', '']
    for project, table in trust.items():
        for topic, entry in sorted(table.items()):
            shown = '—' if entry['rate'] is None else f"{entry['rate']:.0%}"
            lines.append(f"- {topic}: {shown} from {entry['scored']} ({entry['source']})")
    lines += ['', f'## Where the user chose another option than the recommended one ({len(disagreements)})', '']
    for item in disagreements:
        lines.append(f"- {item['episode']} cut {item['cut']} [{item['topic']}{' · ' + item['character'] if item['character'] else ''}] {item['question']}\n"
                     f"  - recommended: {item['recommended']}\n  - user: {item['user']}\n  - why: {item['why']}")
    report = common.write_report('review_answers', {
        'recommended': {f'{topic}|{kind}': entry for (topic, kind), entry in rec.items()},
        'decider': {f'{topic}|{kind}': entry for (topic, kind), entry in decider.items()},
        'resolver': {topic: dict(counts) for topic, counts in resolver.items()}, 'free_text': dict(free),
        'trust': trust, 'disagreements': disagreements, 'shadowed_cuts': shadowed}, '\n'.join(lines), Path(args.out))
    print(report)
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
