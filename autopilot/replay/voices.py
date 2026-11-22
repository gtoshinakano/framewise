"""Replay of the voice resolver (M5) against the speakers the user already settled.

Ground truth:

* **labels**: a diarization label whose lines the user linked to one character (``LABEL_MIN`` lines or more, at least 95%
  one character; Episodes 4-7 were linked label by label). Each is hidden in turn (its lines lose their character, so
  its own links disappear, as for a real unknown voice) and the resolver decides it with every other label as it is.
* **lines**: lines whose speaker a person set (``speaker_source='manual'``, or a ``manual_edit``/``srt_edit`` revision
  whose parent had another character). Each is hidden alone.

Voice scores are ``computed`` (Community-1 embeddings against the project's voice samples, on the CPU of the audio image:
``REPLAY_IMAGE=framewise-audio-rocm:local autopilot/replay/run.sh voices --scores computed``) or ``stored`` (the scores the
transcription saved on line records; old word rows have none). The voice samples were cut from Episodes 1-2, so labels
of Episodes 4-7 never score against their own audio.

Also prints the plan for each episode's real open voices (what a run would apply and ask), writing nothing.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

GOLD_SHARE = 0.95


def gold_labels(lines, minimum):
    by_label = collections.defaultdict(collections.Counter)
    for row in lines:
        if row['speaker_label'] and row['character_context_id']:
            by_label[row['speaker_label']][row['character_context_id']] += 1
    gold = {}
    for label, counter in by_label.items():
        character, count = counter.most_common(1)[0]
        total = sum(counter.values())
        if total >= minimum and count / total >= GOLD_SHARE:
            gold[label] = character
    return gold


def gold_lines(conn, episode_id):
    """Current lines whose character a person set (and the character they set)."""
    found = {}
    for row in conn.execute("""SELECT t.id,t.character_context_id,t.speaker_source,t.origin,p.character_context_id AS before
            FROM transcript_segments t LEFT JOIN transcript_segments p ON p.id=t.parent_id
            WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.character_context_id IS NOT NULL""", (episode_id,)):
        if row['speaker_source'] == 'manual' or (row['origin'] in ('manual_edit', 'srt_edit') and row['before'] != row['character_context_id']):
            found[row['id']] = row['character_context_id']
    return found


def hidden(data, ids):
    return {**data, 'lines': [{**row, 'character_context_id': None} if row['id'] in ids else row for row in data['lines']]}


def score(scorer, data, label=None):
    """``(phrase scores, label scores)`` for the open phrases of ``data`` (only ``label``'s when given)."""
    if scorer is None:
        return None, None
    from app import voice_resolver as vr
    items = [item for item in vr.phrases(data['lines'], data['sequences']) if not item['in_sequence'] and (label is None or item['label'] == label)]
    return scorer.phrases(items), scorer.labels(items)


def outcome(plan, ids):
    """What the plan did with the hidden lines: ``{line_id: (action, character, scope)}``."""
    result = {}
    for unit in plan['applied']:
        for line_id in unit['line_ids']:
            if line_id in ids:
                result[line_id] = ('apply', unit['character'], unit['scope'], unit['why'])
    for group in plan['asked']:
        for item in group['phrases']:
            for line_id in item['line_ids']:
                if line_id in ids:
                    result[line_id] = ('ask', None, 'new voice' if group['new_voice'] else 'ask', group['why'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3')
    parser.add_argument('--project', default='Seishun')
    parser.add_argument('--episodes', default='', help='comma-separated titles (default: every episode of the project)')
    parser.add_argument('--scores', default='stored', choices=('stored', 'computed'))
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import db, voice_resolver as vr
    db.DB_PATH = Path(args.db)
    conn = common.open_copy(Path(args.db))
    project = conn.execute('SELECT id,title FROM projects WHERE title LIKE ?', (f'%{args.project}%',)).fetchone()
    wanted = [title.strip() for title in args.episodes.split(',') if title.strip()]
    episodes = [row for row in conn.execute('SELECT id,title FROM episodes WHERE project_id=? ORDER BY title', (project['id'],))
                if not wanted or row['title'] in wanted]
    names = {row['id']: row['title'] for row in conn.execute("SELECT id,title FROM context_items WHERE kind='character'")}
    labels_report, lines_report, open_report = [], [], []
    totals = collections.Counter()
    started = time.time()
    for episode in episodes:
        data = vr.gather(conn, {'episode_id': episode['id']})
        if not data['lines']:
            continue
        scorer = None
        if args.scores == 'computed':
            scorer = vr.AudioScorer(conn, data)
            if not scorer.usable:
                print(f"{episode['title']}: no audio or no voice samples; stored scores only", flush=True)
                scorer = None
        # 1. every label the user linked, hidden in turn
        for label, truth in sorted(gold_labels(data['lines'], vr.LABEL_MIN).items()):
            ids = {row['id'] for row in data['lines'] if row['speaker_label'] == label}
            test = hidden(data, ids)
            phrase_scores, label_scores = score(scorer, test, label)
            plan = vr.resolve(test, phrase_scores, label_scores)
            got = outcome(plan, ids)
            counts = collections.Counter()
            for line_id in ids:
                action, character, scope, _ = got.get(line_id, ('left', None, '', ''))
                key = 'right' if action == 'apply' and character == truth else 'wrong' if action == 'apply' else action
                counts[key] += 1
                counts[f'{scope}:{key}'] += 1
            totals.update({f'label_{k}': v for k, v in counts.items() if ':' not in k})
            unit = next((u for u in plan['applied'] if u['label'] == label and u['scope'] == 'label'), None)
            group = next((g for g in plan['asked'] if g['label'] == label), None)
            label_scores = (label_scores or {}).get(label) or {}
            top = sorted(label_scores.items(), key=lambda kv: -kv[1])[:2]
            labels_report.append({'episode': episode['title'], 'label': label, 'truth': names.get(truth), 'lines': len(ids), 'counts': dict(counts),
                                  'label_decision': (names.get(unit['character']), unit['why']) if unit else None,
                                  'asked_why': vr.named(group['why'], names) if group else None,
                                  'label_voice': [[names.get(c, c), round(s, 3)] for c, s in top]})
            print(f"{episode['title']} {label} ({names.get(truth)}, {len(ids)} lines): {dict(counts)} · voice {labels_report[-1]['label_voice']}"
                  f" · {labels_report[-1]['label_decision'] or labels_report[-1]['asked_why']}", flush=True)
        # 2. lines a person set, hidden one at a time (stored scores: one line is too little audio to embed reliably on its own)
        for line_id, truth in gold_lines(conn, episode['id']).items():
            plan = vr.resolve(hidden(data, {line_id}))
            action, character, scope, why = outcome(plan, {line_id}).get(line_id, ('left', None, '', ''))
            key = 'right' if action == 'apply' and character == truth else 'wrong' if action == 'apply' else action
            totals[f'line_{key}'] += 1
            lines_report.append({'episode': episode['title'], 'line_id': line_id, 'truth': names.get(truth), 'result': key,
                                 'character': names.get(character), 'scope': scope, 'why': why})
        # 3. the real open voices: what a run would do now
        phrase_scores, label_scores = score(scorer, data)
        plan = vr.resolve(data, phrase_scores, label_scores)
        for unit in plan['applied']:
            open_report.append({'episode': episode['title'], 'action': 'apply', 'label': unit['label'], 'character': names.get(unit['character']),
                                'scope': unit['scope'], 'lines': len(unit['line_ids']), 'why': unit['why'],
                                'label_voice': [[names.get(c, c), s] for c, s in (unit.get('detail') or {}).get('voice_top') or []]})
        for group in plan['asked']:
            open_report.append({'episode': episode['title'], 'action': 'ask', 'label': group['label'], 'lines': sum(len(i['line_ids']) for i in group['phrases']),
                                'candidates': [names.get(c, c) for c in group['candidates']], 'new_voice': group['new_voice'], 'why': vr.named(group['why'], names),
                                'label_voice': [[names.get(c, c), round(s, 3)] for c, s in sorted(((label_scores or {}).get(group['label']) or {}).items(), key=lambda kv: -kv[1])[:3]]})
        print(f"{episode['title']}: open plan {[(item['action'], item['label'], item.get('character') or item.get('candidates'), item['lines']) for item in open_report if item['episode'] == episode['title']]}", flush=True)

    def rate(prefix):
        right, wrong = totals[f'{prefix}_right'], totals[f'{prefix}_wrong']
        return f"{right}/{right + wrong} ({right / (right + wrong):.0%})" if right + wrong else '—'

    md = [f"Project {project['title']}, voice scores **{args.scores}**, {time.time() - started:.0f} s.", '',
          '## Hidden labels (the user linked them label by label)', '',
          f"Applied correctly: **{rate('label')}** of the applied lines; asked {totals['label_ask']}, left {totals['label_left']}.", '',
          '| episode | label | truth | lines | right | wrong | asked | label decision / why asked | label voice |', '|---|---|---|---|---|---|---|---|---|']
    for item in labels_report:
        counts = item['counts']
        decision = f"{item['label_decision'][0]} ({item['label_decision'][1]})" if item['label_decision'] else (item['asked_why'] or '')
        md.append(f"| {item['episode']} | {item['label']} | {item['truth']} | {item['lines']} | {counts.get('right', 0)} | {counts.get('wrong', 0)} | "
                  f"{counts.get('ask', 0)} | {decision} | {', '.join(f'{n} {s}' for n, s in item['label_voice'])} |")
    md += ['', '## Lines a person set, hidden one at a time (stored scores)', '',
           f"Applied correctly: **{rate('line')}**; asked {totals['line_ask']}, left {totals['line_left']}.", '']
    for item in lines_report:
        if item['result'] == 'wrong':
            md.append(f"- wrong: {item['episode']} {item['line_id'][:8]}: truth {item['truth']}, gave {item['character']} ({item['why']})")
    md += ['', '## What a run would do with the open voices now', '']
    for item in open_report:
        if item['action'] == 'apply':
            md.append(f"- {item['episode']} {item['label'] or '(no label)'}: **give {item['lines']} line(s) to {item['character']}** ({item['scope']}: {item['why']})"
                      + (f"; label voice {item['label_voice']}" if item.get('label_voice') else ''))
        else:
            md.append(f"- {item['episode']} {item['label'] or '(no label)'}: ask about {item['lines']} line(s){' (may be new)' if item['new_voice'] else ''}: "
                      f"{item['why']}; candidates {', '.join(item['candidates']) or 'none'}; label voice {item['label_voice']}")
    path = common.write_report('voices', {'scores': args.scores, 'totals': dict(totals), 'labels': labels_report, 'lines': lines_report, 'open': open_report},
                               '\n'.join(md), Path(args.out))
    print(path)


if __name__ == '__main__':
    main()
