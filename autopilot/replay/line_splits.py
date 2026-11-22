"""Replay of the line splitter (M3) against the user's own fixes of lines that ran across a hard cut.

Ground truth, from the revision chains of ``transcript_segments``:

* **splits**: a line the user divided (two ``manual_edit`` revisions with the same parent) that crossed a current hard
  cut. The splitter, run on the parent with the answer hidden, is right when it divides the words at the same place
  (the same words on each side), and its time is compared with the user's.
* **trims**: a line the user only retimed (same text) so that it no longer crosses a cut. Right when the splitter trims
  the same edge to within ``TOLERANCE`` of the user's.

The user's own splits are sometimes workarounds (HARNESS.md D11), so disagreements are listed for a person to judge.
The report also gives what the splitter would do today with every current line that crosses a cut, per episode.

    autopilot/replay/run.sh line_splits [--project Seishun]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TOLERANCE = 0.1
TRIM_TOLERANCE = 0.15   # a trim only has to stop the crossing and keep the voice
SUBTITLE_NEAR = 0.3     # a burned-in subtitle line changing this close to a split agrees with it


def words_left_of(row, at, transcript_parts):
    return sum(1 for part in transcript_parts.parts(dict(row)) if (part['start_time'] + part['end_time']) / 2 < at)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3')
    parser.add_argument('--data', default='/data')
    parser.add_argument('--project', default='')
    parser.add_argument('--reach', type=float, help='override line_splits.REACH')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import line_splits, media, sequence_timeline, transcript_parts
    media.DATA_DIR = Path(args.data)
    if args.reach:
        line_splits.REACH = args.reach
    conn = common.open_copy(Path(args.db))
    episodes = {row['id']: dict(row) for row in conn.execute("""SELECT e.*,p.title AS project FROM episodes e JOIN projects p ON p.id=e.project_id
        WHERE p.title LIKE ?""", (f'%{args.project}%',))}

    def plan_for(row, cut):
        audio = Path(args.data) / 'projects' / episodes[row['episode_id']]['project_id'] / 'episodes' / row['episode_id'] / 'audio.wav'
        start = max(0.0, row['start_time'] - line_splits.CONTEXT)
        envelope = sequence_timeline.audio_envelope(audio, start, row['end_time'] + line_splits.CONTEXT)
        return line_splits.plan_line(row, cut, envelope, start) if envelope else {'action': 'leave', 'why': 'no usable audio'}

    children = collections.defaultdict(list)
    for row in conn.execute("SELECT * FROM transcript_segments WHERE origin='manual_edit' AND parent_id IS NOT NULL"):
        if row['episode_id'] in episodes:
            children[row['parent_id']].append(dict(row))
    starts = {episode_id: [row[0] for row in conn.execute('SELECT start_time FROM cuts WHERE episode_id=? AND active=1 AND cut_index>0 ORDER BY cut_index', (episode_id,))]
              for episode_id in episodes}
    splits, trims = [], []
    for parent_id, kids in children.items():
        parent = conn.execute('SELECT * FROM transcript_segments WHERE id=?', (parent_id,)).fetchone()
        if parent is None:
            continue
        parent = dict(parent)
        crossed = line_splits.crossings(parent, starts[parent['episode_id']])
        if not crossed:
            continue
        title = episodes[parent['episode_id']]['title']
        if len(kids) >= 2:
            kids.sort(key=lambda row: row['start_time'])
            user_at = kids[0]['end_time']
            cut = min(crossed, key=lambda moment: abs(moment - user_at))
            plan = plan_for(parent, cut)
            # The user's own left piece says where they divided the words (their split time can sit a frame inside a word).
            same_words = plan['action'] == 'split' and (''.join(plan['left'].split()) == ''.join(kids[0]['text'].split())
                                                        or words_left_of(parent, plan['at'], transcript_parts) == words_left_of(parent, user_at, transcript_parts))
            splits.append({'episode': title, 'line': [round(parent['start_time'], 3), round(parent['end_time'], 3)], 'text': parent['text'][:60],
                           'user_left': kids[0]['text'],
                           'cut': round(cut, 3), 'user_at': round(user_at, 3), 'plan': plan, 'same_words': same_words,
                           'delta': round(plan['at'] - user_at, 3) if plan['action'] == 'split' else None})
        elif len(kids) == 1 and kids[0]['text'] == parent['text'] and not kids[0]['deleted']:
            kid = kids[0]
            still = line_splits.crossings(kid, starts[parent['episode_id']])
            fixed = [moment for moment in crossed if moment not in still]
            if not fixed:
                continue
            cut = fixed[0]
            plan = plan_for(parent, cut)
            edge = 'end' if abs(kid['end_time'] - parent['end_time']) > abs(kid['start_time'] - parent['start_time']) else 'start'
            right = plan['action'] == 'trim' and abs(plan[edge] - kid[f'{edge}_time']) <= TRIM_TOLERANCE
            trims.append({'episode': title, 'line': [round(parent['start_time'], 3), round(parent['end_time'], 3)], 'text': parent['text'][:60],
                          'cut': round(cut, 3), 'user': [round(kid['start_time'], 3), round(kid['end_time'], 3)], 'plan': plan, 'right': right,
                          'delta': round(plan[edge] - kid[f'{edge}_time'], 3) if plan['action'] == 'trim' else None})
    today = {}
    for episode_id, episode in episodes.items():
        lines = [dict(row) for row in conn.execute('SELECT * FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0', (episode_id,))]
        audio = Path(args.data) / 'projects' / episode['project_id'] / 'episodes' / episode_id / 'audio.wav'
        counts, reasons, samples = collections.Counter(), collections.Counter(), []
        run = conn.execute("SELECT id FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1", (episode_id,)).fetchone()
        changes = sorted({moment for row in conn.execute('SELECT start_time,end_time FROM subtitle_lines WHERE run_id=?', (run['id'],))
                          for moment in (row['start_time'], row['end_time'])}) if run else []
        agree = checked = 0
        for row in lines:
            crossed = line_splits.crossings(row, starts[episode_id])
            if not crossed:
                continue
            start = max(0.0, row['start_time'] - line_splits.CONTEXT)
            envelope = sequence_timeline.audio_envelope(audio, start, row['end_time'] + line_splits.CONTEXT)
            plan = line_splits.plan_line(row, crossed[0], envelope, start) if envelope else {'action': 'leave', 'why': 'no usable audio'}
            counts[plan['action']] += 1
            if plan['action'] == 'split' and changes:
                checked += 1
                agree += any(abs(plan['at'] - moment) <= SUBTITLE_NEAR for moment in changes)
            if plan['action'] == 'leave':
                reasons[plan['why'].split(':')[0]] += 1
            elif len(samples) < 8:
                samples.append({'line': [round(row['start_time'], 2), round(row['end_time'], 2)], 'cut': round(crossed[0], 2), 'text': row['text'][:50],
                                **{key: plan.get(key) for key in ('action', 'at', 'start', 'end', 'why')}})
        if counts:
            today[episode['title']] = {'counts': dict(counts), 'left_because': dict(reasons), 'samples': samples,
                                       'subtitles_agree': [agree, checked]}
    split_right = sum(item['same_words'] for item in splits)
    trim_right = sum(item['right'] for item in trims)
    rows = ['| Episode | crossing lines | split | trim | left | splits a subtitle change agrees with |', '|---|---|---|---|---|---|']
    for title, item in sorted(today.items()):
        counts = item['counts']
        agree, checked = item['subtitles_agree']
        rows.append(f"| {title} | {sum(counts.values())} | {counts.get('split', 0)} | {counts.get('trim', 0)} | {counts.get('leave', 0)} | "
                    + (f"{agree}/{checked} |" if checked else "no subtitles |"))
    lines = [f"User splits reproduced (same words on each side): **{split_right}/{len(splits)}**; user trims reproduced: "
             f"**{trim_right}/{len(trims)}** (same edge within {TRIM_TOLERANCE}s). Reach {line_splits.REACH}s, pause {line_splits.MIN_PAUSE}s.", '## What it would do today', '\n'.join(rows),
             '## User splits', '\n'.join(f"- {'OK ' if item['same_words'] else 'DIFF'} {item['episode']} line {item['line']} cut {item['cut']}: user {item['user_at']}, "
                                         f"plan {item['plan']['action']} {item['plan'].get('at', '')} ({item['plan']['why']}) “{item['text']}” user left “{item['user_left']}”" for item in splits),
             '## User trims', '\n'.join(f"- {'OK ' if item['right'] else 'DIFF'} {item['episode']} line {item['line']} cut {item['cut']}: user {item['user']}, "
                                         f"plan {item['plan']['action']} {item['plan'].get('start', '')}-{item['plan'].get('end', '')} ({item['plan']['why']})"
                                         + (f" Δ {item['delta']:+.3f}s" if item['delta'] is not None else '') for item in trims)]
    path = common.write_report('line_splits', {'settings': {'reach': line_splits.REACH, 'min_pause': line_splits.MIN_PAUSE, 'cross': line_splits.CROSS},
                                               'splits': splits, 'trims': trims, 'today': today}, '\n\n'.join(lines), Path(args.out))
    print(lines[0])
    print(f'report: {path}')


if __name__ == '__main__':
    main()
