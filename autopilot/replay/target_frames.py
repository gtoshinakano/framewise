"""Replay of the target-frame rule (M6, ``character_decisions.decide_frame``) against the user's corrections.

For every target run, every flagged frame is decided as the agent would (with the keyframe states of the cut's current
keyframe analysis). A frame the rule marks *reviewed* is wrong when the user later added a box on that frame for one of
the people the rule said had left (``target_corrections`` action ``add``/``move`` on that run and frame), or when it is
listed in a correction of any kind on that frame. Frames the rule asks about are counted per flag set.

    autopilot/replay/run.sh target_frames
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import character_decisions as cd
    conn = common.open_copy(Path(args.db))
    touched = collections.defaultdict(set)   # run id -> frame ordinals the user corrected
    for row in conn.execute('SELECT run_id,action,before_json,after_json FROM target_corrections'):
        for item in (json.loads(row['before_json'] or '[]') + json.loads(row['after_json'] or '[]')):
            if isinstance(item, dict) and item.get('frame_ordinal') is not None:
                touched[row['run_id']].add(item['frame_ordinal'])
    tally, asked, wrong, samples = collections.Counter(), collections.Counter(), [], []
    for episode in conn.execute('SELECT id,title FROM episodes ORDER BY title').fetchall():
        cuts = cd.episode_cuts(conn, episode['id'])
        for item in cd.target_runs(conn, cuts):
            frames = item['frames']
            if not frames:
                continue
            opening = item['here'].get(frames[0].get('ordinal'), set())
            states = cd.keyframe_states(conn, item['cut']['id'])
            crops = collections.defaultdict(list)
            for crop in json.loads(item['run']['candidates_json'] or '[]'):
                crops[crop.get('frame_ordinal')].append(crop)
            for number, frame in enumerate(frames, 1):
                review = frame.get('review') or {}
                if review.get('reviewed'):
                    # Score the rule on frames the user already looked at too: hide the mark.
                    frame = {**frame, 'review': {**review, 'reviewed': False}}
                decision = cd.decide_frame(frame, opening=opening, here=item['here'].get(frame.get('ordinal'), set()), slots=item['slots'],
                                           states=states.get(frame.get('ordinal'), {}), crops=crops.get(frame.get('ordinal'), []))
                if decision is None:
                    continue
                if decision['action'] == 'reviewed':
                    bad = frame['ordinal'] in touched.get(item['run']['id'], set())
                    tally['wrong' if bad else 'right'] += 1
                    entry = {'episode': episode['title'], 'cut': item['cut']['cut_index'] + 1, 'frame': f'F{number}', 'user_reviewed': bool(review.get('reviewed')),
                             'rule': decision['rule']}
                    tally[f"{decision['rule']}_{'wrong' if bad else 'right'}"] += 1
                    if review.get('reviewed'):
                        tally[f"{decision['rule']}_user_reviewed"] += 1
                    (wrong if bad else samples).append(entry)
                else:
                    asked[','.join(sorted((frame.get('review') or {}).get('flags') or {}))] += 1
    total = tally['right'] + tally['wrong']
    lines = [f"Frames the rule marks reviewed: {total}; the user corrected {tally['wrong']} of them afterwards "
             f"(precision {tally['right'] / total:.3f})." if total else 'The rule marks no frame reviewed on this data.',
             'Per rule: ' + ', '.join(f'{key} {value}' for key, value in sorted(tally.items()) if key not in ('right', 'wrong')) + '.',
             '', '## Asked, by flags', *(f'- {key}: {count}' for key, count in asked.most_common()),
             '', '## Wrong', *(f"- {item['episode']} cut {item['cut']} {item['frame']}" for item in wrong),
             '', '## Right (first 30)', *(f"- {item['episode']} cut {item['cut']} {item['frame']}{' (user also marked it)' if item['user_reviewed'] else ''}" for item in samples[:30])]
    report = common.write_report('target_frames', {'tally': dict(tally), 'asked': dict(asked), 'wrong': wrong, 'right': samples}, '\n'.join(lines), Path(args.out))
    print('\n'.join(lines[:14]))
    print(f'Report: {report}')


if __name__ == '__main__':
    main()
