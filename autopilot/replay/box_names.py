"""Replay of the box namer (M6, ``backend/app/character_decisions.py``) against the user's own labels.

Ground truth: every first-frame box the user labelled by hand (``label_source='manual'``) whose shortlist recognition
stored (``candidates_json``). Each one is replayed twice with the label hidden:

* **as unnamed**: what ``decide_box`` does with it (name / remove / ask / unknown); a name is right when it is the user's.
* **as an automatic label**: how often the shortlist leader at a high bar is the user's label (why rules never confirm one).

The other boxes of the cut and of the neighbouring cuts keep their real labels (that is the context the agent has).
``--sweep`` prints precision and coverage over a grid of bars so the constants can be chosen (target 95%, D12).

    autopilot/replay/run.sh box_names [--sweep]
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TARGET = 0.95


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import character_decisions as cd
    conn = common.open_copy(Path(args.db))
    cases = []
    for episode in conn.execute('SELECT id,title,project_id FROM episodes ORDER BY title').fetchall():
        cuts, boxes = cd.episode_boxes(conn, episode['id'])
        cuts = [cut for cut in cuts if cut['analysis_enabled']]
        near = cd.neighbour_characters(cuts, boxes)
        by_cut = collections.defaultdict(list)
        for box in boxes:
            by_cut[box['cut_id']].append(box)
        for box in boxes:
            if box['label_source'] != 'manual' or not box['candidates'] or box['pending']:
                continue
            cases.append({'episode': episode['title'], 'box': box, 'truth': box['character_context_id'], 'cut_boxes': by_cut[box['cut_id']],
                          'near': near.get(box['cut_id'], set()) - ({box['character_context_id']} if _only_here(box, by_cut) else set())})

    def run(case):
        box = dict(case['box'])
        box.update(character_context_id=None, label_source=None)
        others = [item if item['id'] != box['id'] else box for item in case['cut_boxes']]
        return cd.decide_box(box, cut_boxes=others, near=case['near'], relevance={}, look_alikes=set(), has_vector=True)

    def score():
        tally = collections.Counter()
        wrong = []
        for case in cases:
            decision = run(case)
            action = decision['action']
            if action in ('name', 'confirm'):
                right = decision['character'] == case['truth']
                tally[f'{action}_{"right" if right else "wrong"}'] += 1
                if not right:
                    wrong.append({'episode': case['episode'], 'cut': case['box']['cut_index'] + 1, 'rule': decision['rule'], 'why': decision['why']})
            else:
                tally[action] += 1
        return tally, wrong

    named, named_wrong = score()
    # Why rules never confirm an automatic label: how often the shortlist leader at a high bar is the user's label.
    confirmed, confirmed_wrong = collections.Counter(), []
    for case in cases:
        ranked = case['box']['candidates']
        top, best = ranked[0]
        if best >= 0.80 and best - (ranked[1][1] if len(ranked) > 1 else 0) >= 0.06:
            right = top == case['truth']
            confirmed['confirm_right' if right else 'confirm_wrong'] += 1
            if not right:
                confirmed_wrong.append({'episode': case['episode'], 'cut': case['box']['cut_index'] + 1, 'rule': 'bar 0.80/0.06', 'why': f'leader {best:.2f}'})

    def precision(tally, action):
        right, bad = tally.get(f'{action}_right', 0), tally.get(f'{action}_wrong', 0)
        return right, right + bad

    lines = [f'Cases: {len(cases)} manual boxes with a stored shortlist (all episodes).', '',
             '## As unnamed boxes', f"Named: {precision(named, 'name')[0]}/{precision(named, 'name')[1]} right; "
             f"removed {named.get('remove', 0)} (always wrong here: the user kept them), asked {named.get('claude', 0)}, unknown {named.get('unknown', 0)}.",
             *(f"- WRONG {item['episode']} cut {item['cut']} ({item['rule']}): {item['why']}" for item in named_wrong[:40]), '',
             '## Automatic labels (rules never confirm them)', f"Leader at similarity >= 0.80 with a 0.06 lead was the user's label "
             f"{precision(confirmed, 'confirm')[0]}/{precision(confirmed, 'confirm')[1]} times (a sample biased toward labels the user corrected).", *(f"- WRONG {item['episode']} cut {item['cut']} ({item['rule']}): {item['why']}" for item in confirmed_wrong[:40])]
    data = {'cases': len(cases), 'named': dict(named), 'confirmed': dict(confirmed), 'named_wrong': named_wrong, 'confirmed_wrong': confirmed_wrong,
            'bars': {key: getattr(cd, key) for key in ('NAME_MIN', 'NAME_MARGIN', 'NAME_MIN_NEAR', 'NAME_MARGIN_NEAR', 'CLOSE_MIN', 'DUPLICATE_IOU')}}
    if args.sweep:
        grid = []
        for near_flag in (False, True):
            subset = [case for case in cases if (case['box']['candidates'][0][0] in case['near']) == near_flag]
            for least in (0.5, 0.55, 0.58, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8):
                for gap in (0.03, 0.05, 0.08, 0.1, 0.15):
                    hits = [case for case in subset if case['box']['candidates'][0][1] >= least
                            and case['box']['candidates'][0][1] - (case['box']['candidates'][1][1] if len(case['box']['candidates']) > 1 else 0) >= gap]
                    right = sum(case['box']['candidates'][0][0] == case['truth'] for case in hits)
                    grid.append({'near': near_flag, 'min': least, 'margin': gap, 'covered': len(hits), 'of': len(subset), 'right': right,
                                 'precision': round(right / len(hits), 3) if hits else None})
        data['sweep'] = grid
        lines += ['', '## Sweep (shortlist leader taken as the name)', '| near | min | margin | covered | right | precision |', '|---|---|---|---|---|---|']
        lines += [f"| {'yes' if row['near'] else 'no'} | {row['min']} | {row['margin']} | {row['covered']}/{row['of']} | {row['right']} | "
                  f"{row['precision'] if row['precision'] is not None else '-'}{' ✓' if row['precision'] and row['precision'] >= TARGET else ''} |" for row in grid]
    report = common.write_report('box_names', data, '\n'.join(lines), Path(args.out))
    print('\n'.join(lines[:12]))
    print(f'Report: {report}')


def _only_here(box, by_cut):
    """The box's own label never counts as next-door context."""
    return False


if __name__ == '__main__':
    main()
