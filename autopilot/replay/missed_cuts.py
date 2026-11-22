"""Replay of the missed-cut checker (M2) against the user's final cut timelines.

For each episode: the episode-wide detection is replayed on the proxy's scene scores at the episode's threshold, which
gives the cuts the user started from. Every boundary of the user's current timeline that this detection did not find
is a *missed cut* the user added (targeted re-detection, a manual split or a frame step). The checker runs on each
replayed cut with the answers hidden, and is scored:

* **apply**: each boundary it would split at is right when a missed cut lies within ``TOLERANCE`` of it, else a false
  split (the target is 95% precision, HARNESS.md D12).
* **grey**: peaks it would send to the second reviewer, and how many of them are real cuts (what Claude would see).
* **missed**: missed cuts found by neither; those whose scene score is under ``FLOOR`` are reported apart, because no
  scene measure can find them (the user split them by eye, e.g. inside a dissolve).

Episodes the user never went through have unfixed misses that count here as false splits; the report is per episode
so they can be told apart. ``--sweep`` prints the precision of other ``apply`` settings from the stored features.

    autopilot/replay/run.sh missed_cuts --project "Seishun" [--episodes "Episode 2,Episode 7"] [--sweep]
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TOLERANCE = 0.1
AUDIT = Path(__file__).resolve().parent / 'missed_cuts_audit.json'


def audited():
    """Real cuts the user's timeline lacks, found by checking the strips by eye (missed_cuts_audit.json)."""
    try:
        return {title: sorted(times) for title, times in json.loads(AUDIT.read_text())['real_cuts'].items()}
    except (OSError, ValueError, KeyError):
        return {}


def detect(times, scores, threshold, duration):
    """media.detect_cuts on stored scores: the starts after 0 where a frame scores above the threshold."""
    starts = [0.0]
    for moment, score in zip(times, scores):
        if score > threshold and starts[-1] + 0.15 < moment < duration - 0.15:
            starts.append(moment)
    return starts


def near(value, ordered, tolerance):
    i = bisect.bisect_left(ordered, value)
    return any(abs(value - ordered[j]) <= tolerance for j in (i - 1, i) if 0 <= j < len(ordered))


def episode_replay(missed_cuts, conn, episode, data_dir, limit=None, audit=()):
    proxy = Path(data_dir) / 'projects' / episode['project_id'] / 'episodes' / episode['id'] / 'proxy.mp4'
    if not proxy.is_file():
        return {'title': episode['title'], 'skipped': 'no proxy'}
    settings = json.loads(episode['settings_json'] or '{}')
    threshold = settings.get('scene_threshold', 0.32)
    times, scores = missed_cuts.scene_scores(proxy)
    duration = episode['duration'] or times[-1]
    cuts = conn.execute('SELECT start_time,end_time,analysis_enabled FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index', (episode['id'],)).fetchall()
    truth = sorted(row['start_time'] for row in cuts[1:])
    off = [(row['start_time'], row['end_time']) for row in cuts if not row['analysis_enabled']]

    def analysis_off(start, end):
        """The job skips cuts with analysis off; a replayed cut counts as off when the user's cut at its middle is."""
        middle = (start + end) / 2
        return any(left <= middle < right for left, right in off)
    replayed = detect(times, scores, threshold, duration)
    missed = [moment for moment in truth if not near(moment, replayed[1:], 0.06)]
    size = missed_cuts._dimensions(proxy)
    segments = [segment for segment in list(zip(replayed, replayed[1:] + [duration]))[:limit] if not analysis_off(*segment)]
    peaks, splits, greys, outcomes = [], [], [], {'apply': 0, 'grey': 0, 'clean': 0}
    for start, end in segments:
        check = missed_cuts.check_cut(proxy, times, scores, start, end, size=size)
        outcomes[check['verdict']] += 1
        for moment in check['split_at']:
            splits.append({'time': moment, 'cut': [round(start, 3), round(end, 3)], 'right': near(moment, missed, TOLERANCE),
                           'audited': near(moment, list(audit), TOLERANCE)})
        for moment in check['grey']:
            greys.append({'time': moment, 'right': near(moment, missed, TOLERANCE)})
        for peak in check['peaks']:
            picture = peak.get('picture') or {}
            camera = picture.get('camera') or {}
            peaks.append({'time': peak['time'], 'score': peak['score'], 'isolation': peak['isolation'], 'verdict': peak['verdict'],
                          'correlation': picture.get('correlation'), 'flash': picture.get('flash'), 'move': camera.get('move'),
                          'sharpness': picture.get('sharpness'),
                          'inliers': camera.get('inliers'), 'ratio': camera.get('ratio'),
                          'real': near(peak['time'], missed, TOLERANCE) or near(peak['time'], list(audit), TOLERANCE)})
    in_range = [moment for moment in missed if (not limit or moment < segments[-1][1]) and any(start < moment < end for start, end in segments)]
    found = [moment for moment in in_range if near(moment, [split['time'] for split in splits], TOLERANCE)]
    sent = [moment for moment in in_range if moment not in found and near(moment, sorted(grey['time'] for grey in greys), TOLERANCE)]
    unscored = []
    for moment in in_range:
        i = bisect.bisect_left(times, moment - 1e-4)
        best = max(scores[j] for j in range(max(0, i - 1), min(len(scores), i + 2)))
        if best < missed_cuts.FLOOR:
            unscored.append(round(moment, 3))
    return {'title': episode['title'], 'episode_id': episode['id'], 'threshold': threshold, 'cuts_replayed': len(segments),
            'truth_boundaries': len(truth), 'missed_cuts': len(in_range), 'outcomes': outcomes,
            'apply': {'splits': len(splits), 'right': sum(split['right'] for split in splits), 'audited': sum(split['audited'] and not split['right'] for split in splits),
                      'wrong': [split for split in splits if not split['right'] and not split['audited']],
                      'unverified': [split for split in splits if not split['right'] and not split['audited']]},
            'grey': {'peaks': len(greys), 'real': sum(grey['right'] for grey in greys)},
            'found': len(found), 'sent_to_review': len(sent), 'not_found': len(in_range) - len(found) - len(sent),
            'not_a_scene_peak': unscored, 'peaks': peaks}


def sweep(peaks):
    """Peak-level precision and recall of other apply settings (picture checks as stored). A peak is real when the user's
    timeline or the audit has a cut there; the audit only covers splits the current rule made, so looser settings are
    undercounted."""
    lines = ['| score ≥ | isolation ≥ | correlation < | camera fit rejects | right | wrong | precision | of real peaks |', '|---|---|---|---|---|---|---|---|']
    real_total = sum(peak['real'] for peak in peaks)
    for score in (0.12, 0.15, 0.18, 0.2, 0.25):
        for alone in (2, 2.5, 3, 4, 6):
            for correlation in (0.9, 0.97, 1.01):
                for camera in (True, False):
                    chosen = [peak for peak in peaks if peak['score'] >= score and peak['isolation'] >= alone and peak['correlation'] is not None
                              and peak['correlation'] < correlation and not peak['flash']
                              and not (camera and peak['move'] not in (None, 'unclear') and (peak['inliers'] or 0) >= 40 and (peak['ratio'] or 0) >= 0.5)]
                    right = sum(peak['real'] for peak in chosen)
                    wrong = len(chosen) - right
                    if right:
                        lines.append(f'| {score} | {alone} | {correlation} | {"yes" if camera else "no"} | {right} | {wrong} | {right / len(chosen):.2f} | {right}/{real_total} |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='/replay/app.sqlite3', help='a copy of the database (never the live file)')
    parser.add_argument('--data', default='/data', help='DATA_DIR with the projects (read-only)')
    parser.add_argument('--cache', default='/replay/cache', help='where scene scores are cached')
    parser.add_argument('--project', default='', help='only episodes of projects whose title contains this')
    parser.add_argument('--episodes', default='', help='comma-separated episode titles')
    parser.add_argument('--limit', type=int, help='only the first N replayed cuts of each episode (a quick run)')
    parser.add_argument('--sweep', action='store_true', help='add a table of other apply settings')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'reports'))
    args = parser.parse_args(argv)
    from app import db, missed_cuts
    db.DATA_DIR = Path(args.cache).parent if Path(args.cache).name == 'cache' else Path(args.cache)
    conn = common.open_copy(Path(args.db))
    wanted = {title.strip() for title in args.episodes.split(',') if title.strip()}
    episodes = [row for row in conn.execute("""SELECT e.*,p.title AS project FROM episodes e JOIN projects p ON p.id=e.project_id
        WHERE p.title LIKE ? ORDER BY e.title""", (f'%{args.project}%',)) if not wanted or row['title'] in wanted]
    results = []
    for episode in episodes:
        result = episode_replay(missed_cuts, conn, episode, args.data, args.limit, audited().get(episode['title'], []))
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('peaks', 'apply')} | ({'apply': {k: v for k, v in result['apply'].items() if k != 'wrong'}} if 'apply' in result else {})), flush=True)
    scored = [result for result in results if 'apply' in result]
    total = {key: sum(result[key] for result in scored) for key in ('missed_cuts', 'found', 'sent_to_review', 'not_found')}
    right = sum(result['apply']['right'] for result in scored)
    audit_right = sum(result['apply']['audited'] for result in scored)
    splits = sum(result['apply']['splits'] for result in scored)
    grey = sum(result['grey']['peaks'] for result in scored)
    grey_real = sum(result['grey']['real'] for result in scored)
    rows = ['| Episode | replayed cuts (analysis on) | missed cuts | applied splits (user / audit / unverified) | found | to review (real/peaks) | not found | not a scene peak |', '|---|---|---|---|---|---|---|---|']
    for result in scored:
        rows.append(f"| {result['title']} | {result['cuts_replayed']} | {result['missed_cuts']} | {result['apply']['splits']} ({result['apply']['right']} / {result['apply']['audited']} / {len(result['apply']['unverified'])}) | {result['found']} | "
                    f"{result['sent_to_review']} ({result['grey']['real']}/{result['grey']['peaks']}) | {result['not_found']} | {len(result['not_a_scene_peak'])} |")
    precision = right / splits if splits else None
    audited_precision = (right + audit_right) / splits if splits else None
    summary = (f"Apply precision against the user's timeline **{right}/{splits}" + (f" = {precision:.2f}" if precision is not None else '') + "**; "
               f"with the audited real cuts **{right + audit_right}/{splits}" + (f" = {audited_precision:.2f}" if audited_precision is not None else '') + f"** (target 0.95). "
               f"Missed cuts found by apply: {total['found']}/{total['missed_cuts']}; sent to review: {total['sent_to_review']}; "
               f"not found: {total['not_found']}. Review load: {grey} peaks, {grey_real} of them real cuts.")
    wrong = [f"- {result['title']} {split['time']:.3f}s (cut {split['cut'][0]}-{split['cut'][1]})" for result in scored for split in result['apply']['unverified']]
    markdown = '\n\n'.join(part for part in (summary, '\n'.join(rows), '## Splits neither the user nor the audit confirms (look at their strips)\n\n' + ('\n'.join(wrong) or 'none'),
                                              '## Settings sweep (peak level)\n\n' + sweep([peak for result in scored for peak in result['peaks']]) if args.sweep else '') if part)
    path = common.write_report('missed_cuts', {'settings': {'apply': missed_cuts.APPLY, 'grey': missed_cuts.GREY, 'continuous': missed_cuts.CONTINUOUS,
                                                            'floor': missed_cuts.FLOOR, 'tolerance': TOLERANCE},
                                               'totals': {**total, 'splits': splits, 'right': right, 'audited': audit_right, 'grey_peaks': grey, 'grey_real': grey_real},
                                               'episodes': results}, markdown, Path(args.out))
    print(summary)
    print(f'report: {path}')


if __name__ == '__main__':
    main()
