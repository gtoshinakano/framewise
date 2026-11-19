#!/usr/bin/env python3
"""Offline Qwen judge and paired motion benchmark for a `target_eval capture`."""
import argparse
import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from target_eval import precision_recall, read_jsonl, replay


def judge_task(run, root):
    from app import contact_sheet, target_judge_contract as contract

    outcomes = replay(run, 'split')
    if not any(frame['flags'] for frame in outcomes):
        return None
    paths, sizes, boxes, cells, unmatched, strip = {}, {}, {}, {}, {}, []
    for i, (frame, outcome) in enumerate(zip(run['frames'], outcomes, strict=True), 1):
        label = f'F{i}'
        paths[label] = root / 'images' / frame['image']
        sizes[label] = frame['image_size']
        for box in outcome['boxes']:
            slot = box['slot']
            boxes[slot, label] = box['box']
            if i > 1:
                cells[f'S{slot + 1}·{label}'] = {**box, 'frame': label}
        merges = {item['raw_index'] for item in outcome['merged']}
        for index in sorted(set(outcome['unmatched']) | merges):
            is_merge = index in merges
            ident = f'{"M" if is_merge else "U"}{i}-{index + 1}'
            candidate = {'id': ident, 'frame': label, 'box': frame['raw'][index]['box'], 'raw_index': index}
            (cells if is_merge else unmatched)[ident] = candidate
            strip.append(candidate)
    slots = sorted({slot for slot, _ in boxes})
    opening = sorted({slot for slot, label in boxes if label == 'F1'})
    image = root / 'sheets' / f"{run['id']}.jpg"
    image.parent.mkdir(exist_ok=True)
    if not image.exists():
        contact_sheet.build(image, paths, sizes, slots, boxes, strip)
    grammar = contract.schema(cells, unmatched, opening)
    return {'cut': run['number'], 'cut_id': run['id'], 'image': str(image.relative_to(root)), 'grammar': grammar,
            'cells': cells, 'unmatched': unmatched, 'evidence': contract.evidence(cells, unmatched, opening),
            'prompt_version': contract.PROMPT_VERSION}


def judge_call(server, task, root):
    from app import keyframe_analysis, target_judge_contract as contract

    image = base64.b64encode((root / task['image']).read_bytes()).decode('ascii')
    messages = [{'role': 'system', 'content': contract.SYSTEM}, {'role': 'user', 'content': [
        {'type': 'text', 'text': task['evidence']}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}}]}]
    reply = keyframe_analysis.chat(server, messages, task['grammar'])
    try:
        answer = contract.validate(json.loads(reply['content']), task['grammar'])
        return {**reply, 'answer': answer}
    except (ValueError, TypeError) as exc:
        return {**reply, 'error': str(exc)}


def motion_sets(runs, archive):
    samples = {row['cut_id']: row for row in json.loads((archive / 'samples.json').read_text())}
    for run in runs:
        sample = samples.get(run['id'])
        if not sample:
            continue
        labels = archive / 'labels' / (sample['id'] + '.json')
        if labels.is_file() and [f['ordinal'] for f in run['frames']] == [f['source_frame_index'] for f in sample['frames']]:
            yield run, json.loads(labels.read_text())


def prepare_motion(run, mode, root, pose):
    import cv2
    from app import keyframe_analysis as analysis, keyframe_rules, keyframe_schema, media

    folder = root / 'motion-frames' / run['id']
    folder.mkdir(parents=True, exist_ok=True)
    frames, paths = [], {}
    for i, row in enumerate(run['frames'], 1):
        label = f'F{i}'
        frames.append({'label': label, 'ordinal': row['ordinal'], 't': round(row['relative_time'], 3), 'time': row['time'],
                       'reason': 'initial_state' if i == 1 else None})
        path = folder / f'{label}.jpg'
        if not path.exists():
            media.scale_image(root / 'images' / row['image'], path, analysis.IMAGE_WIDTH)
        paths[label] = path
    slots = [{**slot, 'character_name': slot['name']} for slot in run['slots']]
    detections = [{**box, 'frame_ordinal': row['ordinal']} for row in run['frames'] for box in row['saved']]
    if mode == 'first_frame':
        names = {slot['slot']: slot for slot in slots}
        detections = [{**box, **names[box['slot']], 'position': 'first'} for box in detections if box['frame_ordinal'] == frames[0]['ordinal']]
        first = {'frames': [{'image_size': run['frames'][0]['image_size'], 'position': 'first'}], 'detections': detections}
        roster, anchors, sizes = analysis.first_frame_roster(first, frames)
    else:
        roster, anchors, sizes = analysis.target_roster({'frames': run['frames'], 'slots': slots, 'detections': detections}, frames)
    images = [cv2.imread(str(paths[frame['label']])) for frame in frames]
    people, pairs = analysis.prepare(images, frames, anchors, sizes, run['subtitle_band'], pose)
    states = analysis.dwpose_states(frames, people, [[image.shape[1], image.shape[0]] for image in images])
    windows = keyframe_schema.make_windows(frames, analysis.WINDOW)
    rules = [keyframe_rules.answer_for_window(states, window, pairs) for window in windows]
    sample = {'duration': run['end_time'] - run['start_time'], 'no_characters': False, 'roster': roster, 'subtitle_band': run['subtitle_band']}
    return sample, windows, paths, rules, states, pairs


def motion_call(server, task):
    from app import keyframe_analysis as analysis

    run, mode, window, sample, paths, rules, states, pairs = task
    images = [base64.b64encode(paths[frame['label']].read_bytes()).decode('ascii') for frame in window['frames']]
    reply = analysis.describe_window(lambda messages, grammar: analysis.chat(server, messages, grammar), sample, window, images)
    answer = reply.pop('answer')
    return {'kind': 'motion', 'cut': run['number'], 'cut_id': run['id'], 'mode': mode, 'window': window['index'],
            **reply, 'answer': analysis.merge_window(answer, rules, states, pairs) if answer is not None else None}


def motion_scores(runs, records, archive):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multiframe_experiment'))
    import schema

    report = {}
    for mode in ('first_frame', 'corrected'):
        tp = fp = fn = answered = pairs = 0
        for run, labels in motion_sets(runs, archive):
            found = {(segment['from'], segment['to']): schema.segment_categories(segment)
                     for row in records if row['kind'] == 'motion' and row['cut_id'] == run['id'] and row['mode'] == mode and row['answer']
                     for segment in row['answer']['segments']}
            for pair in labels['pairs']:
                flags = {flag for flag, value in pair['flags'].items() if value} & set(schema.REFERENCE_FLAGS)
                key = pair['from'], pair['to']
                predicted = found.get(key, set()) & set(schema.REFERENCE_FLAGS)
                tp += len(flags & predicted)
                fp += len(predicted - flags)
                fn += len(flags - predicted)
                answered += key in found
                pairs += 1
        report[mode] = {**precision_recall(tp, fp, fn), 'pairs': pairs, 'answered_pairs': answered}
    return report


def judge_scores(results, labels):
    """Per-class metrics on independently labelled judge responses, counting omissions as misses."""
    from collections import Counter
    from app.target_judge_contract import CELL, UNMATCHED

    predicted = {row['cut_id']: row.get('answer') for row in results if row['kind'] == 'judge' and row.get('answer')}
    counters = {'cells': {name: Counter() for name in CELL}, 'unmatched': {name: Counter() for name in UNMATCHED},
                'cast_changes': {name: Counter() for name in ('no', 'yes')}}
    for cut_id, target in labels.items():
        response = predicted.get(cut_id) or {}
        sections = [(section, truth, response.get(section) or {}) for section, truth in target.items() if section != 'cast_changes']
        if 'cast_changes' in target:
            sections.append(('cast_changes', {'cut': target['cast_changes']}, {'cut': response.get('cast_changes')}))
        for section, actual, guessed in sections:
            for key, value in actual.items():
                answer = guessed.get(key)
                counters[section].setdefault(value, Counter())
                if value == answer:
                    counters[section][value]['tp'] += 1
                else:
                    counters[section][value]['fn'] += 1
                    if answer is not None:
                        counters[section].setdefault(answer, Counter())['fp'] += 1
    return {section: {value: precision_recall(counts['tp'], counts['fp'], counts['fn']) for value, counts in rows.items()}
            for section, rows in counters.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sheets', 'run', 'motion-scores', 'judge-scores'))
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[1] / 'multiframe_experiment/archive/runs/mf-20261118-final')
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--results', type=Path, help='Read archived JSONL results for a scoring command')
    args = parser.parse_args()
    root = args.scratch
    result_path = root / 'vision.jsonl'
    if args.results and args.command not in ('motion-scores', 'judge-scores'):
        parser.error('--results is for scoring commands')
    source = args.results or result_path
    previous = read_jsonl(source) if source.exists() else []
    if args.command == 'judge-scores':
        if not args.labels:
            parser.error('judge-scores requires --labels')
        print(json.dumps(judge_scores(previous, json.loads(args.labels.read_text())), indent=2))
        return
    sample = read_jsonl(root / 'capture.jsonl')
    if args.command == 'motion-scores':
        print(json.dumps(motion_scores(sample, previous, args.archive), indent=2))
        return
    tasks_path = root / 'judge-tasks.json'
    if tasks_path.exists():
        judge = json.loads(tasks_path.read_text())
    else:
        judge = [task for run in sample if (task := judge_task(run, root)) is not None]
        tasks_path.write_text(json.dumps(judge, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.command == 'sheets':
        print(f'Built {len(judge)} contact sheets')
        return

    from app import frame_model, keyframe_analysis, keyframe_prompts, keyframe_pose, visual_models

    def key(row):
        return row['kind'], row['cut_id'], row.get('mode'), row.get('window')

    done = {key(row) for row in previous if row.get('answer') is not None}
    judge = [task for task in judge if ('judge', task['cut_id'], None, None) not in done]
    pose = keyframe_pose.DWPose(threads=4)
    motion = []
    for run, _ in motion_sets(sample, args.archive):
        for mode in ('first_frame', 'corrected'):
            data, windows, paths, rules, states, pairs = prepare_motion(run, mode, root, pose)
            motion.extend((run, mode, window, data, paths, rule, states, pairs)
                          for window, rule in zip(windows, rules, strict=True) if ('motion', run['id'], mode, window['index']) not in done)
    if not (judge or motion):
        return
    print(f'Qwen 3.6: {len(judge)} contact sheets and {len(motion)} paired motion windows', flush=True)
    model = visual_models.MODELS['qwen36-35b']
    with frame_model.frame_server(*model.server_files(), parallel=args.parallel, context_per_slot=keyframe_analysis.CONTEXT_PER_SLOT,
                                 extra_args=model.server_args) as server, ThreadPoolExecutor(args.parallel) as pool, result_path.open('a') as out:
        def save(row):
            print(json.dumps({**row, 'model': model.model_id, 'revision': model.revision}, ensure_ascii=False), file=out, flush=True)
            print(f"{row['kind']} cut {row['cut']}: {'ok' if row.get('answer') else 'failed validation'}", flush=True)

        for task, result in zip(judge, pool.map(lambda task: judge_call(server, task, root), judge), strict=True):
            save({'kind': 'judge', 'cut': task['cut'], 'cut_id': task['cut_id'], 'prompt_version': task['prompt_version'], **result})
        for result in pool.map(lambda task: motion_call(server, task), motion):
            save({**result, 'prompt_version': keyframe_prompts.PROMPT_VERSION})


if __name__ == '__main__':
    main()
