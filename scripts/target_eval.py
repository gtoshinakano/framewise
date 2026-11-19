#!/usr/bin/env python3
"""Read-only replay of Target frames on cuts with saved user corrections.

Run `inventory` with stdlib Python, and `capture` in the inference environment with
the existing weights mounted. Capture uses CPU sessions and appends one JSON line
per cut, so a later invocation can resume. Saved automatic boxes are evidence for
inspection only; only hand labels can establish precision or correction savings.
"""
import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


def readonly(path):
    conn = sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def records(conn, sql, args=()):
    return [dict(row) for row in conn.execute(sql, args)]


def inventory(conn, cut_numbers=()):
    """Snapshot the surviving edits and their original frame selection."""
    has_log = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='target_corrections'").fetchone()
    logged = '''OR EXISTS (SELECT 1 FROM target_corrections h WHERE h.run_id=r.id AND h.action!='mirror_migrated')''' if has_log else ''
    runs = records(conn, f'''SELECT r.id AS run_id, r.selection_key, r.frames_json,
        r.model_revision, c.id, c.cut_index + 1 AS number, c.start_time, c.end_time,
        e.id AS episode_id, e.title AS episode, e.project_id, e.original_ext,
        k.result_json AS keyframe_result
        FROM target_detection_runs r JOIN cuts c ON c.id=r.cut_id
        JOIN episodes e ON e.id=c.episode_id
        LEFT JOIN keyframe_results k ON k.cache_key=r.selection_key
        WHERE c.active=1 AND c.start_time=r.cut_start AND c.end_time=r.cut_end AND (
            EXISTS (SELECT 1 FROM target_detections d WHERE d.run_id=r.id AND d.source='manual') OR
            EXISTS (SELECT 1 FROM target_slots s WHERE s.run_id=r.id AND s.label_source='manual') {logged})
        ORDER BY e.title, e.id, c.cut_index, r.rowid''')
    for run in runs:
        if cut_numbers and run['number'] not in cut_numbers:
            continue
        run['slots'] = records(conn, '''SELECT s.slot, s.origin, s.character_context_id, s.label_source, c.title AS name
            FROM target_slots s LEFT JOIN context_items c ON c.id=s.character_context_id
            WHERE s.run_id=? ORDER BY s.slot''', (run['run_id'],))
        by_frame = {}
        for row in records(conn, '''SELECT frame_ordinal, slot, source, box_json FROM target_detections
                WHERE run_id=? ORDER BY frame_ordinal, slot''', (run['run_id'],)):
            ordinal = row.pop('frame_ordinal')
            row['box'] = json.loads(row.pop('box_json'))
            by_frame.setdefault(ordinal, []).append(row)
        run['frames'] = [{**frame, 'saved': by_frame.get(frame['ordinal'], [])} for frame in json.loads(run.pop('frames_json'))]
        actions = records(conn, '''SELECT action,before_json,after_json,details_json FROM target_corrections
            WHERE cut_id=? AND selection_key=? ORDER BY id''', (run['id'], run['selection_key'])) if has_log else []
        run['corrections'] = [{name.removesuffix('_json'): json.loads(value) if name.endswith('_json') else value
                               for name, value in action.items()} for action in actions]
        keyframes = json.loads(run.pop('keyframe_result') or '{}')
        mask = keyframes.get('subtitle_mask') or {}
        run['subtitle_band'] = {k: mask[k] for k in ('top', 'bottom') if k in mask} if mask.get('source') == 'detected' else None
        yield run


def capture_paths(run):
    for i, frame in enumerate(run['frames'], 1):
        yield frame, f"{run['number']}-{run['id']}/F{i}.jpg"


class Capture:
    def __init__(self, conn, images, with_pose):
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/framewise-target-eval-yolo')
        import cv2
        import torch
        from app import anime_models, keyframe_pose

        # Keep CPU replay concurrent with the regular worker without occupying its GPU.
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        self.conn = conn
        self.images = images
        self.detector = anime_models.AniRefDetector(os.environ.get('ANIREF_MODEL_PATH', '/models/aniref/AniRef40000-s-epoch40.pt'), 'cpu')
        self.model = anime_models.CcipEmbedder(os.environ.get('CCIP_MODEL_DIR', '/models/ccip'),
                                               os.environ.get('CCIP_VARIANT') or anime_models.CCIP_DEFAULT_VARIANT,
                                               provider='CPUExecutionProvider')
        self.pose = keyframe_pose.DWPose(threads=3) if with_pose else None
        self.galleries = {}

    def gallery(self, run):
        from app import anime_gallery, anime_matching

        project = run['project_id']
        if project not in self.galleries:
            # Same source eligibility as the app. Read the cache directly to avoid updating gallery verdicts.
            rows = records(self.conn, '''SELECT c.id AS cut_id, d.character_context_id AS character_id,
                v.dimensions, v.embedding FROM character_detection_embeddings v
                JOIN character_detections d ON d.id=v.detection_id
                JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id
                JOIN episodes e ON e.id=c.episode_id JOIN context_items ci ON ci.id=d.character_context_id
                WHERE v.model_id=? AND v.model_revision=? AND v.preprocessing=? AND e.project_id=?
                AND ci.kind='character' AND ci.project_id=e.project_id AND d.label_source='manual'
                AND (d.recognition_use=1 OR (d.recognition_use IS NULL AND d.gallery_status='used'))
                AND (r.rowid=(SELECT max(latest.rowid) FROM character_detection_runs latest WHERE latest.cut_id=c.id)
                    OR EXISTS (SELECT 1 FROM character_image_references x WHERE x.detection_id=d.id))''',
                (self.model.model_id, self.model.revision, self.model.preprocessing, project))
            self.galleries[project] = [dict(cut_id=row['cut_id'], character_id=row['character_id'],
                                            vector=anime_matching.unpack_embedding(row['embedding'], row['dimensions'])) for row in rows]
        return anime_gallery.Gallery([row for row in self.galleries[project] if row['cut_id'] != run['id']])

    def run(self, run, data_dir):
        import numpy as np
        from app import anime_gallery, keyframe_camera, keyframe_pose, target_characters

        source = data_dir / 'projects' / run['project_id'] / 'episodes' / run['episode_id'] / f"original{run['original_ext']}"
        frames = [{'source_frame_index': frame['ordinal'], 'timestamp_seconds': frame['time']} for frame in run['frames']]
        images = target_characters._extract_frames(source, run, frames)
        for (frame, rel), image in zip(capture_paths(run), images, strict=True):
            if list(image.size) != frame['image_size']:
                raise ValueError(f"Cut {run['number']} frame {frame['ordinal']} dimensions changed")
            path = self.images / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, quality=93)
            frame['image'] = rel
        pending = [image for frame, image in zip(run['frames'], images) if not frame.get('mirror')]
        results = iter(self.detector.detect(pending))
        gallery = self.gallery(run)
        crops, owners = [], []
        for frame, image in zip(run['frames'], images):
            boxes = [row['box'] for row in frame['saved']] if frame.get('mirror') else next(results)
            frame['raw'] = [dict(box=box, rank=i, small=bool(anime_gallery.box_problem(box, image.size)), vector=None, gallery={})
                            for i, box in enumerate(boxes)]
            for item in frame['raw']:
                if not item['small'] or frame.get('mirror'):
                    owners.append(item)
                    crops.append(anime_gallery.open_jpeg(anime_gallery.crop_jpeg(image, item['box'])))
        for owner, vector in zip(owners, self.model.embed(crops, batch_size=4), strict=True):
            owner.update(vector=vector, gallery=dict(gallery.rank(vector)))
        bgrs = [np.asarray(image)[:, :, ::-1].copy() for image in images]
        cameras = keyframe_camera.pair_cameras([keyframe_camera.gray(image) for image in bgrs], run['subtitle_band'],
            [[d['box'] for d in frame['raw']] for frame in run['frames']], [image.size for image in images])
        for frame, bgr, camera in zip(run['frames'], bgrs, [None, *cameras], strict=True):
            frame['camera'] = camera
            frame['skeletons'] = []
            if self.pose and not frame.get('mirror'):
                for box in self.pose.detect(bgr):
                    points = self.pose.keypoints(bgr[:, :, ::-1], box)
                    if keyframe_pose.body_usable(points):
                        frame['skeletons'].append({'box': box[:4], 'score': box[4], 'points': points[:17]})
        run['capture'] = dict(detector_revision=self.detector.revision, embedding_model=self.model.model_id,
                              embedding_revision=self.model.revision, same_person=self.model.defaults['track'],
                              gallery_examples=len(gallery), pose=bool(self.pose))
        return run


def read_jsonl(path):
    with path.open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def replay(run, mode, no_match=None, *, cast_locked=False, images=None):
    from app import frame_confidence, keyframe_pose, target_candidates, target_tracking as tracking

    first = run['frames'][0]
    slot_rows = {slot['slot']: slot for slot in run['slots']}
    slots = []
    for box in first['saved']:
        slot = slot_rows[box['slot']]
        # Older runs predate the mirror marker. Their inherited F1 slots still record the first-frame order.
        if first.get('mirror') or slot['origin'] == 'first_frame':
            raw = max(first['raw'], key=lambda item: tracking.iou(item['box'], box['box']), default={})
            slots.append(tracking.Slot(box['slot'], 'first_frame', slot['character_context_id'], box=box['box'], vector=raw.get('vector')))
    first_count = len(slots)
    same_person = run['capture']['same_person']
    no_match = tracking.UNMATCHED_COST if no_match is None else no_match
    previous, verified = None, set(slot.number for slot in slots)
    if images:
        import cv2
        from app import keyframe_camera
        cv2.setNumThreads(2)
    result = []
    for frame in run['frames']:
        gray = keyframe_camera.gray(cv2.imread(str(images / frame['image']))) if images else None
        if frame.get('mirror'):
            result.append({'boxes': frame['saved'], 'unmatched': [], 'merged': [], 'flags': {}, 'score': 1.})
            previous = gray
            continue
        candidates = [{'raw_index': i, **raw} for i, raw in enumerate(frame['raw'])]
        split = []
        if mode == 'legacy':
            predicted = {slot.number: keyframe_pose.predict(slot.box, frame['camera'], *frame['image_size']) for slot in slots if slot.box}
            placed = tracking.assign_frame(slots, candidates, same_person)
            used = {item['detection'] for item in placed}
            unmatched = [i for i in range(len(candidates)) if i not in used]
        else:
            candidates, predicted, split = target_candidates.prepare(candidates, slots, frame['image_size'], frame['camera'], run['subtitle_band'],
                frame['skeletons'], split=mode in ('split', 'flow'), previous=previous, current=gray, verified=verified)
            placed, unmatched = tracking.assign_global(slots, candidates, same_person, predicted, locked=cast_locked, no_match=no_match)
            seen = {entry['slot'] for entry in placed}
            verified = seen
            for slot in slots:
                if slot.number not in seen and slot.number in predicted:
                    slot.box = predicted[slot.number]
        confidence = frame_confidence.review(placed, candidates, unmatched, slots, first_count, predicted, frame['image_size'], split, no_match=no_match)
        boxes = [{**{key: value for key, value in candidates[entry['detection']].items() if key not in ('vector', 'gallery')},
                  'slot': entry['slot'], 'scores': entry.get('scores')} for entry in placed]
        result.append({**confidence, 'boxes': boxes, 'unmatched': sorted({candidates[i]['raw_index'] for i in unmatched
                                                                          if candidates[i]['raw_index'] is not None}), 'merged': split})
        previous = gray
    return result


def match_boxes(predicted, truth, threshold=.50):
    """Maximum number of same-slot, high-overlap matches, with IoU as tie-breaker."""
    from app.target_tracking import hungarian, iou

    weights = [[iou(p['box'], t['box']) if p['slot'] == t['slot'] else 0. for p in predicted] for t in truth]
    bonus = len(truth) + 1
    peers = hungarian([[-(bonus + score) if score >= threshold else 1. for score in row] + [0.] * len(truth) for row in weights])
    return {index: peer for index, peer in enumerate(peers) if peer < len(predicted) and weights[index][peer] >= threshold}


def precision_recall(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def equivalent_late_slots(frames, truth, first_ordinal):
    """A new person's colour is arbitrary, but one colour must stay one person across the cut."""
    from app.target_tracking import hungarian, iou

    opening = {box['slot'] for box in truth.get(str(first_ordinal), {}).get('boxes', [])}
    expected = sorted({box['slot'] for data in truth.values() for box in data['boxes']} - opening)
    observed = sorted({box['slot'] for result in frames.values() for box in result['boxes']} - opening)
    aggregates = {slot: dict.fromkeys(expected, 0.) for slot in observed}
    for ordinal, result in frames.items():
        targets = truth.get(str(ordinal), {}).get('boxes', [])
        for box in result['boxes']:
            if box['slot'] not in aggregates:
                continue
            for target in targets:
                if target['slot'] in expected:
                    overlap = iou(box['box'], target['box'])
                    if overlap >= .50:
                        aggregates[box['slot']][target['slot']] += overlap
    costs = [[-aggregates[slot][target] for target in expected] + [0.] * len(observed) for slot in observed]
    return {slot: expected[index] for slot, index in zip(observed, hungarian(costs), strict=True)
            if index < len(expected) and aggregates[slot][expected[index]] > 0.}


def corrections(proposed, truth):
    """Edits needed to turn predictions into the reviewed frame, given its chosen colours."""
    from app.target_tracking import iou

    same = match_boxes(proposed, truth)
    leftover = [p for i, p in enumerate(proposed) if i not in same.values()]
    absent = [t for i, t in enumerate(truth) if i not in same]
    # Recolouring or moving an existing box can cover one false positive and one miss in a single edit.
    from app.target_tracking import hungarian
    costs = [[-1. if (p['slot'] == t['slot'] or iou(p['box'], t['box']) >= .50) else 1. for p in leftover] + [0.] * len(absent)
             for t in absent]
    changed = sum(col < len(leftover) and costs[i][col] < 0 for i, col in enumerate(hungarian(costs)))
    return len(leftover) + len(absent) - changed


def compare(runs, labels=None, no_match=None, *, images=None, candidate_labels=None):
    """Track saved manual positives and optionally score independently hand-reviewed complete frames."""
    from collections import Counter

    outcomes, totals = [], {'legacy': Counter(), 'global': Counter(), 'split': Counter()}
    split_counts, action_counts = Counter(), Counter()
    if images:
        totals['flow'] = Counter()
    for run in runs:
        by_ordinal = {f['ordinal']: f for f in run['frames']}
        truth = ((labels or {}).get(run['id']) or
                 {str(frame['ordinal']): {'boxes': frame['saved']} for frame in run['frames'] if (frame.get('review') or {}).get('reviewed')})
        action_counts.update(event['action'] for event in run.get('corrections', ()))
        tested = {'cut_id': run['id'], 'cut': run['number'], 'selection_key': run['selection_key'], 'frames': []}
        modes = {mode: replay(run, mode, no_match, images=images if mode == 'flow' else None) for mode in totals}
        colour_mapping = {mode: equivalent_late_slots(dict(zip(by_ordinal, replayed, strict=True)), truth, run['frames'][0]['ordinal'])
                          for mode, replayed in modes.items()} if truth else {}
        for i, (ordinal, frame) in enumerate(by_ordinal.items()):
            checked = truth.get(str(ordinal))
            known = [row for row in frame['saved'] if row['source'] == 'manual']
            result = {'ordinal': ordinal, 'manual': known}
            for mode, replayed in modes.items():
                proposed = replayed[i]
                result[mode] = proposed
                if frame.get('mirror'):
                    continue
                boxes = proposed['boxes']
                matched = match_boxes(boxes, known)
                totals[mode].update(manual=len(known), recovered=len(matched), flagged=bool(proposed['flags']),
                                    proposed_splits=len(proposed['merged']))
                if checked is not None:
                    mapping = colour_mapping[mode]
                    original_first = {box['slot'] for box in truth.get(str(run['frames'][0]['ordinal']), {}).get('boxes', [])}
                    boxes = [{**box, 'slot': box['slot'] if box['slot'] in original_first else mapping.get(box['slot'], box['slot'] + 100)}
                             for box in boxes]
                    good = len(match_boxes(boxes, checked['boxes']))
                    totals[mode].update(tp=good, fp=len(boxes) - good, fn=len(checked['boxes']) - good, reviewed_frames=1,
                                        corrections=corrections(boxes, checked['boxes']))
            tested['frames'].append(result)
            candidates = (candidate_labels or {}).get(run['id'])
            if candidates and not frame.get('mirror'):
                labels_here = candidates['raw'][i]
                if len(labels_here) != len(frame['raw']):
                    raise ValueError(f"Cut {run['number']} raw detection count no longer matches the hand labels")
                gold = {j for j, label in enumerate(labels_here) if label == 'M'}
                actual = {item['raw_index'] for item in result['split']['merged']}
                split_counts.update(tp=len(gold & actual), fp=len(actual - gold), fn=len(gold - actual))
        outcomes.append(tested)
    return {'summary': {mode: {**counts, **precision_recall(counts['tp'], counts['fp'], counts['fn'])} for mode, counts in totals.items()},
            'splitter': precision_recall(split_counts['tp'], split_counts['fp'], split_counts['fn']), 'user_actions': dict(action_counts), 'cuts': outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'capture', 'replay'])
    parser.add_argument('--db', type=Path, default=Path(os.environ.get('DATA_DIR', '/data')) / 'app.sqlite3')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--cuts', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--without-pose', action='store_true')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--candidate-labels', type=Path)
    parser.add_argument('--no-match', type=float)
    args = parser.parse_args()
    if args.command == 'capture' and not (args.output and args.images):
        parser.error('capture requires --output and --images')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'replay':
        if not args.capture:
            parser.error('replay requires --capture')
        runs = [run for run in read_jsonl(args.capture) if not args.cuts or run['number'] in args.cuts]
        result = compare(runs, json.loads(args.labels.read_text()) if args.labels else None, args.no_match, images=args.images,
                         candidate_labels=json.loads(args.candidate_labels.read_text()) if args.candidate_labels else None)
        if args.output:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: result[key] for key in ('summary', 'splitter', 'user_actions')}, indent=2))
        return
    with closing(readonly(args.db)) as conn:
        runs = list(inventory(conn, args.cuts))
        if args.command == 'inventory':
            for run in runs:
                print(json.dumps({'episode': run['episode'], 'cut': run['number'], 'frames': len(run['frames']),
                                  'manual_boxes': sum(box['source'] == 'manual' for f in run['frames'] for box in f['saved']),
                                  'manual_labels': sum(slot['label_source'] == 'manual' for slot in run['slots'])}))
            return
        done = {run['run_id'] for run in read_jsonl(args.output)} if args.output.exists() else set()
        if any(run['run_id'] not in done for run in runs):
            capture = Capture(conn, args.images, not args.without_pose)
            with args.output.open('a') as out:
                for run in runs:
                    if run['run_id'] in done:
                        continue
                    data = capture.run(run, args.data_dir or args.db.parent)
                    print(json.dumps(data, separators=(',', ':')), file=out, flush=True)
                    print(f"Captured {run['episode']} cut {run['number']} ({len(run['frames'])} frames)", file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
