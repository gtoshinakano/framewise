"""Locate people in a cut's boundary frames; detections do not imply identity."""
import math
from pathlib import Path

from . import media


def detect_boundary_people(original, cut, scratch, model, on_frame=None):
    """Run WeVisOne on both original-video boundary frames.

    Boxes use pixel coordinates in the returned image size. Raw model output is
    retained for diagnostics; malformed or empty boxes are not usable labels.
    """
    from PIL import Image
    from wevisone_vl.visualization import parse_predictions

    first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
    frames = []
    for position, timestamp in (('first', first), ('last', last)):
        if on_frame:
            on_frame(position)
        image_path = Path(scratch) / f'{position}.jpg'
        actual = media.extract_cut_frame(Path(original), image_path, timestamp, cut['start_time'], cut['end_time'], width=1280, last=position == 'last')
        with Image.open(image_path) as source:
            image = source.convert('RGB')
            detection = model.predict(image, task='detection', query=['person'], max_new_tokens=512, temperature=0.0, decode_matting=False)
            predictions = parse_predictions(detection.response, image.width, image.height)
            boxes = []
            for values in predictions.values():
                for item in values:
                    coords = item.get('coords')
                    if item.get('type') != 'box' or not isinstance(coords, (list, tuple)) or len(coords) != 4:
                        continue
                    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in coords):
                        continue
                    x1, y1, x2, y2 = coords
                    x1, x2 = max(0, min(image.width, x1)), max(0, min(image.width, x2))
                    y1, y2 = max(0, min(image.height, y1)), max(0, min(image.height, y2))
                    if x1 < x2 and y1 < y2:
                        boxes.append([x1, y1, x2, y2])
            frames.append({'position': position, 'time': actual, 'image_size': [image.width, image.height],
                           'boxes': boxes, 'raw_response': detection.response, 'prompt': detection.prompt})
    return frames


def reference_snapshots(conn, project_id):
    """Freeze references and source coordinates for a queued detection job."""
    import json
    rows = conn.execute('''SELECT r.id,r.character_context_id,ci.title,d.position,d.box_json,
                                 run.frames_json,run.cut_start,run.cut_end,e.id AS episode_id,e.project_id,e.original_ext
                          FROM character_image_references r JOIN context_items ci ON ci.id=r.character_context_id
                          JOIN character_detections d ON d.id=r.detection_id
                          JOIN character_detection_runs run ON run.id=d.run_id JOIN cuts c ON c.id=run.cut_id
                          JOIN episodes e ON e.id=c.episode_id WHERE ci.project_id=? AND ci.kind='character' ORDER BY r.created_at DESC,r.rowid DESC''', (project_id,)).fetchall()
    result, counts = [], {}
    for row in rows:
        item = dict(row)
        cid = item['character_context_id']
        if counts.get(cid, 0) >= 2:
            continue
        frames = json.loads(item.pop('frames_json'))
        frame = next((f for f in frames if f['position'] == item['position']), None)
        if not frame:
            continue
        item['frame'] = frame
        item['box'] = json.loads(item.pop('box_json'))
        result.append(item)
        counts[cid] = counts.get(cid, 0) + 1
    return result


def crop_frame(original, frame, cut_start, cut_end, box, scratch, name):
    """Reconstruct the source frame and return a bounded JPEG crop."""
    from PIL import Image
    import io
    target = Path(scratch) / (name + '.jpg')
    media.extract_cut_frame(Path(original), target, frame['time'], cut_start, cut_end, width=1280, seek_tolerance=.000002)
    with Image.open(target) as image:
        image = image.convert('RGB')
        if list(image.size) != frame['image_size']:
            raise RuntimeError('Reference frame dimensions changed; save a new reference')
        crop = image.crop(tuple(box))
        crop.thumbnail((512, 512))
        output = io.BytesIO()
        crop.save(output, format='JPEG', quality=85)
        return output.getvalue()


def select_analysis_references(references, character_ids, limit=4):
    """Favor one example per selected character before adding extra views."""
    relevant = [ref for ref in references if ref['character_context_id'] in character_ids]
    first, extra, seen = [], [], set()
    for ref in relevant:
        cid = ref['character_context_id']
        if cid in seen:
            extra.append(ref)
        else:
            first.append(ref)
            seen.add(cid)
    return (first + extra)[:limit]
