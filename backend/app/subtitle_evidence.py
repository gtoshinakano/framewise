"""Subtitle-only OCR evidence and conservative temporal consolidation."""
import re
import unicodedata


def _overlaps(a, b):
    """Require the majority of an OCR box to lie in a detected subtitle region."""
    if len(a) != 4 or len(b) != 4:
        return False
    left, top, right, bottom = a
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return False
    overlap = max(0, min(right, b[2]) - max(left, b[0])) * max(0, min(bottom, b[3]) - max(top, b[1]))
    return overlap / (width * height) >= 0.5


def subtitle_readings(detections, ocr):
    """Join OCR words in reading order inside each predicted subtitle region."""
    boxes = [item['coords'] for items in detections.values() for item in items
             if item.get('type') == 'box' and isinstance(item.get('coords'), list) and len(item['coords']) == 4]
    groups = [[] for _ in boxes]
    for text, items in ocr.items():
        if not text.strip():
            continue
        for item in items:
            coords = item.get('coords')
            if item.get('type') != 'box' or not isinstance(coords, list) or len(coords) != 4:
                continue
            region = next((index for index, box in enumerate(boxes) if _overlaps(coords, box)), None)
            if region is not None:
                groups[region].append({'text': text.strip(), 'box': coords})
    readings = []
    for words in groups:
        lines = []
        for word in sorted(words, key=lambda item: (item['box'][1], item['box'][0])):
            box = word['box']
            center, height = (box[1] + box[3]) / 2, box[3] - box[1]
            line = next((line for line in lines if abs(center - line['center']) <= min(height, line['height']) * .5), None)
            if line is None:
                line = {'center': center, 'height': height, 'words': []}
                lines.append(line)
            line['words'].append(word)
        if lines:
            text = '\n'.join(' '.join(word['text'] for word in sorted(line['words'], key=lambda item: item['box'][0])) for line in sorted(lines, key=lambda line: line['center']))
            coords = [min(word['box'][0] for word in words), min(word['box'][1] for word in words),
                      max(word['box'][2] for word in words), max(word['box'][3] for word in words)]
            readings.append({'text': text, 'box': coords})
    return readings


def _key(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip().casefold()


def consolidate(samples):
    """Group consecutive identical subtitle readings; times are observations, not inferred boundaries.

    The caller must provide every selected frame, including frames with no readings;
    an intervening empty frame closes the current observation run.
    """
    events, previous = [], {}
    for sample in sorted(samples, key=lambda item: item['time']):
        active = {}
        for reading in sample['readings']:
            key = _key(reading['text'])
            if not key:
                continue
            event = active.get(key) or previous.get(key)
            if event is None:
                event = {'text': reading['text'], 'language': 'unknown', 'first_observed': sample['time'],
                         'last_observed': sample['time'], 'observations': []}
                events.append(event)
            event['last_observed'] = sample['time']
            event['observations'].append({'time': sample['time'], 'box': reading['box']})
            active[key] = event
        previous = active
    return events
