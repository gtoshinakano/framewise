"""Find where the burned-in subtitle strip is and when its text changes, without a model.

The expensive model should read each distinct subtitle once, not every sampled frame. This module
therefore does the cheap part on the CPU for the whole episode in one decode pass:

1. ``choose_probe_times`` / ``band_from_probes`` turn a few detector boxes into one horizontal strip.
2. ``scan_video`` decodes only that strip as small grayscale frames and keeps the pixels that look like
   subtitle strokes: bright, thin, and still there in the neighbouring frame. Moving highlights fail the
   persistence test; logos and other overlays that are on nearly all the time are removed.
3. Consecutive frames whose stroke pixels agree belong to one subtitle. Each one yields an event with its
   time range and the frame that shows the text most completely, which is the only image a model reads.
"""
import math
import subprocess
import tempfile

import numpy as np

SCAN_WIDTH = 640            # the strip is analysed at this width; thin strokes stay a couple of pixels wide
MAX_SCAN_FPS = 8.           # frame times resolve to 1/8 s; clip duration does not change the cost of decoding
MAX_SCAN_FRAMES = 36000     # caps memory on very long recordings by lowering the scan rate
TOPHAT_SIZE = 7             # strokes narrower than this many pixels survive the white top-hat
TOPHAT_MIN = 48             # stroke brightness above its surroundings
BRIGHT_MIN = 110            # and absolute brightness, which keeps dark-on-dark texture out
STATIC_FREQUENCY = .9       # a pixel that looks like a stroke in this share of frames is an overlay, not text
MIN_TEXT_RATIO = .0025      # share of the strip that must be stroke pixels for a frame to count as text; compression flicker around a logo stays below it
OUTLINE_MAX = 45            # subtitles are usually outlined: a near-black pixel lies within 3 px of every stroke
OUTLINED_MIN_TEXT_RATIO = .002   # with scene clutter removed by the outline test a lower bar still keeps short lines such as "Não."
OUTLINE_SHARE_MIN = .5      # the detector's own subtitle frames must show this share of outlined strokes before the test is used
SAME_TEXT = .72             # two frames show the same line when this much of each one's strokes is within a pixel of the other's
GAP_FRAMES = 2              # a dropout this short inside one line (compression, flicker) does not end it
MIN_EVENT_FRAMES = 3        # about 0.4 s; real subtitles last longer, a flash of changing pixels does not
MAX_SEGMENT_SECONDS = 5.    # a line that stays longer is read again every few seconds as a safeguard
SEGMENT_SECONDS = 4.
DEFAULT_BAND = {'top': .70, 'bottom': 1.}

LANDSCAPE_BAND_PADDING = .02
EXTRA_LINE_ALLOWANCE = .06  # subtitles grow away from the edge they are anchored to when they wrap


def _even(value):
    return max(2, int(value) // 2 * 2)


# -- locating the strip ----------------------------------------------------------------------------

def choose_probe_times(duration, speech, count=16):
    """Times at which a detector looks for the strip: while people speak first, then evenly spaced.

    ``speech`` is a list of ``(start, end)`` ranges (the transcript). Subtitles are most likely to be on screen
    then, which is why this step is meant to follow transcription. Without speech the whole episode is covered.
    """
    if not math.isfinite(duration) or duration <= 0:
        return []
    low, high = duration * .03, duration * .97
    centers = sorted((start + end) / 2 for start, end in speech if end > start and low <= (start + end) / 2 <= high)
    times = []
    if centers:
        wanted = min(count, len(centers))
        times = [centers[min(len(centers) - 1, round((index + .5) * len(centers) / wanted - .5))] for index in range(wanted)]
    missing = count - len(times)
    if missing > 0:
        grid = [low + (high - low) * (index + .5) / missing for index in range(missing)]
        times.extend(time for time in grid if all(abs(time - used) > 1 for used in times))
    return sorted(dict.fromkeys(round(time, 3) for time in times))


def band_from_probes(probes):
    """One horizontal strip (fractions of the frame height) from detector boxes in several frames.

    ``probes`` is a list of ``{'width', 'height', 'boxes'}`` with pixel boxes ``[x0, y0, x1, y1]``. Boxes that
    cannot be a line of text are ignored; the vertical cluster seen in the most frames wins, and the strip
    grows away from the frame edge by one line because wrapped subtitles do.
    """
    sightings = []
    for probe_index, probe in enumerate(probes):
        width, height = probe.get('width') or 0, probe.get('height') or 0
        if width <= 0 or height <= 0:
            continue
        for box in probe.get('boxes') or []:
            if len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
                continue
            x0, y0, x1, y1 = box
            box_width, box_height = (x1 - x0) / width, (y1 - y0) / height
            if box_width < .03 or box_height <= 0 or box_height > .3 or box_width / box_height * width / height < 1.2:
                continue
            sightings.append((((y0 + y1) / 2) / height, max(0., y0 / height), min(1., y1 / height), probe_index))
    if not sightings:
        return {**DEFAULT_BAND, 'source': 'default', 'support': 0, 'probes': len(probes)}
    sightings.sort()
    clusters, current = [], [sightings[0]]
    for item in sightings[1:]:
        if item[0] - current[-1][0] > .06:
            clusters.append(current)
            current = []
        current.append(item)
    clusters.append(current)
    best = max(clusters, key=lambda cluster: (len({item[3] for item in cluster}), np.mean([item[0] for item in cluster])))
    support = len({item[3] for item in best})
    tops, bottoms = sorted(item[1] for item in best), sorted(item[2] for item in best)
    top = tops[int(len(tops) * .1)]
    bottom = bottoms[min(len(bottoms) - 1, int(len(bottoms) * .9))]
    top, bottom = top - LANDSCAPE_BAND_PADDING, bottom + LANDSCAPE_BAND_PADDING
    if (top + bottom) / 2 > .5:
        top -= EXTRA_LINE_ALLOWANCE
    else:
        bottom += EXTRA_LINE_ALLOWANCE
    top, bottom = max(0., top), min(1., bottom)
    if bottom - top < .1:
        middle = (top + bottom) / 2
        top, bottom = max(0., middle - .05), min(1., middle + .05)
    return {'top': round(top, 4), 'bottom': round(bottom, 4), 'source': 'detector', 'support': support, 'probes': len(probes)}


def strip_geometry(source_width, source_height, band, scan_width=SCAN_WIDTH):
    """Pixel rectangle of the strip in the downscaled analysis frame."""
    height = _even(round(source_height * scan_width / source_width))
    top = min(height - 2, max(0, int(band['top'] * height)))
    bottom = min(height, max(top + 2, math.ceil(band['bottom'] * height)))
    return {'width': scan_width, 'frame_height': height, 'y': top, 'height': _even(bottom - top)}


def canvas_box(band, canvas_width, canvas_height):
    """The strip as a pixel box in a frame of ``canvas_width`` x ``canvas_height``."""
    return [0, int(band['top'] * canvas_height), canvas_width, min(canvas_height, math.ceil(band['bottom'] * canvas_height))]


# -- the stroke mask -------------------------------------------------------------------------------

def _tophat_numpy(gray, size):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = size // 2
    out = np.empty_like(gray)
    for index, frame in enumerate(gray):
        padded = np.pad(frame, pad, mode='edge')
        low = sliding_window_view(padded, size, axis=1).min(axis=-1)
        low = sliding_window_view(low, size, axis=0).min(axis=-1)
        padded = np.pad(low, pad, mode='edge')
        opened = sliding_window_view(padded, size, axis=1).max(axis=-1)
        opened = sliding_window_view(opened, size, axis=0).max(axis=-1)
        out[index] = frame - np.minimum(frame, opened)
    return out


def _tophat(gray, size=TOPHAT_SIZE):
    try:
        import cv2
    except ImportError:
        return _tophat_numpy(gray, size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return np.stack([cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel) for frame in gray])


def _local_min_numpy(gray, size):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = size // 2
    out = np.empty_like(gray)
    for index, frame in enumerate(gray):
        padded = np.pad(frame, pad, mode='edge')
        low = sliding_window_view(padded, size, axis=1).min(axis=-1)
        out[index] = sliding_window_view(low, size, axis=0).min(axis=-1)
    return out


def _local_min(gray, size=TOPHAT_SIZE):
    try:
        import cv2
    except ImportError:
        return _local_min_numpy(gray, size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return np.stack([cv2.erode(frame, kernel) for frame in gray])


def stroke_mask(gray, outlined=False):
    """Boolean ``(frames, rows, columns)`` mask of bright, thin strokes in uint8 grayscale frames.

    With ``outlined`` a stroke must also have a near-black pixel within three pixels. Burned-in subtitles
    are drawn with a dark outline or shadow; the thin bright structure of a still anime background (rails,
    straps, window frames) usually is not, which is what separates them on real footage.
    """
    mask = (_tophat(gray) >= TOPHAT_MIN) & (gray >= BRIGHT_MIN)
    if outlined:
        mask &= _local_min(gray) < OUTLINE_MAX
    return mask


def outline_share(gray, boxes):
    """Share of stroke pixels inside ``boxes`` of a frame that have a dark outline, or None without enough strokes.

    ``gray`` is one uint8 grayscale frame resized to the scan width and ``boxes`` are ``[x0, y0, x1, y1]`` in
    its pixels, the places a detector found subtitles. This tells whether the outline test may be used.
    """
    frames = gray[None]
    raw = stroke_mask(frames)[0]
    outlined = raw & (_local_min(frames)[0] < OUTLINE_MAX)
    inside = np.zeros_like(raw)
    for x0, y0, x1, y1 in boxes:
        inside[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = True
    strokes = int((raw & inside).sum())
    if strokes < 120:
        return None
    return int((outlined & inside).sum()) / strokes


def _near(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def similarity(a, b, a_near=None, b_near=None):
    """How much of each frame's strokes lies within a pixel of the other's; the smaller share is returned."""
    count_a, count_b = int(a.sum()), int(b.sum())
    if not count_a or not count_b:
        return 0.
    a_near = _near(a) if a_near is None else a_near
    b_near = _near(b) if b_near is None else b_near
    return min(int((a & b_near).sum()) / count_a, int((b & a_near).sum()) / count_b)


def outline_share_of(image, boxes, scan_width=SCAN_WIDTH):
    """``outline_share`` for a full-size PIL frame and the pixel boxes a detector found subtitles in."""
    from PIL import Image
    scale = scan_width / image.width
    gray = np.asarray(image.convert('L').resize((scan_width, max(2, round(image.height * scale))), Image.BICUBIC), dtype=np.uint8)
    return outline_share(gray, [[value * scale for value in box] for box in boxes])


def outlined_style(shares):
    """Whether the episode's subtitles are outlined, from the per-probe shares (None entries are ignored).

    Returns ``(outlined, share)``. Without a single usable probe the style is unknown, and the outline test is
    left off because wrongly applying it would hide every subtitle, while leaving it off only costs reads.
    """
    usable = sorted(share for share in shares if share is not None)
    if not usable:
        return False, None
    share = usable[len(usable) // 2]
    return share >= OUTLINE_SHARE_MIN, round(share, 3)


# -- events ----------------------------------------------------------------------------------------

def pick_representative(counts, start, end):
    """The frame in [start, end] where the text is most complete, ignoring fades at both ends."""
    length = end - start + 1
    margin = int(length * .2) if length >= 5 else 0
    window = counts[start + margin:end + 1 - margin]
    return start + margin + int(np.argmax(window))


def find_events(masks, static=None, min_pixels=1):
    """Group per-frame stroke masks into subtitle events.

    ``masks`` is a sequence of boolean 2-D arrays, one per frame. A pixel only counts when it is also a stroke
    in a neighbouring frame (moving clutter is dropped) and is not part of ``static``. Returns
    ``(events, counts)`` where each event is ``{'start', 'end', 'rep', 'peak'}`` in frame indexes, ``end`` being
    the last frame that shows the line.
    """
    total = len(masks)
    counts = np.zeros(total, dtype=np.int32)
    events, current, last_mask, last_near, gap = [], None, None, None, 0

    def close():
        nonlocal current
        if current and current['end'] - current['start'] + 1 >= MIN_EVENT_FRAMES:
            current['rep'] = pick_representative(counts, current['start'], current['end'])
            current['peak'] = int(counts[current['rep']])
            events.append(current)
        current = None

    previous, mask = None, masks[0] if total else None
    for index in range(total):
        following = masks[index + 1] if index + 1 < total else None
        keep = np.zeros_like(mask)
        if previous is not None:
            keep |= previous
        if following is not None:
            keep |= following
        frame = mask & keep
        if static is not None:
            frame &= ~static
        counts[index] = int(frame.sum())
        previous, mask = mask, following
        if counts[index] < min_pixels:
            gap += 1
            if current and gap > GAP_FRAMES:
                close()
            continue
        near = _near(frame)
        if current is not None and similarity(last_mask, frame, last_near, near) >= SAME_TEXT:
            current['end'] = index
        else:
            close()
            current = {'start': index, 'end': index}
        last_mask, last_near, gap = frame, near, 0
    close()
    return events, counts


def split_long(events, counts, fps):
    """Break lines on screen for a long time into several events so a missed change is still read."""
    out = []
    for event in events:
        length = (event['end'] - event['start'] + 1) / fps
        if length <= MAX_SEGMENT_SECONDS:
            out.append(event)
            continue
        parts = max(2, round(length / SEGMENT_SECONDS))
        edges = [event['start'] + round((event['end'] - event['start'] + 1) * part / parts) for part in range(parts + 1)]
        for left, right in zip(edges, edges[1:]):
            if right > left:
                rep = pick_representative(counts, left, right - 1)
                out.append({'start': left, 'end': right - 1, 'rep': rep, 'peak': int(counts[rep])})
    return out


def to_seconds(event, fps):
    return {'start_time': round(event['start'] / fps, 3), 'end_time': round((event['end'] + 1) / fps, 3),
            'rep_time': round(event['rep'] / fps, 3), 'peak_pixels': event['peak']}


# -- decoding --------------------------------------------------------------------------------------

def stream_strip(source, geometry, fps, cancel_check=None, batch=64):
    """Yield ``(frames, rows, columns)`` uint8 arrays of the strip, decoded once by ffmpeg."""
    width, rows = geometry['width'], geometry['height']
    size = width * rows
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source), '-an', '-sn', '-dn',
               '-vf', f"fps={fps:.6f},scale={width}:{geometry['frame_height']},crop={width}:{rows}:0:{geometry['y']},format=gray",
               '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL, bufsize=size * batch)
        finished = False
        try:
            while True:
                if cancel_check:
                    cancel_check()
                data = process.stdout.read(size * batch)
                if not data:
                    finished = True
                    break
                usable = len(data) // size * size
                if usable:
                    yield np.frombuffer(data[:usable], dtype=np.uint8).reshape(-1, rows, width)
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            process.wait()
        if finished and process.returncode:
            errors.seek(0)
            raise RuntimeError('Could not decode the video for subtitle scanning: ' + errors.read().decode('utf-8', 'replace')[-800:])


def scan_video(source, source_width, source_height, duration, band, cancel_check=None, progress=None, outlined=False):
    """Scan a whole video's subtitle strip. Returns ``{'events', 'fps', 'diagnostics'}``.

    Event times are episode-relative seconds; ``rep_time`` is the moment to extract for reading. ``outlined`` applies
    the outline test of ``stroke_mask``; use it only when the episode's subtitles are known to be outlined.
    ``counts`` holds the per-frame stroke pixel counts, for diagnosis only.
    """
    fps = max(1., min(MAX_SCAN_FPS, MAX_SCAN_FRAMES / max(duration, 1.)))
    geometry = strip_geometry(source_width, source_height, band)
    area = geometry['width'] * geometry['height']
    minimum = max(8, int(area * (OUTLINED_MIN_TEXT_RATIO if outlined else MIN_TEXT_RATIO)))
    packed, frequency, total = [], np.zeros((geometry['height'], geometry['width']), dtype=np.int32), 0
    expected = max(1, int(duration * fps))
    for frames in stream_strip(source, geometry, fps, cancel_check):
        masks = stroke_mask(frames, outlined)
        frequency += masks.sum(axis=0, dtype=np.int32)
        packed.extend(np.packbits(mask, axis=None) for mask in masks)
        total += len(frames)
        if progress:
            progress(min(1., total / expected))
    shape = (geometry['height'], geometry['width'])
    static = _near(frequency > STATIC_FREQUENCY * total) if total >= 40 else None  # include the fringe that flickers with compression
    masks = _Unpacked(packed, shape, area)
    events, counts = find_events(masks, static, min_pixels=minimum)
    events = split_long(events, counts, fps)
    return {'events': [to_seconds(event, fps) for event in events], 'fps': fps, 'geometry': geometry, 'counts': counts,
            'diagnostics': {'frames': total, 'text_frames': int((counts >= minimum).sum()),
                            'events': len(events), 'static_pixels': int(static.sum()) if static is not None else 0,
                            'strip_pixels': area, 'fps': round(fps, 3), 'outline_test': bool(outlined)}}


class _Unpacked:
    """Sequence view that unpacks one frame mask at a time from packed bits."""

    def __init__(self, packed, shape, area):
        self.packed, self.shape, self.area = packed, shape, area

    def __len__(self):
        return len(self.packed)

    def __getitem__(self, index):
        return np.unpackbits(self.packed[index], count=self.area).astype(bool).reshape(self.shape)

