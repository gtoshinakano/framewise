"""Pictures Cut Assistance looks at: frames of a cut, read from the original video when a model request needs them.

``view_frames`` (``vision_tools.py``) resolves frames by name. A tool message stores only a small reference per picture
in its usage (cut, frame name, optional zoom box), never pixels: the JPEG is cut from the original video again whenever the
conversation is sent to a model that can see (a small in-memory cache keeps the last ones), and the chat page asks for the
same picture through ``GET /api/assistant/messages/{id}/images/{n}``. A picture that no longer fits the context turns back
into its text description (``runner.history_messages``), so the history stays valid for any provider.

A frame is named, never given as a time the model computed by itself:

- ``first`` / ``last``: the cut's boundary frames, exactly the ones its results were made from;
- ``F3``: keyframe 3 of the cut's current keyframe selection (the F1…Fn of the keyframe analysis);
- ``1.25`` or ``1.25s``: the frame at that many seconds from the start of the cut (clip-relative, like every time the assistant reads).
"""
import base64
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from .. import db, frame_cache, media

LONG_EDGE = 1024  # pixels of the wider side; a 16:9 frame is then about 600 tokens for Qwen3.6 (32 px per token)
ZOOM_EDGE = 896  # a zoomed region is scaled to this long side, so small details reach the vision encoder at a readable size
MAX_PER_CALL = 4
MAX_ATTACHED = 6  # pictures that stay in the model's context; older ones turn back into their description
TOKENS_PER_PICTURE = 800  # what history budgeting counts for one attached picture (it errs high)
CACHE_BYTES = 24 * 1024 * 1024
MIN_ZOOM = 0.05  # a zoom box must span at least this share of the width and of the height

_NAME = re.compile(r'^(first|last|f(\d{1,3})|(\d+(?:\.\d+)?)\s*s?)$', re.I)
_cache = OrderedDict()
_lock = threading.Lock()


class FrameError(ValueError):
    """A frame the model asked for cannot be shown; the message says why, in words the model can act on."""


def parse(name):
    """``(kind, value)`` of a frame name: ``('first', None)``, ``('last', None)``, ``('keyframe', 3)`` or ``('time', 1.25)``."""
    text = str(name).strip()
    match = _NAME.match(text)
    if not match:
        raise FrameError(f'"{text}" is not a frame name. Use "first", "last", a keyframe such as "F3", or seconds from the start of the cut such as "1.25"')
    if match.group(2):
        number = int(match.group(2))
        if number < 1:
            raise FrameError('Keyframes start at F1')
        return 'keyframe', number
    if match.group(3) is not None:
        return 'time', float(match.group(3))
    return text.lower(), None


def zoom_box(crop):
    """The validated ``[x1, y1, x2, y2]`` (fractions of the frame) or ``None``."""
    if crop in (None, [], ()):
        return None
    if not isinstance(crop, (list, tuple)) or len(crop) != 4 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in crop):
        raise FrameError('crop must be [x1, y1, x2, y2], four numbers from 0 to 1 (fractions of the frame width and height, from the top left)')
    x1, y1, x2, y2 = (float(value) for value in crop)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise FrameError('crop needs 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1')
    if x2 - x1 < MIN_ZOOM or y2 - y1 < MIN_ZOOM:
        raise FrameError('crop is too small; it must cover at least 5% of the frame in each direction')
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def locate(cut_id, frame, crop=None):
    """Resolve one frame name to what is needed to cut it from the video: ``{spec, number, clip_time, label}``.

    Reads only the database; the pixels come from ``picture``.
    """
    kind, value = parse(frame)
    box = zoom_box(crop)
    cut = db.one('SELECT c.*,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,))
    if not cut:
        raise FrameError('The cut no longer exists')
    source = media.episode_dir(cut['project_id'], cut['episode_id']) / f"original{cut['original_ext']}"
    if not source.is_file():
        raise FrameError('The original video of this cut is not available, so no frame can be shown')
    start, end = cut['start_time'], cut['end_time']
    first, last = media.cut_frame_times(start, end)
    spec = {'source': source, 'start': start, 'end': end, 'width': 4096 if box else LONG_EDGE, 'last': False, 'seek_tolerance': .000002, 'quality': 3}
    if kind == 'first':
        spec['time'] = cut['first_frame_time'] if cut['first_frame_time'] is not None else first
        clip_time, label = 0.0, 'first frame'
    elif kind == 'last':
        saved = cut['last_frame_time']
        spec.update(time=saved if saved is not None else last, last=saved is None, probe_padding=1 if saved is None else 0)
        clip_time, label = round(end - start, 3), 'last frame'
    elif kind == 'keyframe':
        from .. import keyframes  # imports OpenCV; only this branch needs it
        _, _, _, result = keyframes.current_selection(cut_id)
        frames = sorted((result or {}).get('selected_frames') or [], key=lambda item: item['source_frame_index'])
        if not frames:
            raise FrameError('This cut has no current keyframes: the keyframe selection has not run, or it is out of date')
        if value > len(frames):
            raise FrameError(f'This cut has keyframes F1 to F{len(frames)}; F{value} does not exist')
        item = frames[value - 1]
        spec.update(time=item['timestamp_seconds'], ordinal=item['source_frame_index'])
        clip_time, label = round(item['relative_timestamp_seconds'], 3), f'keyframe F{value}'
    else:
        duration = end - start
        if value > duration + 1e-6:
            raise FrameError(f'This cut lasts {duration:.2f} s; {value:g} s is past its end')
        spec['time'] = min(start + value, last)
        clip_time, label = round(value, 3), f'frame at {value:g} s'
    number = cut['cut_index'] + 1 if cut['active'] else None
    name = (f'Cut {number} · ' if number else 'Archived cut · ') + label + (f' ({clip_time:g} s)' if kind in ('first', 'last', 'keyframe') and clip_time else '')
    if box:
        name += ' · zoom ' + ' '.join(f'{value:g}' for value in box)
    return {'spec': spec, 'clip_time': clip_time, 'label': name, 'crop': box, 'frame': frame_name(kind, value)}


def frame_name(kind, value):
    """The canonical spelling of a parsed frame name (what a stored reference keeps)."""
    if kind == 'keyframe':
        return f'F{value}'
    if kind == 'time':
        return f'{value:g}s'
    return kind


def _cache_key(located):
    spec = located['spec']
    stat = Path(spec['source']).stat()
    return (str(spec['source']), stat.st_mtime_ns, stat.st_size, spec['time'], spec.get('ordinal'), spec['last'], tuple(located['crop'] or ()))


def _zoomed(data, box):
    with tempfile.TemporaryDirectory(prefix='framewise-assistant-zoom-') as folder:
        whole, cropped, scaled = (Path(folder) / name for name in ('whole.jpg', 'crop.jpg', 'zoom.jpg'))
        whole.write_bytes(data)
        media.crop_image(whole, cropped, tuple(box))
        media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(cropped), '-vf',
                   f"scale=w='if(gte(iw,ih),{ZOOM_EDGE},-2)':h='if(gte(iw,ih),-2,{ZOOM_EDGE})'", '-frames:v', '1', '-q:v', '3', str(scaled)], 60)
        return scaled.read_bytes()


def picture(cut_id, frame, crop=None, located=None):
    """The JPEG bytes of a frame (cut from the original video, or from the cache of the last few)."""
    located = located or locate(cut_id, frame, crop)
    key = _cache_key(located)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        data = frame_cache.render(dict(located['spec']))
        if located['crop']:
            data = _zoomed(data, located['crop'])
    except (RuntimeError, OSError, ValueError) as exc:
        raise FrameError(f'The frame could not be cut from the video: {str(exc)[:200]}') from exc
    with _lock:
        _cache[key] = data
        while sum(len(value) for value in _cache.values()) > CACHE_BYTES and len(_cache) > 1:
            _cache.popitem(last=False)
    return data


def keyframe_picture(cut_id, ordinal, time, edge=LONG_EDGE):
    """The JPEG of one exact source frame of a cut, by its ordinal (a shot's reference keyframe).

    Cut by frame number, never from a rounded time: a time-based seek returns the first frame at or after the time.
    """
    cut = db.one('SELECT c.*,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,))
    if not cut:
        raise FrameError('The cut no longer exists')
    source = media.episode_dir(cut['project_id'], cut['episode_id']) / f"original{cut['original_ext']}"
    if not source.is_file():
        raise FrameError('The original video of this cut is not available, so no frame can be shown')
    spec = {'source': source, 'time': float(time), 'start': cut['start_time'], 'end': cut['end_time'], 'width': edge, 'last': False,
            'seek_tolerance': .000002, 'quality': 3, 'ordinal': int(ordinal)}
    located = {'spec': spec, 'crop': None, 'clip_time': round(float(time) - cut['start_time'], 3), 'label': f'frame {ordinal}', 'frame': f'#{ordinal}'}
    return picture(cut_id, None, located=located)


def data_url(ref):
    """The ``image_url`` payload of a stored reference."""
    return 'data:image/jpeg;base64,' + base64.b64encode(picture(ref['cut_id'], ref['frame'], ref.get('crop'))).decode('ascii')


def reference(cut_id, located):
    """What a tool message stores for one picture."""
    return {'cut_id': cut_id, 'frame': located['frame'], 'crop': located['crop'], 'label': located['label'], 't': located['clip_time']}
