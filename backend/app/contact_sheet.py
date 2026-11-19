"""A labelled slot-by-keyframe JPEG for the target-frame judge (ffmpeg only)."""
import tempfile
from pathlib import Path

from . import media

TILE = (152, 128)


def build(output, frames, sizes, slots, cells, candidates=()):
    """Write a sheet with a separate F1 reference column and the candidate strip.

    ``frames`` maps frame labels to JPEGs, in time order. ``slots`` are the slot
    numbers in the grid; ``cells`` supplies boxes for ``(slot, label)``. A
    candidate is ``{'id', 'frame', 'box'}`` and may be unmatched or a suspected
    merge. All boxes are in the corresponding source JPEG's pixels.
    """
    if not frames:
        raise ValueError('A target contact sheet needs at least one frame')
    labels = list(frames)
    columns = len(labels) + 1
    with tempfile.TemporaryDirectory(prefix='framewise-target-sheet-') as temp:
        root = Path(temp)
        rows, cropped = [], {}

        def tile(frame, box, label):
            if box is None:
                return None, label
            key = (frame, tuple(box))
            if key not in cropped:
                path = root / f'crop-{len(cropped)}.jpg'
                w, h = sizes[frame]
                media.crop_image(Path(frames[frame]), path, (box[0] / w, box[1] / h, box[2] / w, box[3] / h))
                cropped[key] = path
            return cropped[key], label

        def row(items):
            path = root / f'row-{len(rows)}.jpg'
            media.tile_images(items, path, columns, TILE)
            rows.append(path)

        for slot in slots:
            number = slot + 1
            row([tile(labels[0], cells.get((slot, labels[0])), f'S{number} ref'),
                 *(tile(frame, cells.get((slot, frame)), f'S{number}·{frame}') for frame in labels)])
        strip = [tile(item['frame'], item['box'], f"{item['id']}·{item['frame']}") for item in candidates]
        for offset in range(0, len(strip), columns):
            items = strip[offset:offset + columns]
            row(items + [(None, ' ') for _ in range(columns - len(items))])
        if not rows:
            row([(Path(frames[frame]), frame) for frame in labels] + [(None, ' ')])

        target = output.with_name(f'{output.stem}.partial.jpg')
        try:
            if len(rows) == 1:
                target.write_bytes(rows[0].read_bytes())
            else:
                args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-filter_complex_threads', '1']
                for path in rows:
                    args += ['-threads', '1', '-i', str(path)]
                media.run([*args, '-filter_complex', f'vstack=inputs={len(rows)}', '-frames:v', '1', '-q:v', '3', str(target)], 120)
            target.replace(output)
        finally:
            target.unlink(missing_ok=True)
    return output
