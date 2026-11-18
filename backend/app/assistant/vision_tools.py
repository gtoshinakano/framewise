"""The tool that lets Cut Assistance look at the video: ``view_frames``.

It is offered only to a model that can read images (``Endpoint.vision``: the local Qwen 3.6 when its vision projector is
loaded, a remote or configured server that says it has vision). A call returns text describing what is attached and puts
a small reference per picture on ``ToolContext.images``; the runner stores those with the tool message and sends the pictures
to the model in a user message right after the tool results (OpenAI-style APIs do not accept images inside a tool result).
Frames are resolved by name in ``images.py``; nothing here reads pixels.
"""
from .. import db
from . import images, tools
from .tools import ToolError, tool

NOTE = ('The pictures are attached to the next message, in this order. Describe only what you can see in them. '
        'Who someone is comes from the character boxes, the results and the user, never from a face or an outfit alone. '
        'A single frame cannot show motion: compare frames (for example F1 and F2) to say what changed.')


def resolve_cut(ctx, number):
    """The id of the cut whose frames are shown: this chat's cut, or the cut with that number (in the sequence, or in the episode)."""
    if ctx.sequence_id:
        from . import sequence_tools
        if number is None or isinstance(number, bool) or not isinstance(number, int):
            raise ToolError('Say which cut: give its number as shown in the editor (cut)')
        _, current, _ = sequence_tools.load(ctx)
        cuts = current['timeline']['cuts']
        span = next((cut for cut in cuts if cut['number'] == number), None)
        if span is None:
            raise ToolError(f"Cut {number} is not part of this sequence (cuts {cuts[0]['number']}–{cuts[-1]['number']})")
        return span['id']
    if number is None:
        return ctx.cut_id
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ToolError('cut is the cut number as shown in the editor, starting at 1')
    found = db.one('SELECT id FROM cuts WHERE active=1 AND cut_index=? AND episode_id=(SELECT episode_id FROM cuts WHERE id=?)', (number - 1, ctx.cut_id))
    if not found:
        raise ToolError(f'This episode has no active cut {number}')
    return found['id']


@tool('view_frames', "Look at the video itself. Shows you pictures of a cut in the message that follows this result: "
      "frames names 'first' or 'last' (the cut's boundary frames), 'F1'…'Fn' (the cut's keyframes, the frames the keyframe analysis describes) or seconds from the start of the cut such as '1.25'. "
      "At most 4 per call. crop zooms into the same region of every frame: [x1, y1, x2, y2] as fractions of the width and height from the top left (the upper right quarter is [0.5, 0, 1, 0.5]). "
      "Use it to check a result, a prompt or a user's claim against what is really on screen, and to read small details; ask only for the frames the question needs. cut (a cut number) shows another cut of the episode instead of this one.",
      {'frames': {'type': 'array', 'items': {'type': 'string'}, 'description': "Frame names, for example [\"first\", \"F2\", \"1.25\"]"},
       'crop': {'type': 'array', 'items': {'type': 'number'}, 'description': 'Optional zoom: [x1, y1, x2, y2], fractions from 0 to 1'},
       'cut': {'type': 'integer', 'description': "Cut number as shown in the editor (default: this chat's cut; required in a sequence chat)"}},
      ('frames',), summary=lambda args, value: 'Looked at ' + (', '.join(item['shows'] for item in value['pictures']) if isinstance(value, dict) and value.get('pictures') else 'frames'),
      limit=3000, scope='any', vision=True)
def view_frames(ctx, args):
    names = args['frames']
    if not names:
        raise ToolError('frames needs at least one frame name')
    if len(names) > images.MAX_PER_CALL:
        raise ToolError(f'At most {images.MAX_PER_CALL} frames per call; ask for the rest in another call')
    cut_id = resolve_cut(ctx, args.get('cut'))
    pictures, problems, seen = [], [], set()
    for name in names:
        try:
            located = images.locate(cut_id, name, args.get('crop'))
            images.picture(cut_id, name, args.get('crop'), located)  # cut it now, so a frame that cannot be made is reported to the model here
        except images.FrameError as exc:
            problems.append({'frame': str(name), 'error': str(exc)})
            continue
        reference = images.reference(cut_id, located)
        key = (reference['frame'], tuple(reference['crop'] or ()))
        if key in seen:
            continue
        seen.add(key)
        pictures.append(reference)
    if not pictures:
        raise ToolError('; '.join(f"{item['frame']}: {item['error']}" for item in problems))
    ctx.images.extend(pictures)
    return {'pictures': [{'n': number, 'shows': picture['label'], 'clip_seconds': picture['t']} for number, picture in enumerate(pictures, 1)],
            'errors': problems or None, 'note': NOTE}
