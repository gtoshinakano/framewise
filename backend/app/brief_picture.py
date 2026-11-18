"""The first frame as a picture on the Cut summary request (2026-11-19).

The Cut summary's last section, Cinematic Camera, says which real camera, lens and look would have made the opening
frame. The writer is better at it when it sees the frame, so a model that can read images gets the cut's first frame
after the evidence. The picture is only for that section (``PICTURE_NOTE``): the other sections keep following the
written evidence, as before.

The picture is optional and never a reason to fail the job: ``opening_frame`` answers ``([], reason)`` when the
endpoint cannot see (the caller checks that first), when the request would not leave room for it in one server slot,
or when the frame cannot be cut. Only the reference is stored with the result (``settings.picture``), never pixels.
"""
from .assistant import compact, images

CAPTION = 'opening frame: the first frame of this cut'
FRAME = 'first'
ANSWER_TOKENS = 6144  # what the request asks the model to be able to write (``process_regen``'s ``chat_json`` limit)
SPARE_TOKENS = 512

PICTURE_NOTE = """

## PICTURE OF THE OPENING FRAME

The first frame of this cut is attached after the evidence, as the picture named "opening frame". Use it only to decide the Cinematic Camera section: the framing and perspective, the depth of field, the light and its direction, the palette and the texture it shows. Every other section keeps following the written evidence, as if no picture were attached. Ignore subtitles or other text in the picture."""


def has_room(system, request_json, slot_tokens):
    """True when the request, the answer and one picture fit one server slot."""
    used = compact.estimate_tokens(system) + compact.estimate_tokens(PICTURE_NOTE) + compact.estimate_tokens(request_json)
    return slot_tokens - used - ANSWER_TOKENS - SPARE_TOKENS >= images.TOKENS_PER_PICTURE


def opening_frame(cut_id, system, request_json, slot_tokens):
    """``([(caption, jpeg bytes)], None)``, or ``([], why no picture was sent)``."""
    if not has_room(system, request_json, slot_tokens):
        return [], 'the request leaves no room for a picture'
    try:
        return [(CAPTION, images.picture(cut_id, FRAME))], None
    except (images.FrameError, RuntimeError, OSError, ValueError) as exc:
        return [], f'the first frame could not be cut: {str(exc)[:200]}'
