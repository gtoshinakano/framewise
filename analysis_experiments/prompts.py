"""The prompts of the experiment: the production instruction plus one approach block, and the retry messages.

Needs the application package on the path (``app.visual_contracts`` is pure Python).
"""
import pose_schema as ps
from app.visual_contracts import FRAME_INSTRUCTION, VIDEO_INSTRUCTION, person_context, series_note, subtitle_note


def frame_system(approach):
    return FRAME_INSTRUCTION + '\n\n' + ps.prompt_block(approach, 'frame')


def video_instruction(approach):
    return VIDEO_INSTRUCTION + '\n\n' + ps.prompt_block(approach, 'video')


def frame_user_text(people, hints, regions, language):
    """The per-frame user message exactly as ``frame_model.describe_frame`` builds it."""
    parts = ['This image is the first frame of the cut.', person_context(people, 'first frame')]
    note = series_note(hints)
    if note:
        parts.append(note)
    parts.append(subtitle_note(language, regions, 'image pixels').strip())
    return '\n\n'.join(part for part in parts if part)


def frame_feedback(errors):
    return ('Your answer was rejected by the validator:\n' + ps.format_errors(errors)
            + '\nReturn the complete corrected JSON object again. Change only what this list requires.')


def video_correction(errors):
    return ('\nThe previous response was rejected by the validator:\n' + ps.format_errors(errors)
            + '\nStart again from the video and return one complete JSON object that fixes every item. Be concise and finish the object within the output limit.')
