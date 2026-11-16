"""Keyword suggestions for still-image and shot prompts, read from ``still_vocabulary.yaml``.

The dictionary is a data file so it can be edited without touching code. It describes a frozen picture
in viewer-relative words (screen-left, screen-right). Prompt builders use the sections that help the
writer of a still or of a shot's acting:

* the shot's framing, the characters' placement and gaze, expression, hands and body, layout, setting
  and lighting (``STILL_GUIDE``);
* the verbs a beat can use for interaction with objects and other characters (``BEATS_GUIDE``).

``EXCLUDED`` sections are never offered: ``visual_style`` (the person who generates the images chooses the
style, so no prompt names one), ``reference`` and ``output`` (workflow settings), ``evidence`` and
``fallback`` (values that must never reach a prompt). Everything here is a suggestion: the lists are not
closed and a keyword is used only when the evidence shows it.
"""
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).with_name('still_vocabulary.yaml')
EXCLUDED = ('evidence', 'reference', 'output', 'visual_style', 'fallback')

# Where each part of a still prompt looks for words, as ``section.list`` paths into the dictionary.
STILL_GUIDE = (
    ('still.shot', ('camera.shot_size', 'camera.vertical_angle', 'camera.height', 'camera.horizontal_angle', 'camera.framing', 'composition.crop')),
    ('still.placement', ('character_position.horizontal', 'character_position.vertical', 'character_position.depth', 'character_position.relative_position', 'character_position.distance',
                         'body.orientation', 'body.posture', 'body.torso', 'body.support', 'head.yaw', 'head.pitch', 'head.roll',
                         'eyes.gaze_direction', 'eyes.gaze_target')),
    ('still.subjects', ('emotion.primary', 'emotion.intensity', 'eyebrows.position', 'eyes.gaze_quality', 'mouth.shape', 'face.visible_details', 'arm.position', 'hand.position',
                        'hair.physical_state')),
    ('still.layout', ('composition.occlusion',)),
    ('still.setting', ('environment.type', 'environment.location', 'environment.time_of_day', 'environment.weather', 'environment.atmosphere')),
    ('still.lighting', ('lighting.source', 'lighting.direction', 'lighting.hardness', 'lighting.color_temperature', 'lighting.contrast', 'lighting.shadow')),
)
# Beats are acting over time. Spoken-word lists (action.communication, mouth.activity) stay out: the voices of a shot decide who speaks.
BEATS_GUIDE = (
    ('beats', ('hand.gesture', 'action.locomotion', 'action.object_interaction', 'action.character_interaction')),
)

INTRO = ('Suggested vocabulary, one list per part of the answer. Prefer these words when one of them fits, so that every shot of the sequence speaks the same language; '
         'the lists are suggestions, not a closed set, and you may combine or adapt them. screen-left and screen-right are the viewer\'s left and right, never the character\'s own side. '
         'Use a word only when the evidence shows it: never fill a part because a list offers a word, never write "unknown" or "not visible" (leave the part out), '
         'and never take a medium, an art style or a rendering technique from a list.')


@lru_cache(maxsize=1)
def load():
    """The dictionary as nested dicts of lists; empty when the file or PyYAML is unavailable (prompts then simply omit the keywords)."""
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(FILE.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def values(path):
    """The words of ``section.list`` (for example ``body.orientation``), or [] when it does not exist or is excluded."""
    section, _, name = path.partition('.')
    found = load().get(section) if section not in EXCLUDED else None
    items = found.get(name) if isinstance(found, dict) else None
    return [str(item) for item in items] if isinstance(items, list) else []


def block(guide):
    """A guide as prompt text: one heading per part and one line per list."""
    lines = []
    for part, paths in guide:
        rows = [f"- {path}: {'; '.join(words)}" for path, words in ((path, values(path)) for path in paths) if words]
        if rows:
            lines += [part, *rows]
    return '\n'.join(lines)


def still_block():
    """The keyword section for the writer of a shot's still, or '' when the dictionary could not be read."""
    text = block(STILL_GUIDE)
    return f'## KEYWORDS FOR THE STILL\n{INTRO}\n\n{text}' if text else ''


def beats_block():
    text = block(BEATS_GUIDE)
    return f'## KEYWORDS FOR THE BEATS\nUse these words for what hands and bodies do when one fits; the same rules as above apply, and who speaks is decided by the voices alone.\n\n{text}' if text else ''


def shot_block():
    """Both sections as the shot prompt carries them."""
    return '\n\n'.join(part for part in (still_block(), beats_block()) if part)
