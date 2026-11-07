"""Token-saving projection of tool payloads.

Tool results are sent to the model as minified JSON with empty values dropped,
floats rounded, long strings clipped with an explicit marker, and an overall
size cap. The clip markers tell the model how to ask for more instead of
silently losing data.
"""
import json
import math

DROP_KEYS = {'usage', 'generation_attempts', 'boundary_policy', 'source_file', 'padded_tail', 'detection_response', 'detection_prompt',
             'ocr_prompt', 'ocr_response', 'requested_time', 'selected_for_ocr', 'image_size', 'episode_start_seconds', 'episode_end_seconds'}


def prune(value, max_string=1200, depth=0):
    """Drop empty values and provenance noise; clip long strings."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in DROP_KEYS:
                continue
            item = prune(item, max_string, depth + 1)
            if item in (None, '', [], {}):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        return [item for item in (prune(item, max_string, depth + 1) for item in value) if item not in (None, '', [], {})]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 3)
    if isinstance(value, str):
        text = value.strip()
        if max_string and len(text) > max_string:
            return text[:max_string] + f'…[+{len(text) - max_string} chars; request a narrower field]'
        return text
    return value


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def encode(value, limit=7000, max_string=1200):
    """Serialize a tool result within ``limit`` characters."""
    pruned = prune(value, max_string)
    text = dumps(pruned)
    if len(text) <= limit:
        return text
    # Retry with tighter strings before falling back to a hard cut.
    for size in (600, 300, 160):
        text = dumps(prune(value, size))
        if len(text) <= limit:
            return text
    return dumps({'truncated': True, 'note': 'Result too large; request specific fields.', 'partial': text[:limit - 120]})


def estimate_tokens(text):
    """Rough count that errs high for mixed English/JSON/Japanese text."""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return int(ascii_chars / 3.6 + (len(text) - ascii_chars) * 1.1) + 1
