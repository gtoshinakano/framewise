"""Remove explicit model reasoning blocks before validating an output."""
import re


_BLOCK = re.compile(r'<(think|thinking|analysis|reasoning)\b[^>]*>[\s\S]*?</\1\s*>', re.I)
_TAG = re.compile(r'</?(?:think|thinking|analysis|reasoning)\b', re.I)


def strip_reasoning_blocks(text):
    if not isinstance(text, str):
        raise ValueError('Model output must be text')
    text = _BLOCK.sub('', text).strip()
    if _TAG.search(text):
        raise ValueError('Model output contains an incomplete reasoning block')
    return text
