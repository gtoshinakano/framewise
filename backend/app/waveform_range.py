"""Detailed waveform peaks for one stretch of an episode's audio.wav (16-bit PCM, mono).

The stored overview has about two bins per second, which is too coarse to drag a line's timing.
This reads only the requested slice, so a long episode costs the same as a short one.
"""
import wave
from pathlib import Path

import numpy as np

MAX_SPAN = 300.0
MAX_BINS = 2000


def peaks(audio: Path, start: float, end: float, bins: int) -> dict:
    """Peak (0..1) of each of `bins` equal parts of [start, end); an empty list when there is no audio."""
    if not (end > start >= 0) or end - start > MAX_SPAN or not 1 <= bins <= MAX_BINS:
        raise ValueError('Invalid waveform range')
    if not audio.is_file():
        return {'peaks': [], 'start': start, 'end': end}
    with wave.open(str(audio), 'rb') as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        if width != 2:
            raise ValueError('Unsupported audio format')
        first = min(source.getnframes(), int(start * rate))
        last = min(source.getnframes(), int(end * rate))
        source.setpos(first)
        raw = source.readframes(last - first)
    samples = np.abs(np.frombuffer(raw, dtype='<i2').astype(np.int32)).reshape(-1, channels).max(axis=1) if raw else np.zeros(0, np.int32)
    total = int(round((end - start) * rate))
    edges = np.linspace(0, total, bins + 1).astype(np.int64)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        part = samples[low:max(high, low + 1)]
        result.append(round(float(part.max()) / 32768, 3) if part.size else 0.0)
    return {'peaks': result, 'start': start, 'end': end}
