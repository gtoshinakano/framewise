"""Fake models for `run.sh dry-run` and the tests: the same interfaces as backends.RealModels and qwen.Client, answers canned from the stored data.

Nothing here is a measurement. It only proves that every stage, checkpoint and the report run end to end on the CPU.
"""
import hashlib
import json
import random
import zlib

import numpy as np

import lines as L

DIM = 256


def vec_for(key, noise=0., seed=None):
    base = np.random.default_rng(zlib.crc32(str(key).encode())).normal(size=DIM)
    if noise:
        base = base + np.random.default_rng(seed if seed is not None else 1).normal(size=DIM) * noise
    return base / np.linalg.norm(base)


class FakeModels:
    fake = True

    def __init__(self, prep):
        """`prep` {episode number: prepared episode dict from data.prepare_episode}."""
        self.prep = prep
        self.number = None
        self.load_seconds = {}

    def bind(self, number):
        self.number = number

    def _rows(self):
        return self.prep[self.number]['baseline_rows']

    def _label_character(self):
        votes = {}
        for row in self._rows():
            if row['speaker_label'] and row.get('character'):
                votes.setdefault(row['speaker_label'], {}).setdefault(row['character'], 0)
                votes[row['speaker_label']][row['character']] += 1
        return {label: max(c, key=c.get) for label, c in votes.items()}

    def load_diarization(self):
        pass

    def load_asr(self):
        pass

    def diarize(self, wav, threshold=None, min_speakers=None, max_speakers=None, num_speakers=None):
        labels = sorted({r['speaker_label'] for r in self._rows() if r['speaker_label']})
        fold = {}
        if (threshold or .6) >= .75 and len(labels) > 2:
            fold[labels[-1]] = labels[0]
        if max_speakers and len(labels) > max_speakers:
            for extra in labels[max_speakers:]:
                fold[extra] = labels[0]
        turns = []
        for row in self._rows():
            if not row['speaker_label']:
                continue
            label = fold.get(row['speaker_label'], row['speaker_label'])
            if turns and turns[-1]['speaker'] == label and row['start_time'] - turns[-1]['end_time'] <= .6:
                turns[-1]['end_time'] = row['end_time']
            else:
                turns.append({'start_time': row['start_time'], 'end_time': row['end_time'], 'speaker': label})
        owner = self._label_character()
        used = sorted({t['speaker'] for t in turns})
        centroids = {label: vec_for(owner.get(label) or f'label:{self.number}:{label}', .15, zlib.crc32(label.encode())).tolist() for label in used}
        return {'turns': turns, 'labels': used, 'centroids': centroids, 'seconds': .01}

    def embed(self, clip):
        return vec_for('clip', 0.)

    def embed_span(self, number, start, end, clip):
        middle = (start + end) / 2
        rows = self.prep[number]['baseline_rows']
        owner = None
        for row in rows:
            if row['start_time'] <= middle <= row['end_time']:
                owner = row.get('character') or (f"label:{number}:{row['speaker_label']}" if row['speaker_label'] else None)
                break
        for sample_owner in self.prep.get('_sample_owner', {}).get(number, []):
            if sample_owner[0] <= middle <= sample_owner[1]:
                owner = sample_owner[2]
        return vec_for(owner or f'noise:{start}', .35, int(start * 1000) % 100000)

    def transcribe(self, wav, start, end, context='', language=None, depth=0):
        tokens = []
        for row in self._rows():
            if row['start_time'] >= start and row['end_time'] <= end + .01:
                text = row['text'].strip()
                if language == 'Japanese' and L.looks_non_japanese(text):
                    continue
                if context and text == '今日':
                    text = '今日、'
                tokens.append({'start': row['start_time'], 'end': row['end_time'], 'text': text})
        return {'text': ''.join(t['text'] for t in tokens), 'language': 'Japanese', 'tokens': tokens}

    def release(self):
        pass


class FakeClient:
    """Stands in for qwen.Client. The first attempt of every fifth window omits a line to exercise the retry path."""

    def __init__(self, prep=None):
        self.calls = 0

    def chat(self, messages, max_tokens, schema_=None, hint=None, temperature=.2):
        self.calls += 1
        hint = hint or {}
        if hint.get('task') == 'glossary':
            proposals = [{'name': n, 'japanese': '', 'reading': '', 'sure': False} for n in hint['names']]
            return {'content': json.dumps({'characters': proposals}), 'finish_reason': 'stop', 'usage': {}, 'timings': {}, 'grammar': True}
        if hint.get('task') == 'speaker':
            window, names = hint['window'], hint['names']
            out = []
            for item in window:
                voice = item.get('voice') or []
                if voice and voice[0][1] >= .2:
                    speaker, reason, conf = voice[0][0], 'voice', 'high' if voice[0][1] >= .5 else 'medium'
                elif item.get('on_screen'):
                    speaker, reason, conf = item['on_screen'][0], 'on_screen', 'low'
                else:
                    speaker, reason, conf = 'unknown', 'other', 'low'
                if speaker not in names:
                    speaker = 'unknown'
                out.append({'id': item['id'], 'speaker': speaker, 'confidence': conf, 'reason': reason, 'text_fix': '', 'fix_reason': 'none'})
            if hint.get('attempt') == 1 and zlib.crc32(str(hint.get('key')).encode()) % 5 == 0 and len(out) > 1:
                out = out[:-1]
            return {'content': json.dumps({'lines': out}, ensure_ascii=False), 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1000, 'completion_tokens': 300},
                    'timings': {'predicted_per_second': 50.}, 'grammar': True}
        return {'content': '{}', 'finish_reason': 'stop', 'usage': {}, 'timings': {}, 'grammar': False}
