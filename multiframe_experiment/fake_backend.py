"""Canned model answers for ``--dry`` runs and the tests (no GPU). About a quarter of first answers are broken on purpose."""
import hashlib
import json
import random

import schema

PHRASES = [('head', 'turns her head slightly to the left'), ('eyes', 'blinks once'), ('arms', 'raises the right arm to chest height'),
           ('hands', 'opens the left hand'), ('mouth', 'opens the mouth to speak'), ('expression', 'smiles faintly'), ('legs', 'takes a step forward'),
           ('body_pose', 'leans forward'), ('position', 'moves toward the centre')]


class FakeClient:
    def __init__(self, seconds=None):
        self.seconds = seconds or {'lfm3': 4.0, 'qwen35': 30.0}

    def chat(self, messages, max_tokens, schema_=None, hint=None, temperature=0):
        hint = hint or {}
        rng = random.Random(hashlib.sha1(json.dumps(hint, sort_keys=True, default=str).encode()).hexdigest())
        task = hint.get('task')
        if task == 'answer':
            content = self._answer(hint, rng)
            seconds = self.seconds.get(hint.get('model'), 10.0) * (0.7 + 0.6 * rng.random())
        elif task == 'reference':
            labels = hint['labels']
            pairs = [{'from': a, 'to': b, 'camera': rng.choice(['static', 'static', 'pan_left']),
                      'characters': [{'name': name, **{flag: rng.random() < 0.3 for flag in schema.REFERENCE_FLAGS}} for name in hint.get('roster') or ['person 1']]}
                     for a, b in zip(labels, labels[1:])]
            content, seconds = json.dumps({'pairs': pairs}), 20.0
        else:
            verdicts = [{'id': n, 'verdict': rng.choices(['true', 'false', 'not_visible'], [0.6, 0.3, 0.1])[0]} for n in range(1, hint['n'] + 1)]
            content, seconds = json.dumps({'verdicts': verdicts}), 25.0
        return {'content': content, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 3000, 'completion_tokens': len(content) // 4},
                'timings': {'predicted_per_second': 30.0}, 'grammar': True, 'simulated_seconds': round(seconds, 2)}

    def _answer(self, hint, rng):
        labels = hint['labels']
        names = hint.get('roster') or ['unknown person 1']
        if hint.get('context') == 'frames' and rng.random() < 0.3:
            names = ['Mai']  # an invented name, for the name check
        segments = []
        for a, b in zip(labels, labels[1:]):
            people = []
            for name in names:
                changes = [{'part': part, 'description': text} for part, text in rng.sample(PHRASES, rng.randint(0, 3))]
                people.append({'name': name, 'walking': rng.choice(['no', 'no', 'yes']), 'speaking': rng.choice(['no', 'yes', 'unclear']),
                               'blinking': rng.choice(['no', 'no', 'yes']), 'changes': changes})
            segments.append({'from': a, 'to': b, 'camera': {'move': rng.choice(['static', 'static', 'pan_right', 'zoom_in']), 'detail': ''},
                             'characters': people, 'other_changes': [{'kind': 'lighting', 'description': 'gets brighter'}] if rng.random() < 0.2 else []})
        if hint.get('attempt') == 1 and rng.random() < 0.25:
            segments = segments[:-1] or segments  # wrong count
            if len(labels) == 2:
                return '{"segments": ['  # cut off JSON
        frames = [{'frame': label, 'characters': [{'name': name, 'state': 'standing, mouth closed'} for name in names]} for label in labels]
        return json.dumps({'frames': frames, 'segments': segments, 'summary': 'Fake summary.'})
