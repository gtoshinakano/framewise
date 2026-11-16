"""Canned model answers for the CPU dry run (``run.sh dry-run``): the whole pipeline without a GPU.

The answers are plausible and deterministic per (sample, approach, variant). A quarter of the first
attempts are deliberately invalid so that the validator, the retry loop and the failure counters are
exercised; a mirrored frame mostly echoes the mirrored normal answer, with some noise.
"""
import hashlib
import json
import random

import pose_schema as ps


def rng(*parts):
    return random.Random(int(hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def pose(approach, generator, targets):
    """One plausible pose object (without ids) for ``approach``."""
    pick = generator.choice
    grid = approach == 'strict_detailed'
    yaw = pick(ps.ANGLES['head_yaw']['grid']) if grid else generator.randint(-120, 150)
    value = {'body_yaw': pick(ps.ANGLES['body_yaw']['grid']) if grid else generator.randint(-90, 90), 'body_pose': pick(ps.ENUMS['body_pose']), 'head_yaw': yaw,
             'head_pitch': pick(ps.ANGLES['head_pitch']['grid']) if grid else generator.randint(-40, 30), 'head_roll': pick(ps.ENUMS['head_roll']),
             'eye_gaze': pick(ps.ENUMS['eye_gaze']), 'eye_openness': pick(ps.ENUMS['eye_openness']), 'gaze_target': pick([*targets, 'camera', 'none']),
             'mouth_state': pick(ps.ENUMS['mouth_state']), 'eyebrow_state': pick(ps.ENUMS['eyebrow_state']), 'facial_expression': pick(ps.ENUMS['facial_expression']),
             'camera_elevation': pick(ps.ENUMS['camera_elevation'][:3]), 'shot_size': pick(ps.ENUMS['shot_size'])}
    if generator.random() < 0.08:
        value['head_yaw'], value['eye_gaze'] = None, ps.ABSTAIN
    return with_confidence(approach, generator, value)


def with_confidence(approach, generator, value):
    if approach == 'strict_detailed':
        return value
    names = list(value)
    if approach == 'top5_confidence':
        names = generator.sample(ps.FIELD_ORDER, ps.TOP_N)
        levels = sorted((round(generator.uniform(0.35, 0.98), 2) for _ in names), reverse=True)
        return {**{name: value[name] for name in names}, 'confidence': dict(zip(names, levels))}
    return {**value, 'confidence': {name: round(generator.uniform(0.3, 0.98), 2) for name in names}}


def corrupt(approach, entry):
    """Break one entry the way the approach's validator would notice."""
    entry = json.loads(json.dumps(entry))
    if approach == 'strict_detailed':
        entry['head_yaw'] = 30
    elif approach == 'full_confidence':
        entry.pop('confidence', None)
    else:
        extra = next(name for name in ps.FIELD_ORDER if name not in entry)
        entry[extra] = 'unclear'
    return entry


def normal_answer(sample_id, approach, ids):
    generator = rng(sample_id, approach, 'normal')
    return [{'character_id': ref, **pose(approach, generator, [i for i in ids if i != ref])} for ref in ids]


def mirrored_answer(sample_id, approach, ids, mirror_map):
    """The flipped frame's answer: the mirrored normal answer, 20% of the values replaced."""
    original_ids = [mirror_map[ref] for ref in ids]
    normal = {entry['character_id']: entry for entry in normal_answer(sample_id, approach, original_ids)}
    forward = {old: new for new, old in mirror_map.items()}
    generator = rng(sample_id, approach, 'mirror')
    answer = []
    for ref in ids:
        entry = ps.mirror_entry(normal[mirror_map[ref]], forward)
        fresh = pose('strict_detailed' if approach == 'strict_detailed' else 'full_confidence', generator, [])
        for name in ps.FIELD_ORDER:
            if name in entry and name != 'gaze_target' and generator.random() < 0.2:
                entry[name] = fresh[name]
        answer.append(entry)
    return answer


class FakeFrameClient:
    """Stands in for ``runners.LlamaClient``."""

    def chat(self, messages, max_tokens, schema=None, hint=None, thinking=False):
        hint = hint or {}
        approach, ids = hint['approach'], hint['ids']
        if hint['variant'] == 'mirror' and hint.get('mirror_map'):
            answer = mirrored_answer(hint['sample'], approach, ids, hint['mirror_map'])
        else:
            answer = normal_answer(hint['sample'], approach, ids)
        if hint['attempt'] == 1 and answer and rng(hint['sample'], approach, hint['variant'], 'broken').random() < 0.25:
            answer[0] = corrupt(approach, answer[0])
        body = {'composition': 'fake composition', 'setting': 'fake', 'lighting': 'fake', 'camera': 'fake', 'mood': 'fake', 'visible_characters': [],
                'background_people': 'none', 'props': 'none', 'notes': 'none', 'pose_detail': answer}
        speed = 28 + rng(hint['sample'], approach).random() * 20 + (6 if approach != 'strict_detailed' else 0)
        return {'content': json.dumps(body), 'finish_reason': 'stop', 'usage': {'prompt_tokens': 3000, 'completion_tokens': 900 + 150 * len(ids)}, 'simulated_seconds': speed}


class FakeVideoBackend:
    """Stands in for the Qwen3-VL generation inside ``runners.make_generator``."""

    def generate(self, approach, prompt, attempt, window, names):
        names = names or ['girl with red umbrella']
        generator = rng(approach, window, *names, 'video')
        entries = []
        for name in names:
            others = [other for other in names if other != name]
            entries.append({'character': name, 'start': pose(approach, generator, others), 'end': None if generator.random() < 0.1 else pose(approach, generator, others)})
        if attempt == 1 and rng(approach, window, 'broken', *names).random() < 0.25:
            entries[0]['start'] = corrupt(approach, entries[0]['start'])
        body = {'overview': 'fake overview', 'camera': {'movement': 'static', 'detail': 'none'},
                'characters': [{'name': name, 'activity': 'standing', 'travel': 'none', 'motion': 'holds still', 'speaking': 'no', 'emotion': 'calm', 'end': 'same place'} for name in names],
                'events': [], 'uncertainties': 'none', 'pose_detail': entries}
        return json.dumps(body), 2500, 800 + 200 * len(names), 22 + rng(approach, window).random() * 16


class FakeRuntime:
    """Stands in for ``video_model.VideoRuntime`` (nothing is loaded)."""

    def load(self, model_path):
        return None, None, None

    def close(self):
        pass


def fake_chat(system, user):
    """A canned, valid conclusion for the dry run."""
    if 'best_approach' in user:
        return json.dumps({'best_approach': 'strict_detailed', 'headline': 'Dry run: no real model was used.', 'reasoning': 'These words come from the fake backend, not from Qwen 3.6.',
                           'fields_to_add': ['head_yaw', 'eye_gaze'], 'fields_to_avoid': ['head_pitch'], 'next_steps': ['Run the real experiment.'], 'caveats': ['Fake data.']})
    return json.dumps({'verdict': 'adopt_with_changes', 'headline': 'Dry run: no real model was used.', 'strengths': ['Pipeline works end to end.'], 'weaknesses': ['Numbers are fake.'],
                       'reliable_fields': ['body_pose'], 'unreliable_fields': ['head_pitch'], 'recommendation': 'Run the real experiment before deciding.'})
