"""What each sequence-direction call is given, built from current data when its job starts.

The director model reads compact evidence, never database rows: per-cut digests for the whole
sequence, full first-frame and motion evidence only for the cuts of the shot it is writing, spoken
lines by id on the right clock, and a table of who could own each voice. ``fit`` shrinks an input
step by step until it fits one server slot, so a long sequence degrades in detail instead of failing.
"""
import hashlib
import json
import os

from . import prompt_context, subtitle_store
from .assistant import compact
from .composition_brief import shot_props, short

STRUCTURED = ('initial_frame', 'video_analysis', 'consistency_summary')
SUMMARY_TITLE = 'Cut summary'
LEVELS = 4  # detail levels 0 (everything) to 3 (bare minimum)


def _loads(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return None


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip().lower().rstrip('.') not in ('', 'none', 'unknown', 'not visible') else ''


# -------------------------------------------------------------------- loading

def load(conn, sequence, cuts):
    """Project context and every cut's current results, in one pass."""
    episode = dict(conn.execute('SELECT e.id,e.title,e.project_id,e.subtitle_language,p.title AS project_title,p.synopsis FROM episodes e JOIN projects p ON p.id=e.project_id WHERE e.id=?', (sequence['episode_id'],)).fetchone())
    items = [dict(row) for row in conn.execute('SELECT id,kind,title,content FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title', (episode['project_id'], episode['id']))]
    marks = ','.join('?' for _ in cuts)
    results = {cut['id']: {} for cut in cuts}
    for row in conn.execute(f'SELECT id,cut_id,title,content,origin,settings_json FROM analysis_results WHERE superseded_by IS NULL AND cut_id IN ({marks}) ORDER BY rowid DESC', [cut['id'] for cut in cuts]):
        kind = (_loads(row['settings_json']) or {}).get('structured_type')
        if kind in STRUCTURED:
            content = _loads(row['content'])
            if isinstance(content, dict):
                results[row['cut_id']].setdefault(kind, {'id': row['id'], 'content': content})
        elif row['title'] == SUMMARY_TITLE and not kind:
            results[row['cut_id']].setdefault('summary', {'id': row['id'], 'content': row['content']})
    for row in conn.execute(f'SELECT cut_id,content FROM cut_brief_instructions WHERE cut_id IN ({marks})', [cut['id'] for cut in cuts]):
        if row['content'].strip():
            results[row['cut_id']]['answers'] = row['content'].strip()
    return {'project': {'title': episode['project_title'], 'synopsis': episode['synopsis']}, 'episode': {'id': episode['id'], 'title': episode['title']},
            'subtitle_language': episode['subtitle_language'],
            'characters': [{'id': item['id'], 'name': item['title'], 'notes': item['content']} for item in items if item['kind'] == 'character'],
            'scenarios': [{'id': item['id'], 'title': item['title'], 'content': item['content']} for item in items if item['kind'] == 'scenario'],
            'notes': [{'kind': item['kind'], 'title': item['title'], 'content': item['content']} for item in items if item['kind'] not in ('character', 'scenario')],
            'results': results}


def is_ready(results):
    """A cut can be directed once it has a first-frame description and a motion direction."""
    return isinstance((results.get('initial_frame') or {}).get('content'), dict) and prompt_context.is_direction((results.get('video_analysis') or {}).get('content'))


def missing_cuts(ctx, cuts):
    return [cut['cut_index'] + 1 for cut in cuts if not is_ready(ctx['results'].get(cut['id'], {}))]


def people_in(results):
    """Named characters of a cut as ``{name: character_context_id}``: marked in the first frame or followed by the motion direction."""
    people = {}
    frame = (results.get('initial_frame') or {}).get('content') or {}
    direction = ((results.get('video_analysis') or {}).get('content') or {}).get('direction') or {}
    for person in [*(frame.get('visible_characters') or []), *(direction.get('characters') or [])]:
        if isinstance(person, dict) and _text(person.get('character')) and person.get('character_context_id'):
            people.setdefault(person['character'].strip(), person['character_context_id'])
    return people


def speaking_in(results):
    """Named characters whose mouth visibly speaks in the cut."""
    direction = ((results.get('video_analysis') or {}).get('content') or {}).get('direction') or {}
    return [person['character'].strip() for person in direction.get('characters') or []
            if isinstance(person, dict) and person.get('speaking') == 'yes' and _text(person.get('character')) and person.get('character_context_id')]


# -------------------------------------------------------------------- digests

def cut_digest(span, results, level=0):
    """One cut in a few dozen tokens: framing, place, who is on screen, what happens."""
    frame = (results.get('initial_frame') or {}).get('content') or {}
    video = prompt_context.video_view((results.get('video_analysis') or {}).get('content'))
    video = video if isinstance(video, dict) else {}
    props = shot_props(frame) if frame else {}
    action = _text(video.get('summary')) or _text(((results.get('consistency_summary') or {}).get('content') or {}).get('summary'))
    camera = video.get('camera') if isinstance(video.get('camera'), dict) else {}
    entry = {'cut': span['number'], 't': [span['t0'], span['t1']],
             'shot': ' '.join(part for part in (props.get('size'), f"{props['angle']} angle" if props.get('angle') else None) if part),
             'camera': _text(camera.get('movement')),
             'setting': short(_text(frame.get('setting')), 16 if level < 2 else 8),
             'on_screen': list(people_in(results)),
             'unnamed_people': bool(_text(frame.get('background_people'))) or None,
             'speaking': speaking_in(results),
             'action': short(action, 32 if level < 2 else 14)}
    if level < 1:
        entry.update(lighting=short(_text(frame.get('lighting')), 10), mood=short(_text(frame.get('mood')), 8), props=short(_text(frame.get('props')), 14))
    if not is_ready(results):
        entry['analysis'] = 'missing'
    return {key: value for key, value in entry.items() if value not in (None, '', [], {})}


def speaker_of(line, bible=None):
    """Who says a line: the transcript's own character link, else the bible's decision for that voice label."""
    if line.get('character_context_id') and line.get('character_name'):
        return {'name': line['character_name'], 'character_context_id': line['character_context_id'], 'attribution': 'transcript'}
    voice = next((voice for voice in (bible or {}).get('voices') or [] if voice.get('speaker_label') == line.get('speaker_label')), None)
    if voice and voice.get('character'):
        return {'name': voice['character'], 'character_context_id': voice.get('character_context_id'), 'attribution': 'director', 'confidence': voice.get('confidence')}
    if voice and voice.get('kind') == 'narrator':
        return {'name': 'Narrator', 'character_context_id': None, 'attribution': 'director', 'confidence': voice.get('confidence')}
    return {'name': line.get('speaker_label') or 'Unknown speaker', 'character_context_id': None, 'attribution': 'unknown'}


def transcript_text(lines, bible=None, clip=None):
    """Spoken lines as text, one per row: ``L3 12.30-14.05 SPEAKER_01 / Mai: words``."""
    rows = []
    for line in lines:
        who = speaker_of(line, bible)
        label = line.get('speaker_label') or 'Unknown speaker'
        name = f"{label} / {who['name']}" if who['name'] != label else label
        text = line['text'] if clip is None or len(line['text']) <= clip else line['text'][:clip] + '…'
        rows.append(f"{line['id']} {line['t0']:.2f}-{line['t1']:.2f} {name}: {text}")
    return '\n'.join(rows) or 'No spoken lines in this range.'


def voice_evidence(ctx, timeline):
    """Per voice label: how much it says, which characters the user linked it to, and who is visibly speaking meanwhile."""
    voices = {}
    for line in timeline['lines']:
        label = line['speaker_label']
        if not label:
            continue  # unlabelled lines may come from several people; each shot decides them one by one
        voice = voices.setdefault(label, {'speaker_label': label, 'lines': 0, 'seconds': 0.0, 'linked_by_user': {}, 'unlinked_lines': 0, 'visibly_speaking_meanwhile': {}, 'only_person_on_screen': {}, 'first_words': line['text'][:60]})
        voice['lines'] += 1
        voice['seconds'] = round(voice['seconds'] + line['t1'] - line['t0'], 1)
        if line.get('character_name'):
            voice['linked_by_user'][line['character_name']] = voice['linked_by_user'].get(line['character_name'], 0) + 1
            continue
        voice['unlinked_lines'] += 1
        middle = (line['t0'] + line['t1']) / 2
        cut = next((cut for cut in timeline['cuts'] if cut['t0'] <= middle < cut['t1']), None)
        results = ctx['results'].get(cut['id'], {}) if cut else {}
        for name in speaking_in(results):
            voice['visibly_speaking_meanwhile'][name] = voice['visibly_speaking_meanwhile'].get(name, 0) + 1
        on_screen = list(people_in(results))
        if len(on_screen) == 1:
            voice['only_person_on_screen'][on_screen[0]] = voice['only_person_on_screen'].get(on_screen[0], 0) + 1
    return [{key: value for key, value in voice.items() if value not in ({}, 0)} for voice in voices.values()]


def _notes(ctx, level):
    limit = (400, 240, 120, 0)[level]
    return [{'kind': note['kind'], 'title': note['title'], 'content': note['content'][:limit]} for note in ctx['notes'] if limit and note['content'].strip()][:8]


def roster(ctx, level=0, timeline=None):
    """Character records; with a timeline, also where each one appears and how the opening frames describe them.

    The frames' own words are the evidence for anchors and outfits: record notes describe the series,
    not what a character wears in these cuts.
    """
    limit = (300, 200, 120, 60)[level]
    people = [{'name': character['name'], 'notes': character['notes'][:limit]} for character in ctx['characters']]
    if timeline is None or level >= 3:
        return people
    seen = {}
    for cut in timeline['cuts']:
        frame = (ctx['results'].get(cut['id'], {}).get('initial_frame') or {}).get('content') or {}
        for person in frame.get('visible_characters') or []:
            if isinstance(person, dict) and _text(person.get('character')) and person.get('character_context_id'):
                entry = seen.setdefault(person['character'].strip(), {'cuts': [], 'looks': {}})
                entry['cuts'].append(cut['number'])
                look = short(_text(person.get('appearance')), 45)
                if look:
                    entry['looks'].setdefault(look, cut['number'])
    for person in people:
        entry = seen.get(person['name'])
        if entry:
            person['seen_in_cuts'] = entry['cuts']
            # The longest descriptions say the most; a few are enough to spot what stays constant.
            person['appearance_in_first_frames'] = [f'cut {number}: {look}' for look, number in sorted(entry['looks'].items(), key=lambda item: -len(item[0]))[:(4, 3, 2)[level]]]
    return people


# --------------------------------------------------------------------- inputs

def bible_input(ctx, sequence, timeline, level=0):
    return {'project': {'title': ctx['project']['title'], 'synopsis': ctx['project']['synopsis'][:(1200, 800, 400, 200)[level]]},
            'episode': ctx['episode']['title'],
            'director_instruction': sequence['instruction'].strip() or None,
            'sequence': {'seconds': timeline['duration'], 'cuts': [timeline['cuts'][0]['number'], timeline['cuts'][-1]['number']]},
            'characters': roster(ctx, level, timeline),
            'scenario_notes': [{'title': item['title'], 'content': item['content'][:(300, 200, 100, 0)[level]]} for item in ctx['scenarios']],
            'context_notes': _notes(ctx, level),
            'cuts': [cut_digest(cut, ctx['results'].get(cut['id'], {}), level) for cut in timeline['cuts']],
            'transcript': transcript_text(timeline['lines'], clip=None if level < 3 else 80),
            'voices': voice_evidence(ctx, timeline),
            'pauses': timeline['pauses'] if level < 2 else None}


def bible_brief(bible, level=0):
    """The parts of a bible later calls need."""
    brief = {key: bible.get(key) for key in ('logline', 'synopsis', 'style', 'spoken_language', 'arc', 'pacing', 'continuity')}
    brief['scenes'] = [{key: scene.get(key) for key in ('id', 'from_cut', 'to_cut', 'location', 'time_of_day', 'lighting', 'mood', 'scenario')} for scene in bible.get('scenes') or []]
    brief['characters'] = [{key: person.get(key) for key in (('name', 'role', 'anchors', 'outfit', 'arc') if level < 2 else ('name', 'anchors', 'outfit'))} for person in bible.get('characters') or []]
    return brief


def plan_input(ctx, sequence, timeline, bible, level=0):
    candidates = [{key: candidate[key] for key in ('id', 't', 'kind', 'cut', 'pause', 'crosses_line') if candidate.get(key) is not None} for candidate in timeline['candidates']]
    return {'director_instruction': sequence['instruction'].strip() or None,
            'bible': bible_brief(bible, max(level, 1)),
            'sequence_seconds': timeline['duration'],
            'cuts': [cut_digest(cut, ctx['results'].get(cut['id'], {}), max(level, 1)) for cut in timeline['cuts']],
            'transcript': transcript_text(timeline['lines'], bible, clip=(None, 120, 60, 40)[level]),
            'pauses': timeline['pauses'] if level < 3 else None,
            'candidates': candidates}


def _shift_events(view, offset, duration):
    """A motion view with its event times moved onto the shot's clock; events outside the shot are left out."""
    if not isinstance(view, dict):
        return view
    events = []
    for event in view.get('events') or []:
        try:
            start, end = (float(part) + offset for part in str(event.get('seconds')).split('-'))
        except (ValueError, TypeError):
            events.append(event)
            continue
        if end > 0 and start < duration:
            events.append({**event, 'seconds': f'{max(0.0, start):.1f}-{min(duration, end):.1f}'})
    return {key: value for key, value in {**view, 'events': events, 'kind': None}.items() if value not in (None, [], '')}


def _without_ids(frame):
    if not isinstance(frame, dict):
        return frame
    people = [{key: value for key, value in person.items() if key != 'character_context_id'} if isinstance(person, dict) else person for person in frame.get('visible_characters') or []]
    return {**frame, 'visible_characters': people}


def scene_for(bible, cut_number):
    return next((scene for scene in bible.get('scenes') or [] if isinstance(scene.get('from_cut'), int) and isinstance(scene.get('to_cut'), int)
                 and scene['from_cut'] <= cut_number <= scene['to_cut']), None)


def shot_roster(ctx, bible, shot):
    """Named characters a shot may put on screen, in first-seen order: the frame cut first, then the other covered cuts."""
    order = [shot['frame']['cut_id'], *[cut_id for cut_id in shot['cut_ids'] if cut_id != shot['frame']['cut_id']]]
    people = {}
    for cut_id in order:
        for name, character_id in people_in(ctx['results'].get(cut_id, {})).items():
            people.setdefault(name, character_id)
    by_name = {person.get('name'): person for person in bible.get('characters') or []}
    notes = {character['id']: character['notes'] for character in ctx['characters']}
    return [{'name': name, 'character_context_id': character_id, 'anchors': (by_name.get(name) or {}).get('anchors'), 'outfit': (by_name.get(name) or {}).get('outfit'),
             'notes': (notes.get(character_id) or '')[:200] or None} for name, character_id in people.items()]


def shot_dialogue(shot, bible):
    """A shot's spoken lines with the speaker resolved, on the shot's clock."""
    lines = []
    for line in shot['lines']:
        who = speaker_of(line, bible)
        lines.append({**line, 'character_name': who['name'] if who['attribution'] != 'unknown' else None, 'character_context_id': who['character_context_id'],
                      'attribution': who['attribution'], 'confidence': who.get('confidence')})
    return lines


def shot_parts(ctx, bible, shot):
    """The named characters, the scenario picture and the resolved dialogue of one shot."""
    scene = scene_for(bible, shot['frame']['cut'])
    scenario = next((item for item in ctx['scenarios'] if scene and scene.get('scenario') and item['title'] == scene['scenario']), None)
    return shot_roster(ctx, bible, shot), scenario, shot_dialogue(shot, bible)


def shot_input(conn, ctx, sequence, timeline, bible, shots, index, instruction='', previous=None, level=0):
    """Everything for one shot. ``shots`` are the expanded plan entries; ``previous`` is the shot written before it, if any."""
    shot = shots[index]
    duration = shot['duration']
    cuts = []
    for span in (cut for cut in timeline['cuts'] if cut['id'] in shot['cut_ids']):
        results = ctx['results'].get(span['id'], {})
        offset = span['t0'] - shot['t0']
        is_frame = span['id'] == shot['frame']['cut_id']
        frame = _without_ids(prompt_context.frame_view((results.get('initial_frame') or {}).get('content')))
        if isinstance(frame, dict) and not is_frame and level >= 1:
            frame = {key: frame.get(key) for key in ('composition', 'setting', 'camera', 'visible_characters')}
        summary = (results.get('summary') or {}).get('content')
        consistency = (results.get('consistency_summary') or {}).get('content') or {}
        entry = {'cut': span['number'], 'plays_at': [round(max(0.0, offset), 2), round(min(duration, span['t1'] - shot['t0']), 2)],
                 'supplies_the_initial_frame': is_frame or None,
                 'first_frame': frame,
                 'motion': _shift_events(prompt_context.video_view((results.get('video_analysis') or {}).get('content')), offset, duration),
                 'user_decisions': results.get('answers'),
                 'cut_summary': summary[:(4200, 2400, 1200, 0)[level]] if isinstance(summary, str) and level < 3 else None,
                 'review': {key: consistency.get(key) for key in ('clear', 'confusing')} if level < 1 and not summary and consistency else None}
        cuts.append({key: value for key, value in entry.items() if value not in (None, '', {}, [])})
    scene = scene_for(bible, shot['frame']['cut'])
    scenario = next((item for item in ctx['scenarios'] if scene and item['title'] == scene.get('scenario')), None)
    dialogue = [{'line': line['line'], 't': [line['t0'], line['t1']], 'speaker': line['character_name'] or line['speaker_label'] or 'Unknown speaker',
                 'speaker_known_from': {'transcript': 'user link', 'director': f"sequence bible ({line.get('confidence') or 'unrated'})", 'unknown': 'nobody: unconfirmed voice'}[line['attribution']],
                 'text': line['text'], 'continues_from_previous_shot': line['crosses_start'] or None, 'continues_in_next_shot': line['crosses_end'] or None}
                for line in shot_dialogue(shot, bible)]
    subtitles = []
    if level < 2:
        subtitles = [{'t': round(max(0.0, line['start_time'] - shot['start_time']), 1), 'text': line['text']}
                     for line in subtitle_store.lines(sequence['episode_id'], shot['start_time'], shot['end_time'], conn=conn)]

    def neighbour(other):
        return {'shot': other['index'], 'cuts': other['source_cuts'], 'seconds': other['duration'], 'intent': other['intent'], 'staging': other['staging'] or None} if other else None

    def ends_with(other):
        # How the analysis says the previous shot's characters end: its last cut's motion direction.
        direction = ((ctx['results'].get(other['cut_ids'][-1], {}).get('video_analysis') or {}).get('content') or {}).get('direction') or {}
        return [{'name': person.get('character'), 'ends': person.get('end')} for person in direction.get('characters') or [] if isinstance(person, dict) and _text(person.get('character')) and _text(person.get('end'))] or None

    def opens_with(other):
        # How the next shot's initial frame shows its characters.
        frame = (ctx['results'].get(other['frame']['cut_id'], {}).get('initial_frame') or {}).get('content') or {}
        return [{'name': person.get('character'), 'position': person.get('position'), 'pose': person.get('pose')} for person in frame.get('visible_characters') or []
                if isinstance(person, dict) and _text(person.get('character'))] or None

    frame_cut = next(cut for cut in timeline['cuts'] if cut['id'] == shot['frame']['cut_id'])
    payload = {
        'sequence': {key: bible.get(key) for key in ('synopsis', 'style', 'spoken_language', 'arc', 'continuity')},
        'director_instruction': sequence['instruction'].strip() or None,
        'instruction_for_this_shot': instruction.strip() or None,
        'scene': {key: scene.get(key) for key in ('id', 'location', 'time_of_day', 'lighting', 'mood')} if scene else None,
        'scenario_reference': {'title': scenario['title'], 'notes': scenario['content'][:300]} if scenario else None,
        'shot': {'number': shot['index'], 'of': len(shots), 'seconds': duration, 'source_cuts': shot['source_cuts'], 'intent': shot['intent'], 'staging': shot['staging'] or None,
                 'merges_cuts': shot['merged'] or None,
                 'initial_frame': {'from_cut': shot['frame']['cut'],
                                   'is': 'the first frame of that cut, described in its first_frame' if shot['frame']['strategy'] == 'cut_first_frame'
                                   else f"the frame {shot['frame']['time'] - frame_cut['start_time']:.1f} s into that cut, where the previous shot stopped: derive its state from first_frame plus the motion up to then"},
                 'ends_before_its_cut_ends': shot['ends_mid_cut'] or None},
        'previous_shot': {**neighbour(shots[index - 1]), 'ends_with': ends_with(shots[index - 1]) if level < 3 else None,
                          'hands_over': ((previous or {}).get('continuity') or {}).get('to_next')} if index else None,
        'next_shot': {**neighbour(shots[index + 1]), 'opens_with': opens_with(shots[index + 1]) if level < 3 else None} if index + 1 < len(shots) else None,
        'characters': [{key: value for key, value in person.items() if key != 'character_context_id' and value} for person in shot_roster(ctx, bible, shot)],
        'cuts': cuts,
        'dialogue': dialogue or 'Nobody speaks during this shot.',
        'subtitles_for_meaning_only': {'language': ctx['subtitle_language'], 'lines': subtitles} if subtitles else None,
    }
    return {key: value for key, value in payload.items() if value is not None}


def review_input(bible, shots, results, level=0):
    """The whole sequence as written, shot by shot, compact enough to read in one call."""
    rows = []
    for shot in shots:
        content = results.get(shot['key']) or {}
        beats = [f"{beat.get('end')}: {short(beat.get('action'), (40, 24, 14, 8)[level])}" for beat in content.get('beats') or [] if isinstance(beat, dict)]
        rows.append({'shot': shot['index'], 't': [shot['t0'], shot['t1']], 'cuts': shot['source_cuts'], 'scene': shot['scene'], 'intent': shot['intent'], 'staging': shot['staging'] or None,
                     'written': bool(content) or None,
                     'opening_still': content.get('picture_line'), 'subjects': [subject.get('name') for subject in content.get('subjects') or [] if isinstance(subject, dict)],
                     'summary': content.get('summary'), 'camera': (content.get('camera') or {}).get('move'), 'beats': beats,
                     'speech': [{'line': item.get('line'), 'delivery': item.get('delivery'), 'subject': item.get('subject'),
                                 'words': next((line['text'][:40] for line in shot['lines'] if line['line'] == item.get('line')), None)} for item in content.get('speech') or [] if isinstance(item, dict)],
                     'continuity': content.get('continuity') if level < 2 else None})
    return {'bible': bible_brief(bible, level), 'voices': bible.get('voices'), 'shots': [{key: value for key, value in row.items() if value not in (None, [], '')} for row in rows]}


# --------------------------------------------------------------------- budget

def slot_tokens(provider):
    """Context of one generation slot: the local server splits its window between the parallel text jobs."""
    if provider != 'local':
        return 60000
    from . import local_text_runtime, remote_text
    remote = remote_text.bound()
    if remote is not None:  # a remote lane: the other PC reports each model's per-slot context
        return remote_text.model_context(remote.model)
    try:
        total = int(os.environ.get('QWEN36_CONTEXT_SIZE', '65536'))
    except ValueError:
        total = 65536
    return total // local_text_runtime.text_job_concurrency()


def fit(build, system, max_tokens, provider):
    """The most detailed input that fits one slot next to the system prompt and the answer; returns ``(input, level)``."""
    room = slot_tokens(provider) - compact.estimate_tokens(system) - max_tokens - 512
    size = 0
    for level in range(LEVELS):
        payload = build(level)
        size = compact.estimate_tokens(json.dumps(payload, ensure_ascii=False))
        if size <= room:
            return payload, level
    raise RuntimeError(f'The evidence for this step is about {size} tokens and only {max(0, room)} fit the text model; choose a shorter range of cuts or raise QWEN36_CONTEXT_SIZE')


# --------------------------------------------------------------------- hashes

def _hash(value):
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def shot_hashes(ctx, bible, shot):
    """Fingerprints of what a shot was written from; a difference later marks the shot stale and says why."""
    scene = scene_for(bible, shot['frame']['cut'])
    evidence = {cut_id: {kind: (value['id'] if isinstance(value, dict) else _hash(value)) for kind, value in sorted(ctx['results'].get(cut_id, {}).items())} for cut_id in shot['cut_ids']}
    return {'plan': _hash({key: shot[key] for key in ('start_time', 'end_time', 'frame', 'scene', 'intent', 'staging', 'lines')}),
            'bible': _hash({'style': bible.get('style'), 'spoken_language': bible.get('spoken_language'), 'characters': bible.get('characters'), 'voices': bible.get('voices'),
                            'scene': scene, 'continuity': bible.get('continuity')}),
            'evidence': _hash(evidence)}
