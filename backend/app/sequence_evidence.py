"""What each sequence-direction call is given, built from current data when its job starts.

The director model reads compact evidence, never database rows: per-cut digests for the whole
sequence, full first-frame and motion evidence only for the cuts of the shot it is writing, spoken
lines by id on the right clock, and a table of who could own each voice. ``fit`` shrinks an input
step by step until it fits one server slot, so a long sequence degrades in detail instead of failing.
"""
import hashlib
import json
import os
import re

from . import outfit_library, pose_detail, prompt_context, sequence_compile, sequence_prompt, sequence_timeline, subtitle_store
from .assistant import compact, h3_prompt
from .composition_brief import shot_props, short

# The motion is read as ``video_analysis``: for a cut with a keyframe analysis that is its motion-direction view
# (``prompt_context.with_motion``), so readers of motion directions work for both.
STRUCTURED = ('initial_frame', 'video_analysis', 'keyframe_analysis', 'performance_beats', 'consistency_summary', 'last_frame')
SUMMARY_TITLE = 'Cut summary'
NON_VIDEO_SUMMARY_SECTIONS = re.compile(r'^###[ \t]+(?:Dialogue|Cinematic Camera)[ \t]*$.*?(?=^###[ \t]|\Z)', re.M | re.S)
CAMERA_SECTION = re.compile(r'^###[ \t]+Cinematic Camera[ \t]*$([\s\S]*?)(?=^###[ \t]|\Z)', re.M)
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
    items = [dict(row) for row in conn.execute('SELECT id,kind,title,content,unregistered FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title', (episode['project_id'], episode['id']))]
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
    for cut_id in results:
        results[cut_id] = prompt_context.with_motion(results[cut_id])
    for row in conn.execute(f'SELECT cut_id,content FROM cut_brief_instructions WHERE cut_id IN ({marks})', [cut['id'] for cut in cuts]):
        if row['content'].strip():
            results[row['cut_id']]['answers'] = row['content'].strip()
    # An Unregistered card (unregistered.py) is a minor person's handle: on the roster so lines and subjects can name it,
    # never a bible character, an outfit owner or a sheet.
    characters = [{'id': item['id'], 'name': item['title'], 'notes': item['content'], **({'unregistered': True} if item.get('unregistered') else {})}
                  for item in items if item['kind'] == 'character']
    outfits = [{key: row[key] for key in ('id', 'character_context_id', 'name', 'summary', 'description')}
               for row in outfit_library.rows(conn, episode['project_id'], [character['id'] for character in characters if not character.get('unregistered')])]
    return {'project': {'title': episode['project_title'], 'synopsis': episode['synopsis']}, 'episode': {'id': episode['id'], 'title': episode['title']},
            'subtitle_language': episode['subtitle_language'],
            'characters': characters, 'outfits': outfits,
            'scenarios': [{'id': item['id'], 'title': item['title'], 'content': item['content']} for item in items if item['kind'] == 'scenario'],
            'notes': [{'kind': item['kind'], 'title': item['title'], 'content': item['content']} for item in items if item['kind'] not in ('character', 'scenario')],
            'results': results}


def is_ready(results):
    """Keyframed cuts need current performance beats; legacy cuts keep their original motion evidence."""
    if not isinstance((results.get('initial_frame') or {}).get('content'), dict):
        return False
    if 'keyframe_analysis' in results:
        return prompt_context.beats_current(results)
    video = (results.get('video_analysis') or {}).get('content')
    return prompt_context.is_direction(video) or bool(video and 'last_frame' in results)


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


def mouth_activity(results, offset, start, end):
    content = (results.get('keyframe_analysis') or {}).get('content') or {}
    return [{'t': [round(a, 3), round(b, 3)], 'speaking': people}
            for segment in content.get('segments') or []
            if (a := offset + segment['start']) < end and (b := offset + segment['end']) > start
            if (people := [person['name'] for person in segment['characters'] if person.get('speaking') == 'yes'])]


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
    beats = performance_view(results, 0, span['t1'] - span['t0'])
    if beats and level < 2:
        entry['performance'] = [{'t': beat['t'], 'intent': short(beat.get('intent') or '', 14)} for beat in beats[:6] if beat.get('intent')]
    if level < 1:
        entry.update(lighting=short(_text(frame.get('lighting')), 10), mood=short(_text(frame.get('mood')), 8), props=short(_text(frame.get('props')), 14))
    if not is_ready(results):
        entry['analysis'] = 'missing'
    return {key: value for key, value in entry.items() if value not in (None, '', [], {})}


def speaker_of(line, bible=None):
    """Who says a line: the transcript's own character link, else the bible's decision for that voice label."""
    if line.get('compiled') and line.get('character_name'):
        # A reviewed line: the user fixed its speaker.
        return {'name': line['character_name'], 'character_context_id': line.get('character_context_id'), 'attribution': 'reviewed'}
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
    people = [{'name': character['name'], 'notes': character['notes'][:limit]} for character in ctx['characters'] if not character.get('unregistered')]
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

def bible_input(ctx, sequence, timeline, level=0, final_voices=None):
    return {'project': {'title': ctx['project']['title'], 'synopsis': ctx['project']['synopsis'][:(1200, 800, 400, 200)[level]]},
            'episode': ctx['episode']['title'],
            'director_instruction': sequence['instruction'].strip() or None,
            'sequence': {'seconds': timeline['duration'], 'cuts': [timeline['cuts'][0]['number'], timeline['cuts'][-1]['number']]},
            'characters': roster(ctx, level, timeline),
            'unregistered_people': unregistered_note(ctx),
            'outfit_library': outfit_library_view(ctx, level),
            'scenario_notes': [{'title': item['title'], 'content': item['content'][:(300, 200, 100, 0)[level]]} for item in ctx['scenarios']],
            'context_notes': _notes(ctx, level),
            'cuts': [cut_digest(cut, ctx['results'].get(cut['id'], {}), level) for cut in timeline['cuts']],
            'transcript': transcript_text(timeline['lines'], clip=None if level < 3 else 80),
            'voices': voice_evidence(ctx, timeline) if final_voices is None else None,
            'voices_are_final': final_voices,
            'dialogue_reviewed': reviewed(timeline),
            'pauses': timeline['pauses'] if level < 2 else None}


def outfit_library_view(ctx, level=0):
    """The outfits that already exist for each character, so the bible reuses them and proposes only what is missing."""
    names = {character['id']: character['name'] for character in ctx['characters']}
    grouped = {}
    for outfit in ctx.get('outfits') or []:
        item = {'name': outfit['name']}
        if level < 3:
            item['summary'] = (outfit['summary'] or '')[:(200, 160, 80)[min(level, 2)]]
        if level == 0:
            item['description'] = (outfit['description'] or '')[:400]
        grouped.setdefault(outfit['character_context_id'], []).append(item)
    return [{'character': names[cid], 'outfits': items} for cid, items in grouped.items() if cid in names]


def reviewed(timeline):
    """True when the lines come from a dialogue compilation the user reviewed (None otherwise, so the key is left out)."""
    return True if timeline['lines'] and all(line.get('compiled') for line in timeline['lines']) else None


def without_dialogue(summary):
    """A cut summary for video direction; the still writer receives its dedicated camera section separately.

    The words and times of speech are the user's dialogue, never a summary's, so a shot is not shown a second version of them.
    """
    return NON_VIDEO_SUMMARY_SECTIONS.sub('', summary).strip() if isinstance(summary, str) else ''


def bible_brief(bible, level=0):
    """The parts of a bible later calls need."""
    brief = {key: bible.get(key) for key in ('logline', 'synopsis', 'spoken_language', 'arc', 'pacing', 'continuity')}
    brief['scenes'] = [{key: scene.get(key) for key in ('id', 'from_cut', 'to_cut', 'location', 'time_of_day', 'lighting', 'mood', 'scenario')} for scene in bible.get('scenes') or []]
    brief['characters'] = [{key: person.get(key) for key in (('name', 'role', 'anchors', 'outfit', 'arc') if level < 2 else ('name', 'anchors', 'outfit'))} for person in bible.get('characters') or []]
    if bible.get('outfits'):
        brief['outfits'] = [{key: item.get(key) for key in ('character', 'name', 'summary', 'scenes') if item.get(key)} for item in bible['outfits']]
    return brief


def window_subtitles(lines, subtitles):
    low, high = lines[0]['t0'] - 1.0, max(line['t1'] for line in lines) + 1.0
    return [sub for sub in subtitles if sub['t1'] > low and sub['t0'] < high]


def compile_input(ctx, timeline, window, subtitles, previous, decided, level=0):
    """One window of the dialogue compilation: the recognised lines, the subtitles beside them and the evidence for who speaks.

    ``window`` has ``index``, ``of`` and ``lines`` (timeline lines). ``subtitles`` are ``{id: 'U1', t0, t1, text}``
    on the sequence clock. ``previous`` are the last compiled lines of the window before and ``decided`` its voices.
    """
    lines = window['lines']
    low, high = lines[0]['t0'] - 1.0, max(line['t1'] for line in lines) + 1.0
    nearby = window_subtitles(lines, subtitles)
    pairs = sequence_compile.pair_subtitles(lines, nearby)
    cuts = [cut for cut in timeline['cuts'] if cut['t1'] > low and cut['t0'] < high]
    hints = sequence_compile.continuity_hints(lines, nearby, [cut for cut in cuts if cut['t0'] > 0])
    voices = voice_evidence(ctx, {**timeline, 'lines': lines})
    digest = []
    for cut in cuts:
        results = ctx['results'].get(cut['id'], {})
        entry = cut_digest(cut, results, level)
        visible_speech = mouth_activity(results, cut['t0'], low, high)
        if visible_speech and level < 2:
            entry['mouth_activity'] = visible_speech
        digest.append({key: value for key, value in entry.items() if value not in (None, [], '')})
    table = '\n'.join(f"{line['id']} {line['t0']:.2f}-{line['t1']:.2f} {line['speaker_label'] or 'Unknown speaker'}"
                      + (f" / {line['character_name']}" if line.get('character_name') else '') + f": {line['text']}" for line in lines)
    payload = {'next_window': [f"{line['speaker_label']} / {line.get('character_name') or '?'}: {line['text']}" for line in window.get('next', [])],
               'scene_context': _notes(ctx, max(level, 1)) if level < 2 else None,
               'window': {'part': window['index'], 'of': window['of'], 'spoken_language': ctx.get('spoken_language') or 'the language of the recognised lines (not the subtitles)', 'subtitle_language': ctx['subtitle_language'] or None},
               'lines': table,
               'subtitles': '\n'.join(f"{sub['id']} {sub['t0']:.2f}-{sub['t1']:.2f}: {sub['text']}" for sub in nearby) or 'No subtitle lines were read for this part.',
               'pairs': {line_id: [f"{item['subtitle']} overlap {item['overlap']}s similarity {item['similarity']}" for item in found] for line_id, found in pairs.items() if found} if level < 3 else None,
               'hints': [f"{hint['a']}+{hint['b']}: {'; '.join(hint['reasons'])}" for hint in hints],
               'cuts': digest,
               'characters': [person['name'] for person in ctx['characters']],
               'voices': voices if level < 2 else None,
               'previous': [f"{line['t0']:.2f}-{line['t1']:.2f} {line['speaker_label']} / {line['character'] or '?'}: {line['text']}" for line in previous[-(4, 3, 2, 1)[level]:]] or None,
               'voices_decided': [{key: voice.get(key) for key in ('speaker_label', 'character', 'kind')} for voice in decided] or None}
    return {key: value for key, value in payload.items() if value not in (None, [], '')}


def _hidden_events(ctx, timeline, drops, level):
    """Motion events of the stretch of a cut's picture that a moved boundary hides (``drops`` is on that cut's clock)."""
    cut = next(item for item in timeline['cuts'] if item['number'] == drops['cut'])
    view = prompt_context.video_view((ctx['results'].get(cut['id'], {}).get('video_analysis') or {}).get('content'))
    found = []
    for event in (view.get('events') or []) if isinstance(view, dict) else []:
        try:
            start, end = (float(part) for part in str(event.get('seconds')).split('-'))
        except (ValueError, TypeError):
            continue
        if end > drops['from'] and start < drops['to']:
            subject = event.get('subject') if isinstance(event.get('subject'), str) and event.get('subject').strip() else None
            found.append(f"{event.get('seconds')} s {subject + ': ' if subject else ''}{short(_text(event.get('action')), (24, 16, 10)[min(level, 2)])}")
    return found[:3]


def decision_view(ctx, timeline, boundary, bible, level=0):
    """One hard cut inside speech as the plan step reads it: the speech, and what each possible choice does to words and pictures."""
    speech = boundary['speech']
    lines = {line['id']: line for line in timeline['lines']}
    first = lines[speech['lines'][0]]
    who = speaker_of(first, bible)['name']
    left, right = boundary['after_cut'], boundary['before_cut']
    if speech['kind'] == 'inside_line':
        heard = f"{first['id']} ({who}, {first['t0']:.2f}-{first['t1']:.2f}) is being spoken at the hard cut: \"{speech['words_before']}\" before it, \"{speech['words_after']}\" after it"
    else:
        second = lines[speech['lines'][1]]
        heard = f"{first['id']} and {second['id']} ({who}) are one sentence and the hard cut falls in the pause between them: \"{speech['words_before']}\" / \"{speech['words_after']}\""
    if len(speech['keeps']) > len(speech['lines']):
        heard += f"; the whole phrase is {speech['keeps'][0]}–{speech['keeps'][-1]} ({speech['phrase'][0]:.2f}-{speech['phrase'][1]:.2f})"
    phrase = speech['keeps'][0] if len(speech['keeps']) == 1 else f"{speech['keeps'][0]}–{speech['keeps'][-1]}"
    options = {}
    for choice, option in boundary['options'].items():
        if choice == 'keep_cut':
            text = f"stays at {boundary['t']:.2f}: the shot of cut {left} ends on \"{speech['words_before']}\" and the shot of cut {right} opens on \"{speech['words_after']}\""
        elif choice == 'before_line':
            text = (f"moves {abs(option['shift']):.2f} s earlier to {option['t']:.2f}: the shot of cut {right} starts before {phrase} and carries it whole; "
                    f"cut {left} is hidden from {option['drops']['from']:.1f} to {option['drops']['to']:.1f} s of its own clock")
        else:
            text = f"moves {option['shift']:.2f} s later to {option['t']:.2f}: the shot of cut {left} runs on until {phrase} has ended and carries it whole; the first {option['shift']:.2f} s of cut {right} are hidden"
        if choice != 'keep_cut' and level < 3:
            hidden = _hidden_events(ctx, timeline, option['drops'], level)
            text += f"; the hidden picture shows {'; '.join(hidden)}" if hidden else '; no motion event is recorded in the hidden picture'
        options[choice] = text
    return {'id': boundary['id'], 'hard_cut': boundary['t'], 'cuts': [left, right], 'speech': heard, 'options': options, 'not_possible': boundary['unavailable'] or None}


def fixed_view(ctx, timeline, boundary, bible, level=0):
    """A hard cut whose shot boundary the user placed, as the plan step reads it: where it sits, what each shot hears, the hidden picture and the later shot's opening frame."""
    fixed = boundary['fixed']
    t, shift = fixed['t'], fixed['shift']
    left, right = boundary['after_cut'], boundary['before_cut']
    cuts = {cut['number']: cut for cut in timeline['cuts']}
    drops = None
    if abs(shift) <= 0.002:
        where = f"on the hard cut at {t:.2f}"
        opening = f"the first frame of cut {right}"
    elif shift < 0:
        drops = {'cut': left, 'from': round(t - cuts[left]['t0'], 3), 'to': round(cuts[left]['t1'] - cuts[left]['t0'], 3)}
        where = (f"{abs(shift):.2f} s before the hard cut, at {t:.2f}: the shot of cut {right} starts early and plays that much of cut {left}'s audio over its own picture; "
                 f"cut {left} is hidden from {drops['from']:.1f} to {drops['to']:.1f} s of its own clock")
        opening = f"the first frame of cut {right}, held until the cut's own motion starts {abs(shift):.2f} s into the shot"
    else:
        drops = {'cut': right, 'from': 0.0, 'to': shift}
        where = f"{shift:.2f} s after the hard cut, at {t:.2f}: the shot of cut {left} runs on over that much of cut {right}'s audio; the first {shift:.2f} s of cut {right} are hidden"
        opening = f"a moment {shift:.2f} s into cut {right}, after what the shot of cut {left} carried"
    lines = {line['id']: line for line in timeline['lines']}
    split = sequence_timeline.split_at(timeline['lines'], t)
    if split:
        who = speaker_of(lines[split['lines'][0]], bible)['name']
        heard = f"divides {' and '.join(split['lines'])} ({who}): \"{split['before']}\" ends the shot of cut {left}, \"{split['after']}\" opens the shot of cut {right}"
    elif boundary['speech']:
        speech = boundary['speech']
        phrase = speech['keeps'][0] if len(speech['keeps']) == 1 else f"{speech['keeps'][0]}–{speech['keeps'][-1]}"
        heard = f"keeps {phrase} whole in the shot of cut {right if t <= speech['phrase'][0] + 0.05 else left}"
    else:
        heard = 'divides no speech'
    entry = {'id': boundary['id'], 'hard_cut': boundary['t'], 'cuts': [left, right], 'boundary': where, 'speech': heard, 'opening_frame': opening}
    if drops and level < 3:
        hidden = _hidden_events(ctx, timeline, drops, level)
        entry['hidden_picture'] = '; '.join(hidden) if hidden else 'no motion event is recorded in the hidden picture'
    return entry


def parts_of(timeline, cut):
    """How many shots the app makes of a cut (more than one only for a cut longer than one take)."""
    try:
        return len(sequence_timeline.partition(timeline, cut, (cut['t0'], cut['start_time']), (cut['t1'], cut['end_time']))) - 1
    except sequence_timeline.PlanError:
        return None


def timing_input(ctx, timeline, level):
    boundaries = timeline['boundaries']
    decisions = [decision_view(ctx, timeline, item, {}, level) for item in boundaries if item['speech'] and not item['fixed']]
    cut_numbers = {number for item in decisions for number in item['cuts']}
    return {'decisions': decisions,
            'fixed_by_user': [{'id': item['id'], 't': item['fixed']['t']} for item in boundaries if item['fixed']],
            'cuts': [cut_digest(cut, ctx['results'].get(cut['id'], {}), level) for cut in timeline['cuts'] if cut['number'] in cut_numbers]}


def plan_input(ctx, sequence, timeline, bible, level=0):
    cuts = []
    for cut in timeline['cuts']:
        entry = cut_digest(cut, ctx['results'].get(cut['id'], {}), max(level, 1))
        parts = parts_of(timeline, cut)
        if parts and parts > 1:
            entry['parts'] = parts
        cuts.append(entry)
    timing = timeline.get('dialogue_timing', {})
    decisions = [{key: value for key, value in decision_view(ctx, timeline, boundary, bible, level).items() if value is not None}
                 for boundary in timeline['boundaries'] if boundary['speech'] and not boundary.get('fixed') and boundary['id'] not in timing]
    fixed = [fixed_view(ctx, timeline, boundary, bible, level) for boundary in timeline['boundaries'] if boundary.get('fixed')]
    if decisions:
        open_text = decisions
    elif any(boundary['speech'] for boundary in timeline['boundaries']):
        open_text = 'Dialogue timing or user placements already cover every boundary inside speech.' if timing else 'Nothing to decide: the user placed the boundary at every hard cut inside speech.'
    else:
        open_text = 'No hard cut falls inside speech: every boundary stays on its hard cut.'
    return {'director_instruction': sequence['instruction'].strip() or None,
            'dialogue_reviewed': reviewed(timeline),
            'bible': bible_brief(bible, max(level, 1)),
            'sequence_seconds': timeline['duration'],
            'cuts': cuts,
            'transcript': transcript_text(timeline['lines'], bible, clip=(None, 120, 60, 40)[level]),
            'decisions': open_text,
            **({'dialogue_timing': [{'cuts': [item['after_cut'], item['before_cut']], 'hard_cut': item['t'],
                                    'start': item['options'][timing[item['id']]['choice']]['t'],
                                    'reason': timing[item['id']]['reason']} for item in timeline['boundaries'] if item['id'] in timing]} if timing else {}),
            **({'fixed_by_user': fixed} if fixed else {})}


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


def shifted_range(value, offset, duration):
    try:
        start, end = (float(part) + offset for part in value.rstrip('s').split('-'))
    except (AttributeError, ValueError):
        return None
    if end > 0 and start < duration:
        return [round(max(0.0, start), 3), round(min(duration, end), 3)]
    return None


def performance_view(results, offset, duration):
    if not prompt_context.beats_current(results):
        return []
    view = prompt_context.beats_view(results['performance_beats']['content'])
    return [{**{key: val for key, val in beat.items() if key not in ('at', 'speech')}, 't': times,
             **({'delivery_hints': [{k: line[k] for k in ('speaker', 'delivery') if k in line} for line in beat['speech']]} if beat.get('speech') else {})}
            for beat in (view or {}).get('beats', []) if (times := shifted_range(beat.get('at'), offset, duration))]


def shot_keyframes(results, offset, duration, level):
    view = prompt_context.keyframe_view((results.get('keyframe_analysis') or {}).get('content'), 'review' if level < 2 else 'brief') or {}
    changes = [{**{key: val for key, val in item.items() if key not in ('pair', 'at')}, 't': times}
               for item in view.get('changes', []) if (times := shifted_range(item.get('at'), offset, duration))]
    if level >= 3:
        return None
    frames = [{**frame, 't': round(frame['t'] + offset, 3)} for frame in view.get('frames', []) if isinstance(frame.get('t'), (int, float))]
    preceding = [frame for frame in frames if frame['t'] < 0]
    following = [frame for frame in frames if frame['t'] > duration]
    states = [*preceding[-1:], *(frame for frame in frames if 0 <= frame['t'] <= duration), *following[:1]]
    return {key: value for key, value in {'states': states, 'changes': changes}.items() if value}


def still_camera(results):
    """The camera and look chosen for this cut's opening frame, copied from its Cut summary.

    A line that names a style or medium (a summary written before the check, or edited by hand) is left out: the still
    is a photograph through this camera, and a style word contradicts it.
    """
    summary = (results.get('summary') or {}).get('content') or ''
    section = CAMERA_SECTION.search(summary)
    if not section:
        return ''
    lines = (line.strip().lstrip('-* ').replace('**', '') for line in section.group(1).splitlines() if line.strip())
    return '\n'.join(line for line in lines if not h3_prompt.style_words(line, camera=True))


def _without_ids(frame, level=0):
    """A first-frame view with the characters named only. The pose facts shrink with the detail level: all of them, then state and what is looked at, then none."""
    if not isinstance(frame, dict):
        return frame
    people = []
    for person in frame.get('visible_characters') or []:
        if isinstance(person, dict):
            person = {key: value for key, value in person.items() if key != 'character_context_id'}
            pose = person.get('pose_detail')
            if isinstance(pose, dict) and level >= 2:
                keep = {key: pose[key] for key in ('state', 'looking_at') if key in pose} if level == 2 else {}
                person = {key: value for key, value in person.items() if key != 'pose_detail'}
                if keep:
                    person['pose_detail'] = keep
        people.append(person)
    return {**frame, 'visible_characters': people}


def scene_for(bible, cut_number):
    return next((scene for scene in bible.get('scenes') or [] if isinstance(scene.get('from_cut'), int) and isinstance(scene.get('to_cut'), int)
                 and scene['from_cut'] <= cut_number <= scene['to_cut']), None)


def shot_roster(ctx, bible, shot):
    """Named characters a shot may put on screen: those of its own cut (a neighbour whose audio it borrows is not shown)."""
    people = dict(people_in(ctx['results'].get(shot['cut_id'], {})))
    by_name = {person.get('name'): person for person in bible.get('characters') or []}
    notes = {character['id']: character['notes'] for character in ctx['characters']}
    cards = {character['id'] for character in ctx['characters'] if character.get('unregistered')}
    named = {}
    for item in bible.get('outfits') or []:
        named.setdefault(item.get('character'), []).append(item.get('name'))
    return [{'name': name, 'character_context_id': character_id, 'anchors': (by_name.get(name) or {}).get('anchors'), 'outfit': (by_name.get(name) or {}).get('outfit'),
             **({'outfit_names': named[name]} if named.get(name) else {}), **({'unregistered': True} if character_id in cards else {}),
             'notes': (notes.get(character_id) or '')[:200] or None} for name, character_id in people.items()]


def unregistered_note(ctx):
    """What an ``Unregistered N`` name means, when the project has any: None otherwise, so inputs without one are unchanged."""
    from . import unregistered
    names = [character['name'] for character in ctx['characters'] if character.get('unregistered')]
    return {'names': names, 'rule': unregistered.RULE + ' They have no character record, no sheet and no outfit; never list them as bible characters.'} if names else None


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
    scene = scene_for(bible, shot['cut'])
    scenario = next((item for item in ctx['scenarios'] if scene and scene.get('scenario') and item['title'] == scene['scenario']), None)
    return shot_roster(ctx, bible, shot), scenario, shot_dialogue(shot, bible)


def shot_speakers(ctx, timeline, bible, shot):
    """Per speaker heard in a shot: what they say, where they can be seen, and the ranges the user fixed.

    Only the shot's own cut is pictured. A neighbouring cut whose audio the shot borrows to keep a phrase
    whole is not shown: the shot's own picture goes on over those seconds, so what that cut showed says
    nothing about who is seen, and it is left out.
    """
    dialogue = shot_dialogue(shot, bible)
    rows = []
    for speaker in sequence_prompt.speakers(dialogue):
        lines = [line for line in dialogue if sequence_prompt.speaker_key(line) == speaker]
        per_cut = []
        for span in (cut for cut in timeline['cuts'] if cut['id'] in shot['cut_ids']):
            plays_at = [round(max(0.0, span['t0'] - shot['t0']), 2), round(min(shot['duration'], span['t1'] - shot['t0']), 2)]
            if span['id'] != shot['cut_id']:
                per_cut.append({'cut': span['number'], 'plays_at': plays_at,
                                'picture': "not shown: only its audio is borrowed, and this shot's own picture goes on over these seconds"})
                continue
            results = ctx['results'].get(span['id'], {})
            direction = ((results.get('video_analysis') or {}).get('content') or {}).get('direction') or {}
            moving = next((person.get('speaking') for person in direction.get('characters') or [] if isinstance(person, dict) and _text(person.get('character')) == speaker), None)
            per_cut.append({'cut': span['number'], 'plays_at': plays_at, 'on_screen': speaker in people_in(results), 'mouth_moves': moving if moving in ('yes', 'no', 'unclear') else None,
                            'picture': "this shot's picture"})
        fixed = [{'from': item['from'], 'to': item['to'], 'delivery': item['delivery']} for item in shot.get('fixed_voices') or [] if item.get('speaker') == speaker]
        rows.append({'speaker': speaker, 'lines': [line['line'] for line in lines], 'seconds': round(sum(line['t1'] - line['t0'] for line in lines), 1),
                     'per_cut': [{key: value for key, value in entry.items() if value is not None} for entry in per_cut],
                     'fixed_by_user': fixed or None})
    return [{key: value for key, value in row.items() if value is not None} for row in rows]


def _phrase(edge):
    keeps = edge.get('keeps') or []
    return 'the phrase' if not keeps else keeps[0] if len(keeps) == 1 else f'{keeps[0]}–{keeps[-1]}'


def _placed_words(edge, shot, opens, shift):
    """How a shot meets a boundary the user placed by hand: where it sits against the shot's own cut, without assuming it keeps a phrase whole."""
    how, keeps = edge['how'], bool(edge.get('keeps'))
    if how == 'keep_cut':
        return f"{'opens' if opens else 'ends'} exactly on the hard cut, where the user placed the boundary"
    if opens and how == 'before_line':
        return (f"starts {shift:.2f} s before its own cut, where the user placed the boundary{', and carries ' + _phrase(edge) + ' whole' if keeps else ''}; "
                f"until {shift:.2f} s the picture holds the opening state of the still, and the cut's own motion starts at {shift:.2f} s")
    if opens:
        return f'starts {shift:.2f} s into its own cut, where the user placed the boundary; what the cut showed before that moment has already happened'
    if how == 'before_line':
        return f'ends {shift:.2f} s before its own cut ends, where the user placed the boundary; the last {shift:.2f} s of the cut are not shown'
    return (f"runs {shift:.2f} s past the end of its own cut, where the user placed the boundary{', to finish ' + _phrase(edge) if keeps else ''}: "
            f"from {shot['duration'] - shift:.2f} s to the end it keeps this framing and the acting of whoever speaks continues to the last word")


def edge_words(shot, side):
    """How a shot meets its neighbour on one side, in plain words with seconds on the shot's clock; None at the ends of the sequence."""
    edge = shot[side]
    how, opens = edge['how'], side == 'opens'
    if how in ('start', 'end'):
        return None
    shift = abs(edge.get('shift') or 0.0)
    if how == 'continuation':
        text = 'continues the take of the previous shot, which showed the earlier part of the same cut' if opens else 'the same cut goes on in the next shot, which continues this take'
    elif edge.get('placed'):
        text = _placed_words(edge, shot, opens, shift)
    elif how == 'keep_cut':
        text = 'opens exactly on the hard cut' if opens else 'ends exactly on the hard cut'
    elif opens and how == 'before_line':
        text = (f'starts {shift:.2f} s before its own cut, in the silence before {_phrase(edge)}, and carries that speech whole; '
                f'until {shift:.2f} s the picture holds the opening state of the still, and the cut\'s own motion starts at {shift:.2f} s')
    elif opens:
        text = f'starts {shift:.2f} s into its own cut, after {_phrase(edge)}, which the previous shot finished; what the cut showed before that moment has already happened'
    elif how == 'before_line':
        text = f'ends {shift:.2f} s before its own cut ends, in the silence before {_phrase(edge)}, which the next shot carries; the last {shift:.2f} s of the cut are not shown'
    else:
        text = (f'runs {shift:.2f} s past the end of its own cut to finish {_phrase(edge)}: from {shot["duration"] - shift:.2f} s to the end it keeps this framing '
                f'and the speaker\'s acting continues to the last word')
    split = edge.get('split')
    if split:
        lines = ' and '.join(split['lines'])
        text += (f'; it opens inside {lines}: "{split["before"]}" was heard in the previous shot and "{split["after"]}" is heard here from 0 s' if opens
                 else f'; it ends inside {lines}: "{split["before"]}" is heard here up to the end and "{split["after"]}" continues in the next shot')
    result = {'at': 0.0 if opens else shot['duration'], 'how': text}
    if edge.get('by') == 'user':
        result['decided_by'] = 'the user'
    if edge.get('reason'):
        result['reason'] = edge['reason']
    return result


def shot_input(conn, ctx, sequence, timeline, bible, shots, index, instruction='', previous=None, level=0, earlier=(), from_review=False, reference=None, shows=None):
    """Everything for one shot. ``shots`` are the expanded plan entries; ``previous`` is the shot written before it, if any.

    ``instruction`` is what this rewrite was asked to do and ``earlier`` the user's instructions of earlier rewrites of the shot (oldest first):
    all of them stay in force. An instruction that came from the automatic review (``from_review``) is a suggestion, below the user's own.
    ``reference`` is the shot's reference keyframe (``sequence_reference.resolve``) and ``shows`` that frame's analysis
    (``sequence_reference.evidence``); ``picture_attached`` is set by the job once it knows whether the picture goes along.
    """
    shot = shots[index]
    duration = shot['duration']
    cuts = []
    for span in (cut for cut in timeline['cuts'] if cut['id'] in shot['cut_ids']):
        offset = span['t0'] - shot['t0']
        plays_at = [round(max(0.0, offset), 2), round(min(duration, span['t1'] - shot['t0']), 2)]
        if span['id'] != shot['cut_id']:
            # A neighbour whose audio this shot borrows so that a phrase stays whole: its picture is not part of this shot.
            cuts.append({'cut': span['number'], 'plays_at': plays_at, 'picture': 'not shown: only its audio plays in this shot'})
            continue
        results = ctx['results'].get(span['id'], {})
        summary = without_dialogue((results.get('summary') or {}).get('content') or '')
        consistency = (results.get('consistency_summary') or {}).get('content') or {}
        entry = {'cut': span['number'], 'plays_at': plays_at,
                 'first_frame': _without_ids(prompt_context.frame_view((results.get('initial_frame') or {}).get('content')), level),
                 'motion': _shift_events(prompt_context.video_view((results.get('video_analysis') or {}).get('content')), offset, duration),
                 'keyframes': shot_keyframes(results, offset, duration, level),
                 'performance_beats': performance_view(results, offset, duration),
                 'legacy_last_frame': _without_ids(prompt_context.frame_view((results.get('last_frame') or {}).get('content')), level)
                                      if not prompt_context.is_direction((results.get('video_analysis') or {}).get('content')) and level < 2 else None,
                 'still_camera': still_camera(results),
                 'user_decisions': results.get('answers'),
                 'cut_summary': summary[:(6200, 3000, 1200, 0)[level]] if summary and level < 3 else None,
                 'review': {key: consistency.get(key) for key in ('clear', 'confusing')} if level < 1 and not summary and consistency else None}
        cuts.append({key: value for key, value in entry.items() if value not in (None, '', {}, [])})
    scene = scene_for(bible, shot['cut'])
    scenario = next((item for item in ctx['scenarios'] if scene and item['title'] == scene.get('scenario')), None)
    dialogue = [{'line': line['line'], 't': [line['t0'], line['t1']], 'speaker': line['character_name'] or line['speaker_label'] or 'Unknown speaker',
                 'speaker_known_from': {'reviewed': 'user-reviewed dialogue', 'transcript': 'user link', 'director': f"sequence bible ({line.get('confidence') or 'unrated'})", 'unknown': 'nobody: unconfirmed voice'}[line['attribution']],
                 'text': line['text'], 'continues_from_previous_shot': line['crosses_start'] or None, 'continues_in_next_shot': line['crosses_end'] or None}
                for line in shot_dialogue(shot, bible)]
    subtitles = []
    # The user's reviewed dialogue was reconciled with the subtitles already: a second version of the words and times would only compete with it.
    if level < 2 and not reviewed(timeline):
        subtitles = [{'t': round(max(0.0, line['start_time'] - shot['start_time']), 1), 'text': line['text']}
                     for line in subtitle_store.lines(sequence['episode_id'], shot['start_time'], shot['end_time'], conn=conn)]

    def neighbour(other):
        return {'shot': other['index'], 'cut': other['cut'], 'seconds': other['duration'], 'intent': other['intent']} if other else None

    def ends_with(other):
        # How the analysis says the previous shot's characters end: its cut's motion direction.
        direction = ((ctx['results'].get(other['cut_id'], {}).get('video_analysis') or {}).get('content') or {}).get('direction') or {}
        return [{'name': person.get('character'), 'ends': person.get('end')} for person in direction.get('characters') or [] if isinstance(person, dict) and _text(person.get('character')) and _text(person.get('end'))] or None

    def opens_with(other):
        # How the next shot's initial frame shows its characters.
        frame = (ctx['results'].get(other['cut_id'], {}).get('initial_frame') or {}).get('content') or {}
        return [{'name': person.get('character'), 'position': person.get('position'), 'pose': person.get('pose'),
                 **({'reading': pose_detail.brief(person.get('pose_detail'))} if pose_detail.brief(person.get('pose_detail')) else {})}
                for person in frame.get('visible_characters') or [] if isinstance(person, dict) and _text(person.get('character'))] or None

    own = next(cut for cut in timeline['cuts'] if cut['id'] == shot['cut_id'])
    later = shot['frame']['time'] - own['start_time']
    # The reference's second is on this shot's clock (``sequence_reference.resolve``: episode time minus the shot's start),
    # like the dialogue and the keyframe states above, whether the shot starts on its hard cut or before or after it.
    landing = sequence_prompt.reference_landing(reference['at'], duration) if reference else None
    lands = ({'beat_ends_at': reference['at']} if landing == 'edge' else {'within_first_beat': True} if landing == 'opening'
             else {'at_end_of_shot': True} if landing == 'end' else {})
    payload = {
        'sequence': {key: bible.get(key) for key in ('synopsis', 'spoken_language', 'arc', 'continuity')},
        'director_instruction': sequence['instruction'].strip() or None,
        'instruction_for_this_shot': None if from_review else instruction.strip() or None,
        'earlier_instructions_for_this_shot': list(earlier) or None,
        'review_suggestion': instruction.strip() or None if from_review else None,
        'scene': {key: scene.get(key) for key in ('id', 'location', 'time_of_day', 'lighting', 'mood')} if scene else None,
        'scenario_reference': {'title': scenario['title'], 'notes': scenario['content'][:300]} if scenario else None,
        'shot': {'number': shot['index'], 'of': len(shots), 'seconds': duration, 'cut': shot['cut'], 'part': f"{shot['part'][0]} of {shot['part'][1]}" if shot.get('part') else None,
                 'audio_from_cuts': shot['source_cuts'] if len(shot['source_cuts']) > 1 else None, 'intent': shot['intent'],
                 'initial_frame': {'from_cut': shot['frame']['cut'],
                                   'is': 'the first frame of that cut, described in its first_frame' if shot['frame']['strategy'] == 'cut_first_frame'
                                   else f"the frame {later:.1f} s into that cut, where the previous shot stopped: derive its state from first_frame plus the motion up to then"
                                   if shot['frame']['continues_previous_shot']
                                   else f"the frame {later:.1f} s into that cut (the previous shot ran past the hard cut to finish a phrase): derive its state from first_frame plus the motion up to then"},
                 'opens': edge_words(shot, 'opens'), 'closes': edge_words(shot, 'closes'),
                 'beats': {'at_most': sequence_prompt.beat_limit(duration, reference_edge=landing == 'edge'), 'each_longer_than': round(sequence_prompt.beat_gap(duration), 3)}},
        'previous_shot': {**neighbour(shots[index - 1]), 'ends_with': ends_with(shots[index - 1]) if level < 3 else None,
                          'hands_over': ((previous or {}).get('continuity') or {}).get('to_next')} if index else None,
        'next_shot': {**neighbour(shots[index + 1]), 'opens_with': opens_with(shots[index + 1]) if level < 3 else None} if index + 1 < len(shots) else None,
        'characters': [{key: value for key, value in person.items() if key != 'character_context_id' and value} for person in shot_roster(ctx, bible, shot)],
        'unregistered_people': unregistered_note(ctx) if any(person.get('unregistered') for person in shot_roster(ctx, bible, shot)) else None,
        'cuts': cuts,
        'dialogue_reviewed': reviewed(timeline),
        'dialogue': dialogue or 'Nobody speaks during this shot.',
        'speakers': shot_speakers(ctx, timeline, bible, shot) or None,
        'subtitles_for_meaning_only': {'language': ctx['subtitle_language'], 'lines': subtitles} if subtitles else None,
        # The user's chosen staging moment is never shrunk away by the detail level: it is what the user asked for.
        'reference_frame': {'frame': reference.get('frame'), 'at': reference['at'], **lands, 'chosen_by': 'user', 'shows': shows or 'not analysed: no keyframe states or boxes for this frame',
                            'picture_attached': False} if reference else None,
    }
    payload['shot'] = {key: value for key, value in payload['shot'].items() if value is not None}
    return {key: value for key, value in payload.items() if value is not None}


def edge_brief(edge):
    """An edge in a few words for the review: ``after_line +0.80 s keeping L5``, ``keep_cut dividing L7 (reason)``."""
    how = edge['how']
    if how in ('start', 'end'):
        return None
    if edge.get('placed'):
        text = 'placed by user on the hard cut' if how == 'keep_cut' else f"placed by user {edge.get('shift', 0):+.2f} s" + (f" keeping {_phrase(edge)}" if edge.get('keeps') else '')
    else:
        text = how if how in ('continuation', 'keep_cut') else f"{how} {edge.get('shift', 0):+.2f} s keeping {_phrase(edge)}"
    if edge.get('split'):
        text += f" dividing {' and '.join(edge['split']['lines'])}"
    if edge.get('reason'):
        text += f" ({short(edge['reason'], 20)})"
    return text


def voices_text(content, shot, bible):
    """A written shot's voices in one line, e.g. ``Mai on_screen 0-4.2, narrative 4.2-9 (user); Ken off_screen``."""
    voices = sequence_prompt.voices_of(content, shot_dialogue(shot, bible), shot['duration'])
    parts = {}
    for entry in voices:
        span = '' if entry['from'] == 0 and entry['to'] >= shot['duration'] - 0.001 else f" {entry['from']:g}-{entry['to']:g}"
        parts.setdefault(entry['speaker'], []).append(f"{entry['delivery']}{span}{' (user)' if entry.get('fixed') else ''}")
    return '; '.join(f"{speaker} {', '.join(items)}" for speaker, items in parts.items()) or None


def shot_states(ctx, shots):
    """How the analysis says each shot's characters open and end: ``{shot index: {opens_with, ends_with}}``.

    The opening state is the first-frame reading, only when the shot opens on its cut's first frame. The
    end state is the motion direction's, only when the shot reaches the end of its cut.
    """
    states = {}
    for shot in shots:
        results = ctx['results'].get(shot['cut_id'], {})
        entry = {}
        if shot['frame']['strategy'] == 'cut_first_frame':
            frame = (results.get('initial_frame') or {}).get('content') or {}
            opens = [f"{person['character'].strip()}: {pose_detail.brief(person.get('pose_detail'))}" for person in frame.get('visible_characters') or []
                     if isinstance(person, dict) and _text(person.get('character')) and pose_detail.brief(person.get('pose_detail'))]
            if opens:
                entry['opens_with'] = opens
        if not shot.get('part') or shot['part'][0] == shot['part'][1]:
            direction = ((results.get('video_analysis') or {}).get('content') or {}).get('direction') or {}
            ends = [f"{person['character'].strip()}: {short(_text(person['end']), 16)}" for person in direction.get('characters') or []
                    if isinstance(person, dict) and _text(person.get('character')) and _text(person.get('end'))]
            if ends:
                entry['ends_with'] = ends
        if entry:
            states[shot['index']] = entry
    return states


def review_input(bible, shots, results, level=0, states=None, instructions=None):
    """The whole sequence as written, shot by shot, compact enough to read in one call.

    ``instructions`` are the user's instructions per shot key: what the user asked for is not for the review to undo.
    """
    rows = []
    for shot in shots:
        content = results.get(shot['key']) or {}
        beats = [f"{beat.get('end')}: {short(beat.get('action'), (40, 24, 14, 8)[level])}" for beat in content.get('beats') or [] if isinstance(beat, dict)]
        rows.append({'shot': shot['index'], 't': [shot['t0'], shot['t1']], 'cut': shot['cut'], 'part': f"{shot['part'][0]}/{shot['part'][1]}" if shot.get('part') else None,
                     'scene': shot['scene'], 'intent': shot['intent'], 'opens': edge_brief(shot['opens']), 'closes': edge_brief(shot['closes']),
                     'written': bool(content) or None,
                     'user_instructions': [short(text, (80, 60, 40, 24)[level]) for text in (instructions or {}).get(shot['key']) or []] or None,
                     'opening_still': content.get('picture_line'), 'subjects': [subject.get('name') for subject in content.get('subjects') or [] if isinstance(subject, dict)],
                     'summary': content.get('summary'), 'camera': (content.get('camera') or {}).get('move'), 'beats': beats,
                     'voices': voices_text(content, shot, bible) if content else None,
                     # The seconds let the review tell where speech really is; "none" says outright that nobody speaks in the shot.
                     'lines': [f"{line['line']} {line['t0']:.1f}-{line['t1']:.1f} {line['character_name'] or line['speaker_label'] or 'Unknown speaker'}: {line['text'][:40]}"
                               for line in shot_dialogue(shot, bible)] or 'none',
                     **({key: (states or {}).get(shot['index'], {}).get(key) for key in ('opens_with', 'ends_with')} if level < 3 else {}),
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


def fit(build, system, max_tokens, provider, reserve=0):
    """The most detailed input that fits one slot next to the system prompt and the answer; returns ``(input, level)``.

    ``reserve`` is what attached pictures take (``assistant.images.TOKENS_PER_PICTURE`` each).
    """
    room = slot_tokens(provider) - compact.estimate_tokens(system) - max_tokens - 512 - reserve
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
    scene = scene_for(bible, shot['cut'])
    evidence = {cut_id: {kind: (value['id'] if isinstance(value, dict) else _hash(value)) for kind, value in sorted(ctx['results'].get(cut_id, {}).items())} for cut_id in shot['cut_ids']}
    plan = {key: shot[key] for key in ('start_time', 'end_time', 'frame', 'scene', 'intent', 'opens', 'closes', 'lines')}
    if shot.get('fixed_voices'):
        plan['fixed_voices'] = shot['fixed_voices']  # only when set, so shots without any keep their fingerprint
    kept = {'spoken_language': bible.get('spoken_language'), 'characters': bible.get('characters'), 'voices': bible.get('voices'),
            'scene': scene, 'continuity': bible.get('continuity')}
    if bible.get('outfits'):
        kept['outfits'] = [[item.get('character'), item.get('name'), item.get('scenes')] for item in bible['outfits']]  # only when set, so earlier bibles keep their fingerprint
    return {'plan': _hash(plan),
            'bible': _hash(kept),
            'evidence': _hash(evidence)}
