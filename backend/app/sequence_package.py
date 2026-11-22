"""The sequence as AI Director reads it: contract ``sequence-v2``.

``document`` assembles the ``full`` view from current data, plus ``assets``: what the ZIP export needs to
cut files (frame times, picture intervals, reference crops, voice samples, subtitle lines). The assets are
internal and never served. ``lean``, ``digest`` and ``select`` are pure projections of the full view:

- ``full``: everything, including the prompt texts.
- ``lean`` (the default and the ZIP's ``sequence.json``): ``full`` without the prose that the ZIP's text files
  carry (prompt, prompt template, still) and without what the prompt repeats word for word.
- ``digest`` (``overview.json``): about 5 KB to read first: flags, bible essentials, grouped issues and one
  compact entry per shot.

Each fact is stated once. Keys whose value is null, empty or false are left out (a ``false`` that matters,
such as ``ready`` or ``check.ok``, stays). Shot times are seconds from the start of that shot; ``shot.at``
is its start on the sequence clock and the top-level ``start_time`` is the sequence's start in the episode.
Whatever the app guesses is labelled: ``kind_guessed``, ``linked_by: name``, ``candidate`` with a
``confidence``, ``description_source: derived``.
"""
import json
import re

from . import db, media, outfit_library, sequence_evidence as evidence, sequence_prompt, sequence_reference, sequence_store as store, sequence_view, subtitle_store
from .visual_contracts import _match_name

CONTRACT = 'sequence-v2'
VIEWS = ('lean', 'full', 'digest')
INCLUDES = {'prompts'}  # GET: prose added to the lean view
EXPORT_INCLUDES = {'sequence_audio', 'references', 'clips'}  # POST exports: optional heavy files
PROMPT_KEYS = ('prompt', 'prompt_template', 'still_prompt')
FULL_ONLY = (*PROMPT_KEYS, 'still', 'key', 'soundscape')
SHOT_FIELDS = {'cut', 'part', 'at', 'duration', 'generate_seconds', 'source_cuts', 'opens', 'closes', 'scene', 'intent', 'camera', 'beats', 'continuity',
               'initial_frame', 'voices', 'lines', 'line_map', 'subjects', 'pictures', 'subtitles', 'check', 'stale', 'written', *FULL_ONLY}
SEVERITY_ORDER = {'error': 0, 'warning': 1, 'info': 2}
SIZE_RANK = {'EWS': 0, 'WS': 1, 'MWS': 2, 'MS': 3, 'MCU': 4, 'CU': 5, 'ECU': 6}
ADJACENT_SECONDS = 2.0
VOICE_WINDOW = (2.0, 15.0)  # the length a generator takes as a voice reference
PERSON_WORDS = {'man', 'men', 'woman', 'women', 'boy', 'boys', 'girl', 'girls', 'person', 'people', 'student', 'students', 'teacher', 'child', 'children', 'kid', 'kids',
                'crowd', 'figure', 'silhouette', 'stranger', 'passerby', 'passersby', 'classmate', 'classmates', 'friend', 'friends', 'attendant', 'officer', 'clerk',
                'driver', 'mother', 'father', 'sister', 'brother', 'guy', 'lady', 'someone', 'somebody', 'customer', 'customers', 'worker', 'nurse', 'doctor'}
STOP_WORDS = {'the', 'a', 'an', 'of', 'and', 'with', 'in', 'on', 'at', 'to', 'his', 'her', 'their', 'its', 'left', 'right', 'from'}


class PackageError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


def pack(mapping):
    """The mapping without null and empty values; numbers (also 0) and booleans stay."""
    return {key: value for key, value in mapping.items() if value is not None and value != '' and value != [] and value != {}}


def flag(value):
    """True, or None so that ``pack`` leaves the key out."""
    return True if value else None


def short_text(value, limit):
    value = re.sub(r'\s+', ' ', str(value or '')).strip()
    return value if len(value) <= limit else value[:limit - 1].rstrip() + '…'


def selection(value, allowed, label):
    if not value:
        return set()
    items = {part.strip() for part in value.split(',') if part.strip()}
    unknown = items - set(allowed)
    if unknown:
        raise PackageError(422, f"Unknown {label}: {', '.join(sorted(unknown))}")
    return items


def _text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False) if value not in (None, '') else ''


# ---------------------------------------------------------------- identity

def speaker_kind(name, character_context_id):
    if character_context_id:
        return 'character'
    if name == sequence_prompt.NARRATOR:
        return 'narrator'
    if name == sequence_prompt.UNKNOWN_SPEAKER:
        return 'unknown'
    return 'voice'


class Speakers:
    """The ``speakers`` table: one id per speaker name the shots' voices use (``sequence_prompt.speaker_key``)."""

    def __init__(self):
        self.ids, self.table = {}, {}

    def ref(self, name, character_context_id=None, label=None):
        name = name or sequence_prompt.UNKNOWN_SPEAKER
        if name not in self.ids:
            self.ids[name] = f'SP{len(self.ids) + 1}'
            self.table[self.ids[name]] = {'name': name, 'kind': speaker_kind(name, character_context_id), 'character_context_id': character_context_id, 'speaker_labels': []}
        entry = self.table[self.ids[name]]
        if character_context_id and not entry['character_context_id']:
            entry.update(character_context_id=character_context_id, kind='character')
        if label and label not in entry['speaker_labels']:
            entry['speaker_labels'].append(label)
        return self.ids[name]

    def view(self):
        return {key: pack(value) for key, value in self.table.items()}


def guess_kind(name, frame):
    """``person``, ``object``, ``place`` or ``unknown`` for a subject without a record: what the opening frame's analysis calls it."""
    words = [word for word in re.findall(r'[^\W\d_]+', str(name or '').casefold()) if word not in STOP_WORDS and len(word) > 2]
    if any(word in PERSON_WORDS for word in words):
        return 'person'

    def mentions(value):
        text = _text(value).casefold()
        return bool(text) and any(re.search(r'(?<!\w)' + re.escape(word), text) for word in words)
    if mentions(frame.get('background_people')):
        return 'person'
    if mentions(frame.get('props')):
        return 'object'
    if mentions(frame.get('setting')):
        return 'place'
    return 'unknown'


def roster_of(ctx, bible):
    """Every character a name may stand for: the bible's and the project's records, by id."""
    people = {}
    for person in (bible or {}).get('characters') or []:
        if person.get('character_context_id'):
            people.setdefault(person['character_context_id'], {'name': person['name'], 'character_context_id': person['character_context_id']})
    for person in ctx['characters'] if ctx else []:
        people.setdefault(person['id'], {'name': person['name'], 'character_context_id': person['id']})
    return list(people.values())


def link_subjects(subjects, roster, frame):
    """The prompt's subjects with a record wherever the name matches one; the stored direction is unchanged."""
    out = []
    for subject in subjects:
        item = {'n': subject['number'], 'name': subject['name'], 'kind': subject['kind'], 'character_context_id': subject.get('character_context_id'),
                'context_id': subject.get('context_id'), 'picture': subject.get('picture'), 'anchors': subject.get('anchors'), 'keep': subject.get('keep')}
        if subject['kind'] == 'character':
            item['linked_by'] = 'prompt'
        elif subject['kind'] != 'scenario':
            person = _match_name(subject['name'], roster)
            if person:
                item.update(kind='character', character_context_id=person['character_context_id'], linked_by='name', record_name=person['name'] if person['name'] != subject['name'] else None)
            else:
                item.update(kind=guess_kind(subject['name'], frame), kind_guessed=True)
        out.append(pack(item))
    return out


def candidate(line, dialogue, on_screen, speaking, mouths, cut_number):
    """Advice for a line nobody attributed: the one character the evidence points to, or None with the reason."""
    scores, reasons = {}, {}

    def add(name, cid, why):
        scores[(name, cid)] = scores.get((name, cid), 0) + 1
        reasons.setdefault((name, cid), []).append(why)
    if len(on_screen) == 1:
        name, cid = next(iter(on_screen.items()))
        add(name, cid, f'the only named character on screen in cut {cut_number}')
    for name in speaking:
        if name in on_screen:
            add(name, on_screen[name], f'visibly speaking in cut {cut_number}')
    for other in dialogue:
        if other is line or other.get('attribution') == 'unknown' or not other.get('character_context_id'):
            continue
        gap = max(other['t0'] - line['t1'], line['t0'] - other['t1'], 0.0)
        if gap <= ADJACENT_SECONDS:
            add(other['character_name'], other['character_context_id'], f"speaks {other['line']} {gap:.1f} s {'before' if other['t1'] <= line['t0'] else 'after'} in this shot")
    if not scores:
        return None, f'nobody named is on screen in cut {cut_number} and no attributed line is within {ADJACENT_SECONDS:g} s'
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        names = ' or '.join(name for (name, _), _ in ranked[:2])
        return None, f'the evidence points equally to {names}'
    (name, cid), score = ranked[0]
    why = reasons[(name, cid)]
    if mouths.get(name) in ('no', 'unclear'):
        why.append(f"the mouth is not seen moving in cut {cut_number} ({mouths[name]}): inner voice or off screen")
    return {'character_context_id': cid, 'name': name, 'confidence': 'medium' if score >= 2 else 'low', 'reason': '; '.join(why)}, None


# ---------------------------------------------------------------- pieces of a shot

def edge_view(edge):
    """An edge without what the shot times already say: the hard cut's position, a zero shift, the split's words."""
    if not edge or edge.get('how') == 'end':
        return None
    split = edge.get('split') or {}
    return pack({'how': edge['how'], 'boundary': edge.get('boundary'), 'shift': edge.get('shift') or None, 'keeps': edge.get('keeps'),
                 'splits': split.get('lines'), 'reason': edge.get('reason'), 'by': edge.get('by'), 'placed': edge.get('placed') or None})


def camera_view(camera):
    if not camera:
        return None
    moving = camera.get('move') not in (None, 'static')
    return pack({'move': camera.get('move'), 'amplitude': camera.get('amplitude') if moving else None, 'speed': camera.get('speed') if moving else None, 'detail': camera.get('detail')})


def beat_views(content, shot):
    out = []
    for beat in sequence_prompt.beat_ranges(content, shot) if content.get('beats') else []:
        out.append({'start': beat['start'], 'end': beat['end'], 'hold': True} if beat.get('hold') else {'start': beat['start'], 'end': beat['end'], 'action': beat['action']})
    return out


def frame_view(shot, own, shots):
    frame = shot['frame']
    observed = frame['strategy'] == 'cut_first_frame'
    derived_from = None
    if not observed:
        later = frame['time'] - own['start_time']
        previous = shots[shot['index'] - 2]['index'] if shot['index'] > 1 else None
        why = f'where shot {previous} stopped' if frame['continues_previous_shot'] else f'shot {previous} ran past the hard cut to finish a phrase'
        derived_from = f"cut {frame['cut']} first frame + its motion up to {later:.1f} s into the cut ({why})"
    return {'description_source': 'observed' if observed else 'derived', 'derived_from': derived_from,
            'continues_previous_shot': flag(frame['continues_previous_shot']), 'at': round(frame['time'] - shot['start_time'], 3)}


def picture_views(references, subjects):
    by_picture = {subject.get('picture'): subject['number'] for subject in subjects if subject.get('picture')}
    out = []
    for number, reference in enumerate((item for item in references if item['role'] != 'original_audio'), 1):
        if reference['role'] == 'reference_frame':
            # The user's staging reference: a source frame at ``at`` seconds of the shot (shot-NN-reference.jpg), for placement only.
            out.append(pack({'role': 'reference_frame', 'at': reference.get('at'), 'frame': reference.get('frame')}))
            continue
        out.append(pack({'role': reference['role'], 'subject': by_picture.get(number) if reference['role'] != 'initial_frame' else None}))
    return out


def subtitle_views(lines, shot):
    out = []
    for line in lines:
        if line['end_time'] <= shot['start_time'] or line['start_time'] >= shot['end_time']:
            continue
        out.append(pack({'t0': round(max(0.0, line['start_time'] - shot['start_time']), 3), 't1': round(min(shot['duration'], line['end_time'] - shot['start_time']), 3),
                         'text': line['text'], 'crosses_shot': flag(line['start_time'] < shot['start_time'] - 1e-6 or line['end_time'] > shot['end_time'] + 1e-6)}))
    return out


# ---------------------------------------------------------------- references for the export

def _box_area(box, size):
    try:
        return max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / max(1.0, float(size[0] * size[1]))
    except (TypeError, IndexError, ValueError):
        return 0.0


CROP_MARGIN = 0.15


def crop_fractions(candidate):
    """The detection box with a margin, as fractions of the frame the detector saw."""
    width, height = candidate['image_size']
    x1, y1, x2, y2 = candidate['box']
    margin_x, margin_y = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    return ((x1 - margin_x) / width, (y1 - margin_y) / height, (x2 + margin_x) / width, (y2 + margin_y) / height)


def _detection_candidate(row, source):
    frames = store.loads(row['frames_json'], [])
    frame = next((item for item in frames if item.get('position') == row['position']), None)
    box = store.loads(row['box_json'], [])
    if not frame or len(box) != 4 or not frame.get('image_size'):
        return None
    return {'source': source, 'detection_id': row['detection_id'], 'project_id': row['project_id'], 'episode_id': row['episode_id'], 'original_ext': row['original_ext'],
            'cut_number': row['cut_index'] + 1 if row['cut_index'] is not None and row['cut_index'] >= 0 else None, 'cut_start': row['cut_start'], 'cut_end': row['cut_end'],
            'time': frame.get('time'), 'image_size': frame.get('image_size'), 'box': box, 'area': round(_box_area(box, frame.get('image_size') or [1, 1]), 4)}


def character_candidates(conn, ids, cut_ids):
    """Per character id: crops Framewise can offer, best first. Saved gallery references first (largest box), else manually labelled boxes in the sequence's cuts."""
    if not ids:
        return {}
    marks = ','.join('?' * len(ids))
    found = {cid: [] for cid in ids}
    for row in conn.execute(f'''SELECT r.character_context_id,d.id AS detection_id,d.position,d.box_json,run.frames_json,run.cut_start,run.cut_end,
            c.cut_index,e.id AS episode_id,e.project_id,e.original_ext FROM character_image_references r JOIN character_detections d ON d.id=r.detection_id
            JOIN character_detection_runs run ON run.id=d.run_id JOIN cuts c ON c.id=run.cut_id JOIN episodes e ON e.id=c.episode_id
            WHERE r.character_context_id IN ({marks}) ORDER BY r.rowid''', list(ids)):
        item = _detection_candidate(row, 'saved_reference')
        if item:
            found[row['character_context_id']].append(item)
    cut_marks = ','.join('?' * len(cut_ids))
    if cut_ids:
        for row in conn.execute(f'''SELECT d.character_context_id,d.id AS detection_id,d.position,d.box_json,run.frames_json,run.cut_start,run.cut_end,
                c.cut_index,e.id AS episode_id,e.project_id,e.original_ext FROM character_detections d JOIN character_detection_runs run ON run.id=d.run_id
                JOIN cuts c ON c.id=run.cut_id JOIN episodes e ON e.id=c.episode_id
                WHERE d.label_source='manual' AND d.character_context_id IN ({marks}) AND run.cut_id IN ({cut_marks})
                AND run.rowid=(SELECT MAX(r2.rowid) FROM character_detection_runs r2 WHERE r2.cut_id=run.cut_id) ORDER BY run.rowid''', [*ids, *cut_ids]):
            item = _detection_candidate(row, 'manual_label')
            if item and not any(other['detection_id'] == item['detection_id'] for other in found[row['character_context_id']]):
                found[row['character_context_id']].append(item)
    for cid, items in found.items():
        items.sort(key=lambda item: (item['source'] != 'saved_reference', -item['area']))
    return found


def wav_seconds(path):
    """Length of a saved sample; its stored ranges may be an envelope with gaps."""
    import wave
    try:
        with wave.open(str(path), 'rb') as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error):
        return None


def voice_samples(conn, ids, project_id):
    """Per character id: every saved voice sample, with the one a generator should get first."""
    if not ids:
        return {}
    marks = ','.join('?' * len(ids))
    found = {cid: [] for cid in ids}
    for row in conn.execute(f'''SELECT id,character_context_id,episode_id,start_time,end_time,ranges_json,source_filename,created_at
            FROM character_voice_samples WHERE character_context_id IN ({marks}) ORDER BY created_at,rowid''', list(ids)):
        ranges = store.loads(row['ranges_json'], None) or ([] if row['source_filename'] else [{'start_time': row['start_time'], 'end_time': row['end_time']}])
        saved = media.voice_reference_path(project_id, row['id'])
        audio = media.episode_dir(project_id, row['episode_id']) / 'audio.wav' if row['episode_id'] else None
        seconds = wav_seconds(saved) if saved.is_file() else (sum(item['end_time'] - item['start_time'] for item in ranges) if ranges else None)
        found[row['character_context_id']].append({'id': row['id'], 'episode_id': row['episode_id'], 'ranges': ranges, 'source_filename': row['source_filename'],
                                                   'seconds': round(seconds, 3) if seconds is not None else None, 'created_at': row['created_at'],
                                                   'preview_available': saved.is_file() or bool(ranges and audio and audio.is_file())})
    for cid, samples in found.items():
        usable = [sample for sample in samples if sample['preview_available']]
        fitting = [sample for sample in usable if sample['seconds'] and VOICE_WINDOW[0] <= sample['seconds'] <= VOICE_WINDOW[1]]
        pool = fitting or usable
        if pool:
            chosen = max(pool, key=lambda sample: (sample['seconds'] or 0.0, sample['created_at'] or ''))
            chosen['chosen'] = True
            chosen['reason'] = (f'the longest sample of {VOICE_WINDOW[0]:g}-{VOICE_WINDOW[1]:g} s' if fitting else 'the longest available sample; none is 2-15 s long')
    return found


def scene_frames(bible, timeline, ctx):
    """Per bible scene: the cut whose opening frame shows the most of the place (widest framing, then the longest cut)."""
    from .composition_brief import shot_props
    out = {}
    for scene in (bible or {}).get('scenes') or []:
        cuts = [cut for cut in timeline['cuts'] if isinstance(scene.get('from_cut'), int) and scene['from_cut'] <= cut['number'] <= scene.get('to_cut', -1)]
        ranked = []
        for cut in cuts:
            frame = (ctx['results'].get(cut['id'], {}).get('initial_frame') or {}).get('content') or {}
            size = shot_props(frame).get('size') if frame else None
            ranked.append((SIZE_RANK.get(size, len(SIZE_RANK)), -(cut['end_time'] - cut['start_time']), cut, size))
        if not ranked:
            continue
        _, _, cut, size = min(ranked, key=lambda item: item[:2])
        first = cut.get('first_frame_time')
        time = first if isinstance(first, (int, float)) and cut['start_time'] <= first < cut['end_time'] else cut['start_time']
        out[scene['id']] = {'cut': cut['number'], 'cut_id': cut['id'], 'time': time, 'cut_start': cut['start_time'], 'cut_end': cut['end_time'],
                            'chosen_by': f"widest framing in the scene ({size or 'framing unknown'}), then the longest cut", 'size': size}
    return out


# ---------------------------------------------------------------- the document

def provenance(conn, sequence_id):
    """Revision facts of every current result, by id."""
    out = {}
    for row in conn.execute('''SELECT s.id,s.origin,s.model_id,s.created_at,s.settings_json,(SELECT COUNT(*) FROM sequence_results p WHERE p.sequence_id=s.sequence_id AND p.type=s.type
            AND p.shot_key=s.shot_key AND p.rowid<=s.rowid) AS revision FROM sequence_results s WHERE s.sequence_id=? AND s.superseded_by IS NULL''', (sequence_id,)):
        settings = store.loads(row['settings_json'], {})
        seconds = settings.get('summary_generation_seconds')
        out[row['id']] = pack({'id': row['id'], 'origin': row['origin'], 'model_id': row['model_id'], 'created_at': row['created_at'], 'revision': row['revision'],
                               'generation_seconds': round(seconds, 1) if isinstance(seconds, (int, float)) else None})
    return out


def document(sequence_id, conn=None):
    """``(full, assets)`` for a sequence, or None when it does not exist."""
    if conn is None:
        with db.connect() as conn:
            conn.execute('BEGIN')
            return document(sequence_id, conn)
    sequence = store.get(conn, sequence_id)
    if not sequence:
        return None
    current = sequence_view.state(conn, sequence)
    episode = dict(conn.execute('SELECT id,project_id,original_ext,has_audio FROM episodes WHERE id=?', (sequence['episode_id'],)).fetchone())
    revisions = provenance(conn, sequence_id)
    cuts, timeline, bible = current['cuts'], current['timeline'], current['bible'] or {}
    run = subtitle_store.current_run(sequence['episode_id'], conn) if cuts else None
    subtitle_lines = subtitle_store.lines(sequence['episode_id'], cuts[0]['start_time'], cuts[-1]['end_time'], conn=conn) if cuts and run else []
    views = sequence_view.shot_views(current)
    roster = roster_of(current['ctx'], bible)
    wanted = {person['character_context_id'] for person in bible.get('characters') or [] if person.get('character_context_id')}
    built = build_shots(current, views, roster, subtitle_lines)
    for shot in built['shots']:
        wanted.update(subject['character_context_id'] for subject in shot.get('subjects') or [] if subject.get('character_context_id'))
    wanted.update(entry['character_context_id'] for entry in built['speakers'].table.values() if entry['character_context_id'])
    wanted.update(voice['character_context_id'] for voice in bible.get('voices') or [] if voice.get('character_context_id'))
    cut_ids = [cut['id'] for cut in cuts] if cuts else []
    candidates = character_candidates(conn, sorted(wanted), cut_ids)
    samples = voice_samples(conn, sorted(wanted), episode['project_id'])
    names = {person['character_context_id']: person['name'] for person in roster}
    header = sequence_view.sequence_view(sequence, cuts, current['changed'])
    full = {'contract': CONTRACT, 'view': 'full', 'id': header['id'], 'episode_id': header['episode_id'], 'title': header['title'], 'instruction': header['instruction'],
            'start_time': header['start_time'], 'duration': header['duration'], 'created_at': header['created_at'], 'updated_at': header['updated_at'],
            'ready': sequence_view.is_ready(current, views), 'pending': [shot['index'] for shot in views if not shot['result'] or shot['stale'] or not (shot['result']['check'] or {}).get('ok')],
            'cuts_changed': flag(header['cuts_changed']), 'cuts_missing': flag(header['cuts_missing']), 'plan_problems': current['plan_problems'],
            'cuts': [{'number': cut['number'], 'id': cut['id'], 'at': cut['t0'], 'duration': round(cut['t1'] - cut['t0'], 3)} for cut in timeline['cuts']] if timeline else [],
            'speakers': built['speakers'].view(),
            'bible': pack({key: value for key, value in bible.items() if key != 'open_questions'}) if bible else None,
            'subtitles': pack({'language': (run or {}).get('language'), 'extracted': bool(run)}),
            'issues': [], 'review': None, 'provenance': None, 'shots': built['shots']}
    if full['bible']:
        full['bible']['scenes'] = [pack(scene) for scene in full['bible'].get('scenes') or []]
        full['bible']['voices'] = [pack(voice) for voice in full['bible'].get('voices') or []]
        full['bible']['characters'] = [pack(person) for person in full['bible'].get('characters') or []]
    outfit_files, outfit_issues = outfit_assets(conn, episode['project_id'], bible)
    if full['bible']:
        full['bible']['outfits'] = [pack(item) for item in settled_outfits(bible, outfit_files)]
    review = store.loads(current['review_row']['content'], {}) if current['review_row'] else None
    full['review'] = pack({'verdict': review.get('verdict'), 'stale_reason': current['review_row']['stale_reason']}) if review else None
    rows = {'bible': current['bible_row'], 'plan': current['plan_row'], 'compilation': current['compilation_row'], 'review': current['review_row']}
    full['provenance'] = pack({**{key: revisions.get(row['id']) for key, row in rows.items() if row},
                               'shots': {str(shot['index']): revisions[shot['result']['id']] for shot in views if shot['result'] and shot['result']['id'] in revisions}})
    references = {cid: {'name': names.get(cid), 'candidates': candidates.get(cid) or []} for cid in sorted(wanted)}
    full['issues'] = sort_issues([*built['issues'], *outfit_issues, *sequence_issues(full, current, bible, review, references, built)])
    assets = {'episode': episode, 'shots': built['assets'], 'characters': references, 'voices': {cid: {'name': names.get(cid), 'samples': items} for cid, items in samples.items() if items},
              'scenes': scene_frames(bible, timeline, current['ctx']) if timeline else {}, 'outfits': outfit_files,
              'subtitles': [{'start_time': line['start_time'], 'end_time': line['end_time'], 'text': line['text']} for line in subtitle_lines]}
    return pack_document(full), assets


def outfit_assets(conn, project_id, bible):
    """The library rows the bible uses, with their full descriptions (they go into the download, not into sequence.json), and an issue for each one that is gone."""
    wanted = [item['outfit_id'] for item in bible.get('outfits') or [] if item.get('outfit_id')]
    by_id = {row['id']: row for row in outfit_library.rows(conn, project_id)}
    files = [{key: by_id[outfit_id][key] for key in ('id', 'character_context_id', 'character_name', 'name', 'summary', 'description', 'origin', 'needs_review', 'updated_at')}
             for outfit_id in dict.fromkeys(wanted) if outfit_id in by_id]
    issues = [{'severity': 'warning', 'kind': 'outfit_missing',
               'text': f"The outfit “{item.get('name')}” of {item.get('character')} was deleted from the outfit library; its full description is not in this download."}
              for item in bible.get('outfits') or [] if item.get('outfit_id') not in by_id]
    issues += [{'severity': 'warning', 'kind': 'outfit_unchecked',
                'text': f"The description of the outfit “{item['name']}” of {item['character_name']} was imported from text that may be cut off; check it in the Outfits dialog."}
               for item in files if item['needs_review']]
    return files, issues


def settled_outfits(bible, files):
    """The bible's outfit entries with the library's current name and summary (the library is the truth after an edit)."""
    live = {item['id']: item for item in files}
    return [{**item, **({'name': live[item['outfit_id']]['name'], 'summary': live[item['outfit_id']]['summary']} if item.get('outfit_id') in live else {})} for item in bible.get('outfits') or []]


def build_shots(current, views, roster, subtitle_lines):
    speakers, shots, issues, assets = Speakers(), [], [], []
    ctx, bible, timeline = current['ctx'], current['bible'] or {}, current['timeline']
    for shot in views:
        result = shot['result'] or {}
        content, derived = result.get('content') or {}, result.get('derived') or {}
        own = next(cut for cut in timeline['cuts'] if cut['id'] == shot['cut_id'])
        results = ctx['results'].get(shot['cut_id'], {})
        frame_content = (results.get('initial_frame') or {}).get('content') or {}
        written = bool(result)
        dialogue = derived.get('dialogue') if written and derived.get('dialogue') is not None else evidence.shot_dialogue(shot, bible)
        subjects = link_subjects(derived.get('subjects') or [], roster, frame_content)
        template, slots = (sequence_prompt.template(shot, content, bible, derived.get('subjects') or [], dialogue, result.get('prompt')) if written
                           else (None, sequence_prompt.line_slots(dialogue)))
        on_screen = dict(evidence.people_in(results))
        on_screen.update({subject['name'] if subject.get('linked_by') == 'prompt' else (subject.get('record_name') or subject['name']): subject['character_context_id']
                          for subject in subjects if subject.get('kind') == 'character'})
        direction = ((results.get('video_analysis') or {}).get('content') or {}).get('direction') or {}
        mouths = {person.get('character'): person.get('speaking') for person in direction.get('characters') or [] if isinstance(person, dict)}
        lines = []
        for line in dialogue:
            key = sequence_prompt.speaker_key(line)
            entry = {'id': line['line'], 'speaker': speakers.ref(key, line.get('character_context_id'), line.get('speaker_label') or None), 't0': line['t0'], 't1': line['t1'],
                     'text': line['text'], 'by': line.get('attribution'), 'crosses_start': flag(line.get('crosses_start')), 'crosses_end': flag(line.get('crosses_end'))}
            if line.get('attribution') == 'unknown':
                guess, why = candidate(line, dialogue, on_screen, evidence.speaking_in(results), mouths, shot['cut'])
                entry['candidate'] = guess
                advice = (f"candidate {guess['name']} ({guess['confidence']}), advice only: see the line's candidate" if guess else f'no candidate: {why}')
                issues.append({'shot': shot['index'], 'line': line['line'], 'severity': 'warning', 'kind': 'unknown_speaker',
                               'text': f"Nobody decided who says {line['line']} “{short_text(line['text'], 40)}”; {advice}."})
            lines.append(pack(entry))
        if written and derived.get('voices') is not None:
            voices = [pack({'speaker': speakers.ref(item['speaker']), 'from': item['from'], 'to': item['to'], 'delivery': item['delivery'],
                            'subject': item.get('subject_number'), 'fixed': flag(item.get('fixed_by_user'))}) for item in derived['voices']]
        else:
            voices = [pack({'speaker': speakers.ref(item['speaker']), 'from': item['from'], 'to': item['to'], 'delivery': item['delivery'], 'fixed': True})
                      for item in shot.get('fixed_voices') or []]
        check = result.get('check') or {}
        frame = frame_view(shot, own, views)
        entry = {'index': shot['index'], 'key': shot['key'], 'cut': shot['cut'], 'part': shot['part'], 'at': shot['t0'], 'duration': shot['duration'],
                 'generate_seconds': shot['generate_seconds'] if shot['generate_seconds'] - shot['duration'] > 0.001 else None,
                 'source_cuts': shot['source_cuts'] if shot['source_cuts'] != [shot['cut']] else None,
                 'opens': edge_view(shot['opens']), 'closes': edge_view(shot['closes']), 'scene': shot['scene'], 'intent': shot['intent'],
                 'camera': camera_view(content.get('camera')), 'beats': beat_views(content, shot), 'soundscape': content.get('soundscape'),
                 'continuity': pack({key: (content.get('continuity') or {}).get(key) for key in ('from_previous', 'to_next')}),
                 'initial_frame': pack({'description': content.get('picture_line'), **frame}), 'voices': voices, 'lines': lines, 'line_map': slots,
                 'subjects': subjects, 'pictures': picture_views(derived.get('references') or [], derived.get('subjects') or []),
                 'subtitles': subtitle_views(subtitle_lines, shot),
                 'check': pack({'ok': bool(check.get('ok')), 'errors': check.get('errors'), 'warnings': check.get('warnings'), 'stats': check.get('stats')}) if written else None,
                 'stale': shot['stale'], 'written': None if written else False,
                 'prompt': result.get('prompt'), 'prompt_template': template, 'still_prompt': derived.get('still_prompt'), 'still': content.get('still')}
        shots.append(pack(entry))
        issues.extend(shot_issues(entry, content, frame, written, template, check))
        reference = sequence_prompt.reference_of(content) if written else None
        if reference and reference.get('ordinal') not in sequence_reference.analysed_ordinals(results):
            issues.append(pack({'shot': shot['index'], 'severity': 'info', 'kind': 'reference_unanalysed',
                                'text': f"The staging reference {reference.get('frame') or ''} (shot-{shot['index']:02d}-reference.jpg) has no keyframe analysis; "
                                        'the prompt describes it from the picture alone, or not at all when the text engine could not see it.'}))
        picture = list(sequence_reference.span(shot, own))
        assets.append({'index': shot['index'], 'start_time': shot['start_time'], 'end_time': shot['end_time'], 'duration': shot['duration'],
                       'generate_seconds': shot['generate_seconds'], 'cut': shot['cut'], 'part': shot['part'], 'at': shot['t0'],
                       'frame_time': shot['frame']['time'], 'cut_start': own['start_time'], 'cut_end': own['end_time'], 'picture': picture,
                       'has_speech': bool(dialogue), 'description_source': frame['description_source'],
                       'reference': {'ordinal': reference['ordinal'], 'time': reference['time'], 'at': reference['at'], 'frame': reference.get('frame')} if reference else None})
    return {'speakers': speakers, 'shots': shots, 'issues': issues, 'assets': assets}


def shot_issues(shot, content, frame, written, template, check):
    index, issues = shot['index'], []

    def add(severity, kind, text, line=None):
        issues.append(pack({'shot': index, 'line': line, 'severity': severity, 'kind': kind, 'text': text}))
    if not written:
        add('error', 'not_written', 'The shot has no direction yet; there is no prompt to generate from.')
    if shot.get('stale'):
        add('error', 'stale', f"Changed since the shot was written: {', '.join(shot['stale'])}. Rewrite it in Framewise before generating.")
    for error in check.get('errors') or []:
        add('error', 'check_error', error)
    for warning in check.get('warnings') or []:
        add('warning', 'check_warning', warning)
    if written and template is None and shot.get('prompt'):
        add('warning', 'template_unavailable', 'The stored prompt no longer renders from its direction (the plan or the bible changed since); no prompt template. Rewrite the shot.')
    if frame['description_source'] == 'derived':
        add('warning', 'derived_frame', 'The initial frame is a later, unanalysed moment of the cut: its description is derived (initial_frame.derived_from). Check it against shot-NN-initial.jpg.')
    if shot.get('generate_seconds'):
        add('info', 'padded_shot', 'Shorter than 5 s: generate generate_seconds (the prompt ends on a hold), trim to duration; shot-NN-ref.wav is the original audio from the start of the shot for that length.')
    for subject in shot.get('subjects') or []:
        if subject.get('linked_by') == 'name' and not subject.get('picture'):
            add('warning', 'no_sheet', f"{subject.get('record_name') or subject['name']} has no character sheet in the prompt: the cut's analysis did not mark them, so <Picture 1> is the only "
                                       'reference. Detect characters on the cut and rewrite the shot in Framewise to get one.')
        elif subject.get('kind') in ('person', 'unknown') and not subject.get('character_context_id'):
            add('info', 'unlinked_subject', f"<Subject {subject['n']}> “{subject['name']}” has no record (kind {subject['kind']}, guessed); it is described from <Picture 1> only.")
    for question in content.get('open_questions') or []:
        add('warning', 'open_question', question)
    return issues


def sequence_issues(full, current, bible, review, references, built):
    issues = []
    if full.get('cuts_missing'):
        issues.append({'severity': 'error', 'kind': 'cuts_missing', 'text': 'The first or last cut of this sequence was archived; it must be created again.'})
    if full.get('cuts_changed'):
        issues.append({'severity': 'error', 'kind': 'cuts_changed', 'text': 'The cuts were edited after the sequence was directed; direct it again before generating.'})
    for problem in current['plan_problems']:
        issues.append({'severity': 'error', 'kind': 'plan_problem', 'text': problem})
    for question in bible.get('open_questions') or []:
        issues.append({'severity': 'warning', 'kind': 'open_question', 'text': question})
    if (reason := (current['review_row'] or {}).get('stale_reason')):
        issues.append({'severity': 'warning', 'kind': 'stale_review', 'text': reason})
    shots = {shot['index'] for shot in full['shots']}
    for item in (review or {}).get('issues') or []:
        if item.get('dismissed'):
            continue
        text = f"Review ({item.get('severity') or 'check'}): {item.get('problem') or ''}" + (f" Suggestion: {item['suggestion']}" if item.get('suggestion') else '')
        issues.append(pack({'shot': item.get('shot') if item.get('shot') in shots else None, 'severity': 'warning', 'kind': 'review', 'text': text}))
    sheets = {}
    for shot in full['shots']:
        for subject in shot.get('subjects') or []:
            if subject.get('kind') == 'character' and subject.get('character_context_id'):
                sheets.setdefault(subject['character_context_id'], []).append(shot['index'])
    for cid, item in references.items():
        if cid in sheets and not item['candidates']:
            issues.append({'severity': 'warning', 'kind': 'missing_reference',
                           'text': f"No reference image of {item['name'] or cid} in Framewise (no saved gallery reference, no manual label in these cuts); supply the character sheet yourself."})
    return issues


def sort_issues(issues):
    """Most severe first; an issue that reads the same in several shots is stated once with all of them in ``shots``."""
    groups = {}
    for item in issues:
        key = (item['severity'], item['kind'], item['text'], item.get('line'))
        group = groups.setdefault(key, pack({'shots': [], 'line': item.get('line'), 'severity': item['severity'], 'kind': item['kind'], 'text': item['text']}))
        if item.get('shot') and item['shot'] not in group.setdefault('shots', []):
            group['shots'].append(item['shot'])
    out = [pack({**group, 'shots': sorted(group.get('shots') or [])}) for group in groups.values()]
    return sorted(out, key=lambda item: (SEVERITY_ORDER[item['severity']], (item.get('shots') or [0])[0], item['kind']))


def pack_document(full):
    return pack({key: value for key, value in full.items()})


# ---------------------------------------------------------------- projections

def lean(full, include=()):
    """The default view: ``full`` without the ZIP's text files and the prose the prompt repeats word for word."""
    keep = set(PROMPT_KEYS) if 'prompts' in include else set()
    doc = {**full, 'view': 'lean', 'shots': []}
    if full.get('provenance'):
        def bare(entry):
            return {key: value for key, value in entry.items() if key != 'generation_seconds'}
        doc['provenance'] = {key: ({number: bare(entry) for number, entry in value.items()} if key == 'shots' else bare(value)) for key, value in full['provenance'].items()}
    for shot in full.get('shots') or []:
        item = {key: value for key, value in shot.items() if key not in FULL_ONLY or key in keep}
        item.pop('opens', None)
        if 'check' in item:
            item['check'] = {key: value for key, value in item['check'].items() if key != 'stats'}
        if 'continuity' in item:
            item['continuity'] = pack({'from_previous': item['continuity'].get('from_previous')})
            if not item['continuity']:
                del item['continuity']
        if 'initial_frame' in item:
            item['initial_frame'] = {key: value for key, value in item['initial_frame'].items() if key != 'at'}
        if 'subjects' in item:
            item['subjects'] = [{key: value for key, value in subject.items() if key not in ('anchors', 'keep')} for subject in item['subjects']]
        doc['shots'].append(item)
    return doc


def public_full(full):
    """The full view as served: each edge once (a shot opens with the previous shot's ``closes``)."""
    return {**full, 'shots': [{key: value for key, value in shot.items() if key != 'opens'} for shot in full.get('shots') or []]}


def project(full, view='lean', include='', shot=None, fields=''):
    """The response for ``GET /api/v1/sequences/{id}``."""
    if view not in VIEWS:
        raise PackageError(422, f"Unknown view: {view}; use one of {', '.join(VIEWS)}")
    includes = selection(include, INCLUDES, 'include')
    chosen = selection(fields, SHOT_FIELDS, 'fields')
    if view == 'digest':
        if shot is not None or chosen or includes:
            raise PackageError(422, 'The digest takes no shot, fields or include; use view=lean for those')
        return digest(full)
    doc = public_full(full) if view == 'full' else lean(full, includes)
    if shot is not None:
        doc = slice_shot(full, doc, shot)
    if chosen:
        doc = {**doc, 'shots': [{key: value for key, value in item.items() if key == 'index' or key in chosen} for item in doc['shots']]}
    return doc


def slice_shot(full, doc, index):
    """One shot with what it needs: the header, its speakers, its issues and the sequence-wide ones, its edge in."""
    position = next((number for number, item in enumerate(doc.get('shots') or []) if item['index'] == index), None)
    if position is None:
        raise PackageError(404, 'That shot is not part of the current plan')
    item = dict(doc['shots'][position])
    item['opens'] = full['shots'][position].get('opens') or {'how': 'start'}
    used = {line['speaker'] for line in item.get('lines') or []} | {voice['speaker'] for voice in item.get('voices') or []}
    numbers = {item['cut'], *(item.get('source_cuts') or [])}
    shot_provenance = ((doc.get('provenance') or {}).get('shots') or {}).get(str(index))
    out = {key: value for key, value in doc.items() if key not in ('bible', 'shots', 'cuts', 'speakers', 'issues', 'provenance', 'review')}
    out.update(cuts=[cut for cut in doc.get('cuts') or [] if cut['number'] in numbers], speakers={key: value for key, value in (doc.get('speakers') or {}).items() if key in used},
               issues=[issue for issue in doc.get('issues') or [] if not issue.get('shots') or index in issue['shots']],
               provenance={'shots': {str(index): shot_provenance}} if shot_provenance else None, shots=[item])
    return pack(out)


DELIVERY_WORD = {'on_screen': 'on', 'off_screen': 'off', 'inner_voice': 'inner', 'narrative': 'over'}


def digest(full):
    """About 5 KB to read first: what is ready, what needs a decision, and one compact entry per shot.

    A shot opens with the previous shot's ``closes``; ``refs`` lists only what the prompt attaches besides
    ``<Picture 1>`` (the initial frame) and ``<Audio 1>`` (shot-NN.wav); ``state`` appears only when the shot is not ready.
    """
    speakers = full.get('speakers') or {}
    firsts = [value['name'].split()[0] for value in speakers.values() if value.get('name') and value.get('kind') == 'character']

    def short(name, kind='character'):
        first = name.split()[0] if name.split() else name
        return first if kind == 'character' and firsts.count(first) == 1 else name

    def who(ref):
        speaker = speakers.get(ref) or {}
        return short(speaker.get('name') or ref, speaker.get('kind'))

    bible = full.get('bible') or {}
    shots = []
    for shot in full.get('shots') or []:
        state = 'not written' if shot.get('written') is False else ('stale: ' + ', '.join(shot['stale']) if shot.get('stale')
                                                                  else (None if (shot.get('check') or {}).get('ok') else 'check errors'))
        subjects = {subject['n']: subject for subject in shot.get('subjects') or []}
        refs = []
        for number, picture in enumerate(shot.get('pictures') or [], 1):
            if picture['role'] == 'reference_frame':
                refs.append(f"P{number} staging reference {picture.get('frame') or ''} @{picture.get('at', 0):.2f}s".replace('  ', ' '))
            elif picture['role'] != 'initial_frame':
                subject = subjects.get(picture.get('subject')) or {}
                refs.append(f"P{number} {'sheet' if picture['role'] == 'character_sheet' else 'scenario'}: {subject.get('record_name') or subject.get('name')}")
        lines = []
        for line in shot.get('lines') or []:
            guess = f" (? {short(line['candidate']['name'])})" if line.get('candidate') else ''
            lines.append(f"{line['id']} {who(line['speaker'])}{guess}: {short_text(line['text'], 16)}")
        shots.append(pack({
            'n': shot['index'], 'cut': f"{shot['cut']} {shot['part'][0]}/{shot['part'][1]}" if shot.get('part') else str(shot['cut']), 'at': shot['at'], 'dur': shot['duration'],
            'gen': shot.get('generate_seconds'), 'closes': edge_word(shot.get('closes') or {'how': 'end'}), 'frame': shot['initial_frame']['description_source'],
            'voices': [f"{who(voice['speaker'])} {DELIVERY_WORD.get(voice['delivery'], voice['delivery'])} {voice['from']:g}-{voice['to']:g}" + (' (user)' if voice.get('fixed') else '')
                       for voice in shot.get('voices') or []],
            'lines': lines, 'refs': ', '.join(refs), 'state': state}))
    issues = [pack({**issue, 'text': short_text(issue['text'], 120)}) for issue in full.get('issues') or []]
    return pack({
        'contract': CONTRACT, 'view': 'digest', 'id': full['id'], 'title': full['title'], 'duration': full['duration'],
        'cuts': f"{full['cuts'][0]['number']}-{full['cuts'][-1]['number']}" if full.get('cuts') else None,
        'ready': full['ready'], 'pending': full.get('pending'), 'cuts_changed': full.get('cuts_changed'), 'cuts_missing': full.get('cuts_missing'),
        'plan_problems': full.get('plan_problems'),
        'bible': pack({'logline': bible.get('logline'), 'spoken_language': bible.get('spoken_language'),
                       'characters': [pack({'name': person.get('name'), 'id': person.get('character_context_id'), 'anchors': person.get('anchors'),
                                            'outfits': [item.get('name') for item in bible.get('outfits') or [] if item.get('character') == person.get('name')]}) for person in bible.get('characters') or []],
                       'voices': [f"{voice.get('speaker_label')} → {voice.get('character') or voice.get('kind')} ({voice.get('confidence')}, {voice.get('source')})"
                                  for voice in bible.get('voices') or []]}),
        'subtitles': full.get('subtitles'), 'issues': issues, 'shots': shots})


def edge_word(edge):
    word = edge['how'] + (f" {edge['boundary']}" if edge.get('boundary') else '')
    if edge.get('shift'):
        word += f" {edge['shift']:+g}s"
    if edge.get('placed'):
        word += ' placed'
    if edge.get('splits'):
        word += ' splits ' + ','.join(edge['splits'])
    return word


# ---------------------------------------------------------------- for the audio and frame routes

def shot_spec(sequence_id, index):
    """A planned shot's episode interval and generated length, without building the document."""
    with db.connect() as conn:
        sequence = store.get(conn, sequence_id)
        if not sequence:
            raise PackageError(404, 'Sequence not found')
        current = sequence_view.state(conn, sequence)
    shot = next((item for item in current['shots'] or [] if item['index'] == index), None)
    if shot is None:
        raise PackageError(404, 'That shot is not part of the current plan')
    return {'episode_id': sequence['episode_id'], 'start_time': shot['start_time'], 'end_time': shot['end_time'], 'generate_seconds': shot['generate_seconds'],
            'audio_file': sequence_prompt.audio_file(shot)}


def sequence_interval(sequence_id):
    with db.connect() as conn:
        sequence = store.get(conn, sequence_id)
        if not sequence:
            raise PackageError(404, 'Sequence not found')
        cuts = store.current_cuts(conn, sequence)
    header = sequence_view.sequence_view(sequence, cuts, False)
    return {'episode_id': sequence['episode_id'], 'start_time': header['start_time'], 'end_time': header['start_time'] + header['duration']}
