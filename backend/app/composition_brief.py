"""Compact, deterministic projection of saved visual evidence for composition.

Model descriptions remain observations, not verified facts. No model is invoked by a read.
"""
import base64
import hashlib
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

from . import db, media, structured_results


class CompositionError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail

INCLUDES = {'initial_frame', 'last_frame', 'video_analysis', 'subtitle_ocr', 'consistency_summary', 'dialogue_segments'}
FIELDS = {'id', 'cut_number', 'duration', 'start_time', 'end_time', 'ready', 'updated_at', 'analysis_rev', 'shot', 'setting', 'lighting', 'subjects', 'beats', 'end_state', 'confidence', 'initial_frame_id', 'last_frame_id', 'dialogue', 'subtitles', 'issues', 'links', 'unavailable'}
REASONS = {'visual_pending': 'Required visual stages are missing or being updated.', 'visual_stale': 'Visual stages do not have matching cut-boundary snapshots.', 'frame': 'Initial frame unavailable.', 'audio': 'Prepared audio unavailable.', 'beats_partial': 'Some video events or windows could not be represented completely.', 'setting_partial': 'Some setting details could not be condensed safely.', 'lighting_partial': 'Some lighting details could not be condensed safely.', 'subjects_partial': 'Some visible subject details could not be condensed safely.', 'end_state_partial': 'Some final-frame subject details could not be condensed safely.', 'subtitles_partial': 'Some subtitle observation times were invalid or outside this cut.', 'issues_partial': 'A consistency issue could not be condensed safely.', 'shot': 'Shot properties could not be established from saved evidence.', 'summary_failed': 'The latest production brief generation failed; no invalid revision was saved.', 'identity_ambiguous': 'Some people cannot be matched reliably between boundary frames.'}
REASON_CODES = {'visual_pending':'vp', 'visual_stale':'vs', 'frame':'f', 'audio':'a', 'beats_partial':'bp',
                'setting_partial':'sp', 'lighting_partial':'lp', 'subjects_partial':'up', 'end_state_partial':'ep',
                'subtitles_partial':'op', 'issues_partial':'ip', 'shot':'s', 'summary_failed':'sf', 'identity_ambiguous':'ia'}


def projection_fingerprint():
    """Fingerprint code that can change the projected facts without DB edits."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('composition_brief.py', 'composition_compaction.py', 'structured_results.py', 'output_hygiene.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def register_projection():
    """Advance incremental watermarks once when the API projection changes."""
    current = projection_fingerprint()
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS composition_projection (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT NOT NULL)')
        previous = conn.execute('SELECT fingerprint FROM composition_projection WHERE id=1').fetchone()
        if not previous or previous['fingerprint'] != current:
            conn.execute("UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')")
            conn.execute('INSERT INTO composition_projection(id,fingerprint) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET fingerprint=excluded.fingerprint', (current,))


def selection(value, allowed, label):
    if not value:
        return set()
    items = {part.strip() for part in value.split(',') if part.strip()}
    unknown = items - allowed
    if unknown:
        raise CompositionError(422, f"Unknown {label}: {', '.join(sorted(unknown))}")
    return items


def reference_subjects(record):
    """Refer to a unique identified opening subject instead of repeating its name."""
    subjects = record.get('subjects', [])
    for item in [*record.get('beats', []), *record.get('end_state', {}).get('subjects', [])]:
        name = item.get('subject')
        matches = [index for index, person in enumerate(subjects)
                   if person.get('character_context_id') and person.get('name') == name]
        if len(matches) == 1:
            item['subject'] = matches[0]


def expand_subject_references(record):
    """Keep a field-selected record usable when its subject table is omitted."""
    subjects = record.get('subjects', [])
    for item in [*record.get('beats', []), *record.get('end_state', {}).get('subjects', [])]:
        index = item.get('subject')
        if isinstance(index, int) and not isinstance(index, bool):
            item['subject'] = subjects[index]['name']


def short(value, limit):
    if not isinstance(value, str):
        return ''
    value = re.sub(r'\s+', ' ', value).strip()
    return ' '.join(value.split()[:limit]).rstrip(' ,;')


def sentence(value, limit):
    return short(re.split(r'(?<=[.!?])\s+', value or '')[0], limit)


def setting_phrase(value, limit, missing=None):
    value = re.sub(r'\s+', ' ', value or '').strip()
    value = re.sub(r'\b(?:Interior of a|Visible elements include|The scene takes place in|The lighting is|typical of a retail environment)\b', '', value, flags=re.I)
    value = re.sub(r'\bindicated by the blurred architectural elements such as\s+(?:a|an)\s+', 'blurred ', value, flags=re.I)
    value = re.sub(r'\ban indoor environment\b', 'indoors', value, flags=re.I)
    value = re.sub(r'\blikely an?\s+', 'likely ', value, flags=re.I)
    value = re.sub(r'\band a window\b', 'and window', value, flags=re.I)
    value = re.sub(r'\bin the background\b', 'background', value, flags=re.I)
    value = re.sub(r'\bon the (left|right)\b', r'\1', value, flags=re.I)
    value = re.sub(r'\b(red|blue|green|teal|yellow|purple|orange|brown|white|black|grey|gray|pink)-colored\b', r'\1', value, flags=re.I)
    value = re.sub(r'\s+', ' ', value).strip(' .,')
    parts = re.split(r'(?<=[.!?;])\s+', value)
    chosen = []
    for part in parts:
        part = part.strip(' .,;')
        if not part: continue
        if len((' '.join(chosen) + ' ' + part).split()) <= limit:
            chosen.append(part)
            continue
        clauses = [clause.strip(' .,;') for clause in part.split(',')]
        index = 0
        while index < len(clauses):
            clause = clauses[index]
            if (len(clause.split()) < 3 or re.match(r'^indicated by\b', clause, re.I)) and index + 1 < len(clauses):
                index += 1
                clause += ', ' + clauses[index]
            if clause and len((' '.join(chosen) + ' ' + clause).split()) <= limit:
                chosen.append(clause)
            else:
                if missing is not None: missing.add('detail')
                break
            index += 1
    return '; '.join(chosen)


def issue_phrase(value):
    text = value or ''
    if re.search(r'subtitle', text, re.I) and re.search(r'transcript|spoken|dialogue', text, re.I) and re.search(r'mismatch|conflict|disagree|inconsisten', text, re.I):
        return 'Subtitle and transcript may disagree'
    first = re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', text).strip())[0].strip(' .')
    if 0 < len(first.split()) <= 15 and not re.match(r'^(?:and|but|while|which|whereas)\b', first, re.I):
        return first
    return ''


def issue_phrases(value):
    """Summarize only explicit, recognizable discrepancies without guessing."""
    text = re.sub(r'\s+', ' ', value or '').strip()
    if not text or re.match(r'^(?:none|no (?:significant|major|clear|obvious))\b', text, re.I):
        return [], False
    issues = []
    first = issue_phrase(text)
    if first: issues.append(first)
    sentences = re.split(r'(?<=[.!?])\s+|\s*[•]\s*', text)
    for part in sentences:
        part = re.sub(r'(?<=\d)\.(?=\d)', '·', part)
        patterns = (
            (r'\b(?:identity is not confirmed|identity (?:is )?unconfirmed|remains unidentified|is unidentified)\b', 'Character identity is unconfirmed'),
            (r'\b(?:camera angle|camera movement|camera)\b[^.!?]{0,120}\b(?:inconsistent(?:ly)?|conflict|discrepanc\w*)\b', 'Camera descriptions differ'),
            (r'\bposition discrepancy\b', 'Character positions differ across analyses'),
            (r'\b(?:transcript|dialogue|speaking|spoken)\b[^.!?]{0,180}\bmouth (?:is )?closed\b[^.!?]{0,100}\b(?:contradiction|conflict|discrepanc\w*)\b', 'Dialogue and video mouth movements disagree'),
        )
        for pattern, issue in patterns:
            if re.search(pattern, part, re.I) and issue not in issues:
                if issue in ('Camera descriptions differ', 'Character positions differ across analyses') and re.search(r'\b(?:no|not|without)\b[^.!?]{0,25}\b(?:inconsisten\w*|conflict\w*|discrepanc\w*)\b', part, re.I):
                    continue
                issues.append(issue)
                break
        if len(issues) >= 2: break
    return issues[:2], not issues


def unchanged(before, after, field=None):
    if before == after: return True
    if not before or not after: return False
    a, b = before.lower(), after.lower()
    if field == 'pose':
        # Framing visibility alone does not establish a change in posture.
        visibility = r'\b(?:legs and feet|legs|feet)\s+(?:are\s+)?not visible\b'
        a, b = re.sub(visibility, '', a), re.sub(visibility, '', b)
    # Similar text can describe opposite poses, directions, and expressions.
    contrasts = r'\b(?:left|right|up|down|raised|lowered|open|closed|smil\w*|frown\w*|away|forward|not|no)\b'
    if set(re.findall(contrasts, a)) != set(re.findall(contrasts, b)):
        return False
    if field == 'hands':
        relations = r'\b(?:in|on|under|over|behind|near|around|through)\b'
        if set(re.findall(relations, a)) != set(re.findall(relations, b)): return False
    def normalize(s):
        s = re.sub(r'\b(?:a|an|the|is|are|her|his|their|of)\b|[^a-z0-9 ]', ' ', s)
        if field == 'pose':
            s = re.sub(r'\b(?:upright|posture|position)\b', ' ', s)
            s = re.sub(r'\btorso\b', 'body', s)
        if field == 'hands':
            s = re.sub(r'\b(?:grips|gripping|grip|holds|holding|held)\b', 'hold', s)
            s = re.sub(r'\b(?:hand|hands|arm|arms)\b', 'limb', s)
            s = re.sub(r'\b(?:shopping|visible|with)\b', ' ', s)
        return ' '.join(s.split())
    a, b = normalize(a), normalize(b)
    if not a or not b: return False
    if a == b: return True
    if field == 'facing' and not re.search(r'\b(?:profile|three quarter view)\b', b):
        less_specific = ' '.join(re.sub(r'\b(?:profile|three quarter view)\b', '', a).split())
        if less_specific and less_specific == b: return True
    if field == 'gaze':
        def target(text):
            return re.sub(r'^(?:towards?|at)\s+', '', text)
        if target(a) == target(b): return True
    if field in ('pose','hands') and set(a.split()) == set(b.split()): return True
    if field == 'hands' and set(b.split()) <= set(a.split()) and (set(a.split()) - set(b.split())) <= {'both','limb'}:
        # A less specific final description does not establish that the
        # previously observed hand placement changed.
        return True
    return SequenceMatcher(None, a, b).ratio() >= .9


def anonymous_pairs(first, last, first_frame, last_frame, first_indices, last_indices):
    """Match only unique, strongly similar anonymous people at one location."""
    descriptors = {'black','brown','white','blonde','red','grey','gray','dark','glasses','hair','shirt','blazer','trousers','skirt','tie','top','coat','dress','bag','jacket','uniform'}
    def features(frame, index):
        people = [person for person in frame.get('visible_characters', []) if isinstance(person, dict)]
        if index >= len(people): return set()
        appearance = people[index].get('appearance', '')
        return set(re.findall(r'[a-z]+', appearance.lower())) & descriptors if isinstance(appearance, str) else set()
    candidates = []
    for i in first_indices:
        if first[i]['character_context_id']: continue
        for j in last_indices:
            if last[j]['character_context_id']: continue
            first_pos = first[i].get('frame_pos', '').split(' ')[0]
            last_pos = last[j].get('frame_pos', '').split(' ')[0]
            if not first_pos or first_pos != last_pos: continue
            shared = features(first_frame, i) & features(last_frame, j)
            if len(shared) >= 4 and shared & {'hair','glasses','bag','tie','skirt'}:
                candidates.append((i,j))
    return [(i,j) for i,j in candidates if sum(a==i for a,_ in candidates)==1 and sum(b==j for _,b in candidates)==1]


def clean_clause(value, limit=12):
    """Select a whole short clause, never a word-truncated sentence."""
    value = re.sub(r'\s+', ' ', value or '').strip()
    value = re.sub(r'^(?:she|he|they|her|his|the character|the person)\s+(?:is\s+)?', '', value, flags=re.I)
    for clause in re.split(r'(?<=[.!?;])\s+|,\s+(?=(?:with|while|but|and|holding|wearing|looking|facing)\b)', value, flags=re.I):
        clause = clause.strip(' .;,')
        if clause and len(clause.split()) <= limit:
            return clause
    return ''


def compact_phrase(key, value):
    """Remove grammatical boilerplate without dropping observed nouns or verbs."""
    value = value.strip(' .')
    value = re.sub(r'^(?:her|his|their)\s+', '', value, flags=re.I)
    if key == 'pose':
        value = re.sub(r'\bstanding posture\b', 'standing', value, flags=re.I)
    if key == 'expression':
        value = re.sub(r'^(?:she|he|they)\s+(?:has|have|shows?|is)\s+', '', value, flags=re.I)
        value = re.sub(r'^(?:expression\s+)?(?:is|appears)\s+', '', value, flags=re.I)
        value = re.sub(r',\s*(?:looking|gazing)\s+(?:directly\s+)?(?:forward|ahead|down|up)\s*$', '', value, flags=re.I)
    if key == 'gaze':
        value = re.sub(r'^(?:looking|gazing)\s+(?:directly\s+)?', '', value, flags=re.I)
    if key == 'facing':
        value = re.sub(r'^(?:turned to face|facing|faces|face)\s+(?:towards?\s+)?(?:the\s+)?', '', value, flags=re.I)
    if key == 'outfit_visible':
        value = re.sub(r'\s+are visible$', '', value, flags=re.I)
    return value.strip(' .')


def shot_props(frame):
    text = ' '.join(str(frame.get(k, '')) for k in ('composition', 'camera')).lower()
    shot = {}
    for pattern, value in [(r'extreme close[- ]up', 'ECU'), (r'medium close[- ]up', 'MCU'), (r'close[- ]up', 'CU'), (r'extreme (?:wide|long) shot', 'EWS'), (r'medium (?:wide|long) shot', 'MWS'), (r'(?:wide|long) shot', 'WS'), (r'medium shot', 'MS')]:
        if re.search(pattern, text):
            shot['size'] = value
            break
    for pattern, value in [(r'\boverhead(?:[- ](?:shot|view|angle|camera))\b|\bbird.s.eye\b', 'overhead'), (r'dutch|canted', 'dutch'), (r'low.angle', 'low'), (r'high.angle', 'high'), (r'eye.level', 'eye')]:
        if re.search(pattern, text):
            shot['angle'] = value
            break
    for pattern, value in [(r'handheld|hand.held', 'handheld'), (r'\bpan(?:ning|s)?\b', 'pan'), (r'\btilt(?:ing|s)?\b', 'tilt'), (r'\bdoll(?:y|ies|ying)\b', 'dolly'), (r'\bzoom(?:s|ing)?\b', 'zoom'), (r'\b(?:static|stationary|fixed)\b', 'static')]:
        match = None
        for candidate in re.finditer(pattern, text):
            before_match = re.split(r'[.;!?]', text[:candidate.start()])[-1]
            negated = r'\b(?:no|not|without)\s+(?:(?:camera|significant|visible|apparent|noticeable)\s+)*(?:(?:panning|tilting|zooming|movement|motion)\s*(?:,\s*)?(?:or|and)\s+)*$'
            if value != 'static' and re.search(negated, before_match): continue
            match = candidate
            break
        if match:
            shot['camera'] = value
            if value != 'static':
                before, after = text[max(0,match.start()-40):match.start()], text[match.end():]
                # Directions and speeds elsewhere can describe a person's
                # position or movement. Bind qualifiers to the camera verb.
                direction = (re.match(r'\s+(?:(?:slowly|quickly)\s+)?(?:to(?:wards?)?\s+)?(?:the\s+)?(left|right|up|down|in|out)\b', after) or
                             re.search(r'\b(left|right|up|down|in|out)\s*$', before))
                speed = (re.match(r'\s+(?:to(?:wards?)?\s+)?(?:the\s+)?(?:(?:left|right|up|down|in|out)\s+)?(slowly|slow|medium|fast|quickly)\b', after) or
                         re.search(r'\b(slowly|slow|medium|fast|quickly)(?:\s+(?:left|right|up|down|in|out))?\s*$', before))
                if direction: shot['direction'] = direction.group(1)
                if speed: shot['speed'] = {'slowly': 'slow', 'quickly': 'fast'}.get(speed.group(1), speed.group(1))
            break
    if 'camera' not in shot and re.search(r'\b(?:no|without)\s+(?:(?:significant|apparent|noticeable)\s+)?(?:camera\s+)?(?:movement|motion)\b', text):
        shot['camera'] = 'static'
    return shot


def video_camera(video):
    candidates = []
    for window in video.get('windows', []):
        analysis = window.get('analysis', {})
        if not isinstance(analysis, dict): continue
        candidates.extend(e.get('camera', '') for e in analysis.get('events', []) if isinstance(e, dict))
        candidates.extend(s for s in re.split(r'(?<=[.!?])\s+', analysis.get('overview', '')) if re.search(r'camera', s, re.I))
    movements = []
    for text in candidates:
        if not isinstance(text, str): continue
        value = shot_props({'camera': text})
        if value.get('camera'): movements.append(value)
    return next((m for m in movements if m['camera'] != 'static'), movements[0] if movements else {})


def phrase(text, pattern, limit=12):
    for part in re.split(r'(?<=[.!?])\s+', text or ''):
        if re.search(pattern, part, re.I):
            return short(part, limit)
    return ''


def horizontal_position(position):
    """Read placement before orientation; 'right ... facing left' is right."""
    location = re.split(r'\b(?:facing|faces|looking|gazing|angled|turned|profile)\b', position or '', maxsplit=1, flags=re.I)[0]
    match = re.search(r'\b(?:center[- ]left|left[- ]center|centre[- ]left|center[- ]right|right[- ]center|centre[- ]right|left|right|center(?:ed)?|centre(?:d)?)\b', location, re.I)
    if not match: return None
    word = match.group().lower()
    return 'left' if 'left' in word else 'right' if 'right' in word else 'center'


def position_delta(before, after, before_text, after_text):
    """Return (changed, uncertain) without inferring depth from a default."""
    if not before or not after:
        return False, bool(before or after)
    first, last = before.split(), after.split()
    depth = r'\b(?:foreground|background|mid[- ]?ground|middle[- ]?ground)\b'
    explicit_depth = bool(re.search(depth, before_text or '', re.I) and re.search(depth, after_text or '', re.I))
    if first[0] != last[0]:
        uncertain = len(first) >= 2 and len(last) >= 2 and first[1] != last[1] and not explicit_depth
        return True, uncertain
    if len(first) < 2 or len(last) < 2 or first[1] == last[1]:
        return False, False
    if explicit_depth:
        return True, False
    return False, True


def hand_details(pose_text, pose):
    """Keep separate visible limb observations within the two phrase budgets."""
    parts = []
    pattern = r'\b(?:hands?|arms?|hold(?:ing|s)?|held|grip(?:s|ping)?|palm)\b'
    for sentence in re.split(r'(?<=[.!?])\s+', pose_text or ''):
        if not re.search(pattern, sentence, re.I): continue
        if re.search(r'\b(?:obscured|not visible)\b', sentence, re.I) and not re.search(r'\b(?:hold(?:s|ing)?|held|grip(?:s|ping)?|raised|extended|rest(?:s|ing)?|linked)\b', sentence, re.I):
            continue
        value = re.sub(r'^(?:her|his|their)\s+', '', sentence, flags=re.I).strip(' .')
        value = re.sub(r'^(?:standing|seated|sitting|upright)\b[^,;]*[,;]\s*', '', value, flags=re.I)
        value = re.sub(r'^(?:standing|seated|sitting)(?:\s+(?:posture|upright))?\s+with\s+', '', value, flags=re.I)
        value = re.sub(r'\bthe handle of (?:a|the) shopping cart\b', 'shopping cart handle', value, flags=re.I)
        value = re.sub(r'\bis holding\b', 'holds', value, flags=re.I)
        value = re.sub(r'\b(?:is|are|a|the)\b\s*', '', value, flags=re.I)
        value = re.sub(r'\s+', ' ', value).strip(' .,;')
        if value and value not in parts: parts.append(value)
    if not parts: return pose, '', False
    combined = '; '.join(parts)
    if len(combined.split()) <= 12:
        return pose, combined, False
    if len(parts) > 1:
        first_pose = '; '.join(x for x in (pose, parts[0]) if x)
        remaining = '; '.join(parts[1:])
        if len(first_pose.split()) <= 12 and len(remaining.split()) <= 12:
            return first_pose, remaining, False
    # A warning accompanies any clause that cannot be represented safely.
    candidates = [part for part in parts if len(part.split()) <= 12]
    return pose, candidates[0] if candidates else '', True


def expression_details(source, value):
    """Include observed eye and mouth state when it fits without guessing."""
    additions = []
    patterns = (
        ('eyes', r'\b(?:the\s+)?eyes?\s+(?:are\s+|is\s+)?((?:wide\s+)?open|closed|half[- ]closed|half[- ]lidded)(\s+or\s+(?:(?:nearly|partially|slightly)\s+)?(?:open|closed))?\b'),
        ('mouth', r'\b(?:the\s+)?mouth\s+(?:is\s+|remains\s+)?((?:slightly\s+)?(?:open|closed))(\s+or\s+(?:(?:nearly|partially|slightly)\s+)?(?:open|closed))?\b'),
    )
    for label, pattern in patterns:
        match = re.search(pattern, source or '', re.I)
        if match and not re.search(r'\b'+label+r'\b', value or '', re.I):
            additions.append(label+' '+(match.group(1)+(match.group(2) or '')).lower())
    result, incomplete = value, False
    for item in additions:
        combined = '; '.join(part for part in (result,item) if part)
        if len(combined.split()) <= 12: result = combined
        else: incomplete = True
    return result, incomplete


def subjects(frame, detections, boundary='first', missing=None):
    out = []
    first = detections
    people = [person for person in frame.get('visible_characters', []) if isinstance(person, dict)]
    zones = [horizontal_position(person.get('position', '')) for person in people]
    def box_zone(detection):
        box, width = detection.get('box'), detection.get('frame_width')
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or not isinstance(width,(int,float)) or not math.isfinite(width) or width<=0:
            return None
        x = (box[0]+box[2])/(2*width)
        return 'left' if x < 1/3 else 'right' if x > 2/3 else 'center'
    for idx, person in enumerate(people):
        evidence = person.get('identity_evidence', '')
        evidence = evidence if isinstance(evidence, str) else ''
        matches = [d for d in first if d['position'] == boundary and d['id'] in evidence]
        pos = person.get('position', '').lower()
        horizontal = horizontal_position(pos)
        if not matches and re.search(r'\b(?:user.verified|confirmed|identified as)\b', evidence, re.I):
            # Some older descriptions name a verified person without citing the
            # box ID. Only link an unambiguous current, verified boundary box.
            candidates = [d for d in first if d['position'] == boundary and not d['stale'] and d.get('character_context_id')
                          and (d.get('verified') or d.get('label_source') == 'manual')
                          and d.get('name') and d['name'].casefold() in evidence.casefold()]
            if len(candidates) == 1:
                candidate = candidates[0]
                box, width = candidate.get('box'), candidate.get('frame_width')
                if horizontal and isinstance(box, list) and len(box) == 4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) and isinstance(width, (int, float)) and math.isfinite(width) and width > 0:
                    x = (box[0] + box[2]) / (2 * width)
                    box_pos = 'left' if x < 1/3 else 'right' if x > 2/3 else 'center'
                    if {horizontal, box_pos} == {'left', 'right'}: candidates = []
                matches = candidates
        if not matches and horizontal and zones.count(horizontal)==1:
            # A single boundary box in a single described frame zone can link
            # even when the model did not repeat its label. Ambiguous zones are
            # left unknown; automatic labels remain only likely.
            in_zone = [d for d in first if d['position']==boundary and not d['stale'] and box_zone(d)==horizontal]
            if len(in_zone)==1 and in_zone[0].get('character_context_id'):
                matches = in_zone
        # The model can cite a verified box; do not infer identity from dialogue.
        match = matches[0] if len(matches) == 1 and not matches[0]['stale'] else None
        depth = 'fg' if 'foreground' in pos else 'bg' if 'background' in pos else 'mid'
        obj = {'character_context_id': match.get('character_context_id') if match else None, 'name': match.get('name') if match and match.get('name') else 'unknown'}
        if match and match.get('character_context_id'):
            obj['identity'] = 'confirmed' if match.get('label_source') == 'manual' or match.get('verified') else 'likely'
        if horizontal: obj['frame_pos'] = horizontal + ' ' + depth
        position_text = person.get('position', '') + ' ' + person.get('pose', '')
        facing = re.search(r'\b(?:facing|faces|face|turned to face)\s+(?:(?:slightly|directly|mostly)\s+)?(?:to(?:wards?)?\s+)?(?:the\s+)?(?:camera|left|right|forward|away|male character|female character|other character|man|woman)\b', position_text, re.I)
        view = re.search(r'\b(?:profile|three-quarter view)\b', position_text, re.I)
        facing_value = facing.group(0).lower() if facing else ''
        if view and view.group(0).lower() not in facing_value:
            facing_value = '; '.join(part for part in (facing_value,view.group(0).lower()) if part)
        pose_text, expression = person.get('pose', ''), person.get('expression', '')
        pose, hand, incomplete_hands = hand_details(pose_text, compact_phrase('pose', clean_clause(pose_text)))
        if not pose and pose_text and missing is not None and not re.match(r'^(?:unknown|not visible|obscured|cannot be (?:seen|determined))\b', pose_text.strip(), re.I):
            missing.add('pose')
        if re.search(r'\b(?:posture|body|torso)\s+(?:(?:is|appears)\s+)?upright\b', pose_text, re.I) and not re.search(r'\bnot\s+upright\b', pose_text, re.I) and 'upright' not in pose.lower():
            updated_pose = '; '.join(part for part in (pose,'upright') if part)
            if len(updated_pose.split()) <= 12: pose = updated_pose
            elif missing is not None: missing.add('pose')
        if hand and ' with ' in pose.lower():
            prefix, suffix = re.split(r'\s+with\s+', pose, maxsplit=1, flags=re.I)
            if suffix.strip(' .,;').casefold() == hand.strip(' .,;').casefold(): pose = prefix
        if incomplete_hands and missing is not None: missing.add('hands')
        outfit = phrase(person.get('appearance', ''), r'wear|shirt|coat|blazer|dress|clothing|uniform', 60)
        outfit = re.sub(r'^.*?\b(?:wearing|wears)\s+', '', outfit, flags=re.I).strip(' .') if re.search(r'\b(?:wearing|wears)\b', outfit, re.I) else clean_clause(outfit)
        outfit = re.sub(r'\b(?:a|an)\s+', '', outfit, flags=re.I)
        if len(outfit.split()) > 12: outfit = clean_clause(outfit)
        gaze = re.search(r'\b(?:looking|gazing)\s+(?:directly\s+)?(?:down\s+at\s+(?:the\s+)?[\w-]+|at\s+(?:the\s+)?[\w-]+|towards?\s+(?:the\s+)?[\w-]+|forward|ahead|down|up)\b', expression + ' ' + pose_text, re.I)
        smile = re.search(r'\b(?:slight|gentle|broad|small)?\s*smil(?:e|es|ing)\b', expression, re.I)
        first_expression = re.split(r'(?<=[.!?])\s+', expression.strip(), maxsplit=1)[0].strip(' .')
        first_expression = re.sub(r'\bwith\s+(?:a|an)\s+', 'with ', first_expression, flags=re.I)
        if first_expression and len(first_expression.split()) <= 12:
            expression_value = first_expression
        else:
            expression_value = smile.group().strip().lower() if smile else clean_clause(expression)
        if re.match(r'^(?:(?:her|his|their)\s+)?(?:face\s+(?:is\s+)?not visible|unknown|not visible|obscured|cannot be (?:seen|determined))\b', expression_value, re.I):
            expression_value = ''
        expression_value, incomplete_expression = expression_details(expression, expression_value)
        if incomplete_expression and missing is not None: missing.add('expression')
        if not expression_value and expression and missing is not None and not re.match(r'^(?:(?:her|his|their)\s+)?(?:unknown|not visible|obscured|face\s+(?:is\s+)?not visible|cannot be (?:seen|determined))\b', expression.strip(), re.I):
            missing.add('expression')
        for key, value in {
            'facing': facing_value,
            'pose': pose,
            'hands': hand,
            'gaze': gaze.group(0).strip().lower() if gaze else 'direct eye contact' if re.search(r'\bdirect eye contact\b', expression, re.I) else '',
            'expression': expression_value,
            'outfit_visible': outfit
        }.items():
            value = compact_phrase(key, value)
            if value and not re.search(r'^(?:not visible|unknown|obscured)\.?$', value, re.I): obj[key] = value
        out.append(obj)
    return out


MOTION = re.compile(r"\b(?:turns?|turning|moves?|moving|raises?|raising|lowers?|lowering|closes|closing|opens|opening|smiles?|smiling|walks?|walking|steps?|stepping|reaches?|reaching|shifts?|shifting|leans?|leaning|bends?|bending|push(?:es|ing)?|pull(?:s|ing)?|gestur(?:es?|ing)|blinks?|blinking|grasps?|grasping|brings?|bringing|looks? (?:up|down|toward|away))\b", re.I)


def event_action(text):
    """Keep visual observations across the event, omitting only caveat prose."""
    if not isinstance(text, str):
        return ''
    sentences = [s.strip(' .') for s in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', text).strip()) if s.strip()]
    if not sentences:
        return ''
    actions = []
    for sentence in sentences:
        if re.fullmatch(r'(?:there (?:is|are) )?no significant uncertainty', sentence, re.I):
            continue
        # Qualifiers can apply to observable motion as well as interpretation.
        # Keep them: removing the rest of the clause could erase an action or
        # turn an uncertain observation into a definite one.
        sentence = re.sub(r'^(?:she|he|they)\s+(?:then\s+)?', '', sentence, flags=re.I)
        if sentence and not any(sentence.casefold() == old.casefold() for old in actions):
            actions.append(sentence)
    return '; '.join(actions)


def static_action_is_in_state(action, state):
    """Only elide a static description whose material words are in that state."""
    if not state or not re.search(r'\b(?:stands?|standing|remains?|maintains?|stationary|still|holds?|holding)\b', action, re.I):
        return False
    if re.search(r'\b(?:turns?|turning|moves?|moving|raises?|raising|lowers?|lowering|closes|closing|opens|opening|smiles?|smiling|walks?|walking|steps?|stepping|reaches?|reaching|shifts?|shifting|leans?|leaning|bends?|bending|push(?:es|ing)?|pull(?:s|ing)?|gestur(?:es?|ing)|blinks?|blinking|grasps?|grasping|brings?|bringing)\b', action, re.I):
        return False
    # Do not elide object, limb, or inter-person relations on a bag-of-words
    # match: swapping their participants could silently change the composition.
    allowed = {'a','the','with','and','in','to','toward','towards','of','the','frame','side','of',
               'stands','stand','standing','remains','remain','maintains','maintain','stationary','still','static',
               'pose','posture','upright','facing','faces','face','looking','looks','look','gazing','gaze',
               'forward','camera','neutral','expression','profile','right','left','directly','slightly','her','his','their'}
    if set(re.findall(r'[a-z]+', action.lower())) - allowed:
        return False
    # A gaze or facing direction must be supported by its own field, not by a
    # coincidentally matching direction in a hand or frame-position field.
    for clause in re.split(r'[;,]|\band\b', action, flags=re.I):
        for direction in re.findall(r'\b(?:left|right|up|down|forward|away)\b', clause, re.I):
            if re.search(r'\b(?:look(?:s|ing)?|gaze)\b', clause, re.I) and direction.lower() not in state.get('gaze','').lower(): return False
            if re.search(r'\b(?:face|faces|facing)\b', clause, re.I) and direction.lower() not in state.get('facing','').lower(): return False
    from .composition_compaction import fact_tokens, PROTECTED
    contents = ' '.join(str(state.get(k, '')) for k in ('pose','hands','gaze','facing','expression','outfit_visible'))
    known = fact_tokens(contents, 'beats.0.action')
    if state.get('gaze'): known.add('look')
    if state.get('facing'): known.add('face')
    if state.get('expression'): known.add('expression')
    if state.get('pose'): known.update(('pose','posture'))
    redundant = {'remain','remains','maintain','maintains','still','stationary'}
    words = fact_tokens(action, 'beats.0.action') - redundant
    protected_action = {m.group().lower() for m in PROTECTED.finditer(action)}
    protected_state = {m.group().lower() for m in PROTECTED.finditer(contents)}
    return bool(words) and words <= known and protected_action <= protected_state


def video_beats(video, start, end, static_states=None):
    events, partial = [], False
    duration = end - start
    windows = video.get('windows', []) if isinstance(video, dict) else []
    if not isinstance(windows, list) or not windows:
        return [], True
    for window in windows:
        if not isinstance(window, dict):
            partial = True; continue
        source, analysis = window.get('source', {}), window.get('analysis', {})
        if not isinstance(source, dict) or not isinstance(analysis, dict):
            partial = True; continue
        ws, we = source.get('source_start'), source.get('source_end')
        if any(isinstance(t, bool) or not isinstance(t, (int,float)) or not math.isfinite(t) for t in (ws,we)) or we <= ws:
            partial = True; continue
        source_events = analysis.get('events', [])
        if not isinstance(source_events, list):
            partial = True; continue
        usable = []
        for event in source_events:
            if not isinstance(event, dict):
                partial = True; continue
            a, b = event.get('start_seconds'), event.get('end_seconds')
            if any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) for t in (a, b)):
                partial = True; continue
            if a < -.05 or b > we - ws + .05:
                partial = True; continue
            a, b = max(0., ws - start + max(0., a)), min(duration, ws - start + min(b, we-ws))
            if b <= a: continue
            a, b = round(a, 3), round(b, 3)
            if b <= a: continue
            action = event_action(event.get('visual_action', ''))
            subject = event.get('subject')
            subject = re.sub(r'\s+', ' ', subject).strip() if isinstance(subject, str) else ''
            if not action:
                partial = True; continue
            if static_states and subject in static_states:
                states = static_states[subject]
                # A later sentence in the same event can be static even when
                # another sentence describes motion. Remove only whole simple
                # sentences already represented by the matching boundary state.
                if ':' not in action:
                    kept = []
                    for part in action.split('; '):
                        duplicate = ((a <= .05 and static_action_is_in_state(part, states.get('first'))) or
                                     (duration-b <= .05 and static_action_is_in_state(part, states.get('last'))))
                        if not duplicate: kept.append(part)
                    action = '; '.join(kept) if kept else 'holds pose'
            usable.append({'t0': a, 't1': b, 'subject': subject or 'unknown', 'action': action})
        if not usable:
            partial = True
        events.extend(usable)
    events.sort(key=lambda e: (e['t0'], e['t1'], e['subject'], e['action']))
    # Keep actor/action pairs until serialization, so simultaneous actions of
    # the same actor do not repeat the actor name or silently disappear.
    grouped = []
    for event in events:
        prior = grouped[-1] if grouped else None
        continuation = prior and len(prior['parts']) == 1 and event['subject'].casefold() == prior['parts'][0][0].casefold() and event['t0'] <= prior['t1'] + .05 and SequenceMatcher(None, event['action'].casefold(), prior['parts'][0][1].casefold()).ratio() >= .92
        if prior and (event['t0'] < prior['t1'] or continuation):
            prior['t1'] = max(prior['t1'], event['t1'])
            if not any(actor.casefold() == event['subject'].casefold() and action.casefold() == event['action'].casefold() for actor,action in prior['parts']):
                prior['parts'].append((event['subject'], event['action']))
        else:
            grouped.append({'t0':event['t0'],'t1':event['t1'],'parts':[(event['subject'],event['action'])]})
    coalesced = []
    for beat in grouped:
        prior = coalesced[-1] if coalesced else None
        signature = lambda item: {(actor.casefold(), action.casefold()) for actor,action in item['parts']}
        if prior and 0 <= beat['t0']-prior['t1'] <= .05 and signature(prior) == signature(beat):
            prior['t1'] = beat['t1']
        else:
            coalesced.append(beat)
    grouped = coalesced
    maximum = max(1, math.ceil(duration / 2))
    while len(grouped) > maximum:
        i = min(range(len(grouped)-1), key=lambda n: grouped[n+1]['t0']-grouped[n]['t1'])
        grouped[i]['t1'] = grouped[i+1]['t1']
        for part in grouped[i+1]['parts']:
            if part not in grouped[i]['parts']: grouped[i]['parts'].append(part)
        grouped.pop(i+1)
    result = []
    for beat in grouped:
        first_actor = beat['parts'][0][0]
        pieces = []
        for actor,action in beat['parts']:
            pieces.append(action if actor == first_actor else actor + ': ' + action)
        result.append({'t0':beat['t0'], 't1':beat['t1'], 'subject':first_actor, 'action':'; '.join(pieces)})
    return result, partial


def utterances(segments, start, end):
    groups = []
    for seg in segments:
        speaker = seg.get('speaker_label') or 'unknown'
        char_id = seg.get('character_context_id')
        text = seg.get('text', '').strip()
        if not text: continue
        if groups and (groups[-1]['speaker_label'], groups[-1]['character_context_id']) == (speaker, char_id):
            groups[-1]['segments'].append(seg)
        else:
            groups.append({'speaker_label': speaker, 'character_context_id': char_id, 'character_name': seg.get('character_name') or 'unknown', 'segments': [seg]})
    out = []
    for group in groups:
        overlap = [s for s in group['segments'] if s['start_time'] < end and s['end_time'] > start]
        if not overlap: continue
        text = ''
        previous_raw = ''
        for seg in overlap:
            raw = seg['text']
            fragment = raw.strip()
            if not fragment: continue
            # Respect explicit whitespace in the transcript. When neither
            # fragment supplies it, separate word-like text in scripts that
            # normally use spaces, while retaining CJK character fragments.
            def unspaced_script(char):
                name = unicodedata.name(char, '')
                return any(script in name for script in ('CJK ', 'HIRAGANA ', 'KATAKANA ', 'HANGUL '))
            explicit = bool(previous_raw and (previous_raw[-1].isspace() or raw[0].isspace()))
            word_boundary = (text and text[-1].isalnum() and fragment[0].isalnum() and
                             not unspaced_script(text[-1]) and not unspaced_script(fragment[0]))
            gap = ' ' if text and (explicit or word_boundary) else ''
            text += gap + fragment
            previous_raw = raw
        out.append({'speaker_label': group['speaker_label'], 'character_context_id': group['character_context_id'], 'character_name': group['character_name'], 'text': text,
                    't0': round(max(0, overlap[0]['start_time']-start), 3), 't1': round(min(end-start, max(s['end_time'] for s in overlap)-start), 3),
                    'crosses_cut': group['segments'][0]['start_time'] < start or max(s['end_time'] for s in group['segments']) > end})
    return out


def subtitle_lines(events, start, end, missing=None):
    """Collapse adjacent OCR variants of the same sampled subtitle line."""
    def protected_words(text):
        # Similar OCR strings can still carry opposite meanings or different
        # quantities. Never deduplicate those solely on edit similarity.
        return tuple(re.findall(r"\b(?:not|no|never|neither|nor|não|nao|nunca|jamais|pas|kein|keine|left|right|esquerda|direita|\d+(?:[.,]\d+)?)\b|n't\b", text.casefold()))

    def observation_time(event):
        value = event.get('first_observed') if isinstance(event, dict) else None
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else math.inf

    lines = []
    for event in sorted(events, key=observation_time) if isinstance(events, list) else []:
        if not isinstance(event, dict): continue
        value = event.get('text', '')
        if not isinstance(value, str): continue
        text = re.sub(r'\s+', ' ', value).strip()
        t = event.get('first_observed')
        if not text or not any(c.isalnum() for c in text): continue
        if isinstance(t, bool) or not isinstance(t, (int,float)) or not math.isfinite(t) or t < start-.05 or t > end+.05:
            if missing is not None: missing.add('time')
            continue
        language = event.get('language') or 'unknown'
        relative = round(max(0, min(end-start, t-start)), 3)
        previous = lines[-1] if lines else None
        same = previous and previous['lang'] == language and relative - previous['_last'] <= 3 and relative >= previous['_last'] and (
            text.casefold() == previous['text'].casefold() or
            (protected_words(text) == protected_words(previous['text']) and min(len(text), len(previous['text'])) >= 12 and SequenceMatcher(None, text.casefold(), previous['text'].casefold()).ratio() >= .9))
        if same:
            # OCR samples are observations, not interval boundaries. Keep the
            # first observation time and the latest reading of the same line.
            previous['text'] = text
            previous['_last'] = relative
        else:
            lines.append({'text': text, 'lang': language, 't': relative, '_last': relative})
    return [{k: line[k] for k in ('text','lang','t')} for line in lines]


def make_composition_brief(episode_id, first, last, *, include='', fields='', only_ready=False, updated_since=None, apply_compaction=True):
    with db.connect() as conn:
        conn.execute('BEGIN')
        return _make_composition_brief(conn, episode_id, first, last, include=include, fields=fields, only_ready=only_ready, updated_since=updated_since, apply_compaction=apply_compaction)


def _make_composition_brief(conn, episode_id, first, last, *, include='', fields='', only_ready=False, updated_since=None, apply_compaction=True):
    def one(sql, args=()):
        row = conn.execute(sql, args).fetchone()
        return dict(row) if row else None
    def all_rows(sql, args=()):
        return [dict(row) for row in conn.execute(sql, args)]
    includes = selection(include, INCLUDES, 'include')
    selected = selection(fields, FIELDS, 'fields')
    since = None
    if updated_since:
        try:
            since = datetime.fromisoformat(updated_since.replace('Z', '+00:00'))
            if since.tzinfo is None: raise ValueError()
            since = since.astimezone(timezone.utc)
        except ValueError:
            raise CompositionError(422, 'updated_since must be an ISO 8601 timestamp with timezone')
    episode = one('SELECT * FROM episodes WHERE id=?', (episode_id,))
    if not episode: raise CompositionError(404, 'Episode not found')
    if first < 1 or last < first or last-first >= 500: raise CompositionError(422, 'Choose an inclusive range of at most 500 cuts')
    cuts = all_rows('''SELECT c.*, ch.updated_at FROM cuts c LEFT JOIN composition_changes ch ON ch.cut_id=c.id
                           WHERE c.episode_id=? AND c.active=1 AND c.cut_index BETWEEN ? AND ? ORDER BY c.cut_index''', (episode_id, first-1, last-1))
    if len(cuts) != last-first+1: raise CompositionError(404, 'One or more requested active cuts do not exist')
    folder = media.episode_dir(episode['project_id'], episode_id)
    audio_available = bool(episode['has_audio'] and (folder/'audio.wav').is_file())
    source_available = (folder/'proxy.mp4').is_file() or (folder / f"original{episode['original_ext']}").is_file()
    active = all_rows("SELECT id,payload_json FROM jobs WHERE episode_id=? AND pipeline='analyze' AND status='running'", (episode_id,))
    all_segments = all_rows('''SELECT t.id,t.start_time,t.end_time,t.text,t.speaker_label,t.character_context_id,c.title AS character_name
        FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
        WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 ORDER BY t.start_time,t.id''', (episode_id,))
    updating = set()
    for job in active:
        try: payload = json.loads(job['payload_json'])
        except (ValueError, TypeError): continue
        for item in payload.get('cuts', []):
            cut_id = item.get('cut_id')
            checkpoint = one('SELECT completed FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job['id'], cut_id))
            if cut_id and (not checkpoint or not checkpoint['completed']): updating.add(cut_id)
    records, pending = [], []
    for cut in cuts:
        rows = structured_results.list_results(cut['id'], conn=conn)
        by_type = {row['type']: row for row in reversed(rows) if row['content'] is not None}
        needed = {'initial_frame', 'last_frame', 'video_analysis', 'subtitle_ocr'}
        stale = False
        for name in needed & by_type.keys():
            stored_context = one('SELECT context_json FROM analysis_results WHERE id=?', (by_type[name]['id'],))
            try:
                source_cut = json.loads(stored_context['context_json'] or '{}').get('cut') or {}
                if source_cut.get('start_time') != cut['start_time'] or source_cut.get('end_time') != cut['end_time']:
                    stale = True
            except (TypeError, ValueError, AttributeError):
                stale = True
        latest_attempt = one("""SELECT j.status,a.completed FROM analysis_job_cuts a JOIN jobs j ON j.id=a.job_id
                                WHERE a.cut_id=? AND j.pipeline='analyze' ORDER BY j.rowid DESC LIMIT 1""", (cut['id'],))
        interrupted = latest_attempt and latest_attempt['status'] != 'done' and not latest_attempt['completed']
        ready = needed.issubset(by_type) and not stale and cut['id'] not in updating and not interrupted and source_available and (not episode['has_audio'] or audio_available)
        if not ready: pending.append(cut['cut_index']+1)
        changed = cut['updated_at'] or '1970-01-01T00:00:00.000Z'
        if only_ready and not ready: continue
        if since and datetime.fromisoformat(changed.replace('Z', '+00:00')) < since: continue
        fingerprint = hashlib.sha256(json.dumps([(key, row['id']) for key,row in sorted(by_type.items())], separators=(',', ':')).encode()).hexdigest()[:16]
        record = {'id': cut['id'], 'cut_number': cut['cut_index']+1, 'duration': round(cut['end_time']-cut['start_time'], 6), 'ready': ready, 'updated_at': changed, 'analysis_rev': fingerprint}
        summary_job = one("SELECT status FROM jobs WHERE episode_id=? AND pipeline='regen' AND json_valid(payload_json) AND json_extract(payload_json,'$.production_brief')=1 AND json_extract(payload_json,'$.cut_id')=? ORDER BY rowid DESC LIMIT 1", (episode_id, cut['id']))
        if not ready:
            record['unavailable'] = {'shot': 'visual_stale' if stale else 'visual_pending'}
            if not source_available: record['unavailable']['frame'] = 'frame'
            if episode['has_audio'] and not audio_available: record['unavailable']['audio'] = 'audio'
        else:
            start, end = cut['start_time'], cut['end_time']
            initial, last_frame = by_type['initial_frame']['content'], by_type['last_frame']['content']
            video = by_type['video_analysis']['content']
            stored = one('SELECT context_json FROM analysis_results WHERE id=?', (by_type['initial_frame']['id'],))
            try:
                snapshot = json.loads(stored['context_json'] or '{}').get('model_context_hints', {}).get('character_detections') or {}
                verified = {(d.get('id'), d.get('character_context_id')) for d in snapshot.get('detections', []) if snapshot.get('labels_confirmed_by_queue') and isinstance(d, dict)}
            except (TypeError, ValueError, AttributeError): verified = set()
            run = one('SELECT * FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['id'],))
            links = []
            if run:
                stale = run['cut_start'] != start or run['cut_end'] != end
                try:
                    widths = {f['position']: f['image_size'][0] for f in json.loads(run['frames_json']) if isinstance(f, dict) and isinstance(f.get('image_size'), list) and len(f['image_size']) >= 2}
                except (TypeError, ValueError, KeyError): widths = {}
                for d in all_rows('''SELECT d.*,c.title AS name FROM character_detections d LEFT JOIN context_items c ON c.id=d.character_context_id WHERE d.run_id=? ORDER BY d.rowid''', (run['id'],)):
                    try: box = json.loads(d['box_json'])
                    except (TypeError, ValueError): continue
                    if isinstance(box, list): box = [round(n, 1) if isinstance(n, (int, float)) else n for n in box]
                    links.append({'id': d['id'], 'position': d['position'], 'box': box, 'character_context_id': d['character_context_id'], 'name': d['name'] or 'unknown', 'label_source': d['label_source'] or 'unknown', 'stale': stale, 'verified': (d['id'], d['character_context_id']) in verified, 'frame_width': widths.get(d['position'])})
            record['initial_frame_id'] = by_type['initial_frame']['id']
            record['last_frame_id'] = by_type['last_frame']['id']
            shot = shot_props(initial)
            video_movement = video_camera(video)
            if video_movement:
                shot.update({key:value for key,value in video_movement.items() if key in ('camera','direction','speed')})
            if shot: record['shot'] = shot
            missing_descriptions = set()
            for name, limit in [('setting', 20), ('lighting', 15)]:
                omitted = set()
                value = setting_phrase(initial.get(name, ''), limit, omitted)
                if value: record[name] = value
                if omitted: missing_descriptions.add(name)
            missing_subjects = set()
            chars = subjects(initial, links, missing=missing_subjects)
            if chars: record['subjects'] = chars
            missing_final = set()
            final = subjects(last_frame, links, boundary='last', missing=missing_final)
            delta, matched_first, matched_last, static_states = [], set(), set(), {}
            initial_people = [person for person in initial.get('visible_characters', []) if isinstance(person, dict)]
            final_people = [person for person in last_frame.get('visible_characters', []) if isinstance(person, dict)]
            def subject_changes(i, j):
                before, after = chars[i], final[j]
                changed, uncertain = position_delta(before.get('frame_pos'), after.get('frame_pos'),
                                                    initial_people[i].get('position', ''), final_people[j].get('position', ''))
                if uncertain: missing_final.add('frame_pos')
                changes = {'frame_pos': after['frame_pos']} if changed else {}
                changes.update({key: after[key] for key in ('facing','pose','hands','gaze','expression')
                                if key in after and not unchanged(before.get(key), after.get(key), key)})
                return changes
            # Known character IDs are stable even when a model changes list order.
            for i, char in enumerate(chars):
                if not char['character_context_id']: continue
                candidates = [j for j, end_char in enumerate(final) if j not in matched_last and end_char['character_context_id'] == char['character_context_id']]
                if len(candidates) != 1: continue
                j = candidates[0]; matched_first.add(i); matched_last.add(j)
                changes = subject_changes(i, j)
                if changes: delta.append({'subject': final[j]['name'], **changes})
                if sum(s.get('name') == char['name'] for s in chars) == 1:
                    # The final state is reconstructed only from fields that
                    # the final observation actually contains.
                    final_state = {key: changes.get(key, char.get(key)) for key in ('facing','pose','hands','gaze','expression','outfit_visible') if key in final[j] and changes.get(key, char.get(key))}
                    static_states[char['name']] = {'first':char, 'last':final_state}
            pairs = anonymous_pairs(chars, final, initial, last_frame,
                                    [i for i in range(len(chars)) if i not in matched_first],
                                    [j for j in range(len(final)) if j not in matched_last])
            for i,j in pairs:
                matched_first.add(i); matched_last.add(j)
                changes = subject_changes(i, j)
                if changes: delta.append({'subject':'unknown', **changes})
            remaining_first = [i for i in range(len(chars)) if i not in matched_first]
            remaining_last = [j for j in range(len(final)) if j not in matched_last]
            # When either frame has an unidentified person, list order does not
            # prove continuity. Do not manufacture entry, exit, or pose changes.
            ambiguous = bool(remaining_first and remaining_last and
                             (any(not chars[i]['character_context_id'] for i in remaining_first) or any(not final[j]['character_context_id'] for j in remaining_last)))
            if not ambiguous:
                delta.extend({'subject': chars[i]['name'], 'change': 'left frame'} for i in remaining_first)
                delta.extend({'subject': final[j]['name'], 'change': 'entered frame'} for j in remaining_last)
            last_shot = shot_props(last_frame)
            end_state = {}
            if delta: end_state['subjects'] = delta
            if last_shot.get('size') and last_shot.get('size') != shot.get('size'): end_state['size'] = last_shot['size']
            if end_state: record['end_state'] = end_state
            beats, partial = video_beats(video, start, end, static_states=static_states)
            if beats: record['beats'] = beats
            segments = [s for s in all_segments if s['start_time'] < end and s['end_time'] > start]
            dialogue = utterances(all_segments, start, end)
            if dialogue: record['dialogue'] = dialogue
            missing_subtitles = set()
            subtitles = subtitle_lines(by_type['subtitle_ocr']['content'].get('events', []), start, end, missing=missing_subtitles)
            if subtitles: record['subtitles'] = subtitles
            consistency = by_type.get('consistency_summary', {}).get('content', {})
            confusing = consistency.get('confusing', '')
            issues, incomplete_issues = issue_phrases(confusing)
            if issues: record['issues'] = issues
            if links:
                grouped = []
                for link in links:
                    key = (link['character_context_id'], link['label_source'], link['stale'])
                    target = next((g for g in grouped if g['_key'] == key), None)
                    if target is None:
                        target = {'_key':key, 'label_source':link['label_source'], 'stale':link['stale']}
                        matching_subjects = [i for i, subject in enumerate(chars) if link['character_context_id'] and subject['character_context_id'] == link['character_context_id']]
                        if len(matching_subjects) == 1:
                            target['subject'] = matching_subjects[0]
                        elif link['character_context_id']:
                            target['character_context_id'] = link['character_context_id']
                        grouped.append(target)
                    if link['position'] in ('first','last'):
                        boxes = target.setdefault(link['position'], [])
                        if link['box'] not in boxes: boxes.append(link['box'])
                record['links'] = [{k:v for k,v in group.items() if k != '_key'} for group in grouped]
            record['confidence'] = {k: 'low' for k in ('shot', 'setting', 'lighting', 'subjects', 'beats', 'end_state') if k in record}
            missing = {}
            if not shot: missing['shot'] = 'shot'
            if summary_job and summary_job['status'] == 'failed': missing['shot'] = 'summary_failed'
            if partial: missing['beats_partial'] = 'beats_partial'
            for name in missing_descriptions: missing[name] = name + '_partial'
            if missing_subjects: missing['subjects'] = 'subjects_partial'
            if missing_subtitles: missing['subtitles'] = 'subtitles_partial'
            if ambiguous: missing['end_state'] = 'identity_ambiguous'
            elif missing_final: missing['end_state'] = 'end_state_partial'
            if incomplete_issues: missing['issues'] = 'issues_partial'
            if not source_available and not (folder/'thumbnails'/f"{cut['id']}_first.jpg").is_file(): missing['frame'] = 'frame'
            if not audio_available: missing['audio'] = 'audio'
            if missing: record['unavailable'] = missing
            if apply_compaction:
                from .composition_compaction import apply_cached
                record = apply_cached(conn, cut['id'], record)
            reference_subjects(record)
            for name in includes:
                if name == 'dialogue_segments': record[name] = segments
                elif name in by_type: record[name] = by_type[name]
        if 'unavailable' in record:
            record['unavailable'] = {key: REASON_CODES[reason] for key, reason in record['unavailable'].items()}
        # Revision changes when any projected fact or readiness changes, including
        # transcript edits and detection relabels, and is independent of selection.
        revision_body = {k:v for k,v in record.items() if k not in {'updated_at','analysis_rev'} and k not in includes}
        revision = hashlib.sha256(json.dumps({'record': revision_body, 'results': fingerprint}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).digest()[:8]
        record['analysis_rev'] = base64.urlsafe_b64encode(revision).decode('ascii').rstrip('=')
        if selected:
            mandatory = {'id','cut_number','ready','updated_at','analysis_rev'}
            for name in ('start_time','end_time'):
                if name in selected: record[name] = cut[name]
            if 'links' in selected and 'subjects' not in selected:
                # Preserve self-contained links when their subject table is not
                # among the selected fields.
                for link in record.get('links', []):
                    if 'subject' in link:
                        index = link.pop('subject')
                        link['character_context_id'] = record['subjects'][index]['character_context_id']
            if 'subjects' not in selected:
                expand_subject_references(record)
            record = {k:v for k,v in record.items() if k in selected | mandatory | includes or (not ready and k == 'unavailable')}
        records.append(record)
    used_reasons = {code for record in records for code in record.get('unavailable', {}).values()}
    return {'episode_id': episode_id, 'from': first, 'to': last, 'ready_count': len(cuts)-len(pending), 'pending': pending,
            'unavailable_reasons': {REASON_CODES[name]: reason for name,reason in REASONS.items() if REASON_CODES[name] in used_reasons}, 'cuts': records}
