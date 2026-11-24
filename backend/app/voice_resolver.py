"""Unknown voices given to characters (docs/autopilot/HARNESS.md, milestone M5; details in docs/autopilot/M5.md).

An *open* line is a current transcript line whose speaker is still undecided (``cut_readiness.speaker_open``): no character,
or a low-confidence character nobody decided, and not left as it is by the person (``left_unknown``). Open lines are
grouped into **phrases**: consecutive
open lines of the same diarization label with no other line between them and gaps of at most ``PHRASE_GAP`` (older
episodes store one row per word, so a phrase is what a person hears as one utterance). Each phrase gets **signals**, each
naming one character or nobody:

* ``voice``: its audio scored against the characters' voice samples (``line_voices.score_clip``: mean of the two best
  per-range cosines), decided at ``VOICE_TOP`` with a lead of ``VOICE_GAP``. The job computes the scores in the audio
  worker; a line transcribed with line records already carries them in ``evidence_json`` (used when nothing is computed).
* ``label``: the other lines of the same diarization label that already have a character, when at least ``LABEL_MIN`` of
  them and ``LABEL_SHARE`` of them name one character.
* ``picture``: the stored picture analysis of the cut it is spoken in (keyframe segments over the phrase's seconds, else
  the cut's video analysis) shows exactly one character speaking.
* ``name_tag``: a burned-in subtitle over it carries a ``(名前)`` / ``[名前]`` tag naming exactly one character.

A character the phrase *addresses by name* (a given or full name of the glossary in its text) is excluded: a line rarely
calls its own speaker by name.

Signals come in **families** (``FAMILY``): ``sound`` (voice, label: diarization clusters the same embeddings, so the two
are not independent), ``picture`` and ``text``. A unit is **applied** when signals of at least two families name the same
character and no signal names another (HARNESS section 5: "at least two independent signals agree"). The order:

1. **per label** (a diarization label is one voice): the label's own signals (its joined audio, its links, the majority of
   its phrases' pictures and tags). Applied → every open line of the label gets the character.
2. **per phrase** for the rest, with the phrase's own signals.
3. **label consensus**: when at least ``CONSENSUS_MIN`` phrases of a label were applied and all to one character, and no
   label-level signal names another, the label's remaining phrases follow (tier ``rules``).

**Unregistered cards** (``unregistered.py``) are never named by a signal. Instead a phrase whose cut has exactly one card
boxed goes to that card (tier ``rules``) when speaker attribution answered ``other`` for it or the picture shows that card
speaking, and no voice sample resembles it; never label-wide (one label may be several minor people).

What is left becomes **one escalation per label** (unlabelled lines: one per cut), for Claude when the run's Claude budget
lasts and a picture can settle it (whose mouth moves; ``voice_sheet`` tiles three frames per phrase), else for the person:
always the person when no voice sample resembles the label (``NEW_VOICE``: it may be a new character). Questions are raised
**once**: only the run that Prepare queues asks (``payload.ask``, HARNESS.md D37); any later run (after a re-transcription,
``fw voices run``) applies what two signals agree on and only counts the rest (``left_unasked``). Answers come back through
``answer``; any answer of the person settles the lines for good, "Leave unknown" included (``leave``: they keep their
speaker and say ``left_unknown``), so an unknown voice is asked once (HARNESS.md D29). Lines inside a sequence's range are
left alone (the user's confirmed dialogue would turn stale).

Every applied unit is one ``agent_decisions`` row of kind ``voices`` (target ``transcript_segments``); the revisions keep
the source line's ``origin`` and say ``speaker_source='agent'`` (speaker attribution treats them as decided). An agent's or
Claude's speaker marks nothing stale (automatic audio work never does); the person's answer and an undo do. Undo puts the
lines back without a character while they still hold what the agent wrote.
"""
import json
import re
import uuid
from collections import Counter, defaultdict

from . import agent_log, cut_readiness, db, job_inputs, line_voices, structured_results, unregistered, worker

KIND = 'voices'                 # agent decision kind and Autopilot profile decision
PIPELINE = 'voice_resolver'     # job kind (audio worker)
ESCALATION = 'voice'
VERSION = 2
PHRASE_GAP = 0.5                # seconds between two open lines of one phrase
MAX_PHRASE = 12.0               # a phrase never grows past this
VOICE_TOP = 0.35                # voice score needed to name a character (line_voices.decide's default threshold)
VOICE_GAP = 0.05                # its lead over the runner-up
VOICE_TOP_PADDED = 0.5          # a phrase under MIN_CLIP is scored with the silence around it: noisier, so a higher bar
NEW_VOICE = 0.25                # below this no voice sample resembles the label: it may be a new character
LABEL_MIN = 5                   # linked lines of a label needed before the label names a character
LABEL_SHARE = 0.8
VOTE_MIN = 2                    # phrase votes a label needs to name a character by picture or tag
VOTE_SHARE = 0.6
ADDRESS_SHARE = 0.2             # a label whose phrases address X this often is not X
CONSENSUS_MIN = 3
JOIN_SECONDS = 30.0             # audio joined to score a whole label
MIN_CLIP = line_voices.MIN_CLIP
PAD_REACH = line_voices.PAD_REACH
MAX_SAMPLES = 8                 # phrases listed in one escalation
SHEET_ROWS = 6
FAMILY = {'voice': 'sound', 'label': 'sound', 'picture': 'picture', 'name_tag': 'text'}
OPEN = cut_readiness.open_speaker_sql()
TAG = re.compile(r'[（(［\[]\s*([^（()）\[\]［］]{1,12}?)\s*[）)\]］]')
COLUMNS = ('id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,'
           'words_json,speaker_source,confidence,evidence_json')


class Refused(Exception):
    """An answer that cannot be applied (wrong reviewer, closed item, unknown character)."""


# --- gathering ---------------------------------------------------------------------------------------------------------

def _load(text, default=None):
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def speaking(title, content):
    """Who the stored picture analysis shows speaking: ``{'segments': [[start, end, [ids]]]}`` (clip seconds, keyframes)
    or ``{'cut': [ids]}`` (a video analysis direction); None when it says nothing."""
    data = _load(content, {}) if isinstance(content, str) else (content or {})
    if not isinstance(data, dict):
        return None
    if title == 'Keyframe analysis':
        segments = []
        for segment in data.get('segments') or []:
            if not isinstance(segment, dict) or not isinstance(segment.get('start'), (int, float)) or not isinstance(segment.get('end'), (int, float)):
                continue
            ids = sorted({item.get('character_context_id') for item in segment.get('characters') or []
                          if isinstance(item, dict) and item.get('speaking') == 'yes' and item.get('character_context_id')})
            segments.append([float(segment['start']), float(segment['end']), ids])
        return {'segments': segments} if segments else None
    direction = data.get('direction') if isinstance(data.get('direction'), dict) else data
    characters = direction.get('characters') if isinstance(direction, dict) else None
    if not isinstance(characters, list):
        return None
    return {'cut': sorted({item.get('character_context_id') for item in characters
                           if isinstance(item, dict) and item.get('speaking') == 'yes' and item.get('character_context_id')})}


def sequence_ranges(conn, episode_id):
    return [(row['start_time'], row['end_time']) for row in conn.execute('SELECT start_time,end_time FROM sequences WHERE episode_id=?', (episode_id,))]


def current_lines(conn, episode_id):
    return [dict(row) for row in conn.execute(f"""SELECT {COLUMNS} FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL
        AND deleted=0 ORDER BY start_time,end_time,id""", (episode_id,))]


def gather(conn, job):
    from . import name_glossary, subtitle_store
    episode_id = job['episode_id']
    episode = dict(conn.execute('SELECT id,project_id,has_audio FROM episodes WHERE id=?', (episode_id,)).fetchone())
    lines = current_lines(conn, episode_id)
    cuts = [dict(row) for row in conn.execute('SELECT id,cut_index,start_time,end_time,no_characters FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time',
                                              (episode_id,))]
    pictures = {}
    for row in conn.execute("""SELECT r.cut_id,r.id,r.title,r.content FROM analysis_results r JOIN cuts c ON c.id=r.cut_id
            WHERE c.episode_id=? AND c.active=1 AND r.superseded_by IS NULL AND r.title IN ('Keyframe analysis','Video analysis')
            ORDER BY r.title='Keyframe analysis'""", (episode_id,)):
        seen = speaking(row['title'], row['content'])
        if seen is not None:  # keyframes come last, so they win over an older video analysis
            pictures[row['cut_id']] = {**seen, 'result_id': row['id']}
    characters = [dict(row) for row in conn.execute("""SELECT id,title,unregistered FROM context_items WHERE project_id=? AND kind='character'
        AND (episode_id IS NULL OR episode_id=?) ORDER BY title""", (episode['project_id'], episode_id))]
    references = [row[0] for row in conn.execute("""SELECT v.id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
        WHERE c.project_id=? ORDER BY v.id""", (episode['project_id'],))]
    dismissed = sorted({row[0] for row in conn.execute("SELECT group_key FROM agent_escalations WHERE episode_id=? AND kind=? AND status='dismissed'",
                                                       (episode_id, ESCALATION))})
    # Lines the person put back by undoing an agent's voice: never given to anyone by a rule again, only asked (HARNESS.md D32)
    undone_rows = {item.get('id') for row in conn.execute("SELECT after_json FROM agent_decisions WHERE episode_id=? AND kind=? AND undone_at IS NOT NULL",
                                                          (episode_id, KIND)) for item in (_load(row[0], {}) or {}).get('rows') or []}
    undone = sorted(line['id'] for line in lines if line.get('parent_id') in undone_rows)
    return {'episode': episode, 'lines': lines, 'cuts': cuts, 'pictures': pictures, 'characters': characters,
            'glossary': [{key: row[key] for key in ('spelling', 'character_context_id', 'part')} for row in name_glossary.rows(conn, episode['project_id'], episode_id)],
            'subtitles': [{key: line[key] for key in ('start_time', 'end_time', 'text')} for line in subtitle_store.lines(episode_id, conn=conn)],
            'references': references, 'sequences': sequence_ranges(conn, episode_id), 'dismissed': dismissed, 'undone_lines': undone,
            'cards_by_cut': cards_by_cut(conn, cuts)}


def cards_by_cut(conn, cuts):
    """Cut id -> the Unregistered cards boxed in it (first-frame boxes of the cut's latest run, Target-frames colours)."""
    found = {}
    for cut in cuts:
        ids = set()
        run = conn.execute('SELECT id FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['id'],)).fetchone()
        if run:
            ids |= {row[0] for row in conn.execute("""SELECT d.character_context_id FROM character_detections d JOIN context_items c
                ON c.id=d.character_context_id WHERE d.run_id=? AND d.position='first' AND c.unregistered=1""", (run['id'],))}
        target = conn.execute('SELECT id FROM target_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['id'],)).fetchone()
        if target:
            ids |= {row[0] for row in conn.execute("""SELECT s.character_context_id FROM target_slots s JOIN context_items c
                ON c.id=s.character_context_id WHERE s.run_id=? AND s.ignored=0 AND c.unregistered=1""", (target['id'],))}
        if ids:
            found[cut['id']] = sorted(ids)
    return found


def inputs(conn, job):
    data = job.get('gathered') or gather(conn, job)
    # The fingerprint is what a decision rests on: the lines (ids and characters), pictures, tags, samples, sequences.
    return job_inputs.fingerprint({**data, 'lines': [[row['id'], row['character_context_id'], row['speaker_source'], row['confidence']] for row in data['lines']],
                                   'pictures': sorted(item['result_id'] for item in data['pictures'].values())})


def invalidate(conn, job):
    """Each applied unit marks the cuts its lines touch stale itself (``assign``)."""


# --- phrases and signals -----------------------------------------------------------------------------------------------

def phrases(lines, sequences=()):
    """Open lines grouped into phrases, in time order. Each: ``{key, label, line_ids, rows, start, end, text, pad,
    in_sequence}``; ``pad`` is the span grown into the silence around it (never into another line)."""
    result, current = [], None

    def close():
        if current:
            first, last = current['first'], current['last']
            left = max(current['start'] - PAD_REACH, lines[first - 1]['end_time'] if first > 0 else 0.0)
            right = min(current['end'] + PAD_REACH, lines[last + 1]['start_time'] if last + 1 < len(lines) else current['end'] + PAD_REACH)
            current['pad'] = [round(min(left, current['start']), 3), round(max(right, current['end']), 3)]
            current['text'] = ''.join(row['text'] for row in current['rows']).strip()
            current['in_sequence'] = any(current['start'] < end and current['end'] > start for start, end in sequences)
            current['key'] = current['line_ids'][0]
            del current['first'], current['last']
            result.append(current)

    for index, row in enumerate(lines):
        if not cut_readiness.speaker_open(row):
            close()
            current = None
            continue
        label = row['speaker_label'] or ''
        if (current and current['label'] == label and row['start_time'] - current['end'] <= PHRASE_GAP
                and row['end_time'] - current['start'] <= MAX_PHRASE and index == current['last'] + 1):
            current['rows'].append(row)
            current['line_ids'].append(row['id'])
            current['end'] = max(current['end'], row['end_time'])
            current['last'] = index
            continue
        close()
        current = {'label': label, 'rows': [row], 'line_ids': [row['id']], 'start': row['start_time'], 'end': row['end_time'],
                   'first': index, 'last': index}
    close()
    return result


def clip_span(phrase):
    """The audio a phrase is scored on: itself, or its padded span when it is too short."""
    if phrase['end'] - phrase['start'] >= MIN_CLIP:
        return [phrase['start'], phrase['end']]
    return phrase['pad'] if phrase['pad'][1] - phrase['pad'][0] >= MIN_CLIP else None


def stored_scores(phrase):
    """Voice scores the transcription saved on a phrase's longest line (line records only), as ``{character: score}``."""
    best = None
    for row in phrase['rows']:
        evidence = (_load(row['evidence_json'], {}) or {}).get('evidence')
        if evidence and (best is None or row['end_time'] - row['start_time'] > best[0]):
            best = (row['end_time'] - row['start_time'], evidence)
    if not best:
        return {}
    return {item[0]: float(item[1]) for item in best[1] if isinstance(item, (list, tuple)) and len(item) == 2}


def voice_pick(scores):
    """``(character or None, top, gap)`` at this resolver's thresholds."""
    return line_voices.decide(scores or {}, VOICE_TOP, VOICE_GAP)


def cut_at(cuts, moment):
    for cut in cuts:
        if cut['start_time'] <= moment < cut['end_time']:
            return cut
    return None


def picture_pick(phrase, cuts, pictures, cards=()):
    """The one character the picture shows speaking during the phrase, or None. Unregistered ``cards`` never count as a
    signal (``card_pick`` handles them): they are left out before the count."""
    ids = picture_ids(phrase, cuts, pictures) - set(cards)
    return next(iter(ids)) if len(ids) == 1 else None


def picture_ids(phrase, cuts, pictures):
    """Every character the picture shows speaking during the phrase."""
    cut = cut_at(cuts, (phrase['start'] + phrase['end']) / 2)
    if not cut or cut['no_characters']:
        return set()
    seen = pictures.get(cut['id'])
    if not seen:
        return set()
    if 'segments' in seen:
        start, end = phrase['start'] - cut['start_time'], phrase['end'] - cut['start_time']
        ids = set()
        for segment_start, segment_end, speakers in seen['segments']:
            if segment_start < end and segment_end > start:
                ids.update(speakers)
    else:
        ids = set(seen.get('cut') or [])
    return ids


def card_pick(phrase, data, cards):
    """The Unregistered card a phrase goes to by rule, or None: its cut has exactly one card boxed, speaker attribution
    answered ``other`` for one of its lines or the picture shows that card speaking, and no voice sample resembles it."""
    cut = cut_at(data['cuts'], (phrase['start'] + phrase['end']) / 2)
    boxed = (data.get('cards_by_cut') or {}).get(cut['id']) if cut else None
    if not boxed or len(boxed) != 1 or phrase['detail']['top'] >= VOICE_TOP:
        return None
    card = boxed[0]
    other = any(((_load(row['evidence_json'], {}) or {}).get('qwen') or {}).get('speaker') == 'other' for row in phrase['rows'])
    seen = picture_ids(phrase, data['cuts'], data['pictures'])
    if other or seen == {card}:
        return card, ('speaker attribution said "other"' if other else 'the picture shows them speaking')
    return None


def name_index(glossary, characters):
    """Spelling -> character id, for given and full names (a family name is shared by relatives: never used)."""
    index = defaultdict(set)
    for row in glossary:
        if row.get('character_context_id') and (row.get('part') or '') != 'family' and len(row.get('spelling') or '') >= 2:
            index[row['spelling']].add(row['character_context_id'])
    for character in characters:
        if not character.get('unregistered'):
            index[character['title'].casefold()].add(character['id'])
    return {spelling: next(iter(ids)) for spelling, ids in index.items() if len(ids) == 1}


def addressed(text, names):
    found = set()
    folded = (text or '').casefold()
    for spelling, character in names.items():
        if spelling and spelling in folded:
            found.add(character)
    return found


def tag_pick(phrase, subtitles, names):
    from .audio_lines import subtitle_overlap
    found = set()
    for cue in subtitle_overlap({'start_time': phrase['start'], 'end_time': phrase['end']}, subtitles):
        for tag in TAG.findall(cue['text'] or ''):
            character = names.get(tag.strip().casefold()) or names.get(tag.strip())
            if character:
                found.add(character)
    return next(iter(found)) if len(found) == 1 else None


def label_links(lines):
    """Diarization label -> ``(character, share, count)`` of its lines that already have a character."""
    by_label = defaultdict(Counter)
    for row in lines:
        # Lines still open (a low-confidence guess) are not evidence of who the label is.
        if row['character_context_id'] and row['speaker_label'] and not cut_readiness.speaker_open(row):
            by_label[row['speaker_label']][row['character_context_id']] += 1
    result = {}
    for label, counter in by_label.items():
        character, count = counter.most_common(1)[0]
        total = sum(counter.values())
        result[label] = (character, count / total, total)
    return result


def link_pick(link):
    if not link:
        return None
    character, share, total = link
    return character if total >= LABEL_MIN and share >= LABEL_SHARE else None


def majority(votes):
    votes = [vote for vote in votes if vote]
    if len(votes) < VOTE_MIN:
        return None
    character, count = Counter(votes).most_common(1)[0]
    return character if count / len(votes) >= VOTE_SHARE else None


def agree(signals, excluded=()):
    """Do signals of two families name one character, with none naming another?

    -> ``{action: 'apply'|'ask', character, families, why, candidates}``."""
    named = {name: value for name, value in signals.items() if value}
    candidates = sorted(set(named.values()))
    if not named:
        return {'action': 'ask', 'character': None, 'families': [], 'candidates': [], 'why': 'no signal names anyone'}
    if len(candidates) > 1:
        return {'action': 'ask', 'character': None, 'families': [], 'candidates': candidates,
                'why': 'the signals disagree: ' + ', '.join(f'{name} → {value}' for name, value in sorted(named.items()))}
    character = candidates[0]
    families = sorted({FAMILY[name] for name in named})
    if character in excluded:
        return {'action': 'ask', 'character': character, 'families': families, 'candidates': candidates,
                'why': f'the line addresses {character} by name'}
    if len(families) < 2:
        return {'action': 'ask', 'character': character, 'families': families, 'candidates': candidates,
                'why': f"only one kind of signal ({', '.join(sorted(named))})"}
    return {'action': 'apply', 'character': character, 'families': families, 'candidates': candidates,
            'why': ' + '.join(sorted(named))}


def phrase_signals(phrase, data, names, voice_scores, cards=()):
    scores = voice_scores.get(phrase['key']) if voice_scores else None
    padded = scores is not None and phrase['end'] - phrase['start'] < MIN_CLIP
    if scores is None:
        scores = stored_scores(phrase)
    voice, top, gap = voice_pick(scores)
    if padded and top < VOICE_TOP_PADDED:
        voice = None
    if voice in cards:
        voice = None
    return {'voice': voice, 'picture': picture_pick(phrase, data['cuts'], data['pictures'], cards),
            'name_tag': tag_pick(phrase, data['subtitles'], names)}, {
        'voice_top': [[character, round(score, 3)] for character, score in line_voices.ranked(scores or {})[:3]],
        'top': round(top, 3), 'gap': round(gap, 3), 'addressed': sorted(addressed(phrase['text'], names))}


def resolve(data, voice_scores=None, label_scores=None):
    """The pure core: what to apply and what to ask, for every open phrase of the episode.

    ``voice_scores``: phrase key -> ``{character: score}``; ``label_scores``: label -> scores of its joined audio. Returns
    ``{applied: [unit], asked: [group], left: {reason: count}}``; a unit is ``{scope: label|phrase|consensus, label,
    character, line_ids, phrases, signals, why, tier}``, a group ``{group_key, label, phrases, candidates, why, new_voice}``."""
    names = name_index(data['glossary'], data['characters'])
    cards = {item['id'] for item in data['characters'] if item.get('unregistered')}
    links = {label: link for label, link in label_links(data['lines']).items() if link[0] not in cards}
    dismissed = set(data.get('dismissed') or ())
    all_phrases = phrases(data['lines'], data.get('sequences') or ())
    left = Counter()
    by_label = defaultdict(list)
    for phrase in all_phrases:
        if phrase['in_sequence']:
            left['in a sequence'] += len(phrase['line_ids'])
            continue
        phrase['signals'], phrase['detail'] = phrase_signals(phrase, data, names, voice_scores, cards)
        by_label[phrase['label']].append(phrase)

    applied, asked = [], []
    for label, items in by_label.items():
        label_detail = {}
        remaining = items
        if label:
            joined = (label_scores or {}).get(label)
            voice, top, gap = voice_pick(joined) if joined else (majority([item['signals']['voice'] for item in items]), None, None)
            if voice in cards:
                voice = None
            label_detail = {'voice_top': [[c, round(s, 3)] for c, s in line_voices.ranked(joined or {})[:3]], 'top': top and round(top, 3),
                            'link': list(links[label]) if label in links else None}
            signals = {'voice': voice, 'label': link_pick(links.get(label)),
                       'picture': majority([item['signals']['picture'] for item in items]),
                       'name_tag': majority([item['signals']['name_tag'] for item in items])}
            spoken_to = Counter(character for item in items for character in item['detail']['addressed'])
            excluded = {character for character, count in spoken_to.items() if count >= 2 or count / len(items) >= ADDRESS_SHARE}
            verdict = agree(signals, excluded)
            label_detail.update(signals=signals, verdict=verdict)
            if verdict['action'] == 'apply':
                applied.append({'scope': 'label', 'label': label, 'character': verdict['character'], 'tier': 'local',
                                'line_ids': [line_id for item in items for line_id in item['line_ids']], 'phrases': items,
                                'signals': signals, 'why': verdict['why'], 'detail': label_detail})
                continue
        done = defaultdict(list)
        remaining = []
        for item in items:
            verdict = agree(item['signals'], set(item['detail']['addressed']))
            item['verdict'] = verdict
            if verdict['action'] == 'apply':
                done[verdict['character']].append(item)
                applied.append({'scope': 'phrase', 'label': label, 'character': verdict['character'], 'tier': 'local',
                                'line_ids': list(item['line_ids']), 'phrases': [item], 'signals': item['signals'], 'why': verdict['why']})
                continue
            picked = None if verdict['candidates'] else card_pick(item, data, cards)
            if picked:
                card, why = picked
                applied.append({'scope': 'phrase', 'label': label, 'character': card, 'tier': 'rules', 'line_ids': list(item['line_ids']),
                                'phrases': [item], 'signals': {'unregistered': card}, 'why': f'the only Unregistered person boxed in the cut; {why}'})
            else:
                remaining.append(item)
        if label and remaining and len(done) == 1:
            character, settled = next(iter(done.items()))
            label_named = {value for value in (label_detail.get('signals') or {}).values() if value}
            if len(settled) >= CONSENSUS_MIN and label_named <= {character}:
                applied.append({'scope': 'consensus', 'label': label, 'character': character, 'tier': 'rules',
                                'line_ids': [line_id for item in remaining for line_id in item['line_ids']], 'phrases': remaining,
                                'signals': {'phrases_agreeing': len(settled)},
                                'why': f'{len(settled)} phrases of this voice were settled, all as this character'})
                remaining = []
        if not remaining:
            continue
        groups = {f'voice:{label}': remaining} if label else defaultdict(list)
        if not label:
            # Lines the diarizer left unlabelled are mostly fragments between two voices. One without any signal is left
            # alone; the rest are grouped by their candidates, so one answer settles every line that points the same way.
            for item in remaining:
                if not item['verdict']['candidates']:
                    left['no voice label and no signal'] += len(item['line_ids'])
                    continue
                groups[f"voice::candidates:{','.join(item['verdict']['candidates'])}"].append(item)
        for group_key, members in groups.items():
            if group_key in dismissed:
                left['left unknown by the person'] += sum(len(item['line_ids']) for item in members)
                continue
            candidates = sorted({c for item in members for c in item['verdict']['candidates']} |
                                {value for value in (label_detail.get('signals') or {}).values() if value} |
                                {row['character_context_id'] for item in members for row in item['rows']
                                 if row['character_context_id'] and row['character_context_id'] not in cards})
            # No voice sample resembles it: maybe a character nobody created yet. Only the person can say (a picture of someone
            # else talking in those cuts does not make it theirs). Unlabelled fragments are too short to judge by voice alone.
            if label and label_detail.get('top') is not None:
                new_voice = label_detail['top'] < NEW_VOICE
            else:
                tops = [item['detail']['top'] for item in members if item['detail']['voice_top']]
                new_voice = bool(tops) and max(tops) < NEW_VOICE and not any(item['signals']['picture'] for item in members)
            why = (label_detail.get('verdict') or members[0]['verdict'])['why']
            boxed = sorted({card for item in members for card in (data.get('cards_by_cut') or {}).get(
                (cut_at(data['cuts'], (item['start'] + item['end']) / 2) or {}).get('id'), [])})
            unsure = any(row['character_context_id'] for item in members for row in item['rows'])
            asked.append({'group_key': group_key, 'label': label, 'label_wide': bool(label), 'phrases': members, 'candidates': candidates,
                          'why': why, 'new_voice': new_voice, 'detail': label_detail, 'cards': boxed, 'unsure': unsure})
    return {'applied': applied, 'asked': asked, 'left': dict(left)}


# --- writing ------------------------------------------------------------------------------------------------------------

def assign(conn, episode_id, line_ids, character, *, speaker_source='agent', manual=False, note=None):
    """A revision of each line, still current and open, with ``character``. Returns ``[(old_row, new_id)]``."""
    if not line_ids:
        return []
    marks = ','.join('?' for _ in line_ids)
    rows = [dict(row) for row in conn.execute(f"""SELECT {COLUMNS} FROM transcript_segments WHERE id IN ({marks}) AND episode_id=?
        AND superseded_by IS NULL AND deleted=0 AND {OPEN} ORDER BY start_time""", [*line_ids, episode_id])]
    made = []
    for row in rows:
        evidence = _load(row['evidence_json'], {}) or {}
        if note:
            evidence['agent'] = note
        new_id = str(uuid.uuid4())
        conn.execute(f'INSERT INTO transcript_segments ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                     (new_id, episode_id, row['start_time'], row['end_time'], row['text'], row['speaker_label'], character,
                      'manual_edit' if manual else row['origin'], row['model_id'], row['job_id'], row['id'], row['words_json'],
                      'manual' if manual else speaker_source, None if manual else 'high', json.dumps(evidence, ensure_ascii=False)))
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, row['id']))
        made.append((row, new_id))
    changed = [row for row, _ in made if row['character_context_id'] != character]
    if changed and manual:  # only the person's answer stales the cuts' text results (HARNESS.md D37)
        structured_results.invalidate_transcript(conn, episode_id, min(row['start_time'] for row in changed), max(row['end_time'] for row in changed),
                                                 reason='Speaker changed')
    return made


def leave(conn, episode_id, line_ids, note=None):
    """The person's "Leave unknown": each line still open keeps its speaker (none, or the unsure guess) and becomes
    ``left_unknown``, so no surface flags it again and speaker attribution never asks about it. Returns the count."""
    if not line_ids:
        return 0
    marks = ','.join('?' for _ in line_ids)
    rows = [dict(row) for row in conn.execute(f"""SELECT {COLUMNS} FROM transcript_segments WHERE id IN ({marks}) AND episode_id=?
        AND superseded_by IS NULL AND deleted=0 AND {OPEN}""", [*line_ids, episode_id])]
    for row in rows:
        evidence = _load(row['evidence_json'], {}) or {}
        evidence['left_unknown'] = {'note': str(note)[:1000]} if note else True
        new_id = str(uuid.uuid4())
        conn.execute(f'INSERT INTO transcript_segments ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                     (new_id, episode_id, row['start_time'], row['end_time'], row['text'], row['speaker_label'], row['character_context_id'],
                      'manual_edit', row['model_id'], row['job_id'], row['id'], row['words_json'], cut_readiness.LEFT_UNKNOWN, None,
                      json.dumps(evidence, ensure_ascii=False)))
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, row['id']))
    return len(rows)


def settle_dismissed(conn, episode_id):
    """Voices the person left unknown before ``leave`` wrote the lines: settle their still-open lines now, once."""
    settled = 0
    for row in conn.execute("SELECT * FROM agent_escalations WHERE episode_id=? AND kind=? AND status='dismissed'", (episode_id, ESCALATION)).fetchall():
        settled += leave(conn, episode_id, group_lines(conn, agent_log.escalation_view(row)))
    return settled


def _cut_of(conn, episode_id, moment):
    return conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND start_time<=? AND end_time>? LIMIT 1',
                        (episode_id, moment, moment)).fetchone()


def titles(conn, episode_id):
    return {row['id']: row['title'] for row in conn.execute("""SELECT c.id,c.title FROM context_items c JOIN episodes e ON e.project_id=c.project_id
        WHERE e.id=? AND c.kind='character'""", (episode_id,))}


def apply_unit(conn, episode_id, unit, *, run_id=None, tier=None, manual=False, escalation_id=None, names=None):
    """Write one unit and log it. Returns the decision id, or None when no line was still open."""
    tier = tier or unit.get('tier') or 'local'
    signals = {key: value for key, value in (unit.get('signals') or {}).items() if value}
    note = {'run_id': run_id, 'tier': tier, 'scope': unit.get('scope'), 'why': unit.get('why'), 'signals': signals}
    made = assign(conn, episode_id, unit['line_ids'], unit['character'], manual=manual, note=note)
    if not made:
        return None
    names = names or titles(conn, episode_id)
    who = names.get(unit['character'], unit['character'])
    first = made[0][0]
    cut = _cut_of(conn, episode_id, (first['start_time'] + first['end_time']) / 2)
    what = f"voice {unit['label']}" if unit.get('scope') in ('label', 'consensus') and unit.get('label') else f'"{"".join(row["text"] for row, _ in made)[:30]}"'
    summary = f"Gave {what} ({len(made)} line{'s' if len(made) != 1 else ''}, {first['start_time']:.1f}s) to {who}: {named(unit.get('why') or tier, names)}"
    evidence = {'scope': unit.get('scope'), 'label': unit.get('label'), 'why': unit.get('why'), 'signals': signals,
                **({'escalation_id': escalation_id} if escalation_id else {}), **({'detail': unit['detail']} if unit.get('detail') else {})}
    return agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=cut['id'] if cut else None, kind=KIND, tier=tier,
                            target_table='transcript_segments', target_id=first['id'],
                            before={'rows': [{'id': row['id'], 'character_context_id': row['character_context_id']} for row, _ in made]},
                            after={'character_context_id': unit['character'], 'manual': manual,
                                   'rows': [{'id': new_id, 'source_id': row['id']} for row, new_id in made]},
                            evidence=evidence, confidence='high' if len(unit.get('families') or signals) > 2 else 'medium', summary=summary)


def undo(conn, decision):
    """Put the lines back as they were (no character, or the unsure guess) while every one still holds what the agent wrote."""
    after = _load(decision['after_json'], {}) or {}
    rows = after.get('rows') or []
    if not rows:
        raise agent_log.UndoConflict('This decision changed no line')
    marks = ','.join('?' for _ in rows)
    current = {row['id']: dict(row) for row in conn.execute(f'SELECT {COLUMNS},superseded_by,deleted FROM transcript_segments WHERE id IN ({marks})',
                                                             [row['id'] for row in rows])}
    for item in rows:
        row = current.get(item['id'])
        if row is None or row['superseded_by'] or row['deleted'] or row['character_context_id'] != after.get('character_context_id'):
            raise agent_log.UndoConflict('A line changed since; undo would overwrite that')
    episode_id = decision['episode_id']
    for item in rows:
        row = current[item['id']]
        source = conn.execute(f'SELECT {COLUMNS} FROM transcript_segments WHERE id=?', (item['source_id'],)).fetchone()
        restored = str(uuid.uuid4())
        conn.execute(f'INSERT INTO transcript_segments ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                     (restored, episode_id, row['start_time'], row['end_time'], row['text'], row['speaker_label'], source['character_context_id'] if source else None,
                      source['origin'] if source else row['origin'], row['model_id'], row['job_id'], row['id'], row['words_json'],
                      source['speaker_source'] if source else None, source['confidence'] if source else None,
                      source['evidence_json'] if source else row['evidence_json']))
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (restored, row['id']))
    starts = [current[item['id']]['start_time'] for item in rows]
    ends = [current[item['id']]['end_time'] for item in rows]
    structured_results.invalidate_transcript(conn, episode_id, min(starts), max(ends), reason='Speaker changed')


agent_log.UNDO_KINDS[KIND] = undo


# --- escalations --------------------------------------------------------------------------------------------------------

def policy_for(conn, episode_id, run_id=None):
    from . import autopilot_profile
    profile = autopilot_profile.profile_for(conn, episode_id, run_id, KIND)
    claude = profile.get('claude') or {}
    return {'mode': profile['modes'].get(KIND, 'apply'), 'claude': bool(claude.get('enabled', True)),
            'claude_items': int(claude.get('max_items_per_run', 0) or 0)}


def claude_budget(conn, episode_id, run_id, policy):
    """How many more voices this run (or, without a run, this episode) may hand to Claude."""
    if not policy.get('claude'):
        return 0
    if run_id:
        used = conn.execute("SELECT COUNT(*) FROM agent_escalations WHERE kind=? AND run_id=? AND instr(evidence_json,'\"asked_claude\": true')>0",
                            (ESCALATION, run_id)).fetchone()[0]
    else:
        used = conn.execute("SELECT COUNT(*) FROM agent_escalations WHERE kind=? AND episode_id=? AND status='open' AND reviewer='claude'",
                            (ESCALATION, episode_id)).fetchone()[0]
    return int(policy.get('claude_items', 0)) - used


def phrase_view(item, cuts, names):
    cut = cut_at(cuts, (item['start'] + item['end']) / 2)
    return {'start': round(item['start'], 3), 'end': round(item['end'], 3), 'text': item['text'][:120],
            'cut': cut['cut_index'] + 1 if cut else None, 'cut_id': cut['id'] if cut else None, 'lines': len(item['line_ids']),
            'signals': {key: names.get(value, value) for key, value in (item.get('signals') or {}).items() if value},
            'voice_top': [[names.get(c, c), s] for c, s in (item.get('detail') or {}).get('voice_top') or []],
            'addressed': [names.get(c, c) for c in (item.get('detail') or {}).get('addressed') or []]}


def samples(members):
    """The phrases a reviewer looks at: the longest ones, in time order."""
    chosen = sorted(members, key=lambda item: item['end'] - item['start'], reverse=True)[:MAX_SAMPLES]
    return sorted(chosen, key=lambda item: item['start'])


def named(text, names):
    """A reason written with character ids, with their names instead."""
    for character, title in names.items():
        text = text.replace(character, title)
    return text


def escalate_group(conn, episode_id, group, data, names, *, reviewer, run_id=None, proposal=None, why=None):
    members = group['phrases']
    line_ids = [line_id for item in members for line_id in item['line_ids']]
    shown = [phrase_view(item, data['cuts'], names) for item in samples(members)]
    cuts = sorted({item['cut'] for item in (phrase_view(m, data['cuts'], names) for m in members) if item['cut']})
    voice = group['label'] or 'unlabelled lines'
    who = ', '.join(names.get(c, c) for c in group['candidates']) or 'nobody'
    cards = sorted(group.get('cards') or [], key=lambda card: (unregistered.number(names.get(card, '')), card))
    if cards:
        who += f"; Unregistered boxed there: {', '.join(names.get(c, c) for c in cards)}"
    why = named(why or group['why'], names)
    question = (f"Who speaks {voice} ({len(line_ids)} line{'s' if len(line_ids) != 1 else ''} in cut{'s' if len(cuts) != 1 else ''} "
                f"{', '.join(str(c) for c in cuts[:12])}{'…' if len(cuts) > 12 else ''})? {why[:1].upper() + why[1:]}. Candidates: {who}."
                + (' No voice sample resembles it: it may be a new character.' if group['new_voice'] else ''))
    # "Leave unknown" keeps an unsure guess as it is: one answer, never asked again.
    options = ([names.get(c, c) for c in group['candidates']] + [names.get(c, c) for c in cards]
               + ['Unregistered', 'New character', 'Keep as it is' if group.get('unsure') else 'Leave unknown'])
    detail = group.get('detail') or {}
    # ``label`` makes an answer cover every open line of the voice; a question about single phrases names only its lines.
    evidence = {'label': group['label'] if group.get('label_wide') else None, 'voice_label': group['label'], 'line_ids': line_ids, 'phrases': shown, 'why': why, 'new_voice': group['new_voice'],
                'candidates': [{'id': c, 'name': names.get(c, c)} for c in group['candidates']],
                'cards': [{'id': c, 'name': names.get(c, c)} for c in cards], 'unsure': bool(group.get('unsure')),
                'label_voice_top': [[names.get(c, c), s] for c, s in detail.get('voice_top') or []],
                'label_link': ([names.get(detail['link'][0], detail['link'][0]), round(detail['link'][1], 2), detail['link'][2]] if detail.get('link') else None),
                'asked_claude': reviewer == 'claude'}
    return agent_log.escalate(conn, episode_id=episode_id, kind=ESCALATION, group_key=group['group_key'], question=question,
                              cut_ids=[item['cut_id'] for item in shown if item['cut_id']], options=options, proposal=proposal,
                              evidence=evidence, run_id=run_id, reviewer=reviewer)


def has_picture(group, data):
    """Could looking at the frames settle it? At least one phrase is spoken in a cut that shows people."""
    for item in group['phrases']:
        cut = cut_at(data['cuts'], (item['start'] + item['end']) / 2)
        if cut and not cut['no_characters']:
            return True
    return False


def open_lines(conn, episode_id, ids):
    if not ids:
        return []
    marks = ','.join('?' for _ in ids)
    return [row[0] for row in conn.execute(f"""SELECT id FROM transcript_segments WHERE id IN ({marks}) AND superseded_by IS NULL AND deleted=0
        AND {OPEN}""", ids)]


def group_lines(conn, escalation):
    """The open lines an escalation is about now: every open line of its label (a label is one voice), else the listed ones."""
    evidence = escalation['evidence']
    episode_id = escalation['episode_id']
    label = evidence.get('label')
    if label:
        sequences = sequence_ranges(conn, episode_id)
        return [row['id'] for row in conn.execute(f"""SELECT id,start_time,end_time FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL
            AND deleted=0 AND {OPEN} AND speaker_label=? ORDER BY start_time""", (episode_id, label))
                if not any(row['start_time'] < end and row['end_time'] > start for start, end in sequences)]
    return open_lines(conn, episode_id, evidence.get('line_ids') or [])


def close_settled(conn, episode_id):
    """Close the voice questions whose lines are all settled now (the person used the app, or a later run settled them)."""
    closed = 0
    for item in agent_log.escalations(conn, episode_id, 'open', ESCALATION):
        if not group_lines(conn, item):
            closed += agent_log.close_escalation(conn, item['id'], 'resolved', {'by': 'settled elsewhere'})
    return closed


def answer(conn, escalation_id, choice, character_context_id=None, tier='human', run_id=None, note=None):
    """Claude's or the person's answer to one voice question.

    ``choice``: ``character`` (with ``character_context_id``, an Unregistered card included), ``unregistered`` (an
    Unregistered card: the given one, else a new number), ``new`` (a character that does not exist yet: the person creates
    it, the lines stay open), ``leave`` (the lines keep their speaker, none or the unsure guess, and are never asked about
    again) or ``unsure`` (Claude: the person answers it). Claude never picks a card: which number is the person's choice.
    Returns ``{status, decision_id, lines}``."""
    if tier not in ('claude', 'human'):
        raise Refused('tier must be claude or human')
    row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
    if row is None:
        raise KeyError(escalation_id)
    if row['kind'] != ESCALATION:
        raise Refused('This is not a voice question')
    if row['status'] != 'open':
        raise Refused('This question is already closed')
    if tier == 'claude' and row['reviewer'] != 'claude':
        raise Refused('This voice waits for a person')
    item = agent_log.escalation_view(row)
    episode_id = row['episode_id']
    run_id = run_id or row['run_id']
    names = titles(conn, episode_id)
    cards = unregistered.card_ids_for(conn, names)
    if tier == 'claude' and choice == 'character' and character_context_id in cards:
        choice = 'unregistered'
    if tier == 'claude' and choice in ('new', 'leave', 'unsure', 'unregistered'):
        # Claude cannot create a character, number an Unregistered person or close a voice for good: the person decides (D16).
        hint = {'new': 'Claude thinks it is a new character', 'leave': 'Claude would leave it unknown', 'unsure': 'Claude was unsure',
                'unregistered': 'Claude thinks it is an unregistered minor person'}[choice]
        agent_log.hand_to_person(conn, escalation_id, f"{hint}{': ' + str(note) if note else ''}")
        return {'status': 'handed_to_person', 'decision_id': None, 'lines': 0}
    if choice == 'new':
        agent_log.close_escalation(conn, escalation_id, 'resolved', {'by': tier, 'new_character': True, **({'note': str(note)[:1000]} if note else {})})
        return {'status': 'new_character', 'decision_id': None, 'lines': 0}
    if choice in ('leave', 'unsure'):
        if choice == 'unsure':
            raise Refused('A person answers with a character, Unregistered, new or leave')
        settled = leave(conn, episode_id, group_lines(conn, item), note)
        agent_log.close_escalation(conn, escalation_id, 'dismissed', {'by': tier, 'lines': settled, **({'note': str(note)[:1000]} if note else {})})
        return {'status': 'left_unknown', 'decision_id': None, 'lines': settled}
    if choice == 'unregistered':
        if not character_context_id:
            project_id = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()[0]
            card = unregistered.create_next(conn, project_id)
            character_context_id = card['id']
            names[card['id']] = card['title']
        elif character_context_id not in cards:
            raise Refused('Choose an Unregistered card of this project')
        choice = 'character'
    if choice != 'character':
        raise Refused('choice must be character, unregistered, new, leave or unsure')
    if character_context_id not in names:
        raise Refused('Unknown character for this project')
    lines = group_lines(conn, item)
    if not lines:
        agent_log.close_escalation(conn, escalation_id, 'resolved', {'by': 'settled elsewhere'})
        return {'status': 'outdated', 'decision_id': None, 'lines': 0}
    unit = {'scope': 'label' if item['evidence'].get('label') else 'phrase', 'label': item['evidence'].get('label'), 'character': character_context_id,
            'line_ids': lines, 'signals': {tier: character_context_id}, 'why': f"answered by {'Claude' if tier == 'claude' else 'the person'}"
                                                                              + (f': {str(note)[:200]}' if note else '')}
    decision_id = apply_unit(conn, episode_id, unit, run_id=run_id, tier=tier, manual=tier == 'human', escalation_id=escalation_id, names=names)
    agent_log.close_escalation(conn, escalation_id, 'resolved', {'by': tier, 'character_context_id': character_context_id, 'decision_id': decision_id,
                                                                 **({'note': str(note)[:1000]} if note else {})})
    return {'status': 'resolved', 'decision_id': decision_id, 'lines': len(lines)}


# --- the job ------------------------------------------------------------------------------------------------------------

def queue(conn, episode_id, *, run_id=None, attribution=True, ask=False):
    """Queue the resolver after the transcript producers. None when the profile switched voices off. ``ask``: the run may
    open questions; only Prepare's run does, so an unknown voice is asked once (HARNESS.md D37)."""
    policy = policy_for(conn, episode_id, run_id)
    if policy['mode'] == 'off':
        return None
    payload = {'policy': policy, 'run_id': run_id, 'attribution': bool(attribution), 'version': VERSION, 'ask': bool(ask)}
    waits = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts', 'missed_cuts', 'transcribe', 'line_splits', 'speaker_attribution'))
    if waits:
        payload['depends_on'] = waits
    job_id = worker.insert_job(conn, episode_id, PIPELINE, payload, priority_class='batch')
    worker.sync_episode_status(conn, episode_id)
    return job_id


def publish(conn, job, payload, data, voice_scores=None, label_scores=None):
    """Apply, escalate (only when ``payload.ask``) and report inside the job's publish transaction. Returns the outcome counts."""
    episode_id = job['episode_id']
    policy = payload.get('policy') or policy_for(conn, episode_id, payload.get('run_id'))
    run_id = payload.get('run_id')
    ask = bool(payload.get('ask'))
    names = {item['id']: item['title'] for item in data['characters']}
    settle_dismissed(conn, episode_id)
    close_settled(conn, episode_id)
    plan = resolve(data, voice_scores, label_scores)
    counts = Counter()
    undone = set(data.get('undone_lines') or ())
    for unit in plan['applied']:
        if policy['mode'] == 'ask' or undone & set(unit['line_ids']):
            if not ask:
                counts['left_unasked'] += 1
                counts['lines_unasked'] += len(unit['line_ids'])
                continue
            wide = unit['scope'] in ('label', 'consensus') and bool(unit['label'])
            group = {'group_key': f"voice:{unit['label']}" if wide else f"voice::line:{unit['line_ids'][0]}", 'label': unit['label'], 'label_wide': wide,
                     'phrases': unit['phrases'], 'candidates': [unit['character']], 'why': f"the agent would give it to {names.get(unit['character'])} ({unit['why']})",
                     'new_voice': False, 'detail': unit.get('detail')}
            escalate_group(conn, episode_id, group, data, names, reviewer='human', run_id=run_id,
                           proposal={'character_context_id': unit['character'], 'name': names.get(unit['character'])})
            counts['asked_person'] += 1
            continue
        if apply_unit(conn, episode_id, unit, run_id=run_id, names=names):
            counts[f"applied_{unit['scope']}"] += 1
            counts['lines_applied'] += len(unit['line_ids'])
    budget = claude_budget(conn, episode_id, run_id, policy)
    for group in plan['asked']:
        if not ask:
            counts['left_unasked'] += 1
            counts['lines_unasked'] += sum(len(item['line_ids']) for item in group['phrases'])
            continue
        reviewer = 'human'
        if not group['new_voice'] and budget > 0 and has_picture(group, data):
            reviewer = 'claude'
            budget -= 1
        proposal = ({'character_context_id': group['candidates'][0], 'name': names.get(group['candidates'][0])} if len(group['candidates']) == 1 else None)
        escalate_group(conn, episode_id, group, data, names, reviewer=reviewer, run_id=run_id, proposal=proposal)
        counts[f'asked_{reviewer}'] += 1
        counts['lines_asked'] += sum(len(item['line_ids']) for item in group['phrases'])
    if counts['lines_applied'] and payload.get('attribution', True):
        counts['attribution_queued'] = int(bool(requeue_attribution(conn, episode_id)))
    return {**dict(counts), 'left': plan['left']}


def requeue_attribution(conn, episode_id):
    """Speaker attribution again, when open line records remain: the new speakers are context for its conversation windows."""
    from . import speaker_attribution
    remaining = conn.execute("""SELECT 1 FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND character_context_id IS NULL
        AND origin='model' AND words_json IS NOT NULL AND coalesce(speaker_source,'') NOT IN ('manual','left_unknown') LIMIT 1""", (episode_id,)).fetchone()
    if not remaining or worker.pending_jobs(conn, episode_id, ('speaker_attribution',)):
        return None
    return speaker_attribution.insert(conn, episode_id)


class AudioScorer:
    """Voice scores of phrases and whole labels against the project's voice samples (audio worker, Community-1 embeddings)."""

    def __init__(self, conn, data):
        from . import audio_worker, media
        self.aw = audio_worker
        episode = data['episode']
        path = media.episode_dir(episode['project_id'], episode['id']) / 'audio.wav'
        self.samples = audio_worker.read_audio(path) if path.is_file() else None
        self.pipeline = None
        self.refs = []
        if self.samples is None:
            return
        references = db.all_rows("""SELECT v.id,v.character_context_id AS character_id,v.episode_id,v.start_time,v.end_time,v.ranges_json,v.source_filename,
            v.range_embeddings_json FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
            WHERE c.project_id=? AND v.model_id=? ORDER BY v.id""", (episode['project_id'], audio_worker.DIARIZATION), conn=conn)
        if not references:
            return
        self.pipeline = audio_worker.load_diarizer()
        self.refs = audio_worker.voice_references(type('Snapshot', (), {'data': {'episode': episode, 'references': references}})(), self.pipeline)

    @property
    def usable(self):
        return bool(self.refs) and self.samples is not None

    def clip(self, start, end):
        rate = self.aw.RATE
        return self.samples[max(0, int(start * rate)):max(0, int(end * rate))]

    def score(self, spans):
        import numpy as np
        pieces = [self.clip(start, end) for start, end in spans]
        pieces = [piece for piece in pieces if len(piece)]
        if not pieces:
            return {}
        return line_voices.score_clip(self.aw.embed(self.pipeline, np.concatenate(pieces)), self.refs)

    def phrases(self, items, check=None):
        result = {}
        for index, item in enumerate(items):
            if check and index % 25 == 0:
                check()
            span = clip_span(item)
            if span:
                result[item['key']] = self.score([span])
        return result

    def labels(self, items):
        by_label = defaultdict(list)
        for item in items:
            if item['label']:
                by_label[item['label']].append(item)
        result = {}
        for label, members in by_label.items():
            chosen, total = [], 0.0
            for item in sorted(members, key=lambda m: m['end'] - m['start'], reverse=True):
                if total >= JOIN_SECONDS:
                    break
                chosen.append((item['start'], item['end']))
                total += item['end'] - item['start']
            if total >= MIN_CLIP:
                result[label] = self.score(sorted(chosen))
        return result


def process(job_id, owner):
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    episode_id = job['episode_id']
    payload = json.loads(job['payload_json'])
    try:
        with db.connect() as conn:
            conn.execute('BEGIN')
            snapshot = job_inputs.JobInputs(conn, job, owner)
        data = snapshot.data
        todo = [item for item in phrases(data['lines'], data['sequences']) if not item['in_sequence']]
        voice_scores = label_scores = None
        scored = 'none'
        if todo and data['references'] and data['episode']['has_audio'] != 0:
            worker.stage(job_id, owner, episode_id, f'Scoring {len(todo)} phrases against the voice samples')
            with db.connect() as conn:
                scorer = AudioScorer(conn, data)
            if scorer.usable:
                check = lambda: worker.stage(job_id, owner, episode_id, f'Scoring {len(todo)} phrases against the voice samples')  # noqa: E731
                voice_scores = scorer.phrases(todo, check)
                label_scores = scorer.labels(todo)
                scored = 'computed'
        worker.stage(job_id, owner, episode_id, 'Giving voices to characters')
        with snapshot.publish(invalidate=False) as conn:
            counts = publish(conn, job, payload, data, voice_scores, label_scores)
            counts['voice_scores'] = scored
            payload['result'] = counts
            stage = (f"Complete · {counts.get('lines_applied', 0)} lines given a character, "
                     f"{counts.get('asked_claude', 0)} voices for Claude, {counts.get('asked_human', 0) + counts.get('asked_person', 0)} for you"
                     + (f", {counts['left_unasked']} left unasked" if counts.get('left_unasked') else ''))
            conn.execute("""UPDATE jobs SET status='done',stage=?,error=NULL,payload_json=?,lease_owner=NULL,lease_until=NULL,
                updated_at=CURRENT_TIMESTAMP WHERE id=?""", (stage, json.dumps(payload), job_id))
            worker.sync_episode_status(conn, episode_id)
        worker.wake_event.set()
    except worker.LeaseLost:
        pass
    except Exception as exc:  # noqa: BLE001 - the audio worker's loop does not fail jobs itself
        worker.fail_job(job_id, owner, episode_id, exc)


# --- the frame sheet Claude looks at ----------------------------------------------------------------------------------

def voice_sheet(conn, escalation_id):
    """JPEG bytes: per sample phrase (up to ``SHEET_ROWS``) three frames over its seconds, labelled P1… with the cut and time."""
    import cv2
    import numpy as np
    from . import missed_cuts
    row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
    if row is None:
        raise KeyError(escalation_id)
    item = agent_log.escalation_view(row)
    shown = (item['evidence'] or {}).get('phrases') or []
    if not shown:
        raise ValueError('This question lists no phrases')
    proxy = missed_cuts.proxy_path(conn, row['episode_id'])
    if not proxy.is_file():
        raise RuntimeError('The preview video is missing')
    rows = []
    for number, phrase in enumerate(shown[:SHEET_ROWS], 1):
        start, end = phrase['start'], phrase['end']
        if end - start < MIN_CLIP:  # a word fragment: spread the frames so a moving mouth can show
            middle = (start + end) / 2
            start, end = max(0.0, middle - MIN_CLIP / 2), middle + MIN_CLIP / 2
        moments = [start + (end - start) * share for share in (0.2, 0.5, 0.8)]
        tiles = []
        for moment in moments:
            frames = missed_cuts.frames_near(proxy, moment, span=0.06, width=320)
            frame = missed_cuts.nearest(frames, moment)
            tile = frame[1].copy() if frame else np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.putText(tile, f"P{number} cut {phrase.get('cut') or '?'} {moment:.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(tile)
        height = min(tile.shape[0] for tile in tiles)
        rows.append(np.hstack([tile[:height] for tile in tiles]))
    width = max(part.shape[1] for part in rows)
    rows = [np.pad(part, ((0, 4), (0, width - part.shape[1]), (0, 0))) for part in rows]
    ok, encoded = cv2.imencode('.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('could not encode the sheet')
    return encoded.tobytes()
