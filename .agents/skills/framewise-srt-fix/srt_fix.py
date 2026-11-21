#!/usr/bin/env python3
"""Fix an episode's transcript from a real subtitle file (.srt): wrong words, and a speaker only when clearly wrong.

    srt_fix.py plan "Show / Episode 2" --srt ep02.ja.srt     # compare, write a plan, print a summary
    srt_fix.py show PLAN --class review                      # the proposals to judge, as inline diffs
    srt_fix.py decide PLAN --accept 3,5-7 --reject 9         # record the decisions in the plan
    srt_fix.py apply PLAN [--dry-run]                        # accepted lines whole, safe parts of the rest

Classes: `safe` (clear mishearings: applied unless rejected), `review` (judge each), `variant` (the captions'
spelling of the same word, e.g. 多分/たぶん: reject unless the readings differ) and `info` (notes, never applied).
A line mixing classes carries `after_safe`, its safe parts alone, which is sent unless the line is accepted whole.

The plan compares content characters only (letters, digits, kana, kanji after NFKC), so the captions' own
punctuation and spacing never become edits; a fix is spliced into the line's own text. Deletions (captions trim
fillers) and words between two lines are never applied: they are listed as `info`. Lines are never created,
deleted, merged, split or retimed. Corrections keep each line's times and aligner words
(POST /api/episodes/{id}/transcript/corrections, origin `srt_edit`).

Server: FRAMEWISE_URL, else http://chopper.local:8080. Only HTTP is used; nothing on this machine is built or
restarted. Exit codes: 0 ok, 1 error, 2 a person is needed (the file does not match the episode).
"""
import argparse
import difflib
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_loader = SourceFileLoader('framewise_fw', str(ROOT / 'scripts' / 'fw'))
fw = module_from_spec(spec_from_loader('framewise_fw', _loader))
_loader.exec_module(fw)
Failure, OK, ERROR, BLOCKED = fw.Failure, fw.OK, fw.ERROR, fw.BLOCKED

MAX_HITS = 3             # a caption's first characters found more often than this are too common to place it
CLUSTER = 1.0            # seconds within which two cues share an offset
MIN_RUN = 3              # cues in a row on one offset before it counts (a lone hit is a repeated phrase)
GAP = 2.0                # silence on both sides that splits the episode into alignment chunks
MIN_MATCH_SHARE = .4     # caption keys found near a transcript line, below which the file is the wrong one
MAX_SPREAD = 1.5         # seconds of spread (IQR) inside one run above which its offset does not fit
SAFE_SIMILARITY, MIN_SIMILARITY = .6, .4
MAX_SAFE_FRAGMENT = 12
MAX_CHANGED = .4
MERGE_GAP = 2
SPEAKER_SHARE = .8
MIN_SPEAKER_CHARS = 3   # shorter lines (はい) align too loosely to call a speaker wrong
BATCH = 200
TIME = re.compile(r'(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})')
LEADING_MARK = re.compile(r'^\s*(?:\[[^\]]*\]|［[^］]*］)')
LEADING_TAG = re.compile(r'^\s*[(（]([^()（）]{1,20})[)）]')
ANY_MARK = re.compile(r'\[[^\]]*\]|［[^］]*］|[(（][^()（）]*[)）]|<[^>]*>|\{\\[^}]*\}')
BRACKETS = re.compile(r'[《》＜＞〈〉<>]')
DIGIT_RUN = re.compile(r'\d+(?:[ \t\u3000]+\d+)+')


# --- text ----------------------------------------------------------------------------------------------------

def keep(char):
    return unicodedata.category(char)[0] in 'LN' or char in 'ー々〆ヶ'


def script(char):
    if char.isdigit():
        return 'digit'
    if '぀' <= char <= 'ゟ':
        return 'hiragana'
    if '゠' <= char <= 'ヿ' or char == 'ー':
        return 'katakana'
    if char in '々〆ヶ' or '㐀' <= char <= '鿿' or '豈' <= char <= '﫿':
        return 'kanji'
    if char.isascii():
        return 'latin'
    return 'other'


def content(text):
    """[(comparable char, index in text)] for the letters and digits of a transcript line; NFKC per character."""
    out = []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize('NFKC', char):
            if keep(folded):
                out.append((folded.lower(), index))
    return out


def caption_content(text):
    """The comparable characters of caption text: NFKC of the whole text (so ｶﾞ becomes ガ), letters and digits."""
    return [char.lower() for char in unicodedata.normalize('NFKC', text) if keep(char)]


def caption_output(text):
    """The same characters as written (case kept), for the text that replaces a transcript fragment."""
    return [char for char in unicodedata.normalize('NFKC', text) if keep(char)]


# --- the .srt ------------------------------------------------------------------------------------------------

def seconds(match):
    hours, minutes, secs, millis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis.ljust(3, '0')) / 1000


def parse_srt(raw):
    """Cues ``{n, start, end, lines: [{text, tag}]}``; a ``(名前)`` at the start of a text line is that line's tag."""
    text = raw.lstrip('﻿').replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    for block in re.split(r'\n\s*\n', text):
        rows = [row for row in block.split('\n') if row.strip()]
        timing = next((index for index, row in enumerate(rows) if '-->' in row), None)
        if timing is None:
            continue
        times = TIME.findall(rows[timing])
        if len(times) < 2:
            continue
        start, end = (seconds(TIME.match(':'.join(parts[:3]) + ',' + parts[3])) for parts in times[:2])
        number = rows[timing - 1].strip() if timing else ''
        lines = []
        for row in rows[timing + 1:]:
            if '♪' in row or '♬' in row:
                continue
            tag, rest = None, row
            while True:
                stripped = LEADING_MARK.sub('', rest, count=1)
                if stripped == rest:
                    break
                rest = stripped
            if match := LEADING_TAG.match(rest):
                tag, rest = match.group(1).strip(), rest[match.end():]
            rest = BRACKETS.sub('', ANY_MARK.sub('', rest)).strip()
            if rest:
                lines.append({'text': rest, 'tag': tag})
        if lines:
            cues.append({'n': int(number) if number.isdigit() else len(cues) + 1, 'start': start, 'end': end, 'lines': lines})
    cues.sort(key=lambda cue: (cue['start'], cue['n']))
    return cues


def caption_chars(cues, offsets):
    """One entry per comparable caption character: char, out (as written), cue index, tag, episode time."""
    out = []
    for cue_index, cue in enumerate(cues):
        chars = [(char, written, line['tag']) for line in cue['lines']
                 for char, written in zip(caption_content(line['text']), caption_output(line['text']))]
        span = cue['end'] - cue['start']
        for position, (char, written, tag) in enumerate(chars):
            out.append({'char': char, 'out': written, 'cue': cue_index, 'tag': tag,
                        't': cue['start'] + offsets[cue_index] + span * position / max(1, len(chars))})
    return out


# --- the transcript ------------------------------------------------------------------------------------------

def char_times(line, chars):
    """Episode seconds of each content character: from the aligner words when they still spell the line."""
    words = line.get('words_json')
    if words:
        words = json.loads(words) if isinstance(words, str) else words
        times = []
        for start, end, word in words:
            folded = caption_content(word)
            times += [start + (end - start) * index / max(1, len(folded)) for index in range(len(folded))]
        if len(times) == len(chars):
            return times
    span = line['end_time'] - line['start_time']
    return [line['start_time'] + span * index / max(1, len(chars)) for index in range(len(chars))]


def transcript_chars(lines):
    out = []
    for line_index, line in enumerate(lines):
        chars = content(line['text'])
        for (char, at), t in zip(chars, char_times(line, chars)):
            out.append({'char': char, 'line': line_index, 'at': at, 't': t})
    return out


# --- offset --------------------------------------------------------------------------------------------------

def estimate_offsets(lines, cues, key=5):
    """Seconds to add to each cue's times, as runs of consecutive cues sharing one offset.

    A release of the same episode can move blocks (an opening song cut or moved, an eyecatch), so the offset is
    piecewise constant. Each cue's first characters are looked up anywhere in the transcript; offsets that many
    cues agree on are clusters, consecutive cues on one cluster are a run, and a cue between runs takes the run
    nearest in time. Returns ``(runs, keyed, matched)`` with runs ``[{first, last, offset, spread, cues}]``
    in cue indices.
    """
    tchars = transcript_chars(lines)
    text = ''.join(item['char'] for item in tchars)
    candidates, keyed = {}, 0
    for index, cue in enumerate(cues):
        chars = caption_content(''.join(line['text'] for line in cue['lines']))
        if len(chars) < key:
            continue
        keyed += 1
        needle, hits, start = ''.join(chars[:key]), [], text.find(''.join(chars[:key]))
        while start != -1 and len(hits) <= MAX_HITS:
            hits.append(tchars[start]['t'] - cue['start'])
            start = text.find(needle, start + 1)
        if 0 < len(hits) <= MAX_HITS:
            candidates[index] = hits
    deltas = sorted(delta for hits in candidates.values() for delta in hits)
    clusters, group = [], []
    for delta in deltas:
        if group and delta - group[-1] > CLUSTER:
            clusters.append(group)
            group = []
        group.append(delta)
    if group:
        clusters.append(group)
    centers = [statistics.median(group) for group in clusters if len(group) >= max(MIN_RUN, .03 * keyed)]
    assigned = []
    for index in sorted(candidates):
        best = min(((abs(delta - center), center, delta) for delta in candidates[index] for center in centers), default=None)
        if best and best[0] <= CLUSTER:
            assigned.append((index, best[1], best[2]))
    runs, current = [], []
    for item in assigned:
        if current and item[1] != current[-1][1]:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    runs = [run for run in runs if len(run) >= MIN_RUN]
    out = []
    for run in runs:
        values = [delta for _, _, delta in run]
        if out and out[-1]['center'] == run[0][1]:
            out[-1]['values'] += values
            out[-1]['last'], out[-1]['cues'] = run[-1][0], out[-1]['cues'] + len(run)
            continue
        out.append({'first': run[0][0], 'last': run[-1][0], 'values': values, 'cues': len(run), 'center': run[0][1]})
    for run in out:
        values = run.pop('values')
        run.pop('center')
        quartiles = statistics.quantiles(values, n=4) if len(values) >= 4 else [values[0]] * 3
        run['offset'], run['spread'] = round(statistics.median(values), 3), round(quartiles[2] - quartiles[0], 3)
    return out, keyed, sum(run['cues'] for run in out)


def cue_offsets(cues, runs):
    """Every cue's offset: its run's, or for a cue between runs the run nearest in caption time."""
    offsets = []
    for index, cue in enumerate(cues):
        inside = next((run for run in runs if run['first'] <= index <= run['last']), None)
        if inside is None:
            def distance(run):
                return min(abs(cue['start'] - cues[run['first']]['start']), abs(cue['start'] - cues[run['last']]['end']))
            inside = min(runs, key=distance)
        offsets.append(inside['offset'])
    return offsets


# --- alignment -----------------------------------------------------------------------------------------------

def chunks(tchars, schars):
    """Index ranges of both streams, split where neither has anything for GAP seconds."""
    events = sorted([(item['t'], 0, index) for index, item in enumerate(tchars)] + [(item['t'], 1, index) for index, item in enumerate(schars)])
    groups, current, last = [], ([], []), None
    for t, side, index in events:
        if last is not None and t - last >= GAP and (current[0] or current[1]):
            groups.append(current)
            current = ([], [])
        current[side].append(index)
        last = t
    if current[0] or current[1]:
        groups.append(current)
    return groups


def align(tchars, schars):
    """Opcodes over the whole episode in stream indices: (op, transcript ids, caption ids, insert_at).

    ``insert_at`` is the transcript index an insertion goes before, or None at a chunk edge (between lines).
    """
    ops = []
    for tids, sids in chunks(tchars, schars):
        tids, sids = sorted(tids), sorted(sids)
        left, right = ''.join(tchars[i]['char'] for i in tids), ''.join(schars[j]['char'] for j in sids)
        found = [(op, tids[a:b], sids[c:d], tids[a] if op == 'insert' and 0 < a < len(tids) else None)
                 for op, a, b, c, d in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes()]
        ops += merge_close(found, tchars)
    return ops


def merge_close(ops, tchars):
    """Join two replacements in one line separated by at most MERGE_GAP equal characters (青[春]気象校軍 →
    思[春]期症候群 is one mishearing, not two), so the fragment is judged as a whole. Insertions and deletions are
    never pulled in: a deletion must stay a note."""
    out = []
    for item in ops:
        if (len(out) >= 2 and item[0] == 'replace' and out[-1][0] == 'equal' and len(out[-1][1]) <= MERGE_GAP
                and out[-2][0] == 'replace'):
            first, gap = out[-2], out[-1]
            tids = first[1] + gap[1] + item[1]
            if len({tchars[i]['line'] for i in tids}) == 1:
                out[-2:] = [('replace', tids, first[2] + gap[2] + item[2], None)]
                continue
        out.append(item)
    return out


# --- proposals -----------------------------------------------------------------------------------------------

def name_spellings(glossary, characters):
    """Tag text -> character id, only where it names exactly one character (梓川 names two, so it is left out)."""
    owners = {}
    for entry in glossary:
        if entry.get('character_context_id'):
            owners.setdefault(unicodedata.normalize('NFKC', entry['spelling']).strip(), set()).add(entry['character_context_id'])
    for character in characters:
        owners.setdefault(unicodedata.normalize('NFKC', character['title']).strip(), set()).add(character['id'])
    return {spelling: ids.pop() for spelling, ids in owners.items() if len(ids) == 1}


def fragment_class(before, after, names):
    """('safe'|'review', why) for one replaced fragment of comparable characters."""
    if any(name in after for name in names if len(name) >= 2) and not any(name in before for name in names if len(name) >= 2):
        return 'safe', 'a name from the glossary'
    if len(before) > MAX_SAFE_FRAGMENT or len(after) > MAX_SAFE_FRAGMENT:
        return 'review', 'a long rewrite'
    left, right = {script(char) for char in before}, {script(char) for char in after}
    if left <= {'digit'} and right <= {'digit'}:
        return 'safe', 'digits'
    if len(before) == 1 and len(after) == 1:
        return 'review', 'one character for another (often a spelling choice: 見/観, ブレイク/ブレーク)'
    if 'kanji' in left and 'kanji' in right:
        return 'safe', 'kanji on both sides'
    if left == right == {'katakana'} or left == right == {'latin'}:
        return 'safe', f'{next(iter(left))} on both sides'
    if left <= {'hiragana'} and right <= {'hiragana'}:
        return 'review', 'kana only (particles and endings: an error or the captions condensing)'
    return 'review', 'kana and kanji differ (a spelling variant or a mishearing; it depends on the reading)'


def variant_like(before, after):
    """A change that is usually the captions' spelling of the same word: kanji written in kana or the other way
    round (多分/たぶん, 喧嘩/ケンカ, 更に/さらに), a kanji numeral as a digit, or one kanji for another (探/捜).
    Only the reading tells it from a mishearing (残しなさい/よこしなさい), so it is not decided here."""
    left, right = {script(char) for char in before}, {script(char) for char in after}
    kana = {'hiragana', 'katakana'}
    if ('kanji' in left) != ('kanji' in right) and (left | right) <= kana | {'kanji', 'digit'}:
        return True
    return len(before) == 1 and len(after) == 1 and left | right <= {'kanji', 'digit'}


def splice(text, chars, edits):
    """Apply edits (sorted, non-overlapping, in content positions) to the line's own text."""
    out, cursor = [], 0
    for edit in sorted(edits, key=lambda item: item['k1']):
        if edit['op'] == 'insert':
            at = chars[edit['k1'] - 1][1] + 1 if edit['k1'] else 0
            out.append(text[cursor:at] + edit['after'])
            cursor = at
        else:
            first, last = chars[edit['k1']][1], chars[edit['k2'] - 1][1] + 1
            out.append(text[cursor:first] + edit['after'])
            cursor = last
    return ''.join(out) + text[cursor:]


def keep_particles(tids, tchars, after):
    """Leave hiragana at either end of a replaced fragment in place when the rest of it is kanji or katakana and
    the captions' side does not end in kana there: は私信気象航空軍 → 思春期症候群 keeps the spoken は that the
    captions condensed away. A fragment of kana only (まゆ → 麻衣) is replaced whole."""
    chars = [tchars[i]['char'] for i in tids]
    if not any(script(char) in ('kanji', 'katakana') for char in chars):
        return tids
    first, last = 0, len(chars)
    # の in the transcript against ノ in the captions is the same sound respelled, not a particle to keep
    if script(after[0]) != 'hiragana' and kana_fold(after[0]) != kana_fold(chars[0]):
        while first < last - 1 and script(chars[first]) == 'hiragana':
            first += 1
    if script(after[-1]) != 'hiragana' and kana_fold(after[-1]) != kana_fold(chars[-1]):
        while last - 1 > first and script(chars[last - 1]) == 'hiragana':
            last -= 1
    return tids[first:last]


def kana_fold(char):
    """Hiragana as katakana, so の and ノ compare equal."""
    return chr(ord(char) + 0x60) if '\u3041' <= char <= '\u3096' else char


def tag_owner(tag, names):
    return names.get(unicodedata.normalize('NFKC', tag).strip()) if tag else None


def build_plan(lines, cues, offsets, glossary, characters):
    """Items per line (text, speaker) and info notes for the whole episode."""
    names = name_spellings(glossary, characters)
    titles = {character['id']: character['title'] for character in characters}
    tchars, schars = transcript_chars(lines), caption_chars(cues, offsets)
    positions, by_line = {}, {}
    for index, item in enumerate(tchars):
        positions[index] = len(by_line.setdefault(item['line'], []))
        by_line[item['line']].append(index)
    per_line = {index: {'edits': [], 'equal': 0, 'caption': 0, 'cues': set(), 'tags': [], 'dropped': []} for index in range(len(lines))}
    notes, matched_cues = [], set()
    for op, tids, sids, insert_at in align(tchars, schars):
        if op in ('equal', 'replace'):
            matched_cues |= {schars[j]['cue'] for j in sids}
        if op == 'equal':
            for i, j in zip(tids, sids):
                state = per_line[tchars[i]['line']]
                state['equal'] += 1
                state['caption'] += 1
                state['cues'].add(schars[j]['cue'])
                state['tags'].append(schars[j]['tag'])
            continue
        before = ''.join(tchars[i]['char'] for i in tids)
        after = ''.join(schars[j]['out'] for j in sids)
        if op == 'delete':
            for line in sorted({tchars[i]['line'] for i in tids}):
                per_line[line]['dropped'].append(''.join(tchars[i]['char'] for i in tids if tchars[i]['line'] == line))
            continue
        if op == 'insert':
            if insert_at is None or tchars[insert_at - 1]['line'] != tchars[insert_at]['line']:
                if len(after) >= 2:
                    notes.append({'kind': 'info', 'class': 'info', 'start': schars[sids[0]]['t'], 'cues': sorted({cues[schars[j]['cue']]['n'] for j in sids}),
                                  'reason': f'the captions have "{after}" between two lines; left alone'})
                continue
            line = tchars[insert_at]['line']
            state = per_line[line]
            state['caption'] += len(sids)
            state['cues'] |= {schars[j]['cue'] for j in sids}
            state['edits'].append({'op': 'insert', 'k1': positions[insert_at], 'k2': positions[insert_at], 'before': '', 'after': after})
            continue
        owners = sorted({tchars[i]['line'] for i in tids})
        if len(owners) > 1:
            first = lines[owners[0]]
            notes.append({'kind': 'info', 'class': 'info', 'start': first['start_time'], 'segment_id': first['id'],
                          'reason': f'"{before}" → "{after}" runs across {len(owners)} lines; left alone',
                          'before': ' | '.join(lines[owner]['text'] for owner in owners)})
            continue
        tids = keep_particles(tids, tchars, after)
        before = ''.join(tchars[i]['char'] for i in tids)
        state = per_line[owners[0]]
        state['caption'] += len(sids)
        state['cues'] |= {schars[j]['cue'] for j in sids}
        state['tags'] += [schars[sids[n * len(sids) // len(tids)]]['tag'] for n in range(len(tids))]
        state['edits'].append({'op': 'replace', 'k1': positions[tids[0]], 'k2': positions[tids[-1]] + 1, 'before': before, 'after': after})
    return assemble(lines, cues, offsets, per_line, notes, matched_cues, names, titles)


def stamp(seconds_value):
    minutes, secs = divmod(max(0.0, seconds_value), 60)
    return f'{int(minutes):02d}:{secs:05.2f}'


def text_item(line, chars, state, base, names):
    """The line's text proposal, or None when the captions agree with it."""
    edits = [dict(edit) for edit in state['edits']]
    for match in DIGIT_RUN.finditer(line['text']):
        k1 = next(position for position, (_, at) in enumerate(chars) if at >= match.start())
        k2 = next(position for position, (_, at) in enumerate(chars) if at >= match.end() - 1) + 1
        if not any(edit['k1'] < k2 and k1 < edit['k2'] or edit['op'] == 'insert' and k1 < edit['k1'] < k2 for edit in edits):
            edits.append({'op': 'space', 'k1': k1, 'k2': k2, 'before': match.group(), 'after': ''.join(match.group().split())})
    if not edits:
        return None
    worst, reasons, variants = 'safe', [], True
    for edit in edits:
        if edit['op'] == 'space':
            edit['class'], edit['why'] = 'safe', 'a space inside a number'
        elif edit['op'] == 'insert':
            edit['class'], edit['why'] = 'review', 'words the transcript lacks inside the line'
        else:
            span = line['text'][chars[edit['k1']][1]:chars[edit['k2'] - 1][1] + 1]
            edit['class'], edit['why'] = fragment_class(edit['before'], edit['after'], names)
            if re.search(r'[、。，,！？!?「」]', span):
                edit['class'], edit['why'] = 'review', 'the change runs across punctuation'
        if edit['class'] == 'review':
            worst = 'review'
            if edit['op'] == 'replace' and variant_like(edit['before'], edit['after']):
                edit['why'] = 'a spelling variant, unless the readings differ'
            else:
                variants = False
        reasons.append(f"{edit['before'] or '∅'}→{edit['after']}: {edit['why']}")
    doubtful = False
    if base['similarity'] < SAFE_SIMILARITY:
        doubtful, reasons = True, reasons + [f"line similarity {base['similarity']:.2f}"]
    changed = sum(max(edit['k2'] - edit['k1'], len(edit['after'])) for edit in edits if edit['op'] != 'space')
    if changed > MAX_CHANGED * len(chars):
        doubtful, reasons = True, reasons + ['much of the line changes']
    after = splice(line['text'], chars, edits)
    if after == line['text']:
        return None
    # The safe parts alone, sent even when the rest of the line is rejected; none when the line itself is doubtful.
    after_safe = None if doubtful else splice(line['text'], chars, [edit for edit in edits if edit['class'] == 'safe'])
    if doubtful:
        worst = 'review'
    elif worst == 'review' and variants:
        worst = 'variant'
    return {**base, 'kind': 'text', 'class': worst, 'before': line['text'], 'after': after,
            'after_safe': after_safe if after_safe != line['text'] else None,
            'edits': [{key: edit[key] for key in ('op', 'before', 'after', 'class', 'why')} for edit in edits],
            'reason': '; '.join(reasons), 'accepted': True if worst == 'safe' else None}


def speaker_item(line, chars, state, base, names, titles):
    """A speaker proposal (or a note) when the captions' name tag covers the line and names someone else."""
    owners = [(tag, tag_owner(tag, names)) for tag in state['tags'] if tag_owner(tag, names)]
    if not owners:
        return None
    top = max({owner for _, owner in owners}, key=lambda owner: sum(1 for _, other in owners if other == owner))
    share = sum(1 for _, owner in owners if owner == top) / len(chars)
    if share < SPEAKER_SHARE or top == line['character_context_id']:
        return None
    tag = next(tag for tag, owner in owners if owner == top)
    name = titles.get(top, top)
    if not line['character_context_id']:
        return {**base, 'kind': 'info', 'class': 'info', 'before': line['text'],
                'reason': f'the captions name {name} ({tag}) for this unassigned line; not filled in'}
    if len(chars) < MIN_SPEAKER_CHARS:
        return {**base, 'kind': 'info', 'class': 'info', 'before': line['text'],
                'reason': f"the captions name {name} ({tag}), but the line is too short to be sure; {base['character']} kept"}
    if line.get('speaker_source') == 'manual':
        return {**base, 'kind': 'info', 'class': 'info', 'before': line['text'],
                'reason': f"the captions name {name} ({tag}), but {base['character']} was set by hand; left alone"}
    return {**base, 'kind': 'speaker', 'class': 'review', 'before': line['text'], 'after': line['text'],
            'character_context_id': top, 'new_character': name, 'accepted': None,
            'reason': f"{share:.0%} of the line is under the captions' ({tag}) tag; now {base['character']} "
                      f"(by {line.get('speaker_source') or 'unknown'}, confidence {line.get('confidence') or 'none'})"}


def assemble(lines, cues, offsets, per_line, notes, matched_cues, names, titles):
    items = []

    def caption_text(cue_ids):
        return ' / '.join(' '.join((f"({row['tag']})" if row['tag'] else '') + row['text'] for row in cues[c]['lines']) for c in sorted(cue_ids))

    for index, line in enumerate(lines):
        state, chars = per_line[index], content(line['text'])
        if not chars:
            continue
        similarity = 2 * state['equal'] / (len(chars) + state['caption']) if state['caption'] else 0.0
        base = {'segment_id': line['id'], 'start': line['start_time'], 'end': line['end_time'],
                'speaker_label': line['speaker_label'], 'character': titles.get(line['character_context_id']),
                'cues': [cues[c]['n'] for c in sorted(state['cues'])], 'srt_text': caption_text(state['cues']),
                'similarity': round(similarity, 3)}
        if similarity < MIN_SIMILARITY:
            if state['caption']:
                notes.append({**base, 'kind': 'info', 'class': 'info', 'before': line['text'],
                              'reason': f'the captions hardly match this line (similarity {similarity:.2f}); left alone'})
            continue
        for found in (text_item(line, chars, state, base, names), speaker_item(line, chars, state, base, names, titles)):
            if found:
                (notes if found['kind'] == 'info' else items).append(found)
        for dropped in state['dropped']:
            if len(dropped) >= 2:
                notes.append({**base, 'kind': 'info', 'class': 'info', 'before': line['text'],
                              'reason': f'the captions do not have "{dropped}"; kept (captions trim words)'})
    for index, cue in enumerate(cues):
        text = ' '.join(row['text'] for row in cue['lines'])
        if index not in matched_cues and len(caption_content(text)) >= 4:
            notes.append({'kind': 'info', 'class': 'info', 'start': cue['start'] + offsets[index], 'cues': [cue['n']], 'srt_text': text,
                          'reason': 'speech in the captions with no transcript line; nothing is created'})
    items.sort(key=lambda item: (item['start'], item['kind']))
    notes.sort(key=lambda item: item.get('start') or 0)
    for number, item in enumerate(items + notes, start=1):
        item['id'] = number
    return items + notes


# --- commands ------------------------------------------------------------------------------------------------

def default_out(episode):
    folder = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'framewise' / 'srt-fix'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{episode['id'][:8]}-{time.strftime('%Y%m%d-%H%M%S')}.json"


def load_plan(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise Failure(f'cannot read the plan {path}: {exc}') from None


def save_plan(path, plan):
    Path(path).write_text(json.dumps(plan, ensure_ascii=False, indent=1))


def cmd_plan(api, args):
    episode = fw.find_episode(api, args.episode, args.latest)
    try:
        raw = Path(args.srt).expanduser().read_bytes().decode('utf-8-sig')
    except (OSError, UnicodeDecodeError) as exc:
        raise Failure(f'cannot read {args.srt} as UTF-8: {exc}') from None
    cues = parse_srt(raw)
    if not cues:
        raise Failure(f'{args.srt} has no subtitle cues')
    lines = sorted(api.get(f"/api/episodes/{episode['id']}/transcript"), key=lambda line: (line['start_time'], line['id']))
    if not lines:
        raise Failure('the episode has no transcript yet: transcribe it first (scripts/fw transcribe EP)')
    glossary = api.get(f"/api/projects/{episode['project_id']}/name-glossary")
    glossary = glossary.get('items', glossary) if isinstance(glossary, dict) else glossary
    characters = [item for item in api.get(f"/api/projects/{episode['project_id']}/context") if item.get('kind') == 'character']
    runs, keyed, matched = estimate_offsets(lines, cues)
    share = matched / keyed if keyed else 0.0
    print(f"{episode['project']} / {episode['title']}: {len(lines)} transcript lines, {len(cues)} caption cues")
    print(f'caption cues placed in the transcript: {matched}/{keyed} ({share:.0%})')
    for run in runs:
        print(f"  cues {cues[run['first']]['n']}-{cues[run['last']]['n']} ({stamp(cues[run['first']]['start'])}-{stamp(cues[run['last']]['end'])} in the file): "
              f"offset {run['offset']:+.2f}s, spread {run['spread']:.2f}s, {run['cues']} cues")
    if args.offset == 'auto':
        if not runs or share < MIN_MATCH_SHARE:
            print(f'stopped: too few captions match this transcript ({share:.0%}); is it the right file for this episode?')
            return BLOCKED
        loose = [run for run in runs if run['spread'] > MAX_SPREAD]
        if loose:
            print(f"stopped: the captions drift inside cues {', '.join(str(cues[run['first']]['n']) for run in loose)}... "
                  f'(spread above {MAX_SPREAD}s), so their offset does not fit; check the file, or pass --offset SECONDS')
            return BLOCKED
        offsets = cue_offsets(cues, runs)
    else:
        runs = [{'first': 0, 'last': len(cues) - 1, 'offset': float(args.offset), 'spread': 0.0, 'cues': len(cues)}]
        offsets = [float(args.offset)] * len(cues)
    items = build_plan(lines, cues, offsets, glossary, characters)
    plan = {'version': 1, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'server': api.url,
            'episode': {key: episode[key] for key in ('id', 'title', 'project', 'project_id')},
            'srt': {'file': Path(args.srt).name, 'path': str(Path(args.srt).expanduser().resolve()), 'cues': len(cues)},
            'offsets': [{'cues': [cues[run['first']]['n'], cues[run['last']]['n']], 'offset': run['offset'], 'spread': run['spread']} for run in runs],
            'match_share': round(share, 3), 'items': items}
    out = Path(args.out) if args.out else default_out(episode)
    save_plan(out, plan)
    counts = {name: sum(1 for item in items if item['class'] == name) for name in ('safe', 'review', 'variant', 'info')}
    speakers = sum(1 for item in items if item['kind'] == 'speaker')
    print(f"proposals: {counts['safe']} safe, {counts['review']} to review ({speakers} speaker change(s)), "
          f"{counts['variant']} spelling variants (reject unless the reading differs), {counts['info']} notes")
    print(f'plan: {out}')
    print(f'next: show {out} --class safe, then --class review and --class variant; record with decide; then apply --dry-run and apply')
    return OK


def diff_text(before, after):
    out = []
    for op, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op == 'equal':
            out.append(before[a:b])
        else:
            out.append((f'[-{before[a:b]}-]' if b > a else '') + (f'{{+{after[c:d]}+}}' if d > c else ''))
    return ''.join(out)


def describe(item):
    head = f"#{item['id']} {item['class']:<6} {stamp(item.get('start') or 0)}"
    if item['kind'] == 'text':
        state = {True: ' accepted', False: ' rejected', None: ''}[item.get('accepted')] + (f" applied" if item.get('applied') else '')
        return '\n'.join([f"{head}  {item.get('character') or item.get('speaker_label') or '?'}{state}",
                          f"  ~ {diff_text(item['before'], item['after'])}"]
                         + ([f"  safe part alone: {diff_text(item['before'], item['after_safe'])}"]
                            if item.get('after_safe') and item['after_safe'] != item['after'] else [])
                         + [
                          f"  why: {item['reason']}", f"  srt: {item['srt_text']}  (cue {','.join(map(str, item['cues']))})"])
    if item['kind'] == 'speaker':
        state = {True: ' accepted', False: ' rejected', None: ''}[item.get('accepted')] + (f" applied" if item.get('applied') else '')
        return '\n'.join([f"{head}  speaker {item.get('character')} → {item['new_character']}{state}",
                          f"  line: {item['before']}", f"  why: {item['reason']}", f"  srt: {item['srt_text']}"])
    return f"{head}  {item['reason']}" + (f"\n  line: {item['before']}" if item.get('before') else '') + (f"\n  srt: {item['srt_text']}" if item.get('srt_text') else '')


def cmd_show(api, args):
    plan = load_plan(args.plan)
    items = [item for item in plan['items'] if (not args.cls or item['class'] == args.cls) and (not args.kind or item['kind'] == args.kind)]
    if args.ids:
        wanted = parse_ids(args.ids)
        items = [item for item in items if item['id'] in wanted]
    if args.open:
        items = [item for item in items if item['kind'] != 'info' and item.get('accepted') is None]
    for item in items[:args.limit or None]:
        print(describe(item))
    if args.limit and len(items) > args.limit:
        print(f'… {len(items) - args.limit} more (raise --limit)')
    return OK


def parse_ids(text):
    wanted = set()
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition('-')
        if not first.isdigit() or (last and not last.isdigit()):
            raise Failure(f'cannot read the ids {text!r}: use 3,7,12-20')
        wanted |= set(range(int(first), int(last or first) + 1))
    return wanted


def cmd_decide(api, args):
    plan = load_plan(args.plan)
    accept, reject = parse_ids(args.accept), parse_ids(args.reject)
    if accept & reject:
        raise Failure(f'accepted and rejected at once: {sorted(accept & reject)}')
    known = {item['id']: item for item in plan['items'] if item['kind'] != 'info'}
    unknown = sorted((accept | reject) - set(known))
    if unknown:
        raise Failure(f'not proposals in this plan: {unknown}')
    for number in accept | reject:
        known[number]['accepted'] = number in accept
        if args.note:
            known[number]['decision_note'] = args.note
    if args.text is not None:
        if len(accept) != 1 or known[next(iter(accept))]['kind'] != 'text':
            raise Failure('--text goes with exactly one accepted text proposal')
        known[next(iter(accept))]['after'] = args.text
    save_plan(args.plan, plan)
    open_items = sum(1 for item in known.values() if item.get('accepted') is None)
    print(f'accepted {len(accept)}, rejected {len(reject)}; {open_items} proposal(s) still undecided')
    return OK


def text_to_send(item, mode):
    """The text a text proposal sends: all of it when accepted, else its safe parts unless it was rejected outright."""
    if mode == 'ids' or (mode == 'decided' and item.get('accepted') is True):
        return item['after']
    if item['class'] == 'safe' and item.get('accepted') is False:
        return None
    return item.get('after_safe')


def cmd_apply(api, args):
    plan = load_plan(args.plan)
    mode = 'ids' if args.ids else 'safe' if args.safe else 'decided'
    pending = [item for item in plan['items'] if item['kind'] in ('text', 'speaker') and not item.get('applied') and not item.get('skipped')]
    if args.ids:
        wanted = parse_ids(args.ids)
        pending = [item for item in pending if item['id'] in wanted]
    skip = parse_ids(args.skip)
    chosen = []
    for item in pending:
        if item['id'] in skip:
            continue
        if item['kind'] == 'text':
            text = text_to_send(item, mode)
            if text is not None:
                chosen.append((item, text))
        elif mode == 'ids' or (mode == 'decided' and item.get('accepted') is True):
            chosen.append((item, None))
    undecided = [item['id'] for item in pending if item['class'] in ('review', 'variant') and item.get('accepted') is None]
    if undecided and mode == 'decided':
        print(f'note: {len(undecided)} proposal(s) are undecided; only their safe parts are sent, and a later decision on '
              'those lines needs a new plan (decide them first)')
    if not chosen:
        print('nothing to apply')
        return OK
    by_segment = {}
    for item, text in chosen:
        correction = by_segment.setdefault(item['segment_id'], {'segment_id': item['segment_id'], 'cues': item['cues'][:50],
                                                                'srt_text': item['srt_text'][:2000], 'items': []})
        correction['items'].append(item)
        if item['kind'] == 'text':
            correction['text'] = text
        else:
            correction['character_context_id'] = item['character_context_id']
        correction['note'] = '; '.join(entry.get('decision_note') or entry['reason'] for entry in correction['items'])[:500]
    corrections = list(by_segment.values())
    source = {'kind': 'srt', 'file': plan['srt']['file'], 'offset': plan['offsets'][0]['offset'] if len(plan['offsets']) == 1 else None}
    api.url = args.url.rstrip('/') if args.url else plan.get('server') or api.url
    path = f"/api/episodes/{plan['episode']['id']}/transcript/corrections"
    applied = skipped = 0
    for start in range(0, len(corrections), BATCH):
        batch = corrections[start:start + BATCH]
        body = {'source': source, 'corrections': [{key: value for key, value in item.items() if key != 'items'} for item in batch]}
        if args.dry_run:
            for item in batch:
                for entry in item['items']:
                    what = ('the whole line' if item.get('text') == entry['after'] else 'its safe part') if entry['kind'] == 'text' else 'the speaker'
                    print(describe(entry).split('\n')[0] + f'  (would send {what})')
            continue
        try:
            result = api.post(path, body=body)
        except Failure as exc:
            if 'HTTP 404' in str(exc) or 'HTTP 405' in str(exc):
                raise Failure('the server has no /transcript/corrections yet: chopper runs older code; '
                              'ask the user to rebuild video-analyzer there') from None
            raise
        done = {entry['segment_id']: entry['id'] for entry in result['applied']}
        reasons = {entry['segment_id']: entry['reason'] for entry in result['skipped']}
        for item in batch:
            for entry in item['items']:
                if item['segment_id'] in done:
                    entry['applied'] = done[item['segment_id']]
                    applied += 1
                else:
                    entry['skipped'] = reasons.get(item['segment_id'], 'not applied')
                    skipped += 1
                    print(f"skipped #{entry['id']}: {entry['skipped']}")
        save_plan(args.plan, plan)
    if args.dry_run:
        print(f'dry run: {sum(len(item["items"]) for item in corrections)} proposal(s) on {len(corrections)} line(s) would be sent')
        return OK
    print(f'applied {applied} proposal(s), skipped {skipped}')
    print_links(api, plan)
    print(f"open the episode: {api.url}/#/episode/{plan['episode']['id']}")
    return OK


def print_links(api, plan):
    """Links to the cuts holding the lines a person should look at: every speaker change that was sent and every line the
    server skipped. Each opens the cut on its Results tab, where the spoken words are listed."""
    episode = {'id': plan['episode']['id']}
    wanted = [(entry, 'speaker changed') for entry in plan['items'] if entry['kind'] == 'speaker' and entry.get('applied')]
    wanted += [(entry, f"skipped: {entry['skipped']}") for entry in plan['items'] if entry.get('skipped')]
    if not wanted:
        return
    print('lines to look at (each link opens the cut that holds the line):')
    try:
        for entry, why in wanted[:fw.LINK_LIMIT]:
            print(f"- {stamp(entry.get('start') or 0)} #{entry['id']} {why}: {fw.url_at_time(api, episode, entry.get('start') or 0) or '(no cut at that time)'}")
    except Failure as exc:
        print(f'(no links: {exc})')
    if len(wanted) > fw.LINK_LIMIT:
        print(f'… {len(wanted) - fw.LINK_LIMIT} more in the plan (show PLAN --ids)')


def parser():
    root = argparse.ArgumentParser(prog='srt_fix.py', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest='command', required=True, metavar='COMMAND')
    command = sub.add_parser('plan', help='compare the episode with an .srt and write a plan')
    command.add_argument('episode', nargs='?', help='episode id, or text of "Project / Episode"')
    command.add_argument('--latest', action='store_true', help='the newest episode')
    command.add_argument('--srt', required=True, help='the subtitle file (UTF-8 .srt)')
    command.add_argument('--offset', default='auto', help='seconds added to the captions\' times (default: estimated)')
    command.add_argument('--out', help='where to write the plan (default ~/.cache/framewise/srt-fix/)')
    command.set_defaults(func=cmd_plan)
    command = sub.add_parser('show', help='print proposals as inline diffs')
    command.add_argument('plan')
    command.add_argument('--class', dest='cls', choices=('safe', 'review', 'variant', 'info'))
    command.add_argument('--kind', choices=('text', 'speaker', 'info'))
    command.add_argument('--ids', help='only these: 3,7,12-20')
    command.add_argument('--open', action='store_true', help='only proposals nobody decided yet')
    command.add_argument('--limit', type=int, default=0)
    command.set_defaults(func=cmd_show)
    command = sub.add_parser('decide', help='accept or reject proposals in the plan')
    command.add_argument('plan')
    command.add_argument('--accept', default='', help='ids to accept: 3,7,12-20')
    command.add_argument('--reject', default='', help='ids to reject')
    command.add_argument('--text', help='with one accepted text proposal: the line as it should read instead')
    command.add_argument('--note', help='why (kept in the plan and sent as evidence)')
    command.set_defaults(func=cmd_decide)
    command = sub.add_parser('apply', help='send proposals to the server')
    command.add_argument('plan')
    group = command.add_mutually_exclusive_group()
    group.add_argument('--safe', action='store_true', help='only the safe parts of every line, ignoring decisions on the rest')
    group.add_argument('--ids', help='exactly these proposals, whole, whatever was decided')
    command.add_argument('--skip', default='', help='leave these out')
    command.add_argument('--dry-run', action='store_true', help='print what would be sent, send nothing')
    command.add_argument('--url', help='server (default: the one the plan was made with)')
    command.set_defaults(func=cmd_apply)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    api = fw.Api(fw.base_url())
    try:
        return args.func(api, args)
    except Failure as exc:
        print(f'error: {exc}', file=sys.stderr)
        return ERROR
    except KeyboardInterrupt:
        return ERROR


if __name__ == '__main__':
    sys.exit(main())
