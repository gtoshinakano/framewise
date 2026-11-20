"""Round 2: character names in the ASR text (stdlib only).

Round 1 showed that names as ASR context barely help (梓川 and 咲太 never appeared) and that Qwen's text_fix invented wrong spellings
(安積川, 相坂, 作矢) because its prompt only had the English names. Here: (1) a deterministic pass turns the kana readings of the verified
names into their spelling, (2) a Qwen pass that is given every spelling and reading may only swap a misheard name for one of them;
the validator rejects any other change.
"""
import json
import re

import glossary

KATAKANA = r'ァ-ヺー'
HONORIFIC = '(?=' + '|'.join(glossary.HONORIFICS) + ')'


def katakana(text):
    return ''.join(chr(ord(c) + 0x60) if 'ぁ' <= c <= 'ゖ' else c for c in text)


def alias_rules():
    """[(compiled pattern, spelling)] longest first: full names in katakana, then each part.

    A part read with three kana or more is replaced wherever it stands alone in katakana (not inside a longer katakana word);
    a two-kana part (マイ, リオ, コガ, サキ) only right after its family name or right before an honorific.
    """
    rules = []
    for parts in glossary.PARTS.values():
        (family, family_kana), (given, given_kana) = parts
        rules.append((katakana(family_kana + given_kana), family + given, None))
        for spelling, reading in parts:
            kana = katakana(reading)
            if kana == spelling:
                continue
            if len(reading) >= 3:
                rules.append((kana, spelling, 'alone'))
            else:
                rules.append((kana, spelling, 'context'))
                rules.append((family + kana, family + spelling, None))
                rules.append((katakana(family_kana) + kana, family + spelling, None))
    seen, compiled = set(), []
    for kana, spelling, mode in sorted(rules, key=lambda r: -len(r[0])):
        if (kana, mode) in seen:
            continue
        seen.add((kana, mode))
        if mode == 'alone':
            pattern = f'(?<![{KATAKANA}]){re.escape(kana)}(?![{KATAKANA}])'
        elif mode == 'context':
            pattern = f'(?<![{KATAKANA}]){re.escape(kana)}{HONORIFIC}'
        else:
            pattern = re.escape(kana)
        compiled.append((re.compile(pattern), spelling))
    return compiled


RULES = alias_rules()


def apply_aliases(text):
    """-> (text with the kana readings of verified names spelled, list of (before, after))."""
    changes = []
    for pattern, spelling in RULES:
        for match in pattern.finditer(text):
            changes.append((match.group(0), spelling))
        text = pattern.sub(spelling, text)
    return text, changes


def spellings():
    """Every spelling a name fix may write: family, given and full names."""
    out = set()
    for (family, _), (given, _) in glossary.PARTS.values():
        out.update((family, given, family + given))
    return sorted(out, key=len, reverse=True)


def cast_text():
    rows = []
    for title, ((family, fk), (given, gk)) in glossary.PARTS.items():
        rows.append(f'{title}: family name {family} ({fk}), given name {given} ({gk})')
    return '\n'.join(rows)


def name_only_change(original, fixed, allowed=None):
    """True when `fixed` equals `original` except that some spans were replaced by allowed spellings (each span at most 3 characters longer)."""
    allowed = allowed or spellings()
    if fixed == original:
        return False
    found = []
    taken = [False] * len(fixed)
    for name in allowed:
        for match in re.finditer(re.escape(name), fixed):
            if not any(taken[match.start():match.end()]):
                for i in range(match.start(), match.end()):
                    taken[i] = True
                found.append((match.start(), match.end()))
    if not found:
        return False
    found.sort()
    pattern, position = '', 0
    for start, end in found:
        pattern += re.escape(fixed[position:start]) + f'(.{{1,{end - start + 3}}}?)'
        position = end
    pattern += re.escape(fixed[position:])
    return re.fullmatch(pattern, original, flags=re.S) is not None


SYSTEM = """You correct misheard character names in the lines of a Japanese speech recognizer for one episode of an anime.

You get the cast with the correct spelling and kana reading of every family and given name, then numbered lines. The recognizer often writes a spoken name with wrong kanji of a similar reading, in katakana, or split across characters.

Rules.
1. Report only lines in which a character of the cast is named with a wrong spelling. For each, give the whole line with only that name replaced by the spelling from the cast list.
2. Change nothing else: no other word, no punctuation, no spacing, no rewording.
3. Do not add a name that was not spoken, and leave a word that only sounds like a name but means something else in its sentence.
4. Lines that need no change are left out; an empty list is a normal answer."""


def schema(ids):
    item = {'type': 'object', 'additionalProperties': False, 'required': ['id', 'ja'],
            'properties': {'id': {'type': 'string', 'enum': list(ids)}, 'ja': {'type': 'string'}}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['fixes'],
            'properties': {'fixes': {'type': 'array', 'items': item, 'maxItems': len(ids)}}}


def validate(answer, window):
    """-> (fixes by id, errors as sentences for the model)."""
    items = answer.get('fixes') if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, ['the answer must be an object with a "fixes" array']
    texts = {line['id']: line['ja'] for line in window}
    fixes, errors = {}, []
    for item in items:
        if not isinstance(item, dict) or item.get('id') not in texts:
            errors.append('every fix needs the id of one of the given lines')
            continue
        lid, fixed = item['id'], (item.get('ja') or '').strip()
        if lid in fixes:
            errors.append(f'{lid}: give each line at most once')
        elif fixed == texts[lid]:
            continue
        elif not name_only_change(texts[lid], fixed):
            errors.append(f'{lid}: the fix changes more than a misheard name, or writes a name that is not in the cast list; '
                          'copy the line exactly and replace only the name with its cast spelling, or leave the line out')
        else:
            fixes[lid] = fixed
    return fixes, errors


def user_message(window):
    return 'Cast:\n' + cast_text() + '\n\nLines:\n' + json.dumps([{'id': l['id'], 'ja': l['ja']} for l in window], ensure_ascii=False)


def hits(text):
    """Occurrences of each verified spelling (family and given names) in a text."""
    return {name: text.count(name) for name in spellings() if text.count(name)}
