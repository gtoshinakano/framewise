"""Conservative spelling changes for names that have user-provided readings."""
import json
import re
import unicodedata
from collections import defaultdict

KATAKANA = r'ァ-ヺー'
HONORIFICS = ('さん', 'くん', '君', 'ちゃん', '先輩', 'せんぱい', 'さま', '様', '先生')
KANA = re.compile(r'[ぁ-ゖァ-ヺー]+')
KANJI = re.compile(r'[㐀-鿿々ヶ]')
NAME_RUN = re.compile(r'[ぁ-ゖァ-ヺー㐀-鿿々ヶ]+')
PARTICLES = ('とは', 'って', 'から', 'より', 'が', 'の', 'を', 'に', 'へ', 'は', 'も', 'で', 'と')
BOUNDARY = re.compile(r'^[\s。、，,.!?！？…「」『』（）()：:]')


def katakana(text):
    return ''.join(chr(ord(c) + 0x60) if 'ぁ' <= c <= 'ゖ' else c for c in text)


def hiragana(text):
    return ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in unicodedata.normalize('NFKC', text))


def cast_parts(glossary):
    cast = defaultdict(lambda: defaultdict(list))
    for row in glossary:
        if row['character_context_id']:
            cast[(row['character_context_id'], row.get('character_name') or '')][row['part']].append((row['spelling'], row['reading']))
    return cast


def alias_rules(glossary):
    rules = defaultdict(set)
    for parts in cast_parts(glossary).values():
        for family, fk in parts.get('family', []):
            for given, gk in parts.get('given', []):
                for prefix in {family, katakana(fk)}:
                    rules[rf'(?<![{KATAKANA}]){re.escape(prefix + katakana(gk))}(?![{KATAKANA}])'].add(family + given)
        for spellings in parts.values():
            for spelling, reading in spellings:
                kana = katakana(reading)
                if kana == spelling:
                    continue
                tail = rf'(?![{KATAKANA}])' if len(reading) >= 3 else '(?=' + '|'.join(HONORIFICS) + ')'
                rules[rf'(?<![{KATAKANA}]){re.escape(kana)}{tail}'].add(spelling)
    return [(re.compile(pattern), next(iter(spellings))) for pattern, spellings in
            sorted(rules.items(), key=lambda row: -len(row[0])) if len(spellings) == 1]


def apply_aliases(text, rules):
    changes = []
    for pattern, spelling in rules:
        changes.extend((match[0], spelling) for match in pattern.finditer(text) if match[0] != spelling)
        text = pattern.sub(spelling, text)
    return text, changes


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def near_reading(raw, reading):
    raw, reading = hiragana(raw), hiragana(reading)
    if raw == reading:
        return True
    if 'ー' in raw:
        # ASR writes both a stretched final vowel and an internal long vowel this way.
        for replacement in ('', 'あ', 'い', 'う', 'え', 'お'):
            if raw.replace('ー', replacement) == reading:
                return True
    # An extra kana attached to an otherwise exact name could belong to the sentence.
    if reading in raw and len(raw) > len(reading):
        return False
    return len(raw) == len(reading) and edit_distance(raw, reading) <= 1


def other_name_spans(text, glossary):
    spans = []
    for row in glossary:
        if row['character_context_id']:
            continue
        spelling, reading = row['spelling'], row['reading']
        spans.extend(match.span() for match in re.finditer(re.escape(spelling), text))
        for match in re.finditer(r'[ァ-ヺー]+|[ぁ-ゖ]+', text):
            raw = match[0]
            if near_reading(raw, reading):
                spans.append(match.span())
                continue
            # Mixed readings such as a place name rendered in kana before its school suffix.
            if len(raw) >= 3 and len(raw) < len(reading) and near_reading(raw, reading[:len(raw)]):
                for index in range(1, len(spelling)):
                    if text.startswith(spelling[index:], match.end()):
                        spans.append((match.start(), match.end() + len(spelling) - index))
                        break
    return spans


def plausible(raw, spelling, reading, suffix, *, linked=False):
    if raw == spelling:
        return True
    if not NAME_RUN.fullmatch(raw):
        return False
    if KANA.fullmatch(raw):
        return near_reading(raw, reading)
    honorific = suffix.startswith(HONORIFICS)
    boundary = not suffix or bool(BOUNDARY.match(suffix))
    if linked or honorific or (KANJI.match(raw[-1]) and spelling.endswith(raw[-1])):
        return True
    if not boundary and not suffix.startswith(PARTICLES):
        return False
    if set(KANJI.findall(raw)) & set(KANJI.findall(spelling)):
        return True
    if KANA.search(raw):
        return True
    # Without an orthographic anchor, keep likely bare names and subjects.
    # A short unrelated kanji term in an enumerated phrase is weak evidence.
    return boundary or suffix.startswith(('は', 'が'))


def name_only_change(original, fixed, glossary):
    """Require an unchanged sentence around each plausible name replacement."""
    if fixed == original:
        return False
    cast = cast_parts(glossary)
    candidates = defaultdict(set)
    for parts in cast.values():
        for entries in parts.values():
            for spelling, reading in entries:
                candidates[spelling].add((spelling, reading, '', ''))
        for family, fk in parts.get('family', []):
            for given, gk in parts.get('given', []):
                candidates[family + given].add((family, fk, given, gk))
    protected = other_name_spans(original, glossary)
    # Explore only the offsets that can diverge at a cast spelling in the fixed text.
    pending, visited = [(0, 0)], set()
    while pending:
        a, b = pending.pop()
        if (a, b) in visited:
            continue
        visited.add((a, b))
        if original[a:] == fixed[b:]:
            return True
        if a == len(original) or b == len(fixed):
            continue
        if original[a] == fixed[b]:
            pending.append((a + 1, b + 1))
        for name, forms in candidates.items():
            if not fixed.startswith(name, b):
                continue
            for end in range(a + 1, min(len(original), a + len(name) + 3) + 1):
                raw, suffix = original[a:end], original[end:]
                if (any(a < right and end > left for left, right in protected)
                        or not NAME_RUN.fullmatch(raw)
                        or any(raw != existing and existing in raw and existing not in name for existing in candidates)):
                    continue
                for family, fk, given, gk in forms:
                    if given:
                        valid = near_reading(raw, fk + gk) if KANA.fullmatch(raw) else any(
                            plausible(raw[:i], family, fk, '', linked=True)
                            and plausible(raw[i:], given, gk, suffix, linked=True)
                            and (raw.startswith(family) or near_reading(raw[:i], fk) or
                                 len(raw[:i]) >= max(2, len(family) - 1) and len(raw[i:]) >= 2)
                            for i in range(1, len(raw)))
                    else:
                        linked = any(fixed[:b].endswith(family_name) for parts in cast.values()
                                     if any(given_name == name for given_name, _ in parts.get('given', []))
                                     for family_name, _ in parts.get('family', []))
                        valid = plausible(raw, family, fk, suffix, linked=linked)
                    if valid:
                        pending.append((end, b + len(name)))
    return False


def cast_text(glossary):
    rows = []
    for (_, title), parts in cast_parts(glossary).items():
        names = [f'{part} name {spelling} ({reading})' for part, entries in parts.items() for spelling, reading in entries]
        rows.append(f"{title}: {', '.join(names)}")
    return '\n'.join(rows)


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


def validate(answer, window, glossary):
    """-> (fixes by id, errors as sentences for the model)."""
    items = answer.get('fixes') if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, ['the answer must be an object with a "fixes" array']
    texts = {line['id']: line['ja'] for line in window}
    fixes, errors, seen = {}, [], set()
    for item in items:
        if not isinstance(item, dict) or item.get('id') not in texts:
            errors.append('every fix needs the id of one of the given lines')
            continue
        lid, fixed = item['id'], (item.get('ja') or '').strip()
        if lid in seen:
            errors.append(f'{lid}: give each line at most once')
        elif fixed == texts[lid]:
            seen.add(lid)
            continue
        elif not name_only_change(texts[lid], fixed, glossary):
            errors.append(f'{lid}: the fix changes more than a misheard name, or writes a name that is not in the cast list; '
                          'copy the line exactly and replace only the name with its cast spelling, or leave the line out')
        else:
            fixes[lid] = fixed
        seen.add(lid)
    return fixes, errors


def user_message(window, glossary):
    others = [f"{row['spelling']} ({row['reading']})" for row in glossary if not row['character_context_id']]
    return ('Cast:\n' + cast_text(glossary) + '\n\nOther proper nouns (these are not characters; never change one into a character name):\n'
            + ', '.join(others) + '\n\nLines:\n' + json.dumps(window, ensure_ascii=False))
