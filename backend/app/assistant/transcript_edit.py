"""Read, plan and apply Cut Assistance corrections of a cut's transcript.

Transcript segments are revision chains like analysis results (``parent_id`` /
``superseded_by``; deletions are tombstone revisions), and may be sentence lines or
legacy word fragments. The assistant works on *lines*: consecutive segments merged
into speaker lines, numbered in time order exactly as ``get_transcript`` shows
them. ``build_plan`` turns line-level edits into fragment-level operations and a
readable before/after preview; ``apply`` writes them as ``assistant_edit``
revisions. Nothing here imports FastAPI because the planner runs in the
inference worker.

A plan is JSON: ``{"episode_id", "ops": [revise | delete | insert, ...]}`` where
``revise`` and ``delete`` name the current segment they replace. ``plan_stale``
and ``apply`` refuse a plan whose segments were revised or deleted since it was
made.
"""
import difflib
import json
import re
import uuid

from .. import db
from ..transcript_parts import join_text, manual_metadata

ORIGIN = 'assistant_edit'
MAX_TEXT = 20000
LINE_KEYS = ('start_time', 'end_time', 'text', 'speaker_label', 'character_context_id')


class PlanError(Exception):
    """A problem the model can fix by changing its edits, or a plan that can no longer be applied."""


# ---------------------------------------------------------------- reading

def transcript_rows(cut):
    return db.all_rows('''SELECT t.id,t.start_time,t.end_time,t.text,t.speaker_label,t.origin,t.character_context_id,c.title AS character_name FROM transcript_segments t
                          LEFT JOIN context_items c ON c.id=t.character_context_id
                          WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>?
                          ORDER BY t.start_time,t.id''', (cut['episode_id'], cut['end_time'], cut['start_time']))


def merged_transcript(rows, gap=0.6):
    """Merge word-level fragments into speaker lines; fewer tokens and easier to quote verbatim.

    Each line lists the ``ids`` of the fragments it was built from.
    """
    lines = []
    for row in rows:
        last = lines[-1] if lines else None
        if last and last['speaker_label'] == row['speaker_label'] and last['character_name'] == row['character_name'] \
                and last['origin'] == row['origin'] and row['start_time'] - last['end_time'] <= gap:
            last['text'] = join_text(last['text'], row['text'].strip())
            last['end_time'] = max(last['end_time'], row['end_time'])
            last['ids'].append(row.get('id'))
        else:
            lines.append({**row, 'text': row['text'].strip(), 'ids': [row.get('id')]})
    return lines


def project_characters(cut):
    return db.all_rows("SELECT id,title FROM context_items WHERE project_id=? AND kind='character' AND (episode_id IS NULL OR episode_id=?) ORDER BY title",
                       (cut['project_id'], cut['episode_id']))


# --------------------------------------------------------------- rendering

def speaker_name(line):
    name, label = line.get('character_name'), line.get('speaker_label')
    if name and label and label != name:
        return f'{name} ({label})'
    return name or label or 'unknown'


def render(cut, lines):
    """One readable row per line, clip-relative like get_transcript; used for the proposal card diff."""
    start = cut['start_time']
    return '\n'.join(f"{line['start_time'] - start:.2f}–{line['end_time'] - start:.2f}  {speaker_name(line)}: {line['text']}" for line in lines) or '(no transcript)'


# ---------------------------------------------------------------- planning

class Planner:
    def __init__(self, cut, episode, rows, characters):
        self.cut, self.duration = cut, episode.get('duration')
        self.rows = rows
        self.by_id = {row['id']: row for row in rows}
        self.lines = merged_transcript(rows)
        self.characters = characters
        self.names = {item['id']: item['title'] for item in characters}
        self.ops, self.changes, self.warnings, self.touched = [], [], [], set()

    # lookups
    def line(self, number, what='line'):
        if isinstance(number, bool) or not isinstance(number, int):
            raise PlanError(f'{what} must be a line number from get_transcript')
        if not 1 <= number <= len(self.lines):
            raise PlanError(f'There is no transcript line {number}; the cut has {len(self.lines)} line(s)' if self.lines else 'This cut has no transcript lines to edit; use op "insert" with start and end times')
        return self.lines[number - 1]

    def parts(self, lines):
        return [self.by_id[fragment] for line in lines for fragment in line['ids']]

    def character(self, name):
        wanted = name.strip().casefold()
        hits = [item for item in self.characters if item['title'].strip().casefold() == wanted]
        if len(hits) == 1:
            return hits[0]['id']
        if len(hits) > 1:
            raise PlanError(f'Several characters are named "{name}"; ask the user which one')
        close = difflib.get_close_matches(name.strip(), [item['title'] for item in self.characters], n=3, cutoff=0.5)
        names = ', '.join(item['title'] for item in self.characters) or 'none yet'
        hint = f' Did you mean: {", ".join(close)}?' if close else ''
        raise PlanError(f'No character named "{name}".{hint} Project characters: {names[:300]}. Use an exact name; if none fits, ask the user (a non-character voice such as a narrator goes in speaker_label).')

    def touch(self, first, last, number):
        for index in range(first, last + 1):
            if index in self.touched:
                raise PlanError(f'Edit {number}: line {index} is already changed by another edit; combine them into one edit')
            self.touched.add(index)

    # operations
    def revise(self, part, start, end, text, label, character):
        self.ops.append({'op': 'revise', 'id': part['id'], 'start_time': start, 'end_time': end, 'text': text, 'speaker_label': label, 'character_context_id': character})

    def delete(self, part):
        self.ops.append({'op': 'delete', 'id': part['id']})

    def insert(self, segment):
        self.ops.append({'op': 'insert', **segment})

    @staticmethod
    def clean(text, where):
        if not isinstance(text, str) or not text.strip():
            raise PlanError(f'{where}: text must be a non-empty string (use op "delete" to remove a line)')
        text = text.strip()
        if len(text) > MAX_TEXT:
            raise PlanError(f'{where}: text may contain at most {MAX_TEXT} characters')
        return text

    def update(self, edit, number):
        first = edit.get('line')
        self.line(first)
        last = edit.get('to_line') if edit.get('to_line') is not None else first
        self.line(last, 'to_line')
        if last < first:
            raise PlanError(f'Edit {number}: to_line must not be before line')
        if last > first and (edit.get('text') is not None or edit.get('replacements')):
            raise PlanError(f'Edit {number}: text and replacements change one line; use rewrite for a range of lines (to_line is only for changing the speaker)')
        self.touch(first, last, number)
        done = [description for n in range(first, last + 1) if (description := self.update_line(n, edit, number, quiet=last > first))]
        if not done:
            raise PlanError(f'Edit {number}: line{"s" if last > first else ""} {first}{f"–{last}" if last > first else ""} already {"have" if last > first else "has"} that text and speaker')
        self.changes.append(done[0] if len(done) == 1 else f'lines {first}–{last}: {len(done)} changed ({done[0].split(": ", 1)[1]})')

    def update_line(self, n, edit, number, quiet):
        """Plan one line of an update. Returns a description, or None when ``quiet`` and nothing would change."""
        line = self.lines[n - 1]
        parts = self.parts([line])
        text = line['text']
        if edit.get('text') is not None and edit.get('replacements'):
            raise PlanError(f'Edit {number}: give either text or replacements, not both')
        if edit.get('text') is not None:
            text = self.clean(edit['text'], f'Edit {number}')
        for item in edit.get('replacements') or []:
            if not isinstance(item, dict) or not isinstance(item.get('find'), str) or not item['find'] or not isinstance(item.get('replace'), str):
                raise PlanError(f'Edit {number}: each replacement needs a non-empty find and a replace string')
            if item['find'] not in text:
                raise PlanError(f'Edit {number}: "{item["find"][:60]}" is not in line {n} ("{text[:120]}"). Copy the exact words from get_transcript.')
            text = text.replace(item['find'], item['replace'])
        text = self.clean(text, f'Edit {number}')
        character, label = line['character_context_id'], line['speaker_label']
        if edit.get('clear_character') and edit.get('character'):
            raise PlanError(f'Edit {number}: give character or clear_character, not both')
        if edit.get('clear_character'):
            character = None
        elif edit.get('character'):
            character = self.character(edit['character'])
        if edit.get('speaker_label') is not None:
            label = str(edit['speaker_label']).strip()[:200]
        elif character and not label:
            label = self.names[character]
        text_changed = text != line['text']
        speaker_changed = (character, label) != (line['character_context_id'], line['speaker_label'])
        if not text_changed and not speaker_changed:
            if quiet:
                return None
            raise PlanError(f'Edit {number}: line {n} already has that text and speaker')
        if text_changed:
            # New wording replaces the word-level fragments with one segment spanning them.
            self.revise(parts[0], min(part['start_time'] for part in parts), max(part['end_time'] for part in parts), text, label, character)
            for part in parts[1:]:
                self.delete(part)
        else:
            for part in parts:
                self.revise(part, part['start_time'], part['end_time'], part['text'], label, character)
        what = [word for word, flag in (('text', text_changed), ('speaker', speaker_changed)) if flag]
        who = f" → {self.names.get(character) or label or 'unknown'}" if speaker_changed else ''
        return f'line {n}: {" and ".join(what)} changed{who}'

    def remove(self, edit, number):
        line = self.line(edit.get('line'))
        self.touch(edit['line'], edit['line'], number)
        for part in self.parts([line]):
            self.delete(part)
        self.changes.append(f"line {edit['line']}: removed")

    def to_seconds(self, value, where):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(f'{where} must be a number of clip-relative seconds')
        return round(self.cut['start_time'] + value, 3)

    def new_segments(self, items, number, inherit, bounds):
        """Turn the model's new lines into segments. Times are all given (clip-relative) or all spread over ``bounds``."""
        if not isinstance(items, list) or not items:
            raise PlanError(f'Edit {number}: lines must be a non-empty list of {{text, character?, speaker_label?, start?, end?}}')
        flags = [item.get('start') is not None or item.get('end') is not None for item in items if isinstance(item, dict)]
        timed = any(flags)
        if timed and not all(flags):
            raise PlanError(f'Edit {number}: give start and end on every new line, or on none (the old time span is then shared by text length)')
        if not timed and not bounds:
            raise PlanError(f'Edit {number}: inserted lines need start and end (clip-relative seconds)')
        weights = [max(1, len(re.sub(r'\s', '', str(item.get('text') or '')))) for item in items if isinstance(item, dict)]
        segments, elapsed = [], 0
        for index, item in enumerate(items):
            where = f'Edit {number} line {index + 1}'
            if not isinstance(item, dict):
                raise PlanError(f'{where} must be an object')
            text = self.clean(item.get('text'), where)
            old = inherit[min(index, len(inherit) - 1)] if inherit else None
            named = self.character(item['character']) if item.get('character') else None
            character = named or (old['character_context_id'] if old else None)
            if item.get('speaker_label') is not None:
                label = str(item['speaker_label']).strip()[:200]
            elif named and (not old or old['character_context_id'] != named):
                label = self.names[named]  # A diarization label of another speaker would be wrong here.
            else:
                label = old['speaker_label'] if old else ''
            if timed:
                start, end = self.to_seconds(item.get('start'), f'{where} start'), self.to_seconds(item.get('end'), f'{where} end')
            else:
                total = sum(weights)
                start = round(bounds[0] + (bounds[1] - bounds[0]) * elapsed / total, 3)
                elapsed += weights[index]
                end = bounds[1] if index == len(items) - 1 else round(bounds[0] + (bounds[1] - bounds[0]) * elapsed / total, 3)
            if end <= start or start < 0 or (self.duration is not None and end > self.duration + 0.2):
                raise PlanError(f'{where}: the time range {start - self.cut["start_time"]:.2f}–{end - self.cut["start_time"]:.2f}s (clip-relative) is empty or outside the video')
            segments.append({'start_time': start, 'end_time': end, 'text': text, 'speaker_label': label, 'character_context_id': character})
        return segments

    def pair(self, parts, segments):
        """Revise fragments in order into the new segments; delete leftovers and insert extras."""
        for index in range(max(len(parts), len(segments))):
            if index < len(parts) and index < len(segments):
                self.revise(parts[index], *(segments[index][key] for key in LINE_KEYS))
            elif index < len(parts):
                self.delete(parts[index])
            else:
                self.insert(segments[index])

    def rewrite(self, edit, number):
        self.line(edit.get('line'))
        end = edit.get('to_line') if edit.get('to_line') is not None else edit['line']
        self.line(end, 'to_line')
        if end < edit['line']:
            raise PlanError(f'Edit {number}: to_line must not be before line')
        self.touch(edit['line'], end, number)
        covered = self.lines[edit['line'] - 1:end]
        parts = self.parts(covered)
        bounds = (min(part['start_time'] for part in parts), max(part['end_time'] for part in parts))
        segments = self.new_segments(edit.get('lines'), number, covered, bounds)
        self.pair(parts, segments)
        self.warn_overlap(segments, set(range(edit['line'], end + 1)), number)
        label = f"line {edit['line']}" if end == edit['line'] else f"lines {edit['line']}–{end}"
        self.changes.append(f'{label}: rewritten as {len(segments)} line(s)')

    def add(self, edit, number):
        segments = self.new_segments(edit.get('lines'), number, None, None)
        for segment in segments:
            self.insert(segment)
        self.warn_overlap(segments, set(), number)
        self.changes.append(f'{len(segments)} new line(s) added')

    def warn_overlap(self, segments, replaced, number):
        for segment in segments:
            for index, line in enumerate(self.lines, 1):
                overlap = min(segment['end_time'], line['end_time']) - max(segment['start_time'], line['start_time'])
                if index not in replaced and overlap > 0.15:
                    self.warnings.append(f'Edit {number}: "{segment["text"][:40]}" overlaps existing line {index} by {overlap:.1f}s. If that line is the same speech, rewrite it instead of adding.')

    HANDLERS = {'update': update, 'rewrite': rewrite, 'delete': remove, 'insert': add}

    # result
    def after_rows(self):
        current = {row['id']: dict(row) for row in self.rows}
        for index, op in enumerate(self.ops):
            if op['op'] == 'delete':
                current.pop(op['id'], None)
                continue
            key = op.get('id') or f'new-{index}'
            base = current.get(key, {})
            current[key] = {**base, 'id': key, **{name: op[name] for name in LINE_KEYS}, 'origin': ORIGIN, 'character_name': self.names.get(op['character_context_id'])}
        return sorted(current.values(), key=lambda row: (row['start_time'], str(row['id'])))


def build_plan(cut, episode, rows, characters, edits):
    """Validate line-level edits and return ``{ops, before, after, changes, warnings}``."""
    planner = Planner(cut, episode, rows, characters)
    if not isinstance(edits, list) or not edits:
        raise PlanError('Provide at least one edit')
    for number, edit in enumerate(edits, 1):
        handler = Planner.HANDLERS.get(edit.get('op')) if isinstance(edit, dict) else None
        if handler is None:
            raise PlanError(f'Edit {number} needs op: update, rewrite, delete or insert')
        handler(planner, edit, number)
    before = render(cut, planner.lines)
    after_rows = [row for row in planner.after_rows() if row['start_time'] < cut['end_time'] and row['end_time'] > cut['start_time']]
    after = render(cut, merged_transcript(after_rows))
    if after == before:
        raise PlanError("The edits do not change this cut's transcript (inserted lines must overlap the cut)")
    return {'episode_id': cut['episode_id'], 'ops': planner.ops, 'before': before, 'after': after, 'changes': planner.changes, 'warnings': planner.warnings}


# ------------------------------------------------------- stale check / apply

def parse(payload_json):
    try:
        plan = json.loads(payload_json or '')
        if isinstance(plan, dict) and isinstance(plan.get('ops'), list) and plan.get('episode_id'):
            return plan
    except (TypeError, ValueError):
        pass
    raise PlanError('This proposal is damaged; ask the assistant to propose it again')


def check_current(conn, plan):
    """Raise PlanError unless every segment the plan replaces is still current."""
    for op in plan['ops']:
        if op['op'] == 'insert':
            continue
        row = conn.execute('SELECT episode_id,superseded_by,deleted FROM transcript_segments WHERE id=?', (op['id'],)).fetchone()
        if not row or row['superseded_by'] or row['deleted'] or row['episode_id'] != plan['episode_id']:
            raise PlanError('The transcript changed after this proposal was made; ask the assistant to propose it again')
    for character in {op['character_context_id'] for op in plan['ops'] if op.get('character_context_id')}:
        if not conn.execute("SELECT 1 FROM context_items WHERE id=? AND kind='character'", (character,)).fetchone():
            raise PlanError('A character in this proposal no longer exists; ask the assistant to propose it again')


def plan_stale(conn, payload_json):
    try:
        check_current(conn, parse(payload_json))
        return False
    except PlanError:
        return True


COLUMNS = '(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,deleted,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json)'


def apply(conn, payload_json):
    """Write the plan as ``assistant_edit`` revisions; returns the new segment ids."""
    from ..structured_results import invalidate_transcript
    plan = parse(payload_json)
    check_current(conn, plan)
    created = []
    for op in plan['ops']:
        new_id = str(uuid.uuid4())
        if op['op'] == 'insert':
            conn.execute(f'INSERT INTO transcript_segments {COLUMNS} VALUES (?,?,?,?,?,?,?,0,?,NULL,NULL,NULL,NULL,\'manual\',NULL,NULL)',
                         (new_id, plan['episode_id'], op['start_time'], op['end_time'], op['text'], op['speaker_label'], op['character_context_id'], ORIGIN))
            invalidate_transcript(conn, plan['episode_id'], op['start_time'], op['end_time'])
        else:
            old = conn.execute('SELECT * FROM transcript_segments WHERE id=?', (op['id'],)).fetchone()
            fields = old if op['op'] == 'delete' else op
            metadata = manual_metadata(old, fields)
            conn.execute(f'INSERT INTO transcript_segments {COLUMNS} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (new_id, plan['episode_id'], fields['start_time'], fields['end_time'], fields['text'], fields['speaker_label'], fields['character_context_id'],
                          int(op['op'] == 'delete'), ORIGIN, old['model_id'], old['job_id'], old['id'], *metadata))
            conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, old['id']))
            invalidate_transcript(conn, plan['episode_id'], old['start_time'], old['end_time'])
            if (old['start_time'], old['end_time']) != (fields['start_time'], fields['end_time']):
                invalidate_transcript(conn, plan['episode_id'], fields['start_time'], fields['end_time'])
        created.append(new_id)
    return created
