"""Tool registry and built-in Cut Assistance tools.

Register a tool with the ``@tool`` decorator. The handler receives a
``ToolContext`` and the validated arguments and returns JSON-serialisable data
(or text for ``raw`` tools). ``execute`` compacts every result so the model
sees minified, pruned payloads. Handlers must not import FastAPI: they run in
the inference worker process.
"""
import difflib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import db, subtitle_store
from . import compact, h3_prompt, skills, store, transcript_edit
from .transcript_edit import join_text, merged_transcript, transcript_rows  # noqa: F401  (re-exported for callers and tests)

STRUCTURED = ('initial_frame', 'last_frame', 'video_analysis', 'subtitle_ocr', 'consistency_summary')
TEXT_RESULTS = {
    'production_brief': {'titles': ('Cut summary',), 'title': 'Cut summary', 'kind': 'cut_description'},
    'h3_prompt': {'titles': ('H3 animation prompt', 'MiniMax animation prompt'), 'title': 'H3 animation prompt', 'kind': 'generation_prompt'},
    'image_prompt': {'titles': ('Still image prompt',), 'title': 'Still image prompt', 'kind': 'generation_prompt'},
}
RESULT_TYPES = (*STRUCTURED, 'production_brief', 'review_answers', 'h3_prompt', 'image_prompt')
LABELS = {'initial_frame': 'initial frame', 'last_frame': 'last frame', 'video_analysis': 'video analysis', 'subtitle_ocr': 'subtitle OCR',
          'consistency_summary': 'consistency summary', 'production_brief': 'production brief', 'review_answers': 'review answers',
          'h3_prompt': 'H3 prompt', 'image_prompt': 'still image prompt', 'transcript': 'transcript'}
TEXT_LIMIT, STRUCTURED_LIMIT = 50000, 5000000


class ToolError(Exception):
    """A recoverable problem the model can fix by changing its arguments."""


@dataclass
class ToolContext:
    cut_id: str
    session_id: str = ''
    turn_id: str = ''
    provider: str = ''
    model: str = ''
    extra: dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[ToolContext, dict], Any]
    summary: Callable[[dict, Any], str] | None = None
    limit: int = 7000
    raw: bool = False
    must_pass: bool = False  # The runner sends the model back if it answers while this tool's last result has ok=false.


REGISTRY: dict[str, Tool] = {}


def tool(name, description, properties=None, required=(), summary=None, limit=7000, raw=False, must_pass=False):
    def register(handler):
        REGISTRY[name] = Tool(name, description, {'type': 'object', 'properties': properties or {}, 'required': list(required)}, handler, summary, limit, raw, must_pass)
        return handler
    return register


def schemas():
    return [{'type': 'function', 'function': {'name': item.name, 'description': item.description, 'parameters': item.parameters}} for item in REGISTRY.values()]


def validate(spec, args):
    if not isinstance(args, dict):
        raise ToolError('Arguments must be a JSON object')
    for name in spec.parameters['required']:
        if args.get(name) in (None, ''):
            raise ToolError(f'Missing required argument: {name}')
    clean = {}
    for name, value in args.items():
        schema = spec.parameters['properties'].get(name)
        if schema is None:
            continue  # Ignore unknown keys instead of failing a whole turn.
        if 'enum' in schema and value not in schema['enum']:
            raise ToolError(f"{name} must be one of: {', '.join(map(str, schema['enum']))}")
        expected = schema.get('type')
        if expected == 'string' and not isinstance(value, str):
            raise ToolError(f'{name} must be a string')
        if expected == 'array' and not isinstance(value, list):
            raise ToolError(f'{name} must be an array')
        if expected == 'boolean' and not isinstance(value, bool):
            raise ToolError(f'{name} must be true or false')
        clean[name] = value
    return clean


def execute(ctx, name, arguments):
    """Run a tool call. Returns (content for the model, UI summary, ok)."""
    spec = REGISTRY.get(name)
    if spec is None:
        return compact.dumps({'error': f'Unknown tool {name}. Available: {", ".join(REGISTRY)}'}), f'Unknown tool {name}', False
    try:
        args = json.loads(arguments or '{}') if isinstance(arguments, str) else arguments
        args = validate(spec, args)
        value = spec.handler(ctx, args)
        content = value if spec.raw else compact.encode(value, spec.limit)
        label = spec.summary(args, value) if spec.summary else name
        return content, label, True
    except json.JSONDecodeError:
        return compact.dumps({'error': 'Arguments were not valid JSON; call the tool again with a JSON object.'}), f'{name}: invalid arguments', False
    except ToolError as exc:
        return compact.dumps({'error': str(exc)}), f'{name}: {exc}', False


# ---------------------------------------------------------------- data access

def load_cut(cut_id):
    cut = db.one('''SELECT c.*,e.title AS episode_title,e.project_id,p.title AS project_title,p.synopsis
                    FROM cuts c JOIN episodes e ON e.id=c.episode_id JOIN projects p ON p.id=e.project_id WHERE c.id=?''', (cut_id,))
    if not cut:
        raise ToolError('The cut no longer exists')
    return cut


def structured_type(settings_json):
    return store.loads(settings_json, {}).get('structured_type')


def current_rows(conn, cut_id):
    return [dict(row) for row in conn.execute('SELECT * FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid DESC', (cut_id,))]


def find_current(conn, cut_id, result_type):
    """Return the current result for a type, or None; review answers are a pseudo-result."""
    if result_type == 'review_answers':
        row = conn.execute('SELECT content,updated_at FROM cut_brief_instructions WHERE cut_id=?', (cut_id,)).fetchone()
        if not row or not row['content'].strip():
            return None
        return {'id': None, 'content': row['content'], 'origin': 'manual', 'model_id': None, 'created_at': row['updated_at'], 'settings_json': '{}', 'context_json': '{}', 'kind': None, 'title': 'Review answers'}
    for row in current_rows(conn, cut_id):
        if result_type in STRUCTURED and structured_type(row['settings_json']) == result_type:
            return row
        if result_type in TEXT_RESULTS and row['title'] in TEXT_RESULTS[result_type]['titles'] and not structured_type(row['settings_json']):
            return row
    return None


def clip(seconds, cut):
    return round(seconds - cut['start_time'], 2)


def span(start, end):
    return f'{start:.2f}–{end:.2f}'


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def project_video(content, cut):
    windows = content.get('windows') if isinstance(content.get('windows'), list) else []
    views = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        source = window.get('source') or {}
        offset = (source.get('source_start') if is_number(source.get('source_start')) else cut['start_time']) - cut['start_time']
        analysis = window.get('analysis')
        if not isinstance(analysis, dict):
            views.append({'raw': analysis})
            continue
        events = []
        for event in analysis.get('events') or []:
            if not isinstance(event, dict):
                continue
            start, end = event.get('start_seconds'), event.get('end_seconds')
            uncertainty = event.get('uncertainty')
            events.append({'t': span(offset + start, offset + end) if is_number(start) and is_number(end) else 'unknown',
                           'subject': event.get('subject'), 'action': event.get('visual_action'), 'camera': event.get('camera'),
                           'uncertainty': None if str(uncertainty).strip().lower() in ('none', '') else uncertainty})
        view = {'overview': analysis.get('overview'), 'events': events, 'uncertainties': None if str(analysis.get('uncertainties')).strip().lower() == 'none' else analysis.get('uncertainties')}
        if len(windows) > 1 and is_number(source.get('source_end')):
            view = {'window': span(offset, source['source_end'] - cut['start_time']), **view}
        views.append(view)
    out = {'model': content.get('model')}
    if len(views) == 1:
        out.update(views[0])
    else:
        out['windows'] = views
    return out


def project(result_type, row, cut):
    """Compact, model-friendly view of a result. Times become clip-relative."""
    if result_type in STRUCTURED:
        try:
            content = json.loads(row['content'])
        except (TypeError, ValueError):
            return {'text': row['content']}
        if not isinstance(content, dict):
            return {'value': content}
        if result_type == 'video_analysis':
            return project_video(content, cut)
        if result_type in ('initial_frame', 'last_frame'):
            view = {key: value for key, value in content.items() if key not in ('subtitle_regions', 'frame_time')}
            if is_number(content.get('frame_time')):
                view['clip_time'] = clip(content['frame_time'], cut)
            return view
        if result_type == 'subtitle_ocr':
            events = [{'text': event.get('text'), 'first': clip(event['first_observed'], cut) if is_number(event.get('first_observed')) else None,
                       'last': clip(event['last_observed'], cut) if is_number(event.get('last_observed')) else None, 'language': event.get('language')}
                      for event in content.get('events') or [] if isinstance(event, dict)]
            return {'samples': len(content.get('samples') or []), 'ocr_performed': content.get('ocr_performed'), 'events': events or 'no subtitle text detected', 'language_note': content.get('language_note')}
        return content
    return {'text': row['content']}


def meta(row):
    return {'origin': row.get('origin'), 'model': row.get('model_id'), 'created': (row.get('created_at') or '')[:16]}


def detection_rows(cut):
    run = db.one('SELECT id,cut_start,cut_end FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['id'],))
    if not run:
        return None, []
    stale = run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time']
    rows = db.all_rows('''SELECT d.position,d.label_source,c.title AS name FROM character_detections d
                          LEFT JOIN context_items c ON c.id=d.character_context_id WHERE d.run_id=? ORDER BY d.rowid''', (run['id'],))
    return stale, rows


def cut_jobs(cut, limit=6):
    rows = db.all_rows('''SELECT pipeline,status,stage,error,created_at FROM jobs WHERE episode_id=? AND payload_json LIKE ?
                          ORDER BY rowid DESC LIMIT ?''', (cut['episode_id'], f'%{cut["id"]}%', limit))
    return [{'pipeline': row['pipeline'], 'status': row['status'], 'stage': row['stage'] if row['status'] != 'done' else None,
             'error': (row['error'] or '')[:240] or None, 'at': row['created_at'][:16]} for row in rows]


def status(cut_id):
    cut = load_cut(cut_id)
    with db.connect() as conn:
        found = {result_type: find_current(conn, cut_id, result_type) for result_type in RESULT_TYPES}
        revisions = {row['parent_id'] for row in conn.execute('SELECT parent_id FROM analysis_results WHERE cut_id=? AND parent_id IS NOT NULL', (cut_id,))}
    results = {}
    for result_type, row in found.items():
        if row is None:
            results[result_type] = 'missing'
        else:
            results[result_type] = {**meta(row), 'edited': row['id'] in revisions or row.get('origin') in ('manual_edit', 'assistant_edit') or None}
    stale, detections = detection_rows(cut)
    names = sorted({row['name'] for row in detections if row['name']})
    return cut, {
        'cut': {'number': cut['cut_index'] + 1 if cut['active'] else None, 'archived': not cut['active'] or None, 'start': cut['start_time'], 'end': cut['end_time'],
                'duration': round(cut['end_time'] - cut['start_time'], 3), 'analysis_enabled': bool(cut['analysis_enabled']), 'no_characters': bool(cut['no_characters']) or None,
                'title': cut['title'], 'notes': cut['notes'][:400]},
        'results': results,
        'characters': {'boxes': len(detections), 'named': names, 'unlabeled': sum(1 for row in detections if not row['name']), 'stale': stale or None} if stale is not None else 'not detected',
        'transcript_segments': len(transcript_rows(cut)),
        'subtitle_lines': subtitle_lines(cut),
        'recent_jobs': cut_jobs(cut),
    }


def subtitle_lines(cut):
    """Count of the episode's subtitle lines overlapping the cut, or None when subtitles were never extracted."""
    with db.connect() as conn:
        context = subtitle_store.cut_context(conn, cut, cut['episode_id'])
    return len(context['lines']) if context['extracted'] else None


def cut_card(cut_id):
    """Terse status block for the system prompt so simple questions need no tool call."""
    cut, data = status(cut_id)
    info = data['cut']
    label = f"Cut {info['number']}" if info['number'] else 'Archived cut'
    lines = [f"{label} of episode \"{cut['episode_title']}\" (project \"{cut['project_title']}\"): episode time {cut['start_time']:.3f}–{cut['end_time']:.3f}s, duration {info['duration']:.3f}s."]
    if cut['title'] or cut['notes']:
        lines.append(f"Cut title/notes: {cut['title']} {cut['notes'][:200]}".strip())
    parts = []
    for result_type, value in data['results'].items():
        if value == 'missing':
            parts.append(f'{result_type}=missing')
        else:
            source = value['origin'] if value['origin'] != 'model' else (value['model'] or 'model').split('/')[-1]
            parts.append(f"{result_type}=yes({source}{', edited' if value.get('edited') else ''})")
    lines.append('Results: ' + ', '.join(parts) + '.')
    characters = data['characters']
    if characters == 'not detected':
        lines.append('Character boxes: not detected.' if not cut['no_characters'] else 'Marked as having no characters.')
    else:
        lines.append(f"Character boxes: {characters['boxes']} ({', '.join(characters['named']) or 'none named'}; {characters['unlabeled']} unlabeled{'; stale' if characters.get('stale') else ''}).")
    lines.append(f"Transcript segments overlapping: {data['transcript_segments']}.")
    lines.append('Subtitle lines overlapping: ' + (str(data['subtitle_lines']) if data['subtitle_lines'] is not None else 'subtitles not extracted for this episode') + '.')
    active = [job for job in data['recent_jobs'] if job['status'] in ('queued', 'running')]
    failed = [job for job in data['recent_jobs'] if job['status'] == 'failed'][:2]
    if active:
        lines.append('Running/queued: ' + ', '.join(f"{job['pipeline']} ({job['stage']})" for job in active) + '.')
    if failed:
        lines.append('Recent failures: ' + '; '.join(f"{job['pipeline']}: {(job['error'] or '')[:120]}" for job in failed) + '.')
    return '\n'.join(lines)


# --------------------------------------------------------------------- tools

TYPE_PARAM = {'type': 'string', 'enum': list(RESULT_TYPES)}


@tool('get_cut_status', 'Analysis status for this cut: which results exist (origin/model/date), character boxes, transcript count, recent jobs and errors.',
      summary=lambda args, value: 'Checked analysis status')
def get_cut_status(ctx, args):
    return status(ctx.cut_id)[1]


@tool('get_result', 'Read one current result of this cut. Times are clip-relative seconds. Use fields to fetch only some top-level keys; raw=true returns the stored JSON (needed for set paths).',
      {'type': TYPE_PARAM, 'fields': {'type': 'array', 'items': {'type': 'string'}}, 'raw': {'type': 'boolean'}}, ('type',),
      summary=lambda args, value: f"Read {LABELS[args['type']]}" + (' (missing)' if isinstance(value, dict) and value.get('missing') else ''), limit=9000)
def get_result(ctx, args):
    cut = load_cut(ctx.cut_id)
    with db.connect() as conn:
        row = find_current(conn, ctx.cut_id, args['type'])
    if row is None:
        return {'type': args['type'], 'missing': True}
    if args.get('raw') and args['type'] in STRUCTURED:
        view = store.loads(row['content'], {})
    else:
        view = project(args['type'], row, cut)
    fields = args.get('fields')
    if fields and isinstance(view, dict) and set(view) != {'text'}:  # Text results have no fields.
        if isinstance(view.get('windows'), list):
            # Long cuts have several video windows; select inside each one.
            view = {**view, 'windows': [{key: value for key, value in window.items() if key in fields or key == 'window'} for window in view['windows']]}
        else:
            selected = {key: view[key] for key in fields if key in view}
            if not selected:
                raise ToolError(f"None of those fields exist. Available: {', '.join(view)}")
            view = selected
    return {'type': args['type'], **meta(row), 'content': view}


@tool('get_transcript', "Transcript lines overlapping this cut (clip-relative times; negative or beyond duration means the line crosses the cut boundary). `line` numbers are what propose_transcript_edit refers to.",
      summary=lambda args, value: f"Read transcript ({len(value.get('segments') or [])} segments)")
def get_transcript(ctx, args):
    cut = load_cut(ctx.cut_id)
    rows = merged_transcript(transcript_rows(cut))
    return {'cut_duration': round(cut['end_time'] - cut['start_time'], 3),
            'segments': [{'line': number, 't': span(clip(row['start_time'], cut), clip(row['end_time'], cut)), 'speaker': row['speaker_label'] or 'unknown',
                          'character': row['character_name'], 'source': row['origin'], 'text': row['text']} for number, row in enumerate(rows, 1)] or 'no transcript segments'}


@tool('get_subtitles', "Burned-in subtitle lines overlapping this cut (clip-relative times; negative or beyond duration means the line crosses the cut boundary). They were read from the whole episode in a single language that the user stated, independent of any cut. Read-only.",
      summary=lambda args, value: f"Read subtitles ({len(value.get('lines') or [])} lines)" if isinstance(value.get('lines'), list) else 'Read subtitles')
def get_subtitles(ctx, args):
    cut = load_cut(ctx.cut_id)
    with db.connect() as conn:
        context = subtitle_store.cut_context(conn, cut, cut['episode_id'])
    if not context['extracted']:
        return {'extracted': False, 'lines': 'subtitles have not been extracted for this episode; the user runs that step for the whole video'}
    return {'extracted': True, 'language': context['language'], 'cut_duration': round(cut['end_time'] - cut['start_time'], 3),
            'lines': [{'t': span(clip(line['start_time'], cut), clip(line['end_time'], cut)), 'text': line['text'], 'confidence': line['confidence'],
                       'crosses_cut': line['crosses_cut'] or None} for line in context['lines']] or 'no subtitle overlaps this cut'}


@tool('get_characters', 'Characters of this project (exact names, short notes) and the character boxes detected in this cut.',
      summary=lambda args, value: 'Read characters')
def get_characters(ctx, args):
    cut = load_cut(ctx.cut_id)
    rows = db.all_rows("SELECT title,content FROM context_items WHERE project_id=? AND kind='character' AND (episode_id IS NULL OR episode_id=?) ORDER BY title", (cut['project_id'], cut['episode_id']))
    stale, detections = detection_rows(cut)
    return {'characters': [{'name': row['title'], 'notes': row['content'][:240]} for row in rows] or 'no character notes',
            'boxes_in_cut': [{'frame': row['position'], 'name': row['name'] or 'unlabeled', 'label': row['label_source']} for row in detections] or 'none',
            'boxes_stale': stale or None, 'no_characters': bool(cut['no_characters']) or None}


@tool('get_context_notes', 'Project synopsis and context notes (world, scenario, chapter, object, general).',
      {'kind': {'type': 'string', 'enum': ['chapter_synopsis', 'chapter_happenings', 'world', 'scenario', 'object', 'general']}},
      summary=lambda args, value: 'Read context notes')
def get_context_notes(ctx, args):
    cut = load_cut(ctx.cut_id)
    where, params = "kind!='character'", [cut['project_id'], cut['episode_id']]
    if args.get('kind'):
        where, params = 'kind=?', [args['kind'], *params]
    rows = db.all_rows(f'SELECT kind,title,content FROM context_items WHERE {where} AND project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title', tuple(params))
    return {'synopsis': cut['synopsis'][:800] or 'none', 'notes': [{'kind': row['kind'], 'title': row['title'], 'content': row['content'][:500]} for row in rows] or f"no {args.get('kind') or 'context'} notes"}


@tool('get_neighbors', 'Previous/next cut: timing and their brief or summary text, for continuity only.',
      {'direction': {'type': 'string', 'enum': ['previous', 'next', 'both']}},
      summary=lambda args, value: f"Read {args.get('direction', 'both')} cut context")
def get_neighbors(ctx, args):
    cut = load_cut(ctx.cut_id)
    if not cut['active']:
        raise ToolError('Archived cuts have no timeline neighbours')
    direction = args.get('direction', 'both')
    out = {}
    for label, offset in (('previous', -1), ('next', 1)):
        if direction not in (label, 'both'):
            continue
        other = db.one('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (cut['episode_id'], cut['cut_index'] + offset))
        if not other:
            out[label] = None
            continue
        with db.connect() as conn:
            brief = find_current(conn, other['id'], 'production_brief')
            consistency = find_current(conn, other['id'], 'consistency_summary')
            video = find_current(conn, other['id'], 'video_analysis')
        text = brief['content'][:900] if brief else None
        if not text and consistency:
            text = store.loads(consistency['content'], {}).get('summary')
        if not text and video:
            text = project('video_analysis', video, other).get('overview')
        out[label] = {'number': other['cut_index'] + 1, 'duration': round(other['end_time'] - other['start_time'], 2), 'summary': text or 'no analysis'}
    return out


@tool('find_in_results', "Case-insensitive search across this cut's current results; returns where a text occurs with short snippets. Use before correcting a name everywhere.",
      {'query': {'type': 'string'}}, ('query',),
      summary=lambda args, value: f"Searched results for “{args['query'][:40]}” ({sum(item['count'] for item in value['matches']) if isinstance(value.get('matches'), list) else 0} hits)")
def find_in_results(ctx, args):
    query = args['query'].strip()
    if len(query) < 2:
        raise ToolError('Query must contain at least two characters')
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches = []
    with db.connect() as conn:
        rows = {result_type: find_current(conn, ctx.cut_id, result_type) for result_type in RESULT_TYPES}
    for result_type, row in rows.items():
        if not row:
            continue
        hits = list(pattern.finditer(row['content']))
        if hits:
            snippets = ['…' + row['content'][max(0, hit.start() - 50):hit.end() + 50].replace('\n', ' ') + '…' for hit in hits[:3]]
            matches.append({'type': result_type, 'count': len(hits), 'snippets': snippets})
    return {'query': query, 'matches': matches or 'no matches'}


@tool('load_skill', 'Load the full guide for a skill listed in the system prompt. Load it before doing that kind of task.',
      {'name': {'type': 'string', 'enum': skills.names()}}, ('name',),
      summary=lambda args, value: f"Loaded skill: {(skills.get(args['name']) or {}).get('title', args['name'])}", raw=True)
def load_skill(ctx, args):
    skill = skills.get(args['name'])
    if not skill:
        raise ToolError('Unknown skill')
    return skill['body']


# ------------------------------------------------------------------ proposals

def set_path(data, path, value):
    parts = [part for part in path.split('.') if part != '']
    if not parts:
        raise ToolError('set path must not be empty')
    target = data
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(target, list):
            if not part.isdigit() or int(part) >= len(target):
                raise ToolError(f"Path {path}: '{part}' is not a valid list index")
            if last:
                target[int(part)] = value
            else:
                target = target[int(part)]
        elif isinstance(target, dict):
            if last:
                if part not in target:
                    raise ToolError(f"Path {path}: key '{part}' does not exist (available: {', '.join(map(str, target))})")
                target[part] = value
            elif part not in target:
                raise ToolError(f"Path {path}: key '{part}' does not exist (available: {', '.join(map(str, target))})")
            else:
                target = target[part]
        else:
            raise ToolError(f"Path {path}: cannot descend into a {type(target).__name__}")


def replace_text(text, find, replace, count):
    occurrences = text.count(find)
    if count:
        return text.replace(find, replace, count), min(occurrences, count)
    return text.replace(find, replace), occurrences


def replace_in_json(value, find, replace, remaining):
    """Replace inside string values only, so the JSON structure stays valid."""
    if isinstance(value, str):
        if remaining[0] == 0 or find not in value:
            return value
        hits = value.count(find) if remaining[0] < 0 else min(value.count(find), remaining[0])
        if remaining[0] > 0:
            remaining[0] -= hits
        remaining[1] += hits
        return value.replace(find, replace, hits)
    if isinstance(value, list):
        return [replace_in_json(item, find, replace, remaining) for item in value]
    if isinstance(value, dict):
        return {key: replace_in_json(item, find, replace, remaining) for key, item in value.items()}
    return value


def build_change(result_type, base, args):
    replacements = args.get('replacements') or []
    sets = args.get('set') or []
    content = args.get('content')
    if content is None and not replacements and not sets:
        raise ToolError('Provide replacements, set, or content')
    if result_type in STRUCTURED:
        if base is None:
            raise ToolError(f'There is no {LABELS[result_type]} result to correct; run that analysis first')
        if content is not None:
            try:
                data = json.loads(content)
            except ValueError as exc:
                raise ToolError(f'content must be a JSON object for {result_type}: {exc}') from exc
            if not isinstance(data, dict):
                raise ToolError('content must be a JSON object')
        else:
            data = store.loads(base['content'], {})
            if not data:
                raise ToolError(f'The saved {LABELS[result_type]} is not a JSON object; pass the full corrected JSON as content')
        applied = []
        for item in sets:
            if not isinstance(item, dict) or not isinstance(item.get('path'), str) or 'value' not in item:
                raise ToolError('Each set item needs path and value')
            set_path(data, item['path'], item['value'])
            applied.append({'set': item['path']})
        for item in replacements:
            find, replace = validate_replacement(item)
            counter = [int(item.get('count') or 0) or -1, 0]
            data = replace_in_json(data, find, replace, counter)
            if counter[1] == 0:
                raise ToolError(f'Text not found in {LABELS[result_type]}: "{find[:80]}". Use find_in_results or get_result to copy the exact wording.')
            applied.append({'find': find, 'replaced': counter[1]})
        new = json.dumps(data, ensure_ascii=False)
        if len(new) > STRUCTURED_LIMIT:
            raise ToolError('The corrected result is too large')
        return new, applied or [{'content': 'replaced'}]
    if sets:
        raise ToolError(f'set only applies to structured JSON results; use replacements or content for {result_type}')
    text = content if content is not None else (base['content'] if base else None)
    if text is None:
        raise ToolError(f'There is no saved {LABELS[result_type]} yet; pass the full text as content')
    applied = [{'content': 'replaced'}] if content is not None else []
    for item in replacements:
        find, replace = validate_replacement(item)
        text, hits = replace_text(text, find, replace, int(item.get('count') or 0))
        if not hits:
            raise ToolError(f'Text not found in {LABELS[result_type]}: "{find[:80]}". Copy the exact wording from get_result.')
        applied.append({'find': find, 'replaced': hits})
    text = text.strip()
    if not text:
        raise ToolError('The corrected text is empty')
    if len(text) > TEXT_LIMIT:
        raise ToolError(f'Text results may contain at most {TEXT_LIMIT} characters')
    return text, applied


def validate_replacement(item):
    if not isinstance(item, dict) or not isinstance(item.get('find'), str) or not isinstance(item.get('replace'), str) or not item['find']:
        raise ToolError('Each replacement needs non-empty find and a replace string')
    return item['find'], item['replace']


def normalized(result_type, text):
    if result_type in STRUCTURED:
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)
        except (TypeError, ValueError):
            return text or ''
    return (text or '').strip()


def pretty(result_type, text):
    if result_type in STRUCTURED:
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return text or ''
    return text or ''


def diff_segments(old, new, limit=60000):
    """Word-level diff segments for the proposal card."""
    if len(old) + len(new) > limit:
        lines = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=2)
        return [{'op': 'equal', 'text': '\n'.join(list(lines)[2:])}]
    # ASCII words, whitespace runs and single other characters: JSON quotes stay out of
    # word edits and Japanese text (no spaces) is compared character by character.
    split = lambda text: re.findall(r'[A-Za-z0-9_]+|\s+|[^\sA-Za-z0-9_]', text)
    a, b = split(old), split(new)
    opcodes = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    segments, deleted, inserted = [], '', ''

    def close():
        nonlocal deleted, inserted
        if deleted:
            segments.append({'op': 'delete', 'text': deleted})
        if inserted:
            segments.append({'op': 'insert', 'text': inserted})
        deleted = inserted = ''

    for index, (op, i1, i2, j1, j2) in enumerate(opcodes):
        if op == 'equal':
            text = ''.join(a[i1:i2])
            between = 0 < index < len(opcodes) - 1 and (deleted or inserted)
            if between and len(text) <= 2 and not text.strip():
                # Merge "a person" -> "Tomoe Koga" into one edit instead of word-by-word pairs.
                deleted, inserted = deleted + text, inserted + text
                continue
            close()
            segments.append({'op': 'equal', 'text': text})
        else:
            deleted += ''.join(a[i1:i2])
            inserted += ''.join(b[j1:j2])
    close()
    return segments


def short_diff(old, new, limit=700):
    parts = []
    for segment in diff_segments(old, new):
        if segment['op'] == 'delete':
            parts.append(f"-[{segment['text'].strip()[:160]}]")
        elif segment['op'] == 'insert':
            parts.append(f"+[{segment['text'].strip()[:160]}]")
    text = ' '.join(part for part in parts if part not in ('-[]', '+[]'))
    return text[:limit] + ('…' if len(text) > limit else '')


def create_proposal(ctx, result_type, base, content, summary, diff_base=None, diff_label=None, replace_pending=False, before=None, payload=None):
    """Store a pending proposal; ``diff_base`` overrides what the card compares against.

    Transcript proposals have no base result: ``before`` is the readable current text and ``payload`` the plan applied later.
    """
    if result_type == 'review_answers':
        title, kind = 'Review answers', 'review_answers'
    elif result_type == 'transcript':
        title, kind = 'Transcript', 'transcript'
    elif result_type in TEXT_RESULTS:
        title, kind = (base['title'] if base else TEXT_RESULTS[result_type]['title']), (base['kind'] if base else TEXT_RESULTS[result_type]['kind'])
    else:
        title, kind = base['title'], base['kind']
    proposal_id = str(uuid.uuid4())
    with db.connect() as conn:
        if replace_pending and ctx.turn_id:
            # A re-checked draft replaces this turn's earlier unsaved draft of the same kind.
            conn.execute("UPDATE assistant_proposals SET status='discarded',resolved_at=CURRENT_TIMESTAMP WHERE turn_id=? AND target=? AND status='pending'", (ctx.turn_id, result_type))
        conn.execute('''INSERT INTO assistant_proposals(id,session_id,turn_id,cut_id,target,title,kind,base_result_id,base_content,content,summary,model_id,provider,diff_base,diff_label,payload_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                     (proposal_id, ctx.session_id, ctx.turn_id or None, ctx.cut_id, result_type, title, kind, base['id'] if base else None,
                      base['content'] if base else (before or ''), content, summary[:500], ctx.model, ctx.provider, diff_base, diff_label,
                      json.dumps(payload, ensure_ascii=False) if payload is not None else None))
        if ctx.turn_id:
            store.bump(conn, ctx.turn_id)
    return proposal_id


@tool('propose_result_edit', 'Propose a correction or new text for a result. The user must apply it; nothing is saved until then. Prefer replacements (exact find text, all occurrences unless count) or set (JSON path like windows.0.analysis.events.0.subject) over full content.',
      {'type': {'type': 'string', 'enum': [*STRUCTURED, 'production_brief', 'review_answers', 'h3_prompt', 'image_prompt']},
       'replacements': {'type': 'array', 'items': {'type': 'object', 'properties': {'find': {'type': 'string'}, 'replace': {'type': 'string'}, 'count': {'type': 'integer'}}, 'required': ['find', 'replace']}},
       'set': {'type': 'array', 'items': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'value': {}}, 'required': ['path', 'value']}},
       'content': {'type': 'string', 'description': 'Full new text (or JSON object for structured results)'},
       'summary': {'type': 'string', 'description': 'One line describing the change'}},
      ('type', 'summary'),
      summary=lambda args, value: f"Proposed {LABELS[args['type']]} change")
def propose_result_edit(ctx, args):
    result_type = args['type']
    if result_type == 'h3_prompt':
        raise ToolError('H3 prompts go through check_h3_prompt (it validates the format and shows the prompt to the user). Do not propose it again here.')
    with db.connect() as conn:
        base = find_current(conn, ctx.cut_id, result_type)
    new, changed = build_change(result_type, base, args)
    base_text = base['content'] if base else ''
    if new == normalized(result_type, base_text):
        raise ToolError('The proposal does not change the current content')
    proposal_id = create_proposal(ctx, result_type, base, new, args['summary'])
    return {'proposal_id': proposal_id, 'status': 'pending_user_approval', 'target': result_type, 'applied': changed,
            'diff': short_diff(pretty(result_type, base_text), pretty(result_type, new)),
            'note': 'Not saved yet. Tell the user to review and apply it on the proposal card.'}


# ----------------------------------------------------------------- transcript

LINE_ITEM = {'type': 'object', 'properties': {
    'text': {'type': 'string'}, 'character': {'type': 'string', 'description': 'Exact character name from get_characters'},
    'speaker_label': {'type': 'string'}, 'start': {'type': 'number', 'description': 'Clip-relative seconds'}, 'end': {'type': 'number', 'description': 'Clip-relative seconds'}},
    'required': ['text']}
TRANSCRIPT_EDIT = {'type': 'object', 'properties': {
    'op': {'type': 'string', 'enum': ['update', 'rewrite', 'delete', 'insert']},
    'line': {'type': 'integer', 'description': 'Line number from get_transcript (update, delete, rewrite: first line)'},
    'to_line': {'type': 'integer', 'description': 'rewrite: last line replaced; update: last line of a speaker-only change (default: line)'},
    'text': {'type': 'string', 'description': 'update: the full corrected text of the line'},
    'replacements': {'type': 'array', 'items': {'type': 'object', 'properties': {'find': {'type': 'string'}, 'replace': {'type': 'string'}}, 'required': ['find', 'replace']},
                     'description': 'update: fix words inside the line (find must be exact text from get_transcript)'},
    'character': {'type': 'string', 'description': 'update: link the line to this character (exact name)'},
    'clear_character': {'type': 'boolean', 'description': 'update: remove the character link'},
    'speaker_label': {'type': 'string', 'description': 'update: set the speaker label (for a voice that is not a character, e.g. Narrator)'},
    'lines': {'type': 'array', 'items': LINE_ITEM, 'description': 'rewrite / insert: the new lines in time order'}},
    'required': ['op']}


@tool('propose_transcript_edit', "Propose corrections to this cut's audio transcript: fix wrong text, set or fix the speaker character, split/merge/replace lines, remove a bogus line or add a missing one. Lines are the numbered `line`s of get_transcript. The user must apply the proposal; nothing is saved until then. Put every change for the cut into ONE call (a later call in the same turn replaces the earlier unsaved proposal).",
      {'edits': {'type': 'array', 'items': TRANSCRIPT_EDIT, 'description': 'update: line[, to_line for a speaker change over several lines] + text | replacements | character | clear_character | speaker_label. rewrite: line[, to_line] + lines (replaces those lines; new lines without times share the old time span by text length and inherit the old speaker unless character is given). delete: line. insert: lines with start and end.'},
       'summary': {'type': 'string', 'description': 'One line describing the change'}},
      ('edits', 'summary'),
      summary=lambda args, value: f"Proposed transcript change ({len(args.get('edits') or [])} edit{'s' if len(args.get('edits') or []) != 1 else ''})",
      limit=5000)
def propose_transcript_edit(ctx, args):
    cut = load_cut(ctx.cut_id)
    if not cut['active']:
        raise ToolError('Archived cuts have no transcript to edit')
    episode = db.one('SELECT duration FROM episodes WHERE id=?', (cut['episode_id'],)) or {}
    rows = transcript_rows(cut)
    try:
        plan = transcript_edit.build_plan(cut, episode, rows, transcript_edit.project_characters(cut), args['edits'])
    except transcript_edit.PlanError as exc:
        raise ToolError(str(exc)) from exc
    replaced = db.one("SELECT COUNT(*) AS n FROM assistant_proposals WHERE turn_id=? AND target='transcript' AND status='pending'", (ctx.turn_id,))['n'] if ctx.turn_id else 0
    proposal_id = create_proposal(ctx, 'transcript', None, plan['after'], args['summary'], replace_pending=True, before=plan['before'],
                                  payload={'episode_id': plan['episode_id'], 'ops': plan['ops']})
    # Other lines that carry the same diarization label may be the same speaker; tell the model instead of guessing.
    lines = transcript_edit.merged_transcript(rows)
    linked = {op['id'] for op in plan['ops'] if op['op'] == 'revise' and op['character_context_id']}
    touched = {op['id'] for op in plan['ops'] if op.get('id')}
    labels = {line['speaker_label'] for line in lines if line['speaker_label'] and set(line['ids']) & linked}
    siblings = [number for number, line in enumerate(lines, 1) if line['speaker_label'] in labels and not set(line['ids']) & touched and not line['character_name']]
    out = {'proposal_id': proposal_id, 'status': 'pending_user_approval', 'target': 'transcript', 'changes': plan['changes'], 'warnings': plan['warnings'] or None,
           'diff': short_diff(plan['before'], plan['after']),
           'note': 'Not saved yet. Tell the user to review and apply it on the proposal card.'}
    if siblings:
        out['unlinked_lines_with_same_voice_label'] = siblings
    if replaced:
        out['replaced_earlier_unsaved_proposal'] = True
    return out


# ------------------------------------------------------------------ H3 prompts

def original_h3(ctx):
    """The prompt being fixed: the latest one pasted in this chat, else the saved H3 prompt."""
    if ctx.session_id:
        for row in db.all_rows("SELECT content FROM assistant_messages WHERE session_id=? AND role='user' ORDER BY seq DESC", (ctx.session_id,)):
            prompt = h3_prompt.extract(row['content'])
            if prompt:
                return prompt, 'your prompt'
    with db.connect() as conn:
        saved = find_current(conn, ctx.cut_id, 'h3_prompt')
    prompt = h3_prompt.extract(saved['content']) if saved else None
    return (prompt, 'the saved H3 prompt') if prompt else (None, None)


def spoken_key(text):
    return re.sub(r'[\s。、，,．.!！?？…「」『』“”"\'()（）\-–—~〜]', '', text)


def dialogue_errors(ctx, prompt, original):
    """New <d> lines must quote the transcript words exactly; lines kept from the original are the user's."""
    kept = set(re.findall(r'<d>\[[^\]]*\]\s*(.*?)</d>', original or '', re.S))
    lines = merged_transcript(transcript_rows(load_cut(ctx.cut_id)))
    spoken = spoken_key(''.join(line['text'] for line in lines))
    errors = []
    for text in re.findall(r'<d>\[[^\]]*\]\s*(.*?)</d>', prompt, re.S):
        if text in kept or (spoken_key(text) and spoken_key(text) in spoken):
            continue
        quoted = ' / '.join(f"{line['text']}" for line in lines[:6]) or 'no transcript'
        errors.append(f'Dialogue "{text[:80]}" does not match the transcript. Copy the words exactly (transcript lines: {quoted[:400]}).')
    return errors


def speaker_warnings(ctx, prompt):
    """Compare speaking subjects with the cut's transcript links; a visible-only guess is not a link."""
    speaking = h3_prompt.speaking_subjects(prompt)
    if not speaking:
        return []
    cut = load_cut(ctx.cut_id)
    rows = transcript_rows(cut)
    linked = {row['character_name'] for row in rows if row['character_name']}
    names = h3_prompt.subject_names(prompt)
    with db.connect() as conn:
        answers = find_current(conn, ctx.cut_id, 'review_answers')
    confirmed = (answers or {}).get('content', '').lower()
    warnings = []
    for number in sorted(speaking):
        name = names.get(number, f'<Subject {number}>')
        if name in linked or (name.lower() in confirmed and any(word in confirmed for word in ('speak', 'says', 'line', 'voice', 'dialogue'))):
            continue
        speakers = ', '.join(sorted({row['speaker_label'] or 'unknown' for row in rows})) or 'none'
        detail = f"linked to {', '.join(sorted(linked))}" if linked else f'not linked to any character (speaker {speakers})'
        warnings.append(f'<Subject {number}> ({name}) is given speech, but the transcript is {detail} and no review answer confirms it. Use an off-screen/unconfirmed voice with closed lips, or list the attribution under Open questions.')
    return warnings


@tool('check_h3_prompt', "Check a complete corrected MiniMax H3 prompt against the H3 six-section format and the original prompt (found automatically in this chat or the saved H3 prompt): reference labels and numbers, order, beats, length. Fix every error and call again. When it passes the user sees the prompt in a card with Copy and Save, so do not repeat it in your answer.",
      {'prompt': {'type': 'string', 'description': 'The full corrected prompt'},
       'summary': {'type': 'string', 'description': 'One line: what changed'},
       'original': {'type': 'string', 'description': 'Only when the prompt being fixed is not in this chat'}}, ('prompt',),
      summary=lambda args, value: ('H3 prompt check passed' + (f" · {len(value['warnings'])} warning(s)" if value.get('warnings') else '')) if value.get('ok') else f"H3 prompt check: {len(value.get('errors') or [])} error(s)",
      limit=4000, must_pass=True)
def check_h3_prompt(ctx, args):
    # The user's pasted prompt is authoritative; the model's own copy is only a fallback.
    original, source = original_h3(ctx)
    if original is None and args.get('original'):
        original, source = h3_prompt.strip_fences(args['original']), 'the given original'
    result = h3_prompt.check(args['prompt'], original)
    result['errors'].extend(dialogue_errors(ctx, result['prompt'], original))
    result['ok'] = not result['errors']
    result['warnings'].extend(speaker_warnings(ctx, args['prompt']))
    out = {'ok': result['ok'], 'errors': result['errors'], 'warnings': result['warnings'], 'stats': result['stats'], 'compared_with': source or 'nothing (new prompt)'}
    if not result['ok']:
        out['note'] = 'Fix every error, then call check_h3_prompt again with the full prompt.'
        return out
    content = result['prompt']
    with db.connect() as conn:
        base = find_current(conn, ctx.cut_id, 'h3_prompt')
    if base and base['content'].strip() == content:
        out['note'] = 'Identical to the saved H3 prompt; nothing to save. Tell the user it is unchanged.'
        return out
    out['proposal_id'] = create_proposal(ctx, 'h3_prompt', base, content, args.get('summary') or 'Corrected H3 prompt',
                                         diff_base=original, diff_label=source, replace_pending=True)
    out['note'] = ('DONE: the user now sees the full prompt in a card with Copy and Save buttons. Your reply must NOT contain the prompt or a code block, '
                   'and must not call propose_result_edit for it. Reply only with Changes, Kept and Open questions (include the warnings you left).')
    return out
