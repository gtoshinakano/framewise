"""Load cut evidence from one SQLite snapshot for API readers and queue workers."""
import json

from . import db, media, subtitle_store


def json_object_or_empty(value):
    """Keep damaged optional metadata from making the episode editor inaccessible."""
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def serialize_cut(cut: dict):
    cut["results"] = json_object_or_empty(cut.pop("results_json"))
    cut["active"] = bool(cut["active"])
    cut["analysis_enabled"] = bool(cut["analysis_enabled"])
    cut["no_characters"] = bool(cut.get("no_characters"))
    for flag in ("has_analysis_results", "has_summary", "has_boxes", "has_characters", "has_unlabeled_boxes", "has_target_boxes", "has_unlabeled_target_boxes", "has_visual_results", "has_consistency_summary", "last_job_failed"):
        if flag in cut:
            cut[flag] = bool(cut[flag])
    fallback_first, fallback_last = media.cut_frame_times(cut["start_time"], cut["end_time"])
    cut["first_frame_time"] = cut.get("first_frame_time") if cut.get("first_frame_time") is not None else fallback_first
    cut["last_frame_time"] = cut.get("last_frame_time") if cut.get("last_frame_time") is not None else fallback_last
    return cut


def serialize_transcript(segment: dict):
    segment["deleted"] = bool(segment["deleted"])
    return segment


def subtitles(conn, cut, episode_id):
    """Subtitle lines overlapping the cut from the episode-wide extraction, with the language they are in."""
    context = subtitle_store.cut_context(conn, cut, episode_id)
    band = context.pop("band")
    context["strip"] = {"top": band["top"], "bottom": band["bottom"]} if band else None
    return context


def load(conn, cut_id):
    """Return cut context with current neighboring results and overlapping transcript segments."""
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,), conn=conn)
    if not cut:
        return None
    episode = db.one("SELECT id,project_id,title,duration FROM episodes WHERE id=?", (cut["episode_id"],), conn=conn)
    project = db.one("SELECT id,title,synopsis FROM projects WHERE id=?", (episode["project_id"],), conn=conn)
    previous = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] - 1), conn=conn) if cut["active"] else None
    following = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] + 1), conn=conn) if cut["active"] else None
    neighbors = {}
    for label, neighbor in (("previous", previous), ("next", following)):
        neighbors[label] = db.all_rows("SELECT kind,title,content,origin,model_id FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid DESC", (neighbor["id"],), conn=conn) if neighbor else []
    return {
        "project": {"id": project["id"], "title": project["title"], "synopsis": project["synopsis"]},
        "episode": {"id": episode["id"], "title": episode["title"], "duration": episode["duration"]},
        "cut": serialize_cut(cut),
        "previous_cut": serialize_cut(previous) if previous else None,
        "next_cut": serialize_cut(following) if following else None,
        "neighbor_results": neighbors,
        "chunks": db.all_rows("SELECT * FROM chunks WHERE episode_id=? AND start_time<? AND end_time>? ORDER BY chunk_index", (episode["id"], cut["end_time"], cut["start_time"]), conn=conn),
        "context_items": db.all_rows("SELECT * FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title", (project["id"], episode["id"]), conn=conn),
        "transcript_segments": [serialize_transcript(segment) for segment in db.all_rows("SELECT t.*,c.title AS character_name FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? ORDER BY t.start_time,t.id", (episode["id"], cut["end_time"], cut["start_time"]), conn=conn)],
        "subtitles": subtitles(conn, cut, episode["id"]),
    }


def selected(conn, cut_id, config, *, strict=False):
    context = load(conn, cut_id)
    if context is None:
        return None
    valid = {item["id"] for item in context["context_items"]}
    ids = set(config.get('context_ids') or ())
    if strict and not ids.issubset(valid):
        raise ValueError("One or more selected context items no longer apply to this cut")
    detections = character_detections(conn, cut_id)
    detected_ids = {d['character_context_id'] for d in detections['detections'] if d['character_context_id']} if detections and not detections['stale'] else set()
    context["context_items"] = [item for item in context["context_items"] if item["id"] in ids or (item['kind'] == 'character' and item['id'] in detected_ids)]
    context["additional_context"] = config.get('additional_context', '')
    if detections and not detections['stale']:
        context['character_detections'] = {'run_id': detections['id'], 'model_id': detections['model_id'], 'model_revision': detections['model_revision'],
            'frames': [{'position': frame['position'], 'time': frame['time'], 'image_size': frame['image_size']} for frame in detections['frames']],
            'detections': [{'id': d['id'], 'position': d['position'], 'box': d['box'], 'character_context_id': d['character_context_id'],
                            'character_name': d['character_name'], 'label_source': d['label_source'], 'match_reason': d['match_reason'], 'match_model_id': d['match_model_id'], 'match_model_revision': d['match_model_revision']} for d in detections['detections']]}
    return context


def character_detections(conn, cut_id):
    cut = db.one('SELECT * FROM cuts WHERE id=?', (cut_id,), conn=conn)
    if not cut:
        return None
    run = db.one('SELECT * FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,), conn=conn)
    if not run:
        return None
    run['frames'] = json.loads(run.pop('frames_json'))
    run['stale'] = run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time']
    boxes = db.all_rows('''SELECT d.*,c.title AS character_name,EXISTS(SELECT 1 FROM character_image_references r WHERE r.detection_id=d.id) AS is_reference
                           FROM character_detections d LEFT JOIN context_items c ON c.id=d.character_context_id WHERE d.run_id=? ORDER BY d.rowid''', (run['id'],), conn=conn)
    names = {row['id']: row['title'] for row in db.all_rows("SELECT ci.id,ci.title FROM context_items ci JOIN cuts cu ON cu.id=? JOIN episodes e ON e.id=cu.episode_id WHERE ci.project_id=e.project_id AND ci.kind='character'", (cut_id,), conn=conn)}
    for box in boxes:
        box['box'] = json.loads(box.pop('box_json'))
        box['is_reference'] = bool(box['is_reference'])
        # Ranked suggestions from the last recognition pass, for one-click labelling.
        box['candidates'] = [{**item, 'character_name': names[item['character_id']]}
                             for item in json.loads(box.pop('candidates_json') or '[]') if item['character_id'] in names]
    run['detections'] = boxes
    return run


def unsettled_characters(cut, context):
    if cut['no_characters']:
        return None
    detections = context.get('character_detections') or {}
    if not any(frame.get('position') == 'first' for frame in detections.get('frames', [])):
        return 'not_detected'
    if any(person['position'] == 'first' and person['character_context_id'] is None for person in detections['detections']):
        return 'unknown_characters'
    return None


def visual(conn, job):
    """Load one cut's visual inputs with the choice snapshot from the queued request."""
    from . import character_detection, job_kinds, keyframe_analysis, keyframes, structured_results

    payload = job.get('payload') or json.loads(job['payload_json'])
    cut_id = job['unit_id']
    if payload.get('workflow') != 'single_cut_v2':
        return legacy_visual(conn, job, payload, cut_id)
    request = next((item for item in payload['cuts'] if item['cut_id'] == cut_id), None)
    if request is None:
        request = {'config': payload['default_config']}
    config = request['config']
    context = selected(conn, cut_id, config)
    if not context or not context['cut']['active'] or not context['cut']['analysis_enabled']:
        return {'cut_id': cut_id, 'skip': 'Cut archived, removed or disabled'}
    cut = context['cut']
    if payload.get('characters_resolved'):
        problem = unsettled_characters(cut, context)
        if problem:
            return {'cut_id': cut_id, 'skip': 'Character labels still need review' if problem == 'unknown_characters' else 'Character boxes are missing'}
    if job['priority_class'] == 'interactive' and context.get('character_detections'):
        context['character_detections']['labels_confirmed_by_queue'] = True
    rows = db.all_rows("""SELECT id,kind,title,content,origin,settings_json,stale_reason FROM analysis_results
        WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid""", (cut_id,), conn=conn)
    base_ids = {kind: row['id'] for row in rows if (kind := structured_results.structured_type(row['settings_json']))}
    manual_bases = {kind for row in rows if row['origin'] in ('manual', 'manual_edit', 'assistant_edit')
                    if (kind := structured_results.structured_type(row['settings_json']))}
    stages = request.get('stages') or job_kinds.stages_of(payload)
    if payload.get('only_missing'):
        current = {structured_results.structured_type(row['settings_json']) for row in rows if not row['stale_reason']}
        stages = [stage for stage in stages if stage not in current]
    if not stages:
        return {'cut_id': cut_id, 'skip': 'Requested stages are current'}
    context['current_results'] = [{key: row[key] for key in ('id', 'kind', 'title', 'content', 'origin')} for row in rows]
    labeled = {person['character_context_id'] for person in context.get('character_detections', {}).get('detections', []) if person['character_context_id']}
    labeled.update(entry['id'] for entry in context['context_items'] if entry['kind'] == 'character')
    references = character_detection.select_analysis_references(character_detection.reference_snapshots(conn, context['project']['id']), labeled)
    key, selection, targets = None, None, None
    if 'keyframe_analysis' in stages:
        _, _, key, selection = keyframes.current_selection(cut_id, conn)
        if key and not cut['no_characters']:
            targets = keyframe_analysis.target_run(cut_id, key, cut, conn)
    return {'cut_id': cut_id, 'config': config, 'context': context, 'stages': stages, 'base_result_ids': base_ids,
            'manual_bases': manual_bases,
            'character_references': references, 'selection_key': key, 'keyframe_selection': selection, 'target_run': targets}


def legacy_visual(conn, job, payload, cut_id):
    """Old raw frame workflows use current context and preserve the user's explicit prompt overrides."""
    from .inference_worker import FIELDS
    item = next(item for item in payload['cuts'] if item['cut_id'] == cut_id)
    context = selected(conn, cut_id, item['config'])
    if not context or not context['cut']['active'] or not context['cut']['analysis_enabled']:
        return {'cut_id': cut_id, 'skip': 'Cut archived, removed or disabled'}
    cut = context['cut']
    boundaries = [cut['start_time'], cut['end_time']]
    saved = (item.get('context') or {}).get('cut') or {}
    original_boundaries = item.get('source_bounds') or [saved.get('start_time'), saved.get('end_time')]
    checkpoint = db.one('SELECT source_cut_json FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job['id'], cut_id), conn=conn)
    checkpoint_bounds = json.loads(checkpoint['source_cut_json']) if checkpoint and checkpoint['source_cut_json'] else original_boundaries
    if payload.get('text_only') and boundaries != checkpoint_bounds:
        return {'cut_id': cut_id, 'skip': 'Cut boundaries changed since the saved frame evidence'}
    titles = {title for _, title, _ in FIELDS}
    bases = {}
    for row in conn.execute('SELECT id,title,origin FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid', (cut_id,)):
        if row['title'] in titles:
            bases[row['title']] = {'id': row['id'], 'origin': row['origin']}
    return {'cut_id': cut_id, 'context': context, 'bases': bases, 'bounds': boundaries,
            'request_override': item.get('request_override'), 'config': item['config'],
            'reuse_observations': boundaries == checkpoint_bounds}


def visual_fingerprint(conn, job):
    from .job_inputs import fingerprint

    item = job.get('gathered') or visual(conn, job)
    if 'reuse_observations' in item:
        # Recording a checkpoint of this job does not change the underlying inputs.
        return fingerprint({key: value for key, value in item.items() if key != 'reuse_observations'})
    context = item.get('context')
    if not context:
        return fingerprint(item)
    # The visual prompts use only the cut, marked people and the subtitle strip.
    # Fresh neighboring results and transcript content stay available to text jobs.
    cut = {key: context['cut'].get(key) for key in ('id', 'start_time', 'end_time', 'analysis_enabled', 'no_characters')}
    subtitle = {key: context['subtitles'].get(key) for key in ('strip', 'language')}
    return fingerprint({'cut': cut, 'subtitles': subtitle, 'detections': context.get('character_detections'),
                        'context_items': context['context_items'], 'references': item['character_references'],
                        'base_ids': {stage: item['base_result_ids'].get(stage) for stage in item['stages']},
                        'selection': item['selection_key'], 'target_run': item['target_run']})
