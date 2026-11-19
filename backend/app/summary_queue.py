"""Queue consistency summaries and resolve their evidence at execution."""
import json
from . import analysis_context, consistency_evidence, consistency_summary, db, job_inputs, prompt_context, structured_results, worker


def missing_evidence(by_type, pending_beats=False):
    """Why a cut cannot be reviewed yet, or None.

    A cut needs its first frame and its motion: the keyframe analysis with performance beats built from it (since
    2026-11-18), or for a cut analysed before that only an older video analysis. ``pending_beats`` says a chained beats
    job will write them before the summary runs. A cut with keyframes never falls back to an older video analysis.
    """
    if (by_type.get('initial_frame') or {}).get('content') is None:
        return 'Initial frame and keyframe analysis results are required'
    if 'keyframe_analysis' in by_type:
        if pending_beats or prompt_context.beats_current(by_type):
            return None
        return 'The performance beats of this cut are missing or older than its keyframe analysis; write them first'
    video = (by_type.get('video_analysis') or {}).get('content')
    if video is None:
        return 'Initial frame and keyframe analysis results are required'
    if not prompt_context.is_direction(video) and (by_type.get('last_frame') or {}).get('content') is None:
        # A video analysis from before motion directions only describes the cut together with its last frame.
        return 'This cut has an older video analysis without a last frame; run Keyframes'
    return None


def source_ids(results):
    """The results a consistency summary is written from: not itself, and not what a newer motion source replaced."""
    by_type = {item['type']: item for item in results}
    keyframes = 'keyframe_analysis' in by_type
    current_flow = keyframes or prompt_context.is_direction((by_type.get('video_analysis') or {}).get('content'))
    return [item['id'] for item in results if item['type'] != 'consistency_summary'
            and not (item['type'] == 'last_frame' and current_flow) and not (item['type'] == 'video_analysis' and keyframes)]


def payload_for(cut_id, config, context, results):
    by_type = {item['type']: item for item in results}
    problem = missing_evidence(by_type)
    if problem:
        raise ValueError(problem)
    evidence = consistency_summary.build_input(results, context['transcript_segments'], context['context_items'], context.get('subtitles'), prompt_context.series_context(context))
    if context.get('character_detections'):
        evidence['character_detections'] = prompt_context.slim_detections(context['character_detections'])
    return {'cut_id': cut_id, 'context': {'cut': prompt_context.slim_cut(context['cut'])}, 'config': config,
            'evidence': evidence, 'source_result_ids': source_ids(results)}


def prerequisites(conn, episode_id, cut_id, config):
    from . import job_kinds, performance_beats
    jobs = worker.pending_jobs(conn, episode_id,
                               ('prepare', 'hardcuts', 'analyze', 'performance_beats', 'character_detect', 'subtitles', 'transcribe'),
                               [cut_id])
    for job_id in jobs:
        row = conn.execute('SELECT pipeline,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row['pipeline'] == 'analyze' and 'keyframe_analysis' in job_kinds.stages_of(json.loads(row['payload_json'])):
            beats = performance_beats.ensure_queued(conn, episode_id, cut_id, {key: value for key, value in config.items()
                                                  if key in ('text_provider', 'openrouter_model', 'local_thinking')})
            if beats not in jobs:
                jobs.append(beats)
            break
    return jobs


def insert(conn, episode_id, payload, *, retry_of=None):
    cut_id, config = payload['cut_id'], payload['config']
    parents = list(dict.fromkeys([*(payload.get('depends_on') or []), *prerequisites(conn, episode_id, cut_id, config)]))
    return worker.insert_job(conn, episode_id, 'consistency_summary',
                             {'cut_id': cut_id, 'config': config, 'depends_on': parents},
                             priority_class='batch' if payload.get('batch') else 'interactive', retry_of=retry_of)


def gather(conn, job):
    request = json.loads(job['payload_json'])
    cut_id, config = request['cut_id'], request['config']
    cut = db.one('SELECT * FROM cuts WHERE id=? AND active=1', (cut_id,), conn=conn)
    if not cut:
        return {'skip': 'Cut archived or removed'}
    results = structured_results.list_results(cut_id, conn=conn)
    by_type = {result['type']: result for result in results}
    if problem := missing_evidence(by_type):
        return {'missing': problem}
    context = analysis_context.selected(conn, cut_id, config)
    values = payload_for(cut_id, config, context, results)
    episode = consistency_evidence.episode_context(conn, cut)
    if episode:
        values['evidence']['episode'] = episode
    values['pictures'], values['keyframes'], values['aspect'] = consistency_evidence.gather(cut_id, by_type, conn)
    picture_cuts = list(dict.fromkeys(picture['cut_id'] for picture in values['pictures']))
    marks = ','.join('?' for _ in picture_cuts)
    values['picture_timing'] = db.all_rows(f"""SELECT id,start_time,end_time,first_frame_time,last_frame_time
        FROM cuts WHERE id IN ({marks}) ORDER BY id""", picture_cuts, conn=conn)
    values['source_transcript_ids'] = [row['id'] for row in context['transcript_segments']]
    base = by_type.get('consistency_summary')
    values['base'] = {name: base[name] for name in ('id', 'origin')} if base else {}
    return values


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    cut_id = json.loads(job['payload_json'])['cut_id']
    conn.execute("""UPDATE analysis_results SET stale_reason='Consistency summary changed'
        WHERE cut_id=? AND superseded_by IS NULL AND title='Cut summary'""", (cut_id,))
