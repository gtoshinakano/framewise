"""Attribute open transcript lines, then correct cast-name spellings."""
import json
import logging
import uuid
from bisect import bisect_right

from . import audio_lines, db, job_inputs, name_glossary, speaker_prompt, subtitle_store, transcript_names, worker

log = logging.getLogger(__name__)


def rows_in_range(rows, span):
    if not span:
        return rows
    return [row for row in rows if span['start_time'] <= (row['start_time'] + row['end_time']) / 2 < span['end_time']]


def gather(conn, job):
    episode_id = job['episode_id']
    episode = db.one('SELECT id,project_id FROM episodes WHERE id=?', (episode_id,), conn=conn)
    project_id = episode['project_id']
    transcript = db.all_rows("""SELECT id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,
        words_json,speaker_source,evidence_json,confidence FROM transcript_segments
        WHERE episode_id=? AND deleted=0 AND superseded_by IS NULL ORDER BY start_time,end_time,id""", (episode_id,), conn=conn)
    characters = db.all_rows("""SELECT id,title,unregistered FROM context_items WHERE project_id=? AND kind='character'
        AND (episode_id IS NULL OR episode_id=?) ORDER BY episode_id IS NULL DESC,title,id""", (project_id, episode_id), conn=conn)
    cuts = db.all_rows("""SELECT id,start_time,end_time,no_characters,cut_index FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time,cut_index""",
                       (episode_id,), conn=conn)
    boxes = db.all_rows("""SELECT c.id AS cut_id,d.character_context_id FROM cuts c
        JOIN character_detections d ON d.run_id=(SELECT r.id FROM character_detection_runs r WHERE r.cut_id=c.id ORDER BY r.rowid DESC LIMIT 1)
        WHERE c.episode_id=? AND c.active=1 AND d.position='first' AND d.character_context_id IS NOT NULL
        ORDER BY c.cut_index,d.rowid""", (episode_id,), conn=conn)
    subtitle_run = subtitle_store.current_run(episode_id, conn)
    return {'transcript': transcript, 'characters': characters, 'cuts': cuts, 'boxes': boxes,
            'subtitles': subtitle_store.lines(episode_id, conn=conn), 'subtitle_run': subtitle_run['id'] if subtitle_run else None,
            'glossary': name_glossary.rows(conn, project_id, episode_id)}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """Speaker attribution marks nothing stale (HARNESS.md D37): it only fills in model lines nobody decided, and the
    results built on the transcript keep their state until the person changes a line themselves."""


def insert(conn, episode_id, *, span=None, dependencies=()):
    payload = {'config': {'text_provider': 'local', 'local_thinking': True}, 'depends_on': list(dependencies)}
    if span:
        payload['range'] = span
    return worker.insert_job(conn, episode_id, 'speaker_attribution', payload)


def open_items(data, eligible):
    names = {item['id']: item['title'] for item in data['characters']}
    cards = {item['id'] for item in data['characters'] if item.get('unregistered')}
    casts = {}
    for box in data['boxes']:
        title = names.get(box['character_context_id'])
        if title:
            # An Unregistered card is a minor person on screen, never a listed character: "other" is its answer.
            shown_as = f'{title} (unregistered)' if box['character_context_id'] in cards else title
            casts.setdefault(box['cut_id'], {}).setdefault(shown_as, None)
    cuts = data['cuts']
    starts = [cut['start_time'] for cut in cuts]
    shown, by_label = [], {}
    for number, row in enumerate(data['transcript'], 1):
        lid = f'L{number}'
        by_label[lid] = row
        item = {'id': lid, 't': [round(row['start_time'], 1), round(row['end_time'], 1)], 'ja': row['text']}
        if row['speaker_label']:
            item['diar'] = row['speaker_label']
        i = bisect_right(starts, row['start_time']) - 1
        if i >= 0 and row['start_time'] < cuts[i]['end_time']:
            item['cut'] = cuts[i]['cut_index']
            if not cuts[i]['no_characters'] and cuts[i]['id'] in casts:
                item['on_screen'] = list(casts[cuts[i]['id']])
        pt = audio_lines.subtitle_overlap(row, data['subtitles'])
        if pt:
            item['pt'] = ' / '.join(cue['text'].replace('\n', ' ') for cue in pt)[:240]
        if row['id'] not in eligible or row['speaker_source'] in DECIDED:
            item['decided'] = names.get(row['character_context_id'], 'unknown')
        else:
            evidence = json.loads(row['evidence_json'] or '{}').get('evidence', [])
            item['voice'] = [[names[character], round(score, 2)] for character, score in evidence if character in names]
        shown.append(item)
    return shown, by_label


# Speakers this job never re-decides: settled by the voice score, given by the voice resolver (autopilot M5, two signals),
# or left as they are by the person (one question per unknown voice, HARNESS.md D29).
DECIDED = ('voice', 'voice_padded', 'agent', 'left_unknown')


def model_line(row):
    return row['origin'] == 'model' and row['speaker_source'] not in ('manual', 'agent', 'left_unknown') and bool(row['words_json'])


def revise(conn, episode_id, job_id, row, text, speaker, source, confidence, evidence):
    new_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,
        origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json)
        VALUES (?,?,?,?,?,?,?,'model',?,?,?,?,?,?,?)""",
        (new_id, episode_id, row['start_time'], row['end_time'], text, row['speaker_label'], speaker,
         row['model_id'], job_id, row['id'], row['words_json'], source, confidence, json.dumps(evidence, ensure_ascii=False)))
    conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, row['id']))


def process(job_id, owner):
    from . import inference_worker, local_text_runtime

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
        eligible = {row['id'] for row in rows_in_range(data['transcript'], payload.get('range')) if model_line(row)}
        shown, by_label = open_items(data, eligible)
        # The answer's names: registered characters only. An Unregistered person is "other" (voice_resolver gives the line to
        # the card when it is the only one boxed in the cut).
        names = {row['title']: row['id'] for row in reversed(data['characters']) if not row.get('unregistered')}
        system = speaker_prompt.system_open(any('(unregistered)' in name for item in shown for name in item.get('on_screen') or ()))
        models = {}
        failed_windows = []

        def call(system, evidence, schema, validate, tokens, label):
            usage = {}
            try:
                output, model = inference_worker.chat_json('', system, evidence, (), max_tokens=tokens,
                    progress=lambda attempt, total: worker.stage(job_id, owner, episode_id, f'{label} · attempt {attempt}/{total}'),
                    cancel_check=lambda: inference_worker.check_summary_lease(job_id, owner), provider='local', thinking=True,
                    usage_out=usage, validator=validate,
                    response_format={'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True, 'schema': schema}})
            finally:
                variant = usage.get('model_variant')
                if variant:
                    models[local_text_runtime.revision_for(variant)] = local_text_runtime.model_id_for(variant)
            models.setdefault(local_text_runtime.revision_for(usage.get('model_variant')), model)
            return output

        def ask(window, conversation, number, total):
            label = f'Attributing speakers {number}/{total}'

            def validate(answer):
                values, errors = speaker_prompt.validate(answer, window, names)
                if errors:
                    raise ValueError(' '.join(errors[:10]))
                return values

            try:
                return call(system, speaker_prompt.user_message(window, names, conversation),
                            speaker_prompt.answer_schema([item['id'] for item in window], names), validate, 6144, label)
            except RuntimeError as exc:
                if 'returned an invalid analysis after' not in str(exc):
                    raise
                failed_windows.append(label)
                log.warning('%s: %s', label, exc)
                return {}

        speakers = speaker_prompt.attribute_open(shown, ask)
        glossary = data['glossary']
        fixes = {}
        if any(row['character_context_id'] for row in glossary):
            name_items = [{'id': item['id'], 'ja': item['ja']} for item in shown if by_label[item['id']]['id'] in eligible]
            windows = speaker_prompt.make_windows(name_items, size=60, overlap=0)
            for number, (start, stop) in enumerate(windows, 1):
                window = name_items[start:stop]
                accepted = {}

                def validate_names(answer):
                    values, errors = transcript_names.validate(answer, window, glossary)
                    accepted.update(values)
                    if errors:
                        raise ValueError(' '.join(errors[:10]))
                    return values

                label = f'Correcting names {number}/{len(windows)}'
                try:
                    fixes.update(call(transcript_names.SYSTEM, transcript_names.user_message(window, glossary),
                                      transcript_names.schema([item['id'] for item in window]), validate_names, 4096, label))
                except RuntimeError as exc:
                    if 'returned an invalid analysis after' not in str(exc):
                        raise
                    fixes.update(accepted)
                    log.info('%s retained %d valid spelling changes', label, len(accepted))

        provenance = {'fingerprint': snapshot.stamp, 'segment_ids': [row['id'] for row in data['transcript']],
                      'subtitle_run': data['subtitle_run'], 'glossary_ids': [row['id'] for row in glossary],
                      'models': models, 'unanswered_windows': failed_windows}
        with snapshot.publish(invalidate=False) as conn:
            updated = 0
            for lid, row in by_label.items():
                if row['id'] not in eligible or lid not in fixes and lid not in speakers:
                    continue
                speaker, source, confidence = row['character_context_id'], row['speaker_source'], row['confidence']
                evidence = json.loads(row['evidence_json'] or '{}')
                if lid in speakers:
                    answer = speakers[lid]
                    speaker = names.get(answer['speaker'])
                    ranked = evidence.get('evidence') or []
                    source = 'qwen'
                    confidence = 'medium' if speaker and ranked and speaker == ranked[0][0] else 'low'
                    evidence['qwen'] = answer
                text = fixes.get(lid, row['text'])
                if (text, speaker, source, confidence) == (row['text'], row['character_context_id'], row['speaker_source'], row['confidence']):
                    continue
                evidence['attribution_fingerprint'] = snapshot.stamp
                evidence['attribution_models'] = models
                revise(conn, episode_id, job_id, row, text, speaker, source, confidence, evidence)
                updated += 1
            stage = f'Complete · {len(speakers)} attributed, {len(fixes)} spelling fixes'
            if failed_windows:
                stage += f', {len(failed_windows)} unanswered windows'
            conn.execute("""UPDATE jobs SET status='done',stage=?,input_json=?,error=NULL,lease_owner=NULL,lease_until=NULL,
                updated_at=CURRENT_TIMESTAMP WHERE id=?""", (stage, json.dumps(provenance), job_id))
            worker.sync_episode_status(conn, episode_id)
    except (worker.LeaseLost, inference_worker.SummaryCancelled):
        log.info('Speaker attribution %s stopped before publishing', job_id)
    except Exception as exc:
        log.exception('Speaker attribution failed for %s', job_id)
        worker.fail_job(job_id, owner, episode_id, exc)
