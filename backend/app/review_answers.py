"""Draft concise answers to a cut's consistency-summary review items."""
import json
import uuid

from . import consistency_summary, db

SUMMARY_TITLE = 'Cut summary'
FIELDS = ['answers']
MAX_ANSWERS = 50000

SYSTEM = '''You are a production-direction assistant for a pipeline that re-creates an anime episode shot by shot with AI: for each cut a still image of the opening frame is generated, then the dialogue audio, then a video that animates the still. You resolve the open review items for ONE cut so that the final brief for those three steps can be written from decisive facts.

## YOUR TASK

You receive a JSON evidence package. Its "review" object contains the Consistency Summary of the current cut, in particular "questions": multiple-choice questions about the points where the evidence is unsure (each has a topic, optionally a character, a question, options with ids, an optional recommended option id and a short why). Older summaries carry "human_review" (issues needing human judgment) and "recommended_checks" (suggested verification steps) as text instead. Write the answers a careful human reviewer would enter, using only the supplied evidence.

## EVIDENCE AND PRIORITY

All supplied observations are model-generated and may contain errors, except "existing_answers", which are the user's own decisions. Use sources in this priority order:

1. "existing_answers": authoritative. Keep their meaning and wording; never contradict them. Include them in the output when they are still relevant.
2. The current cut: "current_cut" first frame (the opening still: composition, each character's appearance, pose, expression, emotion, gaze and action), keyframes (each character's state at each keyframe and what changed between consecutive keyframes; body parts marked keypoints were measured, the camera was measured from the background), performance beats (the timed acting plan built from them: per beat the camera, each visible character's body, face, gaze and mouth, the beat's intent, and each transcript line's delivery), "transcript", "subtitles" (burned-in lines that overlap the cut, read from the whole episode, all in the one language given by subtitles.language; for comparison only) and "review". "review.contradicted_by_pictures" lists stored readings the reviewer saw contradicted in a picture of this cut (the picture label, the stored claim and what the picture shows): for what is visible at that moment, the picture outranks the descriptions. Cuts analysed before keyframes carry a video analysis instead (the motion direction: camera, each character's activity, travel, motion, speaking, emotion and end state, plus timed events); older saved motion directions may carry an "in last frame" presence confirmation; even older cuts carry a separate last-frame description.
3. "previous_cut" and "next_cut" "consistency_summary", when present (continuity of characters, outfits, positions, props, setting, lighting and time of day). They are model reviews of the neighbouring cut, not confirmations; a neighbour without one has no supplied evidence, so do not infer anything about it.

## RULES

- Answer every question with one bullet: choose the option the evidence supports best (the recommended one when the evidence agrees with it) and copy that option's text word for word after the label. Only when no option is right, write a short statement of your own in the same form. Never answer with the option id. A question whose decision "existing_answers" already states keeps that answer.
- For an older summary, address every concrete item in "human_review" and "recommended_checks" with one bullet each. Merge bullets about the same decision. Skip generic disclaimers.
- State a fact as confirmed only if at least two independent sources agree, or a neighbouring Cut summary or the user's existing answers establish it. Repetition by the same model, or a summary derived from the analysis it summarizes, is not independent confirmation.
- Source rule: the first frame decides how the cut starts (appearance, pose, expression, emotion at the first frame); the keyframes decide what changes afterwards (motion, camera, mouth, emotion change and final position); the performance beats turn them into timed acting and decide each line's delivery, and where a beat contradicts the keyframes, the keyframes win. For older cuts the video analysis plays the keyframes' part. A historical "in last frame" confirmation proves presence for that saved result; its absence proves nothing. When two sources differ only in wording or fine nuance, pick by this rule and say so in a few words instead of leaving it open.
- When evidence supports a choice but does not fully confirm it, give the decisive direction plus the reason in a few words (for example "Use the first frame; the keyframes are vaguer on this point").
- Speaking: use each line's delivery in the performance beats: on_screen is lip-synced, inner_voice is an inner voice (mouth closed), off_screen is a voice from outside the picture, narrative plays over the picture; keep it unless the review gives a reason to think the mouth was missed. For older cuts, a transcript line linked to an on-screen character whose speaking is yes is lip-synced; with speaking no it is an inner voice or narration (mouth closed); a line from someone not in frame is an off-screen voice.
- When evidence cannot settle an item, write "Unresolved: <exactly what is missing>". Do not guess.
- Never invent actions, characters, identities, timestamps, camera moves or dialogue. Take emotion from the visual evidence and the meaning of the words; do not describe vocal tone as observed, because no acoustic analysis was made.
- Treat a difference between the first frame and a later moment as movement, not a contradiction. Characters may speak while stationary.
- Do not reproduce or quote subtitle or on-screen text. Do not tell the reader to treat on-screen text as an object to reproduce.
- Write imperative, final-direction statements for the animation brief, not questions, hedging or meta commentary about the evidence process.

## OUTPUT FORMAT

Return exactly one JSON object with one string field, "answers":

{"answers": "- Topic: statement\\n- Topic: statement"}

- The value is a plain-text list of at most 8 short bullets, each one or two lines, each starting with "- " and a topic label. For a question the label is its topic with a capital letter, followed by the character in brackets when the question names one ("- Gaze (Name): option text"). Other bullets use a label such as Characters, Opening pose, Emotion, Camera, Motion, Speaking or Continuity.
- No preamble, headings, Markdown beyond the "- " bullets, or repetition of the review text.
- If there are no questions (or both "human_review" and "recommended_checks" say "None required.") and there are no existing answers, return {"answers": "No review decisions required."}.

Return JSON only.'''


def transcript_text(segments):
    return consistency_summary.build_input([], segments, [])['transcript_text']


def _neighbour(cut, consistency):
    """A neighbouring cut's boundaries and, when it has one, its Consistency summary. Nothing else about it is sent."""
    if not cut:
        return None
    entry = {'cut_index': cut.get('cut_index'), 'start_time': cut.get('start_time'), 'end_time': cut.get('end_time')}
    if isinstance(consistency, dict):
        consistency = {key: value for key, value in consistency.items() if key not in ('questions', 'visual_check')}  # a neighbour's open questions and picture checks are not facts about it
    if consistency:
        entry['consistency_summary'] = consistency
    return entry


def review_view(content):
    """What the answers prompt reads of a consistency summary: its questions, or the two older text fields, and what its
    pictures contradicted."""
    content = content if isinstance(content, dict) else {}
    view = {key: content.get(key, '') for key in ('confusing', 'clear')}
    seen = [{key: item[key] for key in ('character', 'picture', 'claim', 'seen') if item.get(key)}
            for item in content.get('visual_check') or [] if isinstance(item, dict) and item.get('verdict') == 'contradicts']
    if seen:
        view['contradicted_by_pictures'] = seen
    if isinstance(content.get('questions'), list):
        view['questions'] = [{key: item[key] for key in ('id', 'topic', 'character', 'question', 'options', 'recommended', 'why') if item.get(key) not in (None, '', [])}
                             for item in content['questions'] if isinstance(item, dict)]
    else:
        view.update({key: content.get(key, '') for key in ('human_review', 'recommended_checks')})
    return view


def build_evidence(results, context, existing_answers, neighbours, pending_review=False):
    """neighbours maps previous/next to that adjacent cut's Consistency summary content, when it has one.

    With `pending_review` the consistency summary may be missing or about to be replaced by a chained
    job; `refresh` fills it in when the job starts.
    """
    by_type = {item['type']: item for item in results}
    review = by_type.get('consistency_summary')
    if not review or not isinstance(review['content'], dict):
        if not pending_review:
            raise ValueError('Generate the consistency summary first')
        review = {'content': {}}
    base = consistency_summary.build_input(results, context['transcript_segments'], context['context_items'], context.get('subtitles'))
    evidence = {
        'review': review_view(review['content']),
        'current_cut': {'visual_evidence_text': base['visual_evidence_text'], 'transcript': base['transcript_text'], 'subtitles': base['subtitles']},
        'existing_answers': existing_answers.strip() or None,
    }
    for label, cut in (('previous_cut', context.get('previous_cut')), ('next_cut', context.get('next_cut'))):
        entry = _neighbour(cut, (neighbours or {}).get('previous' if label == 'previous_cut' else 'next'))
        if entry:
            evidence[label] = entry
    return evidence


def has_review_items(evidence):
    """False when the consistency summary leaves nothing to decide or check."""
    review = evidence.get('review', {})
    if review.get('questions'):
        return True
    return any(str(review.get(key) or '').strip().rstrip('.').lower() not in ('', 'none required') for key in ('human_review', 'recommended_checks'))


def validate_answers(text):
    text = text.strip()
    if not text:
        raise ValueError('answers must not be empty')
    return text[:MAX_ANSWERS]


def refresh(payload, results, existing_answers):
    """Update a chained job's snapshot with what the earlier jobs of its chain produced.

    The consistency summary and the saved answers are read again when the job starts, because the jobs
    it waited for wrote the former and the user may have changed the latter. Everything else in the
    evidence is unchanged by the chain.
    """
    review = next((item for item in results if item['type'] == 'consistency_summary'), None)
    if not review or not isinstance(review['content'], dict):
        raise RuntimeError('The consistency summary this job waited for is missing; queue the chain again')
    payload['evidence']['review'] = review_view(review['content'])
    payload['evidence']['existing_answers'] = existing_answers.strip() or None
    payload['existing_answers'] = existing_answers
    payload['source_result_ids'] = [item['id'] for item in results]
    return payload


def payload_for(cut_id, config, evidence, source_result_ids, existing_answers):
    return {'cut_id': cut_id, 'config': config, 'evidence': evidence, 'source_result_ids': source_result_ids,
            'existing_answers': existing_answers}


def insert(conn, episode_id, payload):
    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='review_answers' AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?", (episode_id, payload['cut_id'])).fetchone():
        raise ValueError('Review answers are already being generated for this cut')
    job_id = str(uuid.uuid4())
    conn.execute('INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)', (job_id, episode_id, 'review_answers', json.dumps(payload, ensure_ascii=False)))
    return job_id


def generate(config, evidence, progress=None, cancel_check=None, usage_out=None):
    from .inference_worker import chat_json
    output, model = chat_json(config['openrouter_model'], SYSTEM, json.dumps(evidence, ensure_ascii=False), FIELDS, max_tokens=2048,
                              progress=progress, cancel_check=cancel_check, provider=config.get('text_provider', 'openrouter'),
                              thinking=config.get('local_thinking', False), usage_out=usage_out)
    return validate_answers(output['answers']), model


def save_draft(conn, cut_id, job_id, content, model_id):
    conn.execute('INSERT INTO cut_review_drafts(cut_id,job_id,content,model_id) VALUES (?,?,?,?) '
                 'ON CONFLICT(cut_id) DO UPDATE SET job_id=excluded.job_id,content=excluded.content,model_id=excluded.model_id,created_at=CURRENT_TIMESTAMP',
                 (cut_id, job_id, content, model_id))


def save_answers(conn, cut_id, content, expected):
    """Store a draft as the cut's saved answers unless they changed after the job was queued.

    `expected` is the text the job was drafted from. Returns False and leaves the answers alone when
    they no longer match it, so nothing typed in the meantime is lost.
    """
    row = conn.execute('SELECT content FROM cut_brief_instructions WHERE cut_id=?', (cut_id,)).fetchone()
    if (row['content'] if row else '') != (expected or ''):
        return False
    conn.execute('INSERT INTO cut_brief_instructions(cut_id,content) VALUES (?,?) ON CONFLICT(cut_id) DO UPDATE SET content=excluded.content,updated_at=CURRENT_TIMESTAMP', (cut_id, content))
    return True


def get_draft(cut_id):
    row = db.one('SELECT job_id,content FROM cut_review_drafts WHERE cut_id=?', (cut_id,))
    return {'job_id': row['job_id'], 'content': row['content']} if row else {'job_id': None, 'content': ''}
