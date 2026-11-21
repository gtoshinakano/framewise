"""The ``character_profile`` job: describe a newly created character from the box it was created on.

``POST /api/character-detections/{id}/new-character`` creates the character, labels the box and queues this
job. The chosen character-profile model (``visual_models``: Qwen3.6-35B-A3B by default, snapshotted in
``payload.models`` when queued) sees the whole frame (the box outlined) and a close crop of the box and writes a
character sheet (what identifies the character in any outfit) and an outfit sheet (what they wear here).
The rendered text is appended to the character's Details; the answer, prompt and model are kept on the job
(``payload.result``) as provenance.
"""
import base64
import io
import json
import logging
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from . import db, frame_model, job_inputs, media, visual_models, worker
from .visual_contracts import OUTFIT_RULE

log = logging.getLogger('framewise.character_profile')

MAX_TOKENS = 1024
CROP_SIDE = 1024
FIELD_CHARS = 400
OUTLINE = (255, 0, 255)

CHARACTER_FIELDS = (('hair', 'Hair'), ('eyes', 'Eyes'), ('face', 'Face'), ('skin', 'Skin'), ('build', 'Build'),
                    ('age', 'Apparent age'), ('distinguishing_features', 'Distinguishing features'))
OUTFIT_FIELDS = (('top', 'Top'), ('bottom', 'Bottom'), ('footwear', 'Footwear'), ('accessories', 'Accessories'), ('colours', 'Colours'))
NOT_VISIBLE = 'not visible'

PROFILE_INSTRUCTION = f'''You write the reference description of one character for an artist who will draw a character sheet and an outfit sheet of them.

You receive two images: the whole frame, where the marked character is outlined with a magenta rectangle (the outline is not part of the scene), and a close crop of the same character. The message gives the character's name, the user's notes and the box.

Describe only the marked character, never other people in the frame.
- character_sheet is what identifies the character at any angle and in any outfit: hair (length, style, parting or fringe, colour), eyes (colour, shape), face (shape and notable features), skin (tone), build (body type and height impression), age (apparent age range) and distinguishing_features (marks, scars, glasses or other features that belong to the person rather than to this outfit; "none visible" when there are none).
- outfit_sheet is what the character wears in this frame: top (every visible layer, from outer to inner), bottom, footwear, accessories (jewellery, hair accessories, bags, belts, worn items; not objects held in the hands) and colours (the outfit's main colours as a short palette).

Write short concrete phrases and name colours precisely. Never name a drawing or visual style.
{OUTFIT_RULE}
The user's notes tell who the character is; they are not evidence of how the character looks here. When the notes and the picture disagree, write what the picture shows.
Ignore burned-in subtitles and any other lettering on screen.
Return one JSON object that matches the schema.'''


def schema():
    def section(fields):
        return {'type': 'object', 'properties': {key: {'type': 'string'} for key, _ in fields},
                'required': [key for key, _ in fields], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'character_sheet': section(CHARACTER_FIELDS), 'outfit_sheet': section(OUTFIT_FIELDS)},
            'required': ['character_sheet', 'outfit_sheet'], 'additionalProperties': False}


def validate(raw):
    """Parse the model's answer into both sheets; a missing field reads "not visible", a missing sheet is an error."""
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError('The answer is not a JSON object')
    profile = {}
    for name, fields in (('character_sheet', CHARACTER_FIELDS), ('outfit_sheet', OUTFIT_FIELDS)):
        section = value.get(name)
        if not isinstance(section, dict):
            raise ValueError(f'The answer has no {name}')
        profile[name] = {}
        for key, _ in fields:
            text = section.get(key)
            text = ' '.join(text.split())[:FIELD_CHARS].strip() if isinstance(text, str) else ''
            profile[name][key] = text or NOT_VISIBLE
    return profile


def render(profile, episode_title, cut_number, model_id, day):
    """The block appended to the character's Details."""
    lines = ['## Character sheet']
    lines += [f"{label}: {profile['character_sheet'][key]}" for key, label in CHARACTER_FIELDS]
    lines += ['', f'## Outfit sheet ({episode_title}, cut {cut_number})']
    lines += [f"{label}: {profile['outfit_sheet'][key]}" for key, label in OUTFIT_FIELDS]
    lines += ['', f'_Described by {model_id} from the first frame of cut {cut_number} on {day}._']
    return '\n'.join(lines)


def grid_box(box, size):
    """A pixel box as x1,y1,x2,y2 on the 0-1000 grid the frame prompts use."""
    width, height = size
    return [round(box[0] * 1000 / width), round(box[1] * 1000 / height), round(box[2] * 1000 / width), round(box[3] * 1000 / height)]


def user_text(name, notes, box):
    parts = [f'Character name: {name}', f'Marked box (x1,y1,x2,y2 on a 0-1000 grid over the whole frame): {box}']
    parts.append('User notes:\n' + notes.strip() if notes.strip() else 'User notes: none')
    return '\n\n'.join(parts)


def images(original, frame, cut_start, cut_end, box, scratch):
    """The whole frame with the box outlined and a close crop of the box, as JPEG bytes."""
    from PIL import Image, ImageDraw
    target = Path(scratch) / 'frame.jpg'
    media.extract_cut_frame(Path(original), target, frame['time'], cut_start, cut_end, width=1280, seek_tolerance=.000002)
    with Image.open(target) as image:
        image = image.convert('RGB')
        if list(image.size) != list(frame['image_size']):
            raise RuntimeError('The detected frame changed size; detect characters again')
        crop = image.crop(tuple(round(value) for value in box))
        crop.thumbnail((CROP_SIDE, CROP_SIDE))
        outlined = image.copy()
        ImageDraw.Draw(outlined).rectangle(tuple(round(value) for value in box), outline=OUTLINE, width=max(3, image.size[0] // 320))
        encoded = []
        for picture in (outlined, crop):
            output = io.BytesIO()
            picture.save(output, format='JPEG', quality=90)
            encoded.append(output.getvalue())
        return encoded[0], encoded[1]


def _body(messages, with_schema, thinking=False):
    response_format = {'type': 'json_schema', 'json_schema': {'name': 'character_profile', 'strict': True, 'schema': schema()}} if with_schema else {'type': 'json_object'}
    return {'messages': messages, 'temperature': 0, 'max_tokens': MAX_TOKENS, 'response_format': response_format, 'chat_template_kwargs': frame_model.thinking_kwargs(thinking)}


def describe(server_url, frame_jpeg, crop_jpeg, text, cancel_check=None, usage_out=None):
    """Ask the frame model for both sheets; a server that cannot build the grammar answers in plain JSON."""
    from .inference_worker import request_json
    def url(data):
        return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')}}
    messages = [{'role': 'system', 'content': PROFILE_INSTRUCTION},
                {'role': 'user', 'content': [url(frame_jpeg), url(crop_jpeg), {'type': 'text', 'text': text}]}]
    def ask(body):
        request = urllib.request.Request(server_url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        return request_json(request, cancel_check, timeout=600)
    try:
        payload = ask(_body(messages, True))
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 422, 500):
            raise
        payload = ask(_body(messages, False))
    if usage_out is not None and isinstance(payload.get('usage'), dict):
        usage_out.update({key: value for key, value in payload['usage'].items() if key in ('prompt_tokens', 'completion_tokens', 'total_tokens') and isinstance(value, int)})
    choice = payload['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('The description was truncated')
    return validate(choice['message']['content'])


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    row = conn.execute('''SELECT d.box_json,d.position,d.character_context_id,r.frames_json,r.cut_start,r.cut_end,
                                 c.active,c.cut_index,c.start_time,c.end_time,e.project_id,e.title AS episode_title,e.original_ext,
                                 ci.title AS name,ci.content AS notes,ci.episode_id AS context_episode_id
                          FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id
                          JOIN episodes e ON e.id=c.episode_id LEFT JOIN context_items ci ON ci.id=? AND ci.kind='character'
                          WHERE d.id=? AND c.episode_id=?''',
                       (payload.get('character_context_id'), payload.get('detection_id'), job['episode_id'])).fetchone()
    if not row or row['name'] is None:
        return {'skip': 'box or character removed'}
    if not row['active'] or (row['start_time'], row['end_time']) != (row['cut_start'], row['cut_end']):
        return {'skip': 'the cut or detected frame changed'}
    if row['character_context_id'] != payload['character_context_id']:
        return {'skip': 'the box was labelled with another character'}
    return dict(row)


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    from .structured_results import invalidate_context
    invalidate_context(conn, job['project_id'], job['context_episode_id'], 'Character details changed')


def finish(conn, job_id, episode_id, stage):
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job_id))
    worker.sync_episode_status(conn, episode_id)


def process(job_id, owner):
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    episode_id = job['episode_id']
    cancel_check = lambda: inference_worker.check_summary_lease(job_id, owner)
    try:
        with db.connect() as conn:
            conn.execute('BEGIN')
            live = job_inputs.JobInputs(conn, job, owner)
        row = live.data
        if row.get('skip'):
            with live.publish(invalidate=False) as conn:
                finish(conn, job_id, episode_id, 'Skipped · ' + row['skip'])
            return
        frame = next((item for item in json.loads(row['frames_json']) if item.get('position') == row['position']), None)
        if not frame:
            raise RuntimeError('The detected frame of this box is gone; detect characters again')
        box = json.loads(row['box_json'])
        original = media.episode_dir(row['project_id'], episode_id) / f"original{row['original_ext']}"
        worker.stage(job_id, owner, episode_id, 'Reading the character box')
        with tempfile.TemporaryDirectory(prefix='framewise-character-profile-') as scratch:
            frame_jpeg, crop_jpeg = images(original, frame, row['cut_start'], row['cut_end'], box, scratch)
        text = user_text(row['name'], row['notes'] or '', grid_box(box, frame['image_size']))
        usage, started = {}, time.monotonic()
        model = visual_models.for_job(payload, 'character_profile')
        worker.stage(job_id, owner, episode_id, f'Starting {model.short_name}')
        with frame_model.model_server(model, cancel_check=cancel_check) as server:
            worker.stage(job_id, owner, episode_id, 'Describing features')
            profile = describe(server, frame_jpeg, crop_jpeg, text, cancel_check=cancel_check, usage_out=usage)
        cut_number = row['cut_index'] + 1
        block = render(profile, row['episode_title'], cut_number, model.short_name, date.today().isoformat())
        result = {'model_id': model.model_id, 'model_revision': model.revision, 'visual_model': model.key, 'profile': profile, 'appended': block,
                  'box': box, 'frame_time': frame['time'], 'usage': usage, 'seconds': round(time.monotonic() - started, 1),
                  'prompt': {'system': PROFILE_INSTRUCTION, 'user': text}}
        provenance = {'fingerprint': live.stamp, 'detection_id': payload['detection_id'], 'character_context_id': payload['character_context_id']}
        with live.publish(project_id=row['project_id'], context_episode_id=row['context_episode_id']) as conn:
            conn.execute("""UPDATE context_items SET content=CASE WHEN trim(content)='' THEN ? ELSE rtrim(content) || char(10) || char(10) || ? END,
                           updated_at=CURRENT_TIMESTAMP WHERE id=?""", (block, block, payload['character_context_id']))
            conn.execute("UPDATE jobs SET payload_json=json_set(payload_json,'$.result',json(?)),input_json=? WHERE id=?",
                         (json.dumps(result, ensure_ascii=False), json.dumps(provenance), job_id))
            finish(conn, job_id, episode_id, 'Complete · Details updated')
    except inference_worker.SummaryCancelled:
        log.info('Character description %s was cancelled', job_id)
    except worker.LeaseLost as exc:
        log.info('Character description %s stopped: %s', job_id, exc)
    except Exception as exc:
        log.exception('Character description failed for job %s', job_id)
        inference_worker.fail_job(job_id, owner, episode_id, exc)
