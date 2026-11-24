"""One transaction for the character, their glossary and outfits, and queued voice references."""
import hashlib
import json
import sqlite3
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from . import agent_log, character_decisions, db, media, outfit_library, registration_images, structured_results, unregistered, worker
from .audio_ranges import VoiceRange

router = APIRouter()
PATH = '/api/projects/{project_id}/character-registration'


class GlossaryEntry(BaseModel):
    id: UUID | None = None
    part: Literal['family', 'given']
    spelling: str = Field(min_length=1, max_length=120)
    reading: str = Field(min_length=1, max_length=120, pattern=r'^[ぁ-ゖァ-ヺー]+$')


class OutfitEntry(BaseModel):
    id: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(default='', max_length=400)
    description: str = Field(min_length=1, max_length=6000)


class VoiceEntry(BaseModel):
    episode_id: UUID
    ranges: list[VoiceRange] = Field(min_length=1, max_length=50)


class Registration(BaseModel):
    character_id: UUID
    version: str | None = None
    save_id: UUID
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default='', max_length=50000)
    episode_id: UUID | None = None
    is_main: bool = False
    detection_id: UUID | None = None
    describe: bool = False
    glossary: list[GlossaryEntry] = Field(default_factory=list, max_length=100)
    outfits: list[OutfitEntry] = Field(default_factory=list, max_length=100)
    voice: list[VoiceEntry] = Field(default_factory=list, max_length=20)
    remove_voice: list[UUID] = Field(default_factory=list, max_length=100)


def project(conn, project_id):
    if not conn.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
        raise HTTPException(404, 'Project not found.')


def record(conn, project_id, character_id):
    row = db.one("SELECT * FROM context_items WHERE id=? AND project_id=? AND kind='character' AND unregistered=0",
                 (character_id, project_id), conn=conn)
    if not row:
        raise HTTPException(404, 'Character not found.')
    glossary = db.all_rows('SELECT id,part,spelling,reading FROM project_name_glossary WHERE character_context_id=? ORDER BY id',
                           (character_id,), conn=conn)
    outfits = outfit_library.rows(conn, project_id, [character_id])
    content = {'character': row, 'glossary': glossary, 'outfits': outfits}
    version = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return {**content, 'version': version, 'thumbnail': registration_images.thumbnail(project_id, character_id).is_file()}


@router.get(PATH + '/{character_id}')
def get_registration(project_id: str, character_id: UUID):
    from .main import list_voice_samples

    character_id = str(character_id)
    with db.connect() as conn:
        conn.execute('BEGIN')
        data = record(conn, project_id, character_id)
    voices = list_voice_samples(project_id, character_id)
    return {**data, 'voice': voices}


@router.get(PATH + '/{character_id}/image')
def get_thumbnail(project_id: str, character_id: UUID):
    row = db.one("SELECT id,project_id FROM context_items WHERE id=? AND project_id=? AND kind='character'",
                 (str(character_id), project_id))
    path = registration_images.thumbnail(row['project_id'], row['id']) if row else None
    if not path or not path.is_file():
        raise HTTPException(404, 'Character image not found.')
    return FileResponse(path, media_type='image/jpeg')


def reconcile(conn, project_id, character_id, rows, submitted, kind):
    existing = {row['id']: row for row in rows}
    ids = [str(item.id) for item in submitted if item.id]
    if len(set(ids)) != len(ids) or set(ids) - existing.keys():
        raise HTTPException(422, 'A submitted entry does not belong to this character.')
    table = 'project_name_glossary' if kind == 'glossary' else 'character_outfits'
    for key in existing.keys() - set(ids):
        conn.execute(f'DELETE FROM {table} WHERE id=?', (key,))
    changed = len(existing) != len(ids)
    for item in submitted:
        prior = existing.get(str(item.id))
        if kind == 'glossary':
            spelling = item.spelling.strip()
            if not spelling:
                raise HTTPException(422, 'Enter a spelling and a kana reading.')
            values = (item.part, spelling, item.reading)
            if prior and values == (prior['part'], prior['spelling'], prior['reading']):
                continue
            if prior:
                conn.execute('UPDATE project_name_glossary SET part=?,spelling=?,reading=? WHERE id=?', (*values, prior['id']))
            else:
                conn.execute('INSERT INTO project_name_glossary(id,project_id,character_context_id,part,spelling,reading) VALUES(?,?,?,?,?,?)',
                             (str(uuid4()), project_id, character_id, *values))
        else:
            name, description = item.name.strip(), item.description.strip()
            if not name or not description:
                raise HTTPException(422, 'Give each outfit a name and description.')
            summary = item.summary.strip() or outfit_library.summarize(description)
            if prior and (name, description, item.summary) == (prior['name'], prior['description'], prior['summary']):
                continue
            if prior:
                conn.execute('UPDATE character_outfits SET name=?,description=?,summary=?,needs_review=0,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                             (name, description, summary, prior['id']))
            else:
                outfit_library.insert(conn, project_id, character_id, name, description, summary)
        changed = True
    return changed


def save_voices(conn, project_id, character_id, body):
    from .main import validate_voice_ranges

    jobs, deleted = [], []
    seen = set()
    for sample_id in body.remove_voice:
        row = conn.execute('DELETE FROM character_voice_samples WHERE id=? AND character_context_id=? RETURNING id',
                           (str(sample_id), character_id)).fetchone()
        if row:
            deleted.append(media.voice_reference_path(project_id, row['id']))
    for sample in body.voice:
        episode_id = str(sample.episode_id)
        if episode_id in seen:
            raise HTTPException(422, 'Combine the ranges from each episode into one sample.')
        seen.add(episode_id)
        episode = db.one('SELECT * FROM episodes WHERE id=? AND project_id=?', (episode_id, project_id), conn=conn)
        if not episode or not episode['has_audio'] or not episode['duration'] or not (media.episode_dir(project_id, episode_id) / 'audio.wav').is_file():
            raise HTTPException(422, 'Choose prepared audio from this project.')
        ranges = sorted((item.model_dump() for item in sample.ranges), key=lambda item: item['start_time'])
        validate_voice_ranges(episode, ranges)
        prior = conn.execute("""SELECT id FROM jobs WHERE episode_id=? AND pipeline='voice_sample'
                                AND json_extract(payload_json,'$.registration_save_id')=? AND json_extract(payload_json,'$.character_context_id')=?""",
                             (episode_id, str(body.save_id), character_id)).fetchone()
        jobs.append(prior['id'] if prior else worker.enqueue_in(conn, episode_id, 'voice_sample',
                    {'character_context_id': character_id, 'ranges': ranges, 'registration_save_id': str(body.save_id)}))
    return jobs, deleted


def insert_character(conn, project_id, character_id, body, title, scope):
    if not body.detection_id:
        conn.execute("INSERT INTO context_items(id,project_id,episode_id,kind,title,content,is_main) VALUES(?,?,?,'character',?,?,?)",
                     (character_id, project_id, scope, title, body.content, body.is_main))
        structured_results.invalidate_context(conn, project_id, scope)
        return None
    detection_id = str(body.detection_id)
    episode_id = character_decisions_episode(conn, project_id, detection_id)
    if scope is not None and scope != episode_id:
        raise HTTPException(422, 'Choose the source episode or the entire project.')
    try:
        created = character_decisions.create_character(conn, detection_id,
                    {'title': title, 'content': body.content, 'is_main': body.is_main, 'episode_only': bool(scope)},
                    character_id=character_id, describe=body.describe)
    except character_decisions.Refused as exc:
        raise HTTPException(exc.args[1] if len(exc.args) > 1 else 409, str(exc.args[0])) from exc
    agent_log.note_override(conn, 'character_detections', detection_id)
    character_decisions.close_settled(conn, episode_id)
    return created['job_id']


def character_decisions_episode(conn, project_id, detection_id):
    row = db.one('''SELECT c.episode_id FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id
                    JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id WHERE d.id=? AND e.project_id=?''',
                 (detection_id, project_id), conn=conn)
    if not row:
        raise HTTPException(404, 'Detection not found.')
    return row['episode_id']


@router.post(PATH)
def save_registration(project_id: str, draft: str = Form(), thumbnail: UploadFile | None = File(default=None)):
    try:
        body = Registration.model_validate_json(draft)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_input=False, include_context=False)) from exc
    title, character_id = body.title.strip(), str(body.character_id)
    scope = str(body.episode_id) if body.episode_id else None
    if not title or unregistered.is_card(title):
        raise HTTPException(422, 'Enter a valid character name.')
    if body.version is not None and (thumbnail or body.detection_id or body.describe):
        raise HTTPException(422, 'A frame or box is used when creating a character.')
    if body.describe and not body.detection_id:
        raise HTTPException(422, 'Select a first-frame box to describe.')
    image = registration_images.read_image(thumbnail)[0] if thumbnail else None
    stored_image = None
    profile_job = None
    try:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            project(conn, project_id)
            if scope and not conn.execute('SELECT 1 FROM episodes WHERE id=? AND project_id=?', (scope, project_id)).fetchone():
                raise HTTPException(422, 'Choose an episode in this project.')
            previous = None
            if body.version:
                previous = record(conn, project_id, character_id)
                if previous['version'] != body.version:
                    raise HTTPException(409, 'The character changed. Reopen the editor to load the latest details.')
            elif conn.execute('SELECT 1 FROM context_items WHERE id=?', (character_id,)).fetchone():
                current = record(conn, project_id, character_id)
                if current['character']['title'] != title:
                    raise HTTPException(409, 'This character ID has already been used.')
                jobs = db.all_rows("""SELECT id,pipeline FROM jobs WHERE json_extract(payload_json,'$.character_context_id')=?
                                  AND (json_extract(payload_json,'$.registration_save_id')=? OR pipeline='character_profile')""",
                                  (character_id, str(body.save_id)), conn=conn)
                return {**current, 'voice_job_ids': [job['id'] for job in jobs if job['pipeline'] == 'voice_sample'],
                        'profile_job_id': next((job['id'] for job in jobs if job['pipeline'] == 'character_profile'), None)}

            if not previous or title != previous['character']['title']:
                names = conn.execute("SELECT title FROM context_items WHERE project_id=? AND kind='character' AND id<>?", (project_id, character_id))
                if any(other['title'].strip().casefold() == title.casefold() for other in names):
                    raise HTTPException(409, 'A character with this name already exists in the project.')

            if previous:
                prior = previous['character']
                if (title, body.content, scope, body.is_main) != (prior['title'], prior['content'], prior['episode_id'], bool(prior['is_main'])):
                    conn.execute('UPDATE context_items SET title=?,content=?,episode_id=?,is_main=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                                 (title, body.content, scope, body.is_main, character_id))
                    for changed_scope in {prior['episode_id'], scope} if scope and prior['episode_id'] else {None}:
                        structured_results.invalidate_context(conn, project_id, changed_scope)
            else:
                profile_job = insert_character(conn, project_id, character_id, body, title, scope)

            if reconcile(conn, project_id, character_id, previous['glossary'] if previous else [], body.glossary, 'glossary'):
                from .main import glossary_changed
                glossary_changed(conn, project_id)
            reconcile(conn, project_id, character_id, previous['outfits'] if previous else [], body.outfits, 'outfits')
            jobs, deleted = save_voices(conn, project_id, character_id, body)
            if image:
                stored_image = registration_images.thumbnail(project_id, character_id)
                registration_images.store_thumbnail(stored_image, image)
            result = record(conn, project_id, character_id)
    except Exception as exc:
        if stored_image:
            registration_images.unlink(stored_image)
        if isinstance(exc, sqlite3.IntegrityError):
            raise HTTPException(409, 'A glossary entry or outfit is duplicated.') from exc
        raise
    for path in deleted:
        registration_images.unlink(path)
    if jobs or profile_job:
        worker.wake_event.set()
    return {**result, 'voice_job_ids': jobs, 'profile_job_id': profile_job}
