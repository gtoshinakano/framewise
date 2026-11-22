"""Outfit library routes: the project's named outfits per character (see ``outfit_library``)."""
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db, outfit_library

router = APIRouter()


class OutfitEntry(BaseModel):
    character_context_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(default='', max_length=400)
    description: str = Field(min_length=1, max_length=6000)


def require_project(conn, project_id):
    if not conn.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
        raise HTTPException(404, 'Project not found')


def save(project_id, body, outfit_id=None):
    name, description = body.name.strip(), body.description.strip()
    if not name or not description:
        raise HTTPException(422, 'Give the outfit a name and a description.')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require_project(conn, project_id)
        if not conn.execute("SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character'", (body.character_context_id, project_id)).fetchone():
            raise HTTPException(422, 'Choose a character from this project.')
        summary = body.summary.strip() or outfit_library.summarize(description)
        try:
            if outfit_id:
                # An edit by the user settles any doubt about text that was cut off when the outfit was imported.
                cursor = conn.execute("""UPDATE character_outfits SET character_context_id=?,name=?,summary=?,description=?,needs_review=0,updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND project_id=?""", (body.character_context_id, name, summary, description, outfit_id, project_id))
                if not cursor.rowcount:
                    raise HTTPException(404, 'Outfit not found')
            else:
                outfit_library.insert(conn, project_id, body.character_context_id, name, description, summary, origin='user')
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, 'This character already has an outfit with that name.') from exc
        return outfit_library.rows(conn, project_id)


@router.get('/api/projects/{project_id}/outfits')
def list_outfits(project_id: str):
    with db.connect() as conn:
        require_project(conn, project_id)
        return outfit_library.rows(conn, project_id)


@router.post('/api/projects/{project_id}/outfits')
def create_outfit(project_id: str, body: OutfitEntry):
    return save(project_id, body)


@router.put('/api/projects/{project_id}/outfits/{outfit_id}')
def update_outfit(project_id: str, outfit_id: str, body: OutfitEntry):
    return save(project_id, body, outfit_id)


@router.delete('/api/projects/{project_id}/outfits/{outfit_id}')
def delete_outfit(project_id: str, outfit_id: str):
    with db.connect() as conn:
        cursor = conn.execute('DELETE FROM character_outfits WHERE id=? AND project_id=?', (outfit_id, project_id))
        if not cursor.rowcount:
            raise HTTPException(404, 'Outfit not found')
        return outfit_library.rows(conn, project_id)
