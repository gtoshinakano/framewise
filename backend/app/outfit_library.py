"""Project-wide library of what each character wears: one named outfit with a complete description.

Every episode of the project shares it. The Sequence director proposes outfits in the bible (`adopt` creates the new
ones in the transaction that saves the bible); the user can read and edit all of them in the Outfits dialog. The full
description is what the consumer app needs to make a reusable character sheet, so it travels in the sequence download.
"""
import json
import re
import uuid
from pathlib import Path

from . import db

SEED_FLAG = 'outfit-library-seed-v1'
SEED_FILE = Path(__file__).with_name('outfit_seed.json')
SUMMARY_CHARS = 160
MIN_DESCRIPTION = 80  # a director proposal shorter than this cannot make a character sheet
ORIGINS = ('seed', 'user', 'director')


def summarize(description, limit=SUMMARY_CHARS):
    """The first sentence of a description, cut at a word when it is still too long."""
    text = re.sub(r'\s+', ' ', str(description or '')).strip()
    sentence = re.match(r'.+?[.!?](?=\s|$)', text)
    text = sentence.group(0) if sentence else text
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0].rstrip(' ,;:') + '…'


def rows(conn, project_id, character_ids=None):
    query = """SELECT o.*,c.title AS character_name FROM character_outfits o
        JOIN context_items c ON c.id=o.character_context_id WHERE o.project_id=?"""
    args = [project_id]
    if character_ids is not None:
        if not character_ids:
            return []
        query += f" AND o.character_context_id IN ({','.join('?' for _ in character_ids)})"
        args.extend(character_ids)
    return db.all_rows(query + ' ORDER BY c.title COLLATE NOCASE,o.name COLLATE NOCASE,o.id', args, conn=conn)


def find(conn, character_id, name):
    return conn.execute('SELECT * FROM character_outfits WHERE character_context_id=? AND lower(name)=lower(?)', (character_id, str(name or '').strip())).fetchone()


def insert(conn, project_id, character_id, name, description, summary='', origin='user', sequence_id=None, needs_review=False):
    name, description = str(name).strip(), str(description).strip()
    outfit_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO character_outfits(id,project_id,character_context_id,name,summary,description,origin,source_sequence_id,needs_review)
        VALUES (?,?,?,?,?,?,?,?,?)""", (outfit_id, project_id, character_id, name, str(summary or '').strip() or summarize(description), description, origin, sequence_id, int(needs_review)))
    return outfit_id


def adopt(conn, project_id, sequence_id, bible, origin='director'):
    """Link a bible's outfits to the library, creating the ones that do not exist yet. Returns the bible with its entries settled.

    The library text is the truth: an outfit that exists is never overwritten by a bible, and the full description of a new
    one lives only in the library (the bible keeps the name and the summary). `status` says whether this sequence created it.
    """
    settled = []
    for entry in bible.get('outfits') or []:
        row = find(conn, entry['character_context_id'], entry['name'])
        if row is None and str(entry.get('description') or '').strip():
            outfit_id = insert(conn, project_id, entry['character_context_id'], entry['name'], entry['description'], entry.get('summary'), origin, sequence_id)
            row = conn.execute('SELECT * FROM character_outfits WHERE id=?', (outfit_id,)).fetchone()
        item = {key: value for key, value in entry.items() if key != 'description'}
        if row is not None:
            item.update(outfit_id=row['id'], name=row['name'], summary=row['summary'], status='new' if row['source_sequence_id'] == sequence_id and row['origin'] != 'seed' else 'existing')
        settled.append(item)
    return {**bible, 'outfits': settled}


def adopt_for(conn, sequence, bible, origin='director'):
    """`adopt` for a sequence row: finds its project and does nothing when the bible has no outfits."""
    if not bible.get('outfits'):
        return bible
    project_id = conn.execute('SELECT project_id FROM episodes WHERE id=?', (sequence['episode_id'],)).fetchone()['project_id']
    return adopt(conn, project_id, sequence['id'], bible, origin)


def seed(conn):
    """Insert the outfit sheets written by hand in the consumer app, for the characters that exist by that name."""
    if conn.execute('SELECT 1 FROM app_flags WHERE name=?', (SEED_FLAG,)).fetchone():
        return
    entries = json.loads(SEED_FILE.read_text(encoding='utf-8'))
    for character in conn.execute("SELECT id,project_id,title FROM context_items WHERE kind='character' ORDER BY created_at,id").fetchall():
        for entry in entries:
            if entry['character'].casefold() == character['title'].strip().casefold() and not find(conn, character['id'], entry['name']):
                insert(conn, character['project_id'], character['id'], entry['name'], entry['description'], origin='seed', needs_review=entry['needs_review'])
    conn.execute('INSERT INTO app_flags(name) VALUES(?)', (SEED_FLAG,))
