"""Unregistered characters: numbered, reusable cards for minor people who need a line or a box but no character record.

A card is an ordinary ``context_items`` character row with ``unregistered=1`` and the fixed title ``Unregistered N``,
project-wide and without details, so every picker, check and foreign key that takes a character takes it too. The person
picks the same number for one person's boxes and lines; prompts then use the title as a stable handle, and two numbers
are two people. A card is never a recognition example or candidate, never a voice sample or reference, never an
automatic signal of the voice resolver and never gets a profile, sheet or outfit: those paths filter it with
``NOT_CARD``. No FastAPI here, so every worker may import it.
"""
import re
import uuid

PREFIX = 'Unregistered'
TITLE = re.compile(rf'^{PREFIX} (\d+)$')
HANDLE = re.compile(rf'\b{PREFIX} \d+\b')  # a card's title inside a text
# SQL: the context_items row aliased ``alias`` is not a card.
NOT_CARD = 'COALESCE({alias}.unregistered,0)=0'
RULE = (f'A name of the form "{PREFIX} N" is a placeholder for a minor person who has no character record. Use it exactly as '
        'written, as that person\'s name; different numbers are different people. Never replace it with a guessed name, role '
        'or identity, and describe that person only from what the picture shows.')


def not_card(alias='ci'):
    return NOT_CARD.format(alias=alias)


def is_card(row):
    """True for a card row (a dict or sqlite row with ``unregistered``), or for a card title given as text."""
    if isinstance(row, str):
        return bool(TITLE.match(row.strip()))
    if row is None:
        return False
    try:
        return bool(row['unregistered'])
    except (KeyError, IndexError, TypeError):
        return False


def present(names):
    """Whether any of ``names`` is a card title (the prompts add ``RULE`` only then)."""
    return any(isinstance(name, str) and is_card(name) for name in names or ())


def card_ids(conn, project_id):
    return {row[0] for row in conn.execute("SELECT id FROM context_items WHERE project_id=? AND kind='character' AND unregistered=1",
                                           (project_id,))}


def card_ids_for(conn, character_ids):
    """The cards among ``character_ids``."""
    ids = [item for item in set(character_ids or ()) if item]
    if not ids:
        return set()
    marks = ','.join('?' for _ in ids)
    return {row[0] for row in conn.execute(f'SELECT id FROM context_items WHERE id IN ({marks}) AND unregistered=1', ids)}


def cards(conn, project_id):
    rows = conn.execute("SELECT * FROM context_items WHERE project_id=? AND kind='character' AND unregistered=1", (project_id,)).fetchall()
    return sorted((dict(row) for row in rows), key=number)


def number(card):
    match = TITLE.match((card.get('title') if isinstance(card, dict) else card) or '')
    return int(match.group(1)) if match else 0


def create_next(conn, project_id):
    """A new card with the lowest free number, inside the caller's transaction. Returns its row."""
    taken = {number(card) for card in cards(conn, project_id)}
    taken |= {int(match.group(1)) for (title,) in conn.execute("SELECT title FROM context_items WHERE project_id=? AND kind='character'",
                                                               (project_id,)) if (match := TITLE.match(title or ''))}
    value = next(n for n in range(1, len(taken) + 2) if n not in taken)
    card_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO context_items (id,project_id,episode_id,kind,title,content,is_main,unregistered)
                    VALUES (?,?,NULL,'character',?,'',0,1)""", (card_id, project_id, f'{PREFIX} {value}'))
    return dict(conn.execute('SELECT * FROM context_items WHERE id=?', (card_id,)).fetchone())
