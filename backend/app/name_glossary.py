"""User-maintained readings and spellings of project names."""
import uuid

from . import db

SEED_PROJECT = 'd66e89d5-79ba-4c2b-9f9c-a4285ffa5615'
SEED_NAMES = {
    'Sakuta Azusagawa': [('梓川', 'あずさがわ'), ('咲太', 'さくた')],
    'Mai Sakurajima': [('桜島', 'さくらじま'), ('麻衣', 'まい')],
    'Kaede Azusagawa': [('梓川', 'あずさがわ'), ('かえで', 'かえで')],
    'Rio Futaba': [('双葉', 'ふたば'), ('理央', 'りお')],
    'Yuuma Kunimi': [('国見', 'くにみ'), ('佑真', 'ゆうま')],
    'Tomoe Koga': [('古賀', 'こが'), ('朋絵', 'ともえ')],
    'Shoko Makinohara': [('牧之原', 'まきのはら'), ('翔子', 'しょうこ')],
    'Saki Kamisato': [('上里', 'かみさと'), ('沙希', 'さき')],
    'Fumika Nanjou': [('南条', 'なんじょう'), ('文香', 'ふみか')],
}
SEED_OTHER = ('峰ヶ原高校', 'みねがはらこうこう')


def rows(conn, project_id, episode_id=None):
    query = """SELECT g.*,c.title AS character_name FROM project_name_glossary g
        LEFT JOIN context_items c ON c.id=g.character_context_id WHERE g.project_id=?"""
    args = [project_id]
    if episode_id:
        query += ' AND (c.episode_id IS NULL OR c.episode_id=?)'
        args.append(episode_id)
    return db.all_rows(query + ' ORDER BY c.title,g.part,g.spelling,g.id', args, conn=conn)


def seed(conn):
    flag = 'name-glossary-bunny-girl-senpai-v1'
    if conn.execute('SELECT 1 FROM app_flags WHERE name=?', (flag,)).fetchone():
        return
    if not conn.execute('SELECT 1 FROM projects WHERE id=?', (SEED_PROJECT,)).fetchone():
        return
    characters = {row['title']: row['id'] for row in conn.execute("""SELECT id,title FROM context_items
        WHERE project_id=? AND kind='character' AND episode_id IS NULL ORDER BY id""", (SEED_PROJECT,))}
    values = [(str(uuid.uuid4()), SEED_PROJECT, *SEED_OTHER, None, None)]
    for title, parts in SEED_NAMES.items():
        if title in characters:
            values.extend((str(uuid.uuid4()), SEED_PROJECT, spelling, reading, characters[title], part)
                          for part, (spelling, reading) in zip(('family', 'given'), parts))
    conn.executemany("""INSERT OR IGNORE INTO project_name_glossary
        (id,project_id,spelling,reading,character_context_id,part) VALUES (?,?,?,?,?,?)""", values)
    conn.execute('INSERT INTO app_flags(name) VALUES(?)', (flag,))
