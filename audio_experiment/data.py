"""Reads the database COPY (never the live file) into plain dicts for the experiment. Stdlib only."""
import json
import sqlite3

import common
import lines as L


def connect(path):
    conn = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)   # a private copy on a read-only mount: no -shm/-wal files can be made
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn, sql, args=()):
    return [dict(r) for r in conn.execute(sql, args)]


def episode_number(episode_id):
    for number, value in common.EPISODES.items():
        if value == episode_id:
            return number
    return None


def characters(conn, project_id=common.PROJECT_ID):
    return rows(conn, "SELECT id,title,episode_id,is_main FROM context_items WHERE project_id=? AND kind='character' ORDER BY title", (project_id,))


def voice_samples(conn, project_id=common.PROJECT_ID):
    """Voice samples of the project's characters (an orphan sample of a deleted episode is ignored)."""
    samples = []
    for row in rows(conn, """SELECT v.id,v.character_context_id,c.title,v.episode_id,v.start_time,v.end_time,v.embedding_json,v.ranges_json,v.model_id
            FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
            WHERE c.project_id=? AND c.kind='character' ORDER BY c.title,v.id""", (project_id,)):
        number = episode_number(row['episode_id'])
        if number is None:
            continue
        ranges = json.loads(row['ranges_json']) if row['ranges_json'] else [{'start_time': row['start_time'], 'end_time': row['end_time']}]
        samples.append({'id': row['id'], 'character': row['title'], 'character_id': row['character_context_id'], 'episode': number,
                        'ranges': [{'start': r['start_time'], 'end': r['end_time']} for r in ranges],
                        'embedding': json.loads(row['embedding_json']), 'model_id': row['model_id']})
    return samples


def baseline_rows(conn, episode_id):
    return rows(conn, """SELECT t.id,t.start_time,t.end_time,t.text,t.speaker_label,t.character_context_id AS character_id,c.title AS character,t.origin,t.model_id
            FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
            WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 ORDER BY t.start_time,t.id""", (episode_id,))


def baseline_job_seconds(conn, episode_id):
    """Wall seconds of the last full-episode transcribe job (None when it never recorded a start)."""
    for row in rows(conn, """SELECT payload_json,started_at,finished_at,(julianday(finished_at)-julianday(started_at))*86400 AS s FROM jobs
            WHERE episode_id=? AND pipeline='transcribe' AND status='done' ORDER BY finished_at DESC""", (episode_id,)):
        payload = json.loads(row['payload_json'] or '{}')
        if not payload.get('range') and row['s']:
            return round(row['s'])
    return None


def subtitles(conn, episode_id):
    run = rows(conn, "SELECT id,language FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY created_at DESC LIMIT 1", (episode_id,))
    if not run:
        return []
    return rows(conn, 'SELECT line_index,start_time,end_time,text FROM subtitle_lines WHERE run_id=? ORDER BY start_time,line_index', (run[0]['id'],))


def cuts(conn, episode_id):
    return rows(conn, 'SELECT id,cut_index,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time', (episode_id,))


def cut_roster(conn, episode_id):
    """cut id -> names of the characters the latest detection run labelled in the cut's first frame (may be empty or missing)."""
    result = {}
    for row in rows(conn, """SELECT r.cut_id,d.character_context_id,c.title FROM character_detection_runs r
            JOIN character_detections d ON d.run_id=r.id JOIN context_items c ON c.id=d.character_context_id JOIN cuts k ON k.id=r.cut_id
            WHERE k.episode_id=? AND k.active=1 AND r.created_at=(SELECT MAX(created_at) FROM character_detection_runs WHERE cut_id=r.cut_id)""", (episode_id,)):
        result.setdefault(row['cut_id'], [])
        if row['title'] not in result[row['cut_id']]:
            result[row['cut_id']].append(row['title'])
    return result


def context_breakdown(conn, episode_id):
    row = rows(conn, "SELECT title,length(content) AS n FROM context_items WHERE episode_id=? AND kind='chapter_happenings'", (episode_id,))
    return row


def merged_baseline(brows):
    """The app's merged_transcript over the stored rows (gap 0.6 s), with the character link kept."""
    for row in brows:
        row.setdefault('character_id', None)
    merged = L.merge_rows(brows)
    for number, line in enumerate(merged, 1):
        line['id'] = f'B{number}'
        line['speaker'] = line.get('character') or None
    return merged


def anchors(samples, brows, number):
    """Anchors of one episode: voice-sample ranges (kind 'sample') and hand/assistant corrected rows with a character (kind 'manual')."""
    result = []
    for sample in samples:
        if sample['episode'] != number:
            continue
        for index, rng in enumerate(sample['ranges']):
            result.append({'episode': number, 'kind': 'sample', 'character': sample['character'], 'start': rng['start'], 'end': rng['end'],
                           'range_id': f"{sample['id']}:{index}", 'sample_id': sample['id']})
    for row in brows:
        if row['origin'] != 'model' and row.get('character'):
            result.append({'episode': number, 'kind': 'manual', 'character': row['character'], 'start': row['start_time'], 'end': row['end_time'],
                           'range_id': f"row:{row['id']}", 'text': row['text']})
    return result


def prepare_episode(conn, number, samples):
    episode_id = common.EPISODES[number]
    meta = rows(conn, 'SELECT id,title,duration,subtitle_language FROM episodes WHERE id=?', (episode_id,))[0]
    brows = baseline_rows(conn, episode_id)
    return {'number': number, 'episode_id': episode_id, 'title': meta['title'], 'duration': meta['duration'],
            'baseline_rows': brows, 'baseline_lines': merged_baseline([dict(r) for r in brows]),
            'baseline_job_seconds': baseline_job_seconds(conn, episode_id),
            'subtitles': subtitles(conn, episode_id), 'cuts': cuts(conn, episode_id), 'roster': cut_roster(conn, episode_id),
            'anchors': anchors(samples, brows, number), 'breakdown': context_breakdown(conn, episode_id)}
