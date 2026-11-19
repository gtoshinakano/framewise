"""Episode-level subtitle track storage.

Subtitle lines belong to the episode and to a time range, never to a cut. A cut sees the lines that overlap its
boundaries, so moving or re-detecting cuts changes nothing here. A run becomes current once it completes; older
runs stay for history and are only returned when asked for.
"""
import json

from . import db


def _loads(value, default):
    try:
        result = json.loads(value) if value else default
    except (TypeError, ValueError):
        return default
    return result if isinstance(result, type(default)) else default


def serialize_run(row, with_prompt=False):
    """A run as the API shows it. The prompt is long and only returned with the history view."""
    run = dict(row)
    run['band'] = _loads(run.pop('band_json', None), {})
    run['diagnostics'] = _loads(run.pop('diagnostics_json', None), {})
    run['settings'] = _loads(run.pop('settings_json', None), {})
    run.pop('scan_hash', None)
    if not with_prompt:
        run.pop('prompt', None)
        run.pop('input_fingerprint', None)
        run['settings'].pop('inputs', None)
    return run


def invalidate_probes(conn, episode_id):
    conn.execute("""UPDATE subtitle_runs SET stale_reason='Transcript timing changed' WHERE id=(
        SELECT id FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1)""",
        (episode_id,))


def current_run(episode_id, conn=None):
    """The latest completed run of the episode, or None."""
    if conn is None:
        with db.connect() as conn:
            return current_run(episode_id, conn)
    row = conn.execute("SELECT * FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1", (episode_id,)).fetchone()
    return serialize_run(row) if row else None


def line_view(row):
    return {'id': row['id'], 'start_time': row['start_time'], 'end_time': row['end_time'], 'text': row['text'], 'confidence': row['confidence']}


def lines(episode_id, start=None, end=None, run_id=None, conn=None):
    """Lines of the current run (or ``run_id``) overlapping ``[start, end)``; all of them without a range."""
    if conn is None:
        with db.connect() as conn:
            return lines(episode_id, start, end, run_id, conn)
    if run_id is None:
        run = conn.execute("SELECT id FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1", (episode_id,)).fetchone()
        if not run:
            return []
        run_id = run['id']
    sql, args = 'SELECT * FROM subtitle_lines WHERE run_id=?', [run_id]
    if start is not None and end is not None:
        sql += ' AND start_time<? AND end_time>?'
        args += [end, start]
    return [line_view(row) for row in conn.execute(sql + ' ORDER BY start_time,line_index', args)]


def episode_language(episode_id, conn=None):
    """The language the user stated for this episode's subtitles, as saved by the last run request."""
    if conn is None:
        with db.connect() as conn:
            return episode_language(episode_id, conn)
    row = conn.execute('SELECT subtitle_language FROM episodes WHERE id=?', (episode_id,)).fetchone()
    return row['subtitle_language'] if row and row['subtitle_language'] else None


def cut_context(conn, cut, episode_id):
    """What cut analysis is given: the language, the strip and the lines that overlap this cut.

    Lines may extend beyond the cut; ``crosses_cut`` says so, and their times stay episode-relative like
    the transcript's.
    """
    run = current_run(episode_id, conn)
    if not run:
        return {'extracted': False, 'language': episode_language(episode_id, conn), 'band': None, 'lines': []}
    overlap = lines(episode_id, cut['start_time'], cut['end_time'], run['id'], conn)
    for line in overlap:
        line['crosses_cut'] = line['start_time'] < cut['start_time'] or line['end_time'] > cut['end_time']
    return {'extracted': True, 'language': run['language'], 'band': run['band'] or None, 'lines': overlap}


def reading_rows(run_id, conn):
    return conn.execute('SELECT * FROM subtitle_readings WHERE run_id=? ORDER BY event_index', (run_id,)).fetchall()
