"""Resolve a job kind's declared footprint under the claim transaction."""
import json
from dataclasses import dataclass

from . import job_kinds

REMOTE_OWNER = 'remote:'
PROJECT_DOMAINS = {'context', 'voices'}


def text_provider(payload):
    return payload.get('text_provider') or (payload.get('config') or {}).get('text_provider') or 'openrouter'


def _cut_ids(conn, job, scope):
    if scope.sequence_id:
        from . import sequence_store
        sequence = sequence_store.get(conn, scope.sequence_id)
        cuts = sequence_store.current_cuts(conn, sequence) if sequence else None
        return frozenset(cut['id'] for cut in cuts) if cuts else None
    if scope.cut_range is not None:
        return frozenset(row['id'] for row in conn.execute('''SELECT id FROM cuts
            WHERE episode_id=? AND active=1 AND cut_index BETWEEN ? AND ?''', (job['episode_id'], *scope.cut_range)))
    return scope.cut_ids


def _neighbor_ids(conn, episode_id, cut_ids):
    if not cut_ids:
        return cut_ids
    rows = conn.execute('''SELECT DISTINCT adjacent.id FROM json_each(?) chosen JOIN cuts c ON c.id=chosen.value
        JOIN cuts adjacent ON adjacent.episode_id=c.episode_id AND adjacent.active=1 AND adjacent.cut_index BETWEEN c.cut_index-1 AND c.cut_index+1
        WHERE c.episode_id=? AND c.active=1''', (json.dumps(sorted(cut_ids)), episode_id))
    return cut_ids | {row['id'] for row in rows}


@dataclass(frozen=True)
class Footprint:
    episode_id: str
    project_id: str
    owner: str
    resources: frozenset[str]
    reads: dict
    writes: dict


def footprint(conn, job, owner):
    kind = job_kinds.kind_of(job['pipeline'])
    payload = json.loads(job['payload_json'])
    scope = kind.scope(payload) if kind.scope else job_kinds.Scope()
    cut_ids = _cut_ids(conn, job, scope)
    nearby = _neighbor_ids(conn, job['episode_id'], cut_ids) if scope.adjacent_reads else cut_ids
    resources = kind.resources
    if 'text' in resources:
        resources = resources - {'text'}
        if not owner.startswith(REMOTE_OWNER) and text_provider(payload) == 'local':
            resources |= {'gpu_text'}
    project_id = conn.execute('SELECT project_id FROM episodes WHERE id=?', (job['episode_id'],)).fetchone()['project_id']
    reads = {domain: nearby if domain in scope.adjacent_reads else cut_ids for domain in scope.reads | scope.adjacent_reads}
    writes = dict.fromkeys(scope.writes, cut_ids)
    return Footprint(job['episode_id'], project_id, owner, resources, reads, writes)


def _overlap(write, write_ids, read, read_ids, same_episode, same_project):
    if write == 'episode' or read == 'episode':
        return same_episode
    if write != read and not (write.startswith(read + ':') or read.startswith(write + ':')):
        return False
    if write.partition(':')[0] in PROJECT_DOMAINS:
        return same_project
    return same_episode and (write_ids is None or read_ids is None or not write_ids.isdisjoint(read_ids))


def conflict(first, second):
    same_episode = first.episode_id == second.episode_id
    gpu = {'gpu', 'gpu_text'}
    if first.owner != second.owner and first.resources & gpu and second.resources & gpu:
        # The local text server has multiple compatible slots for work on one episode.
        if 'gpu' in first.resources | second.resources or not same_episode:
            return True
    same_project = first.project_id == second.project_id
    if not same_episode and not same_project:
        return False
    return any(_overlap(output, output_ids, domain, cut_ids, same_episode, same_project)
               for writer, reader in ((first, second), (second, first))
               for output, output_ids in writer.writes.items()
               for domains in (reader.reads, reader.writes)
               for domain, cut_ids in domains.items())


def active_leases(conn, now):
    return {row['id']: footprint(conn, row, row['lease_owner']) for row in conn.execute('''SELECT * FROM jobs
        WHERE lease_owner IS NOT NULL AND lease_until>?''', (now,))}
