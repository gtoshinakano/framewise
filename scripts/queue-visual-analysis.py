#!/usr/bin/env python3
"""Wait for an episode's subtitle extraction, free the audio worker's GPU memory, queue visual analysis.

    scripts/queue-visual-analysis.py                  # latest episode
    scripts/queue-visual-analysis.py "Episode 6"      # by title or id
    scripts/queue-visual-analysis.py --dry-run        # show what would happen, change nothing

Steps (stdlib only, no model calls, safe to leave running in the background):
 1. Poll until the episode has no pending prepare/hardcuts/cutframes/transcribe/voice_sample/
    subtitles/character_detect job. Abort if the latest subtitles job failed or was cancelled.
 2. If VRAM is still held above --vram-threshold-mib, and no audio job is running anywhere,
    restart the audio worker container (it drops its ROCm context; models are loaded per job).
 3. POST /api/episodes/{id}/visual-analysis (first frame, last frame, video analysis).
    Retries on HTTP 409 (the API says "wait for media preparation or detection").

The database is only read (read-only SQLite inside the API container); all writes go through the API.
"""
import argparse
import glob
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BLOCKING = ('prepare', 'hardcuts', 'cutframes', 'transcribe', 'voice_sample', 'subtitles', 'character_detect')
AUDIO = ('transcribe', 'voice_sample')
QUERY = r'''
import json, sqlite3, sys
a = json.loads(sys.argv[1])
c = sqlite3.connect("file:/data/app.sqlite3?mode=ro", uri=True)
c.row_factory = sqlite3.Row
ep = c.execute("SELECT id,title,subtitle_language FROM episodes WHERE id=? OR title=? ORDER BY created_at DESC LIMIT 1", (a["ep"], a["ep"])).fetchone() if a["ep"] else c.execute("SELECT id,title,subtitle_language FROM episodes ORDER BY created_at DESC LIMIT 1").fetchone()
out = {"episode": dict(ep) if ep else None}
if ep:
    marks = ",".join("?" * len(a["blocking"]))
    out["pending"] = [dict(r) for r in c.execute(f"SELECT pipeline,status,stage FROM jobs WHERE episode_id=? AND status IN ('queued','running','interrupted') AND pipeline IN ({marks})", [ep["id"], *a["blocking"]])]
    row = c.execute("SELECT status,error FROM jobs WHERE episode_id=? AND pipeline='subtitles' ORDER BY created_at DESC, rowid DESC LIMIT 1", (ep["id"],)).fetchone()
    out["subtitles"] = dict(row) if row else None
    marks = ",".join("?" * len(a["audio"]))
    out["audio_running"] = c.execute(f"SELECT COUNT(*) FROM jobs WHERE status='running' AND pipeline IN ({marks})", a["audio"]).fetchone()[0]
print(json.dumps(out))
'''


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def read_state(args):
    payload = json.dumps({'ep': args.episode, 'blocking': BLOCKING, 'audio': AUDIO})
    result = subprocess.run(['docker', 'exec', '-i', args.api_container, 'python3', '-c', QUERY, payload], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-300:] or 'docker exec failed')
    return json.loads(result.stdout)


def vram_used_mib():
    values = []
    for path in glob.glob('/sys/class/drm/card*/device/mem_info_vram_used'):
        try:
            values.append(int(open(path).read()) // 1048576)
        except (OSError, ValueError):
            pass
    return max(values) if values else None


def post(url):
    request = urllib.request.Request(url, method='POST', data=b'')
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {'detail': exc.read().decode(errors='replace')[:300]}


def wait_for_subtitles(args):
    last = None
    while True:
        try:
            state = read_state(args)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            log(f'state read failed ({exc}); retrying')
            time.sleep(args.interval)
            continue
        if not state['episode']:
            sys.exit(f'Episode not found: {args.episode or "(latest)"}')
        sub = state['subtitles']
        if not state['pending']:
            if sub and sub['status'] in ('failed', 'cancelled'):
                sys.exit(f'Subtitle extraction {sub["status"]}: {(sub["error"] or "")[:200]}. Nothing queued.')
            if not sub and not args.skip_subtitles:
                sys.exit('This episode has no subtitle job. Run it first, or pass --skip-subtitles.')
            return state
        summary = ', '.join(f'{j["pipeline"]}:{j["status"]} {j["stage"] or ""}'.strip() for j in state['pending'])
        if summary != last:
            log(f'waiting for {state["episode"]["title"]}: {summary}')
            last = summary
        time.sleep(args.interval)


def free_audio_gpu(args, state):
    used = vram_used_mib()
    if args.no_audio_restart:
        log(f'audio restart disabled; VRAM used: {used} MiB')
        return
    if used is None:
        log('VRAM usage unreadable; skipping audio worker restart')
        return
    if used <= args.vram_threshold_mib:
        log(f'VRAM used {used} MiB is under {args.vram_threshold_mib} MiB; audio worker left alone')
        return
    if state['audio_running']:
        log(f'VRAM used {used} MiB, but {state["audio_running"]} audio job(s) are running; not restarting')
        return
    log(f'VRAM used {used} MiB: restarting {args.audio_container} to release its GPU context')
    if args.dry_run:
        return
    result = subprocess.run(['docker', 'restart', args.audio_container], capture_output=True, text=True, timeout=120)
    if result.returncode:
        log(f'restart failed: {result.stderr.strip()[-200:]}')
        return
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(3)
        now = vram_used_mib()
        if now is not None and now <= args.vram_threshold_mib:
            break
    log(f'VRAM used after restart: {vram_used_mib()} MiB')


def queue_visual(args, episode):
    url = f'{args.api}/api/episodes/{episode["id"]}/visual-analysis?only_missing={str(not args.redo).lower()}'
    if args.characters_resolved:
        url += '&characters_resolved=true'
    if args.from_cut:
        url += f'&from_cut={args.from_cut}'
    if args.dry_run:
        log(f'dry run: would POST {url}')
        return
    for attempt in range(1, 11):
        status, body = post(url)
        if status == 200:
            log(f'queued visual analysis: {body["queued"]} cuts, {body["skipped"]} skipped, job(s) {body["job_ids"]}')
            blocked = {k: v for k, v in body['blocked'].items() if v}
            if blocked:
                log(f'blocked (characters unsettled): {blocked}')
            if not body['queued']:
                sys.exit('Nothing was queued (all stages already have results, or cuts are blocked).')
            return
        if status == 409:
            log(f'409 {body.get("detail")}; retry {attempt}/10 in {args.interval}s')
            time.sleep(args.interval)
            continue
        sys.exit(f'HTTP {status}: {body}')
    sys.exit('Gave up after repeated 409 responses.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', nargs='?', help='episode title or id (default: newest episode)')
    parser.add_argument('--api', default='http://192.168.0.6:8080', help='base URL (see APP_BIND_IP in .env)')
    parser.add_argument('--api-container', default='video-analyzer-video-analyzer-1')
    parser.add_argument('--audio-container', default='video-analyzer-audio-worker-1')
    parser.add_argument('--interval', type=int, default=30, help='seconds between polls')
    parser.add_argument('--vram-threshold-mib', type=int, default=4096, help='restart the audio worker if more VRAM than this is still used')
    parser.add_argument('--no-audio-restart', action='store_true')
    parser.add_argument('--skip-subtitles', action='store_true', help='do not require a subtitle job')
    parser.add_argument('--characters-resolved', action='store_true', help='only queue cuts whose characters are settled')
    parser.add_argument('--from-cut', type=int, help='first cut number to analyse (as shown in the editor)')
    parser.add_argument('--redo', action='store_true', help='redo stages that already have results (only_missing=false)')
    parser.add_argument('--dry-run', action='store_true', help='report only: no restart, no queueing')
    args = parser.parse_args()
    state = wait_for_subtitles(args)
    episode = state['episode']
    log(f'{episode["title"]}: media jobs finished; subtitle language {episode["subtitle_language"]}')
    free_audio_gpu(args, state)
    queue_visual(args, episode)


if __name__ == '__main__':
    main()
