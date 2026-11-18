#!/usr/bin/env python3
"""Download a model listed in backend/app/model_manifest.json, pinned and checksummed.

    python3 scripts/install-model.py --list
    python3 scripts/install-model.py lfm3-omni-2b
    python3 scripts/install-model.py lfm3-omni-2b --verify
    python3 scripts/install-model.py dwpose          # DWPose keypoints for the keyframe analysis (CPU)
    HF_TOKEN=... python3 scripts/install-model.py qwen36-vision   # the vision projector of the Qwen 3.6 text model (gated repo)

Files land in the model's directory (``default_dir`` under the project, or the directory named by
its ``dir_env`` variable) as ``<name>.partial`` while downloading, so the inference worker can show
the progress, and are renamed only after their sha256 matches. An interrupted download resumes.
Standard library only: it runs on the host, outside the containers.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / 'backend' / 'app' / 'model_manifest.json'
CHUNK = 8 * 1024 * 1024


def load():
    return json.loads(MANIFEST.read_text())


def target_dir(entry):
    configured = os.environ.get(entry.get('dir_env') or '')
    return Path(configured) if configured else PROJECT / entry['default_dir']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


class KeepTokenOnItsHost(urllib.request.HTTPRedirectHandler):
    """Hugging Face sends big files to a CDN with signed URLs; the access token must not follow the redirect there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def download(url, partial, size, token=None):
    opener = urllib.request.build_opener(KeepTokenOnItsHost)
    for attempt in range(1, 6):
        have = partial.stat().st_size if partial.exists() else 0
        if have >= size:
            return
        headers = {'Range': f'bytes={have}-', 'User-Agent': 'framewise-model-installer'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start over
                with partial.open('ab' if have else 'wb') as handle:
                    done, last = have, 0.0
                    while block := response.read(CHUNK):
                        handle.write(block)
                        done += len(block)
                        now = time.monotonic()
                        if now - last > 2:
                            print(f'\r  {partial.name}: {100 * done / size:5.1f}% of {size / 1e9:.2f} GB', end='', flush=True)
                            last = now
            print(f'\r  {partial.name}: 100.0% of {size / 1e9:.2f} GB')
            return
        except (OSError, urllib.error.URLError) as exc:
            print(f'\n  attempt {attempt} failed: {exc}; retrying', file=sys.stderr)
            time.sleep(5 * attempt)
    raise SystemExit(f'Download of {url} failed')


def install(key, entry, verify_only=False):
    folder = target_dir(entry)
    folder.mkdir(parents=True, exist_ok=True)
    ok = True
    for item in entry['files']:
        final = folder / item['name']
        if final.exists():
            if sha256(final) == item['sha256']:
                print(f'{item["name"]}: installed, checksum ok')
                continue
            print(f'{item["name"]}: checksum mismatch', file=sys.stderr)
            if verify_only:
                ok = False
                continue
            final.rename(final.with_name(final.name + '.bad'))
        if verify_only:
            print(f'{item["name"]}: not installed', file=sys.stderr)
            ok = False
            continue
        partial = final.with_name(final.name + '.partial')
        url = f'https://huggingface.co/{entry["repo"]}/resolve/{entry["revision"]}/{item["name"]}?download=true'
        token = os.environ.get('HF_TOKEN') or None
        if entry.get('gated') and not token:
            raise SystemExit(f'{entry["repo"]} is a gated repository: set HF_TOKEN to an access token and run this again')
        print(f'Downloading {item["name"]} from {entry["repo"]}@{entry["revision"][:10]}')
        download(url, partial, item['size'], token)
        actual = sha256(partial)
        if actual != item['sha256']:
            partial.unlink()
            raise SystemExit(f'{item["name"]}: sha256 {actual} does not match {item["sha256"]}; the partial file was removed')
        partial.rename(final)
        print(f'{item["name"]}: installed, checksum ok')
    print(f'{key}: {"ready" if ok else "incomplete"} in {folder}')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?')
    parser.add_argument('--list', action='store_true', help='list the installable models')
    parser.add_argument('--verify', action='store_true', help='only check the installed files')
    args = parser.parse_args()
    manifest = load()
    if args.list or not args.model:
        for key, entry in manifest.items():
            total = sum(item['size'] for item in entry['files'])
            print(f'{key:16} {entry["repo"]} ({total / 1e9:.2f} GB) -> {target_dir(entry)}')
        return
    if args.model not in manifest:
        raise SystemExit(f'Unknown model {args.model!r}; choose one of: {", ".join(manifest)}')
    if not install(args.model, manifest[args.model], args.verify):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
