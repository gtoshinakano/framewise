# Agent guide

Framewise is a local-first video analysis app. Preserve the hierarchy `Project → Episodes → Original Video` and the original upload. The initial working flow is import → background media preparation → user-triggered hard-cut detection → timeline playback and editable cut notes. AI inference is future work; do not download models without a task requiring them.

## Code map

- `backend/app/main.py`: FastAPI endpoints and input validation.
- `backend/app/db.py`: SQLite schema and connections. Database and media persist under `DATA_DIR` (`/data` in Docker).
- `context_items` stores project or episode scoped chapter, character, world, scenario, object, and general notes for future inference prompts.
- `analysis_results` stores per-cut text with revision links and optional model/job provenance; `transcript_segments` stores timed audio text and speaker/character links with the same append-a-revision approach. Default transcript reads exclude superseded and deleted revisions; `?history=true` returns the full chain. `/api/cuts/{id}/analysis-context` assembles adjacent cuts, approved results, applicable context items, chunks, and approved transcript overlap.
- `backend/app/worker.py`: one background media worker claims jobs from SQLite. Claims use a 60-second renewable lease, three-attempt interruption limit, and queued settings snapshot. Queued and interrupted jobs survive process restarts.
- `chunks` stores timed working segments for long videos; cuts are independent visual boundaries and may cross chunk edges.
- `backend/app/inference_worker.py` also claims `regen` jobs (`POST /api/cuts/{id}/regenerate-prompt`), which append a revision of only the `MiniMax animation prompt` result and fail if that prompt changed while queued. Analyzed frames are saved as `analysis/{result_id}.jpg` beside the original so `/api/analysis-results/{id}/frame` can serve them under detection overlays. Burned-in subtitles are context only: `SUBTITLE_RULE` keeps them out of every generated prompt. The UI lives in `AnalysisDialog.tsx` (centered dialog with `DetectionPreview.tsx` layers and `AnimationPromptPanel.tsx`).
- `backend/app/media.py`: FFprobe/FFmpeg preparation, waveform, scene detection, and per-cut first/last frames. Frame URLs use cut IDs so redetection can retain existing cut records.
- `frontend/src/App.tsx`: React project manager and episode editor. `Timeline.tsx` windows long recordings and clips its lanes to the visible range; `CutGallery.tsx` pages cut cards.
- The episode editor polls `/api/episodes/{id}/status` for job changes and only loads the full episode every 30 seconds while idle. Keep the waveform cache keyed to the latest completed Prepare job when changing this flow.
- `compose.yml`: persistent data and model cache; optional GPU overrides are separate.
- `deploy/framewise.service`: optional systemd unit for Cockpit Services start/stop. Update its `WorkingDirectory` if the checkout path changes.

## Invariants and extension points

- The original video must never be overwritten by proxy or inference work.
- Database IDs are UUIDs; construct media paths only from database rows and fixed filenames.
- Long-running media work belongs in the queue, not an HTTP request.
- Keep `uvicorn --workers 1` for the current deployment. The media worker runs in the API process but claims through SQLite; a future separate inference worker should claim only its own pipeline kinds and use the same lease discipline.
- A failed job records its error; reruns are allowed. Existing cut edits and result IDs are retained when redetection finds the same boundary within 0.25 seconds. Unmatched cuts become inactive archives rather than being deleted; later matching boundaries reactivate their IDs. Do not delete and reinsert cut rows: foreign keys would cascade-delete result history. The `cuts` table keeps negative `cut_index` values for archives and nonnegative indices for active timeline cuts.
- Media and model cache volumes must survive image rebuilds.
- Future ROCm/CUDA inference should be a separate optional worker image with pinned compatible PyTorch/runtime packages. Store model provenance and prompt/settings with each result. Do not put generated text into files that replace originals.
- Read `docs/model-integration.md` before implementing the requested WeVisOne-VL adapter. Its documented tasks are frame-level perception, and ROCm compatibility still needs validation. Do not download weights before a task explicitly calls for model inference.

## Verification

Run `docker compose build`, `docker compose up -d`, check `/api/health`, import a short video, wait for preparation, run hard-cut detection, seek a cut, save a note, and restart the container to confirm persistence. Verify a no-audio video and an invalid upload when changing media handling.

## This Ubuntu host

As of 2026-10-27, `/etc/systemd/system/framewise.service` is installed, enabled, and starts the Compose app. The ignored local `.env` binds port 8080 to `192.168.0.6`; update `APP_BIND_IP` if this host's LAN address changes. The service restart exercised Compose stop and start while keeping `framewise_data` and `framewise_models`. A container-backed import, preparation, cut detection, browser playback, and persistence check passed; test projects were removed afterward. The `bytoshi` shell cannot access `/var/run/docker.sock` or run sudo noninteractively, so use Cockpit or a privileged terminal to change the service. Read-only checks still work with `systemctl status framewise.service`, `journalctl -u framewise.service`, and `curl http://192.168.0.6:8080/api/health`.

- Disk hygiene: on this development host a full disk makes SQLite fail with `disk I/O error` and the API restart-loops. Run `scripts/docker-cleanup.sh` after builds (or install `deploy/framewise-cleanup.timer`); it prunes dangling images and old build cache, never volumes.
