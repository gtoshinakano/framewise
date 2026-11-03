# Framewise

A local video analysis workspace. React handles projects, episodes, playback, timeline, waveform, and editable cut notes. FastAPI and FFmpeg prepare media and detect candidate hard cuts. SQLite and media files live in a persistent Docker volume.

## Start on Ubuntu / Cockpit

From this directory:

```bash
docker compose up -d --build
docker compose ps
```

The app binds to `127.0.0.1:8080` by default. For access from devices on your trusted LAN, copy `.env.example` to `.env`, set `APP_BIND_IP` to this server's LAN IPv4 address, and start or restart the service. Then open `http://SERVER_IP:8080`. To expose Start/Stop in Cockpit's **Services** view, install the supplied systemd unit (edit `WorkingDirectory` first if this checkout moves):

```bash
sudo install -m 0644 deploy/framewise.service /etc/systemd/system/framewise.service
sudo systemctl daemon-reload
sudo systemctl enable --now framewise.service
```

Then use `framewise.service` in Cockpit's Services page. The unit runs `docker compose up -d --no-build` on start and `docker compose down` on stop; both keep the named volumes. Build updated images explicitly with `docker compose --profile inference build` before restarting the service after code changes. This keeps normal service restarts from recreating large GPU build layers. To change the bind address, port, or upload limit, edit `.env` and restart the service. The server has no built-in login: expose it only on a trusted network or behind an authenticated reverse proxy.
The named volumes default to `framewise_data` and `framewise_models`, so they are reused even if the checkout directory is renamed. Set `DATA_VOLUME` and `MODEL_VOLUME` in `.env` only when you intentionally want a different storage set.

To stop without removing data:

```bash
docker compose down
```

Do not use `docker compose down -v` unless you intend to delete the project and model volumes.

## First workflow

1. Choose **New project**, enter a project and episode name, and upload a video.
2. The queued preparation job probes the original, organizes timed chunks, creates a smaller MP4 preview, indexes frame sample times every five seconds, and extracts mono WAV audio plus waveform peaks when audio is present.
3. Open the episode editor and choose **Detect hard cuts**. FFmpeg's scene score identifies boundaries and records first and last frame timestamps for each cut. Click a chunk, cut, or thumbnail to seek the preview. Scroll over the timeline to zoom around the pointer, and drag with the middle mouse button to move the visible window. You can also use the zoom slider, overview bar, and Earlier/Later controls to navigate. The timeline defaults to a 10-minute window for videos longer than 30 minutes. The cut gallery shows 24 cuts per page. The inspector shows both cut frames, can play just the selected chunk or cut, and can edit its notes. Add manual per-cut analysis fields for first/last frames, camera movement, composition, audio matching, or generation prompts. Edits create revisions, and earlier versions remain available under **Show history**.
4. Change processing settings in the sidebar, save, then rerun **Prepare** or **Redetect cuts**. Preview width, CRF quality, encoding speed, frame interval, chunk duration, and scene threshold are configurable. Cut edits and result revisions follow boundaries that remain within 0.25 seconds of their previous positions. Boundaries that disappear move to **Archived cuts** with their notes and result history intact; a later matching redetection restores the same cut. Chunk titles and notes are retained when their start times remain the same.
5. Add characters, worlds, scenarios, or chapter notes in the project's **Context library**. Items can apply to the entire project or a specific episode.
6. Use the editor's **Audio** tab to add timed speech notes, speaker labels, and optional character links. Current transcript segments appear in overlapping cuts' input context. Editing or deleting a segment preserves earlier revisions under **Show history**.
7. Once the inference worker is set up, select a cut, choose first/last or evenly spaced frames, specify visual categories to detect, select relevant context, add cut-specific notes, and save the preset. **Queue cut** queues that cut; **Queue enabled cuts** queues all enabled timeline cuts using each one's saved preset. Disable or re-enable a cut in its inspector. A change affects future runs; already queued runs retain their original selection and settings. You can save presets, edit notes and adjust cut boundaries while visual analysis runs. The episode's **Runs, queue & history** lists media, audio and visual runs, errors, linked outputs and elapsed processing times; queued runs can be cancelled and failed runs retried. Review and revise the resulting fields. Current results are available to later analysis. Results include frame observations, raw OCR, evidence assessments with qualitative confidence, and a cut summary for still-image and animation planning. Each summary follows the same ordered topics, including characters, camera, setting, lighting, on-screen text, audio context, animation direction, and uncertainties. Each topic has a qualitative confidence label.

For long uploads, keep the browser tab open until the upload reaches 100%. Media preparation then runs in the background. The project and editor show current stage and any FFmpeg error. One media job runs at a time. Jobs are claimed from SQLite with a renewable 60-second lease; queued jobs start after a restart, and interrupted jobs resume when their lease expires. After three interrupted attempts, the job is marked failed and can be run again from the editor. Each job keeps the processing settings that were active when it was queued.

The editor checks a small episode status endpoint every 2.5 seconds. It reloads the full cut and transcript data while processing or after edits, and every 30 seconds while idle. Waveform data reloads only after a completed Prepare job. Job summaries omit queued context snapshots; the results panel requests compact records while the complete frozen context remains available from the analysis-results API.

## Storage and architecture

```
video_data:/data/
  app.sqlite3
  tmp/                         # large multipart upload spool
  projects/<project-id>/episodes/<episode-id>/
    original.<extension>
    proxy.mp4
    audio.wav
    waveform.json
    frames/manifest.json
model_cache:/models/
  huggingface/
  torch/
  cache/
```

From the project list, open **Storage & cleanup** to inspect project files, their sizes and links to results, runs and voice samples. The page identifies missing media, broken database links, unreadable metadata and orphan analysis or transcript records. Use its search and cleanup filter to review candidates before selecting them. Unreferenced analysis frames, old temporary voice uploads and other recognized leftovers can be removed after confirmation. Prepared playback and audio media may also be removed individually; run **Prepare** to recreate them. Cut, interval and analysis frames are regenerated on demand without keeping permanent JPEGs. Cleanup is blocked while the episode has queued or running work and rechecks references and file changes immediately before deletion. Originals, unknown files, the database and model cache are protected. Linked legacy frames are deleted only after a matching replacement has been verified. Unlinked or unknown files are listed for inspection, but only recognized generated files are eligible for deletion.

In an episode, expand **Runs, queue & history** to inspect errors, output revisions and per-cut saved evidence. You can remove saved evidence for a finished run without deleting its results; incomplete cuts may need to be sampled again on retry. Saved descriptions and timestamps stay in the database, and their JPEGs can be regenerated without rerunning the model. Deleting a finished run removes its evidence and disconnects its provenance from surviving results and transcripts. Delete or edit results and transcripts from the cut and Audio editors. Orphan revision chains whose cut or episode is missing can be inspected and deleted from the storage page; links from valid records require database repair instead.

Frame previews are generated on demand in a bounded background queue, deduplicated, and kept only in a short-lived memory cache (up to 64 MiB; entries expire after 120 seconds and are swept periodically). Existing JPEGs are removed after an identical replacement has been verified; if reconstruction fails or differs, the existing image is retained. The original upload is never replaced. To verify and reclaim legacy frame JPEGs in bulk, run `docker compose exec video-analyzer python -m app.frame_gc --apply`; omit `--apply` for a verification-only pass. The SQLite database holds metadata, per-cut presets, job checkpoints, context and revisioned results. Analysis samples the original without modifying it. Each queued run freezes its settings, selected library context, synopsis, neighboring cuts, chunk notes and current transcript; completed results retain the snapshot, sampled frame descriptions, prompts and model provenance. Inference uses a separate worker and the same renewable lease discipline as media jobs. A resumed batch skips completed cuts and reuses saved frame descriptions for an interrupted cut. Retrying a failed run creates a new history entry with the original snapshot and checkpoints, keeping the failure and its time. If an unfinished cut was disabled, archived or its timing changed, start a new analysis instead.

OpenRouter requests show their current evidence or summary step in the job stage. Transient connection and provider errors, empty or truncated replies, and invalid structured output are retried up to three times. Truncated or empty replies get a larger output-token limit on the next attempt; paid models may charge more for retries and additional tokens. The selected model does not change. If the provider still cannot return the requested structured evidence, the run fails with its error in history and can be retried. Sampling fewer frames reduces the amount of evidence sent to the provider; save a new preset and queue a new run to apply it.

## WeVisOne and OpenRouter setup (ROCm)

The inference image must be validated on the target machine. Its pinned ROCm wheels target the Radeon AI PRO R9700 (`gfx1201`); another GPU architecture needs matching AMD wheels. The media API remains CPU-only. It sends the chosen context and visual descriptions to OpenRouter. Provider pricing and availability depend on the model you select; `openrouter/free` is the default, but its availability and rate limits can vary.

1. Download the pinned WeVisOne-VL 4B detection model (about 9.7 GB) and Qwen3-VL 2B frame-caption model (about 4.3 GB). They are stored in `.models`, which is ignored by Git and persists across image rebuilds. The scripts resume partial files and verify the large model files. Set `WEVISONE_MODEL_DIR` and `CAPTION_MODEL_DIR` to use other persistent directories; use absolute paths when sharing them with Compose.

   ```bash
   ./scripts/download-wevisone.sh
   ./scripts/download-caption-model.sh
   ```

2. In the ignored `.env`, set `OPENROUTER_API_KEY` to your key. The key is used only by the inference worker and is not stored in job snapshots or results. Ensure `.env` is readable only by trusted users. Set `VIDEO_GID` and `RENDER_GID` to the host's device group IDs if they differ from the defaults.
3. From a Docker-privileged terminal, build and start the app and worker:

   ```bash
   docker compose -f compose.yml -f compose.inference.rocm.yml --profile inference up -d --build
   docker compose -f compose.yml -f compose.inference.rocm.yml --profile inference logs -f inference-worker
   ```

   To include the worker when the existing systemd unit starts Compose, add `COMPOSE_FILE=compose.yml:compose.inference.rocm.yml` and `COMPOSE_PROFILES=inference` to `.env` and restart the service from Cockpit or a privileged terminal. Both containers share the existing data volume. The model bind mount and named data/model cache volumes survive rebuilds.

For code-only changes after both images already exist, these lightweight rebuilds reuse the installed GPU runtime. Use the full Compose build above when changing runtime dependencies or the base images.

```bash
docker build -f deploy/Dockerfile.refresh-api -t video-analyzer:local .
docker build -f deploy/Dockerfile.refresh-inference -t framewise-inference-rocm:local .
sudo systemctl restart framewise.service
```

WeVisOne supplies spatial detections, while the separate Qwen model supplies frame descriptions. Both can make mistakes; character matching, motion, and camera movement can also be inaccurate. A sequence of stills does not establish the true action between samples. Review generated analysis and transcripts before using them. The app does not generate images, speech, or video. OCR and confidence assessments are uncalibrated model outputs and need review. See [the integration notes](docs/model-integration.md).

## Development

For Docker development with frontend hot reload and automatic API reload, run `COMPOSE_FILE=compose.yml:compose.inference.rocm.yml:compose.dev.yml docker compose up -d` (omit the inference override if you do not use it). Open `http://<APP_BIND_IP>:8080` for Vite hot reload (or use `APP_PORT` if set). Set `APP_BIND_IP` to a LAN-accessible address in `.env` to access Vite from another device. Restart the inference worker after backend changes with `docker compose restart inference-worker`.

The Docker image builds the frontend and serves it from FastAPI. For local development with Node and FFmpeg installed, start the API on port 8000 (`uvicorn app.main:app --reload` from `backend`, with `DATA_DIR` set to a writable directory) and run `npm install && npm run dev` from `frontend`. Vite proxies `/api` to the API. The API schema is at `/docs`.

Useful checks:

```bash
docker compose build
docker compose up -d
curl http://localhost:8080/api/health
docker compose logs -f video-analyzer
```

With local Python, FFmpeg, and FFprobe installed, install `backend/requirements-dev.txt`, set `PYTHONPATH=backend`, and run `python -m unittest discover -s tests -v`. The test creates temporary videos and checks import, preparation, seekable preview, waveform, hard cuts, edits, context records, and no-audio handling.

## Next pipelines

Hard-cut detection may miss fades or split on rapid motion. Temporal motion analysis and production of images, audio, or video remain separate future work.

### Local audio transcription (ROCm)

The optional audio worker uses Qwen3-ASR 1.7B, Qwen3-ForcedAligner 0.6B, and pyannote Community-1. It runs separately from visual inference and keeps the downloaded checkpoints in the persistent model cache. Accept the conditions at https://huggingface.co/pyannote/speaker-diarization-community-1 and add a Hugging Face read token as `HF_TOKEN` in the ignored `.env` file. Do not commit the token. Then include `compose.audio.rocm.yml` alongside the usual Compose files and start the `audio` profile, for example:

```sh
COMPOSE_FILE=compose.yml:compose.inference.rocm.yml:compose.audio.rocm.yml:compose.dev.yml docker compose --profile audio up -d --build audio-worker
```

The `audio-worker` image uses a separate ROCm environment to isolate audio dependencies from the visual worker. In the episode's Audio workspace, choose a character and add one or more non-overlapping excerpts of that character's voice, totaling 2–60 seconds, then save them as a project-wide reference. The playhead time is displayed in seconds; focusing a start or end field fills it from the playhead. Preview the combined excerpts before saving, and play saved references in the same panel. Alternatively, upload and preview a 2–60 second audio recording (up to 25 MB); this also works when the episode has no audio. Use excerpts or a recording containing only that character. The worker combines the selected excerpts and extracts a voice embedding; enrollment fails if it detects multiple speakers or no usable voice. The audio for new references is retained with the project; older uploaded references cannot be previewed if their audio was discarded. Multiple saved references can improve matching. Transcription can be queued independently of hard-cut detection. Current transcription text, timings, and suggested character links are included in overlapping cut analysis; they can be edited or deleted. Transcription reruns replace untouched generated segments through revision history; edited or manually created speech is retained and overlapping new output is omitted. The forced aligner supports fewer languages than the ASR model; a job fails rather than inventing timestamps when alignment is unavailable. Queued inference jobs can be cancelled from the episode editor.

With `compose.dev.yml`, the audio worker also bind-mounts the backend code; restart `audio-worker` after Python changes. The regular audio image includes the code at build time.

## External API

Framewise provides a read-only integration API for applications on the trusted network. Open `/docs` for Swagger UI and the integration guide, `/openapi.json` for the machine-readable OpenAPI document, or `/api/integration-guide` for the guide as Markdown. Start with `/api/v1/projects` and follow the guide to select an episode, page through cuts, read analysis and download audio and character voice references. The existing `/api` also exposes editing operations. No authentication is required, so expose the service only to trusted clients. A reusable prompt for creating an agent skill is in [docs/agent-skill-prompt.md](docs/agent-skill-prompt.md).
