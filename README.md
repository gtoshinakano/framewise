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
2. The queued preparation job probes the original, organizes timed chunks, creates a smaller MP4 preview, extracts JPEG frames every five seconds, and extracts mono WAV audio plus waveform peaks when audio is present.
3. Open the episode editor and choose **Detect hard cuts**. FFmpeg's scene score identifies boundaries and saves first and last sampled frames for each cut. Click a chunk, cut, or thumbnail to seek the preview. The timeline defaults to a 10-minute window for videos longer than 30 minutes; use the overview bar, window menu, and Earlier/Later controls to move through the video. The cut gallery shows 24 cuts per page. The inspector shows both cut frames, can play just the selected chunk or cut, and can edit its notes. Add manual per-cut analysis fields for first/last frames, camera movement, composition, audio matching, or generation prompts. Edits create revisions, and earlier versions remain available under **Show history**.
4. Change processing settings in the sidebar, save, then rerun **Prepare** or **Redetect cuts**. Preview width, CRF quality, encoding speed, frame interval, chunk duration, and scene threshold are configurable. Cut edits and result revisions follow boundaries that remain within 0.25 seconds of their previous positions. Boundaries that disappear move to **Archived cuts** with their notes and result history intact; a later matching redetection restores the same cut. Chunk titles and notes are retained when their start times remain the same.
5. Add characters, worlds, scenarios, or chapter notes in the project's **Context library**. Items can apply to the entire project or a specific episode.
6. Use the editor's **Audio** tab to add timed speech notes, speaker labels, and optional character links. Approved segments appear in overlapping cuts' input context. Editing or deleting a segment preserves earlier revisions under **Show history**.
7. Once the inference worker is set up, select a cut, choose first/last or evenly spaced frames, specify visual categories to detect, select relevant context, add cut-specific notes, and save the preset. **Analyze cut** queues that cut; **Analyze all cuts** queues all active cuts using each one's saved preset. Review and revise the resulting draft fields. Results include frame observations, character and outfit prompts, a first-still prompt, a MiniMax H3 animation prompt, and voice-over notes.

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
    frames/frame_000001.jpg ...
    frames/manifest.json
    thumbnails/cut_000000.jpg ...
    thumbnails/<cut-id>_first.jpg
    thumbnails/<cut-id>_last.jpg
model_cache:/models/
  huggingface/
  torch/
  cache/
```

The SQLite database holds metadata, per-cut presets, job checkpoints, context and revisioned results. Analysis samples the original without modifying it. Each queued run freezes its settings, selected library context, synopsis, neighboring cuts, chunk notes and approved transcript; completed results retain the snapshot, sampled frame descriptions, prompts and model provenance. Inference uses a separate worker and the same renewable lease discipline as media jobs. A resumed batch skips completed cuts and reuses saved frame descriptions for an interrupted cut. If a provider or worker error stops a run, the editor offers **Retry analysis** with the original snapshot. If an unfinished cut was archived or its timing changed, start a new analysis instead.

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

WeVisOne supplies spatial detections, while the separate Qwen model supplies frame descriptions. Both can make mistakes; character matching, motion, and camera movement can also be inaccurate. A sequence of stills does not establish the true action between samples. Review each generated draft and approved transcript before using it. The app does not transcribe audio or generate images, speech, or video. MiniMax H3 prompts are drafts for its first-frame workflow; its supported output duration is 4–15 seconds, so longer or shorter source cuts may need splitting or holding. See [the integration notes](docs/model-integration.md) and [MiniMax H3 guidance](https://www.minimax.io/news/minimax-h3-open-source).

## Development

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

Hard-cut detection may miss fades or split on rapid motion. Automated transcription, voice identification, temporal motion analysis, and production of images, audio, or video remain separate future work.
