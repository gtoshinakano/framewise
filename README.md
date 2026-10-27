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

Then use `framewise.service` in Cockpit's Services page. The unit runs `docker compose up -d --build` on start and `docker compose down` on stop; both keep the named volumes. To change the bind address, port, or upload limit, edit `.env` and restart the service. The server has no built-in login: expose it only on a trusted network or behind an authenticated reverse proxy.
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

For long uploads, keep the browser tab open until the upload reaches 100%. Media preparation then runs in the background. The project and editor show current stage and any FFmpeg error. One media job runs at a time. Jobs are claimed from SQLite with a renewable 60-second lease; queued jobs start after a restart, and interrupted jobs resume when their lease expires. After three interrupted attempts, the job is marked failed and can be run again from the editor. Each job keeps the processing settings that were active when it was queued.

The editor checks a small episode status endpoint every 2.5 seconds. It reloads the full cut and transcript data while processing or after edits, and every 30 seconds while idle. Waveform data reloads only after a completed Prepare job.

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

The `projects`, `episodes`, `jobs`, `chunks`, `cuts`, `analysis_results`, `context_items`, and `transcript_segments` SQLite tables hold metadata. Chunks are timed working segments; cut boundaries are detected independently and may span chunk boundaries. Per-cut analysis results and transcript segments keep origin, approval status, optional model/job provenance, and revision links. Manual transcript segments can be edited in the browser; automated transcription is not present yet. `/api/cuts/{id}/analysis-context` collects the project synopsis, applicable context items, adjacent cuts and approved results, overlapping chunks, and approved transcript segments for that cut. No model weights or AI services are installed or downloaded in this version. FFmpeg processing works on CPU. ROCm and CUDA device overrides are supplied for later model pipelines:

```bash
docker compose -f compose.yml -f compose.rocm.yml up -d --build
# or, on a CUDA host with NVIDIA Container Toolkit:
docker compose -f compose.yml -f compose.cuda.yml up -d --build
```

The base image is not a ROCm or CUDA PyTorch image yet. When an inference pipeline is added, pin a compatible runtime and model version in its own worker image. Keep the current `/models` volume for weights so rebuilds do not redownload them.
The ROCm override maps `/dev/kfd` and `/dev/dri` and adds the host's `video` and `render` group IDs. The supplied defaults match this Ubuntu host; on another machine, set `VIDEO_GID` and `RENDER_GID` in `.env` from `getent group video` and `getent group render`.

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

The current hard-cut pass is visual scene-score detection, so fades and motion may yield false boundaries or missed cuts. Future work should add model-driven first/last frame analysis, camera motion and composition, audio transcription, voice and character grouping, and prompt generation. Model runs should append records to `analysis_results`, preserve the model ID, job ID, prompt, settings, and frozen context snapshot, and leave manual revisions intact. Keep inference on proxy or extracted frames and reference the original for final quality checks. The requested [WeVisOne-VL paper](https://huggingface.co/papers/2610.31329) is planned for frame-level visual perception; its weights are intentionally not downloaded. See [the integration plan](docs/model-integration.md).
