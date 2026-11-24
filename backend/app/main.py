import asyncio
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import shutil
import time
import uuid
import sqlite3
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field

from . import analysis_context, db, episode_requests, media, worker, cut_history, cut_previews, storage, frame_source, frame_cache, subtitle_store, summary_jobs, audio_worker, prompt_context, job_kinds, runs, structured_results, episode_events, episode_jobs
from . import agent_log, character_decisions, audio_ranges, name_glossary, speaker_attribution, line_edit, waveform_range, transcript_corrections, unregistered, cut_readiness
from .transcript_parts import manual_metadata
from .analysis_context import json_object_or_empty, serialize_cut, serialize_transcript
from .external_api import AnalysisContext, AnalysisResult, ContextItem, EpisodeSubtitles, Transcript, VoiceSample, router as external_router
from .inference_worker import SUMMARY_TITLE, SUMMARY_TEXT_SYSTEM, SYSTEM_PROMPT, SUMMARY_SYSTEM, PRODUCTION_BRIEF_SYSTEM

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_GB", "40")) * 1024 ** 3
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_SETTINGS = {"proxy_width": 960, "proxy_crf": 28, "proxy_preset": "veryfast", "frame_interval": 5.0, "scene_threshold": 0.32, "chunk_duration": 60}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    from .composition_brief import register_projection
    register_projection()
    worker.start_worker()
    frame_cache.clear_abandoned_temporary_files()
    async def clear_cache_periodically():
        ticks = 0
        while True:
            if ticks % 20 == 0:  # the disk cache every ten minutes, starting now
                try:
                    await asyncio.to_thread(frame_cache.prune_disk)
                except OSError as exc:
                    frame_cache.log.warning('Could not prune the frame cache: %s', exc)
            ticks += 1
            await asyncio.sleep(30)
            frame_cache.clear_expired()
    cleaner = asyncio.create_task(clear_cache_periodically())
    try:
        yield
    finally:
        cleaner.cancel()
        worker.stop_worker()


INTEGRATION_GUIDE = (Path(__file__).parent / "integration-guide.md").read_text(encoding="utf-8")
app = FastAPI(title="Framewise API", version="1.0.0", description=INTEGRATION_GUIDE, lifespan=lifespan,
              docs_url=None, redoc_url=None)
from .json_gzip import JsonGzip
app.add_middleware(JsonGzip)
from .composition_brief import CompositionError

@app.exception_handler(CompositionError)
async def composition_error_handler(_request, exc):
    return JSONResponse(status_code=exc.status_code, content={'detail':exc.detail})

app.include_router(storage.router)
from .keyframes_api import router as keyframes_router
app.include_router(keyframes_router)
from .target_characters_api import router as target_characters_router
app.include_router(target_characters_router)
from .assistant.api import router as assistant_router
from .assistant.audio_transcription import DEFAULT_AUDIO_MODEL
app.include_router(assistant_router)
from .chat_gateway import router as chat_gateway_router
app.include_router(chat_gateway_router)
from .sequences import router as sequences_router
app.include_router(sequences_router)
from .queue_api import router as queue_router
app.include_router(queue_router)
from .outfits_api import router as outfits_router
app.include_router(outfits_router)
from .autopilot_api import router as autopilot_router
app.include_router(autopilot_router)
from .missed_cuts_api import router as missed_cuts_router
app.include_router(missed_cuts_router)
from .review_decisions_api import router as review_decisions_router
app.include_router(review_decisions_router)
from .voice_resolver_api import router as voice_resolver_router
from .character_decisions_api import router as character_decisions_router
app.include_router(character_decisions_router)
app.include_router(voice_resolver_router)
from .visual_models_api import router as visual_models_router
app.include_router(visual_models_router)
from .point_detect_api import first_frame_spec, router as point_detect_router
app.include_router(point_detect_router)
from . import character_registration, web_search_api
app.include_router(character_registration.router)
app.include_router(web_search_api.router)
from .review_report import CUT_FLAGS, router as review_router
app.include_router(review_router)
from .cut_needs import router as cut_needs_router
app.include_router(cut_needs_router)
app.mount("/api/swagger-assets", StaticFiles(directory=Path(__file__).parent / "swagger-ui"), name="swagger-assets")


@app.get("/docs", include_in_schema=False)
def swagger_docs():
    return get_swagger_ui_html(openapi_url=app.openapi_url, title="Framewise API documentation",
                               swagger_js_url="/api/swagger-assets/swagger-ui-bundle.js",
                               swagger_css_url="/api/swagger-assets/swagger-ui.css",
                               swagger_favicon_url="data:,", swagger_ui_parameters={"validatorUrl": None})


@app.get("/api/integration-guide", response_class=PlainTextResponse, tags=["External integration v1"],
         summary="Read the external integration guide as Markdown")
def integration_guide():
    return INTEGRATION_GUIDE


class ProjectInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    synopsis: str = Field(default="", max_length=20000)


class EpisodeInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SettingsInput(BaseModel):
    proxy_width: int = Field(ge=320, le=1920)
    proxy_crf: int = Field(ge=18, le=35)
    proxy_preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"]
    frame_interval: float = Field(ge=0.5, le=60)
    scene_threshold: float = Field(ge=0.05, le=0.9)
    chunk_duration: int = Field(ge=10, le=3600)


class PipelineInput(BaseModel):
    pipeline: Literal["prepare", "hardcuts"]


class CutDetectionInput(BaseModel):
    scene_threshold: float = Field(ge=0.05, le=0.9, allow_inf_nan=False)


class CharacterDetectionLabelInput(BaseModel):
    character_context_id: str | None = None


class NewCharacterInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=50000)
    is_main: bool = False
    # True scopes the character to the box's episode; otherwise it belongs to the whole project.
    episode_only: bool = False


class RecognitionUseInput(BaseModel):
    # None: decided automatically; True: always a matching example; False: never.
    use: bool | None = None


class CharacterDetectionBoxInput(BaseModel):
    position: Literal['first', 'last']
    box: tuple[float, float, float, float]


class AnalysisConfig(BaseModel):
    frame_mode: Literal["per_second", "endpoints", "uniform"] = "per_second"
    frame_count: int = Field(default=1, ge=1)
    video_width: int = Field(default=640, ge=128, le=1920, multiple_of=2)
    video_height: int = Field(default=360, ge=128, le=1080, multiple_of=2)
    video_fps: float = Field(default=4, gt=0, le=30, allow_inf_nan=False)
    video_frame_budget: int | None = Field(default=None, ge=2, le=60)
    context_ids: list[str] = Field(default_factory=list, max_length=200)
    detection_queries: list[Annotated[str, Field(min_length=1, max_length=100, pattern=r".*\S.*")]] = Field(default_factory=lambda: ["person", "clothing", "animal", "vehicle", "chair", "table", "subtitles"], min_length=1, max_length=20)
    additional_context: str = Field(default="", max_length=20000)
    text_provider: Literal["openrouter", "local"] = "openrouter"
    local_thinking: bool = False
    openrouter_model: str = Field(default="openrouter/free", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_./:-]+$")


class CutInput(BaseModel):
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=20000)
    notes: str = Field(default="", max_length=20000)
    results: dict = Field(default_factory=dict)


class CutAnalysisEnabledInput(BaseModel):
    analysis_enabled: bool


class CutSplitInput(BaseModel):
    time: float


class CutTimingInput(BaseModel):
    start_time: float
    end_time: float
    expected_start_time: float | None = None
    expected_end_time: float | None = None


class CutBoundaryStepInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    direction: Literal['earlier', 'later']
    anchor: Literal['start', 'end'] = 'end'


class ChunkInput(BaseModel):
    title: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=20000)


ResultKind = Literal["first_frame", "camera_movement", "composition", "last_frame", "cut_description", "audio_visual_match", "generation_prompt", "other"]


class ResultCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ResultKind
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=50000)


class StructuredAnalysisResult(BaseModel):
    id: str
    cut_id: str
    type: Literal['initial_frame', 'keyframe_analysis', 'performance_beats', 'video_analysis', 'last_frame', 'subtitle_ocr', 'consistency_summary']
    content: dict[str, Any] | None
    frame_url: str | None
    origin: str
    model_id: str | None
    job_id: str | None
    parent_id: str | None
    superseded_by: str | None
    created_at: str


class PromptRegenInput(BaseModel):
    mode: Literal["edit", "regenerate"]
    instruction: str = Field(default="", max_length=5000)
    current_prompt: str | None = Field(default=None, max_length=50000)
    request_override: dict | None = None
    preview_base_id: str | None = None
    preview_checked: bool = False
    source_job_id: str | None = None


class AnalysisRequestRetry(BaseModel):
    source_job_id: str
    request_override: dict


class ResultUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=5000000)


class TranscriptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=20000)
    speaker_label: str = Field(default="", max_length=200)
    character_context_id: str | None = None
    left_unknown: bool = Field(default=False, description="Keep the speaker as it is (none, or an unsure guess) and never ask about it again.")


class TranscriptMergeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_ids: list[str] = Field(min_length=2, max_length=50, description="The current lines to join, in any order.")
    text: str = Field(min_length=1, max_length=20000, description="The text of the joined line.")
    speaker_label: str = Field(default="", max_length=200)
    character_context_id: str | None = None


class SpeakerConfirmInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_ids: list[str] = Field(min_length=1, max_length=200, description="Current lines whose speaker the person confirms as it is.")


class SplitPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000)


class TranscriptSplitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at: float = Field(gt=0, allow_inf_nan=False, description="Episode seconds where the line is divided.")
    left: SplitPart
    right: SplitPart


class CorrectionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal['srt'] = 'srt'
    file: str = Field(default="", max_length=500, description="The source file's name, kept as evidence.")
    offset: float | None = Field(default=None, allow_inf_nan=False, description="Seconds added to the source's times to match the episode.")


class TranscriptCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str = Field(min_length=1)
    text: str | None = Field(default=None, max_length=20000, description="The corrected text; omitted keeps the text.")
    character_context_id: str | None = Field(default=None, description="Another character for the line; omitted keeps the speaker.")
    cues: list[int] = Field(default_factory=list, max_length=50, description="Numbers of the source cues the line matched.")
    srt_text: str = Field(default="", max_length=2000, description="The source's words for the line.")
    note: str = Field(default="", max_length=500, description="Why the line was changed.")


class TranscriptCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: CorrectionSource = Field(default_factory=CorrectionSource)
    corrections: list[TranscriptCorrection] = Field(min_length=1, max_length=transcript_corrections.MAX_CORRECTIONS)


class AssignVoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker_label: str = Field(default="", max_length=200, description="The diarization label of the voice; empty selects lines with no label.")
    character_context_id: str = Field(min_length=1, description="The character who speaks it.")


VoiceRange = audio_ranges.VoiceRange


class VoiceSampleInput(BaseModel):
    character_context_id: str
    ranges: list[VoiceRange] | None = Field(default=None, min_length=1, max_length=50)
    # Preserve requests from clients using the original single-range API.
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class VoicePreviewInput(BaseModel):
    ranges: list[VoiceRange] = Field(min_length=1, max_length=50)


class TranscribeRangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_time: float = Field(ge=0, allow_inf_nan=False, description="Start of the audio to transcribe again, in seconds.")
    end_time: float = Field(gt=0, allow_inf_nan=False, description="End of the audio to transcribe again, in seconds.")
    replace_manual: bool = Field(default=False, description="Also replace segments that were edited or added by hand inside the interval. By default only untouched model output is replaced.")


class ContextInput(BaseModel):
    kind: Literal["chapter_synopsis", "chapter_happenings", "character", "world", "scenario", "object", "general"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=50000)
    episode_id: str | None = None
    is_main: bool = False


def require_project(project_id: str):
    project = db.one("SELECT * FROM projects WHERE id=?", (project_id,))
    if not project:
        raise HTTPException(404, "Project not found")
    return project


def require_episode(episode_id: str):
    episode = db.one("SELECT * FROM episodes WHERE id=?", (episode_id,))
    if not episode:
        raise HTTPException(404, "Episode not found")
    episode["settings"] = {**DEFAULT_SETTINGS, **json_object_or_empty(episode.pop("settings_json"))}
    episode["has_audio"] = bool(episode["has_audio"])
    return episode


def episode_response(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id)
    episode["proxy_ready"] = (path / "proxy.mp4").is_file()
    episode["cuts"] = [serialize_cut(cut) for cut in db.all_rows(f"""SELECT c.*, {CUT_FLAGS}
        FROM cuts c WHERE c.episode_id=? AND c.active=1 ORDER BY c.cut_index""", (episode_id,))]
    episode["archived_cuts"] = [serialize_cut(cut) for cut in db.all_rows("SELECT * FROM cuts WHERE episode_id=? AND active=0 ORDER BY start_time,cut_index", (episode_id,))]
    episode["chunks"] = db.all_rows("SELECT * FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,))
    with db.connect() as conn:
        episode["audio_ranges"] = audio_ranges.public(conn, episode_id)
        episode["cut_history"] = cut_history.state(conn, episode_id)
        run = subtitle_store.current_run(episode_id, conn)
        episode["subtitles"] = {"language": episode.get("subtitle_language"),
                                "run": {**run, "line_count": conn.execute("SELECT COUNT(*) FROM subtitle_lines WHERE run_id=?", (run["id"],)).fetchone()[0]} if run else None}
    # The queued payload can contain every cut's context and may be very large, and an old episode holds thousands of
    # jobs: return the bounded list of episode_jobs.listed (metadata only). The dialog reads a cut's own jobs from /api/cuts/{id}/jobs.
    current_cuts = {cut['id']: cut for cut in episode['cuts'] + episode['archived_cuts']}
    with db.connect() as conn:
        episode["jobs"] = episode_jobs.listed(conn, episode_id, current_cuts)
    return episode


def serialize_result(result: dict):
    result["settings"] = json_object_or_empty(result.pop("settings_json"))
    result["context"] = json_object_or_empty(result.pop("context_json"))
    return result


def require_character(episode: dict, character_context_id: str | None):
    """Accept a transcript line that names a character of this project."""
    if not character_context_id:
        return
    item = db.one("SELECT project_id,kind FROM context_items WHERE id=?", (character_context_id,))
    if not item or item["project_id"] != episode["project_id"] or item["kind"] != "character":
        raise HTTPException(400, "Choose a character from this project")


def validate_transcript(episode: dict, body: TranscriptInput):
    if body.end_time <= body.start_time:
        raise HTTPException(400, "Transcript end must be after its start")
    if episode["duration"] is not None and body.end_time > episode["duration"] + 0.2:
        raise HTTPException(400, "Transcript segment ends after the video")
    require_character(episode, body.character_context_id)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/projects")
def list_projects():
    projects = db.all_rows("SELECT * FROM projects ORDER BY created_at DESC")
    for project in projects:
        project["episodes"] = db.all_rows("SELECT id, title, status, stage, error, duration, created_at FROM episodes WHERE project_id=? ORDER BY created_at", (project["id"],))
    return projects


@app.post("/api/projects")
def create_project(body: ProjectInput):
    project_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO projects (id,title,synopsis) VALUES (?,?,?)", (project_id, body.title.strip(), body.synopsis))
    return require_project(project_id)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    project = require_project(project_id)
    project["episodes"] = db.all_rows("SELECT id, title, status, stage, error, duration, created_at FROM episodes WHERE project_id=? ORDER BY created_at", (project_id,))
    return project


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, body: ProjectInput):
    require_project(project_id)
    with db.connect() as conn:
        title = body.title.strip()
        changed = conn.execute('UPDATE projects SET title=?,synopsis=? WHERE id=? AND (title!=? OR synopsis!=?)',
                               (title, body.synopsis, project_id, title, body.synopsis))
        if changed.rowcount:
            structured_results.invalidate_context(conn, project_id, None, 'Project title or synopsis changed')
    return require_project(project_id)


@app.get("/api/projects/{project_id}/context", responses={200: {"model": list[ContextItem]}})
def list_context(project_id: str):
    """List project-wide and all episode-specific context items for this project."""
    require_project(project_id)
    return db.all_rows("SELECT * FROM context_items WHERE project_id=? ORDER BY kind,title", (project_id,))


@app.post("/api/projects/{project_id}/unregistered-characters", responses={200: {"model": ContextItem}})
def create_unregistered_character(project_id: str):
    """A new numbered Unregistered card (the lowest free number) for a minor person who needs a line or a box but no record."""
    require_project(project_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        card = unregistered.create_next(conn, project_id)
    return card


@app.post("/api/projects/{project_id}/context")
def create_context(project_id: str, body: ContextInput):
    require_project(project_id)
    if body.kind == 'character' and unregistered.is_card(body.title):
        raise HTTPException(400, "Names of the form 'Unregistered N' are reserved for Unregistered cards")
    if body.episode_id and require_episode(body.episode_id)["project_id"] != project_id:
        raise HTTPException(400, "Episode belongs to another project")
    context_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO context_items (id,project_id,episode_id,kind,title,content,is_main) VALUES (?,?,?,?,?,?,?)",
                     (context_id, project_id, body.episode_id, body.kind, body.title.strip(), body.content, int(body.is_main and body.kind == 'character')))
        structured_results.invalidate_context(conn, project_id, body.episode_id)
    return db.one("SELECT * FROM context_items WHERE id=?", (context_id,))


@app.put("/api/context/{context_id}")
def update_context(context_id: str, body: ContextInput):
    item = db.one("SELECT * FROM context_items WHERE id=?", (context_id,))
    if not item:
        raise HTTPException(404, "Context item not found")
    if body.episode_id and require_episode(body.episode_id)["project_id"] != item["project_id"]:
        raise HTTPException(400, "Episode belongs to another project")
    if unregistered.is_card(item) and (body.kind != 'character' or body.title.strip() != item['title'] or body.content.strip()
                                       or body.is_main or body.episode_id):
        # Prompts read the fixed title as the person's handle; a card has no details, episode or main flag.
        raise HTTPException(400, "An Unregistered card cannot be renamed, scoped or given details; create a character for that person instead")
    with db.connect() as conn:
        conn.execute("UPDATE context_items SET episode_id=?,kind=?,title=?,content=?,is_main=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (body.episode_id, body.kind, body.title.strip(), body.content, int(body.is_main and body.kind == 'character'), context_id))
        structured_results.invalidate_context(conn, item['project_id'], item['episode_id'])
        if item['episode_id'] is not None and body.episode_id != item['episode_id']:
            structured_results.invalidate_context(conn, item['project_id'], body.episode_id)
    return db.one("SELECT * FROM context_items WHERE id=?", (context_id,))


@app.delete("/api/context/{context_id}")
def delete_context(context_id: str):
    with db.connect() as conn:
        item = conn.execute("SELECT project_id,episode_id FROM context_items WHERE id=?", (context_id,)).fetchone()
        if not item:
            raise HTTPException(404, "Context item not found")
        samples = conn.execute("SELECT id FROM character_voice_samples WHERE character_context_id=?", (context_id,)).fetchall()
        conn.execute("DELETE FROM context_items WHERE id=?", (context_id,))
        structured_results.invalidate_context(conn, item['project_id'], item['episode_id'])
    for sample in samples:
        media.voice_reference_path(item['project_id'], sample['id']).unlink(missing_ok=True)
    from . import registration_images
    registration_images.unlink(registration_images.thumbnail(item["project_id"], context_id))
    return {"ok": True}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    require_project(project_id)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute("""SELECT 1 FROM jobs j JOIN episodes e ON e.id=j.episode_id WHERE e.project_id=?
            AND (j.status IN ('queued','running') OR j.lease_until>unixepoch()) LIMIT 1""", (project_id,)).fetchone()
        if active:
            raise HTTPException(409, "Wait for analysis jobs to finish before deleting this project")
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    shutil.rmtree(db.DATA_DIR / "projects" / project_id, ignore_errors=True)
    return {"ok": True}


async def save_episode(project_id: str, title: str, file: UploadFile):
    require_project(project_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "Upload an MP4, MOV, MKV, WebM, AVI, or M4V video")
    title = title.strip()
    if not title or len(title) > 200:
        raise HTTPException(400, "Episode title must be 1–200 characters")
    episode_id = str(uuid.uuid4())
    path = media.episode_dir(project_id, episode_id)
    target = path / f"original{ext}"
    total = 0
    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the configured upload size limit")
                output.write(chunk)
        if not total:
            raise HTTPException(400, "Video file is empty")
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes (id,project_id,title,original_filename,original_ext,settings_json) VALUES (?,?,?,?,?,?)",
                         (episode_id, project_id, title, Path(file.filename or "video").name, ext, json.dumps(DEFAULT_SETTINGS)))
        worker.enqueue(episode_id, "prepare")
    except Exception:
        with db.connect() as conn:
            conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
        shutil.rmtree(path, ignore_errors=True)
        raise
    finally:
        await file.close()
    return episode_response(episode_id)


@app.post("/api/import")
async def import_video(file: UploadFile = File(...), project_title: str = Form(...), episode_title: str = Form("Episode 1")):
    title = project_title.strip()
    if not title or len(title) > 200:
        raise HTTPException(400, "Project title must be 1–200 characters")
    project = create_project(ProjectInput(title=title))
    try:
        episode = await save_episode(project["id"], episode_title, file)
    except Exception:
        with db.connect() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project["id"],))
        raise
    return {"project": project, "episode": episode}


@app.post("/api/projects/{project_id}/episodes")
async def add_episode(project_id: str, file: UploadFile = File(...), title: str = Form(...)):
    return await save_episode(project_id, title, file)


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str):
    return episode_response(episode_id)


@app.get("/api/episodes/{episode_id}/status")
def get_episode_status(episode_id: str):
    episode = db.one("SELECT status,stage,error FROM episodes WHERE id=?", (episode_id,))
    if not episode:
        raise HTTPException(404, "Episode not found")
    # The event counters as they stand now: a client that reads them before loading knows which first event is news.
    revision = episode_events.snapshot(episode_id)
    latest = db.one("SELECT id,status FROM jobs WHERE episode_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1", (episode_id,))
    with db.connect() as conn:
        queue = runs.summary(conn, episode_id)
    return {**episode, "job_id": latest["id"] if latest else None, "job_status": latest["status"] if latest else None, "queue": queue, "revision": revision}


@app.patch("/api/episodes/{episode_id}")
def update_episode(episode_id: str, body: EpisodeInput):
    episode = require_episode(episode_id)
    with db.connect() as conn:
        title = body.title.strip()
        if conn.execute('UPDATE episodes SET title=? WHERE id=? AND title!=?', (title, episode_id, title)).rowcount:
            structured_results.invalidate_context(conn, episode['project_id'], episode_id, 'Episode title changed')
    return episode_response(episode_id)


@app.put("/api/episodes/{episode_id}/settings")
def update_settings(episode_id: str, body: SettingsInput):
    require_episode(episode_id)
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET settings_json=? WHERE id=?", (body.model_dump_json(), episode_id))
    return episode_response(episode_id)


@app.post("/api/episodes/{episode_id}/run")
def run_pipeline(episode_id: str, body: PipelineInput):
    require_episode(episode_id)
    try:
        job_id = worker.enqueue(episode_id, body.pipeline)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id}


@app.delete("/api/episodes/{episode_id}")
def delete_episode(episode_id: str):
    episode = require_episode(episode_id)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute("""SELECT 1 FROM jobs WHERE episode_id=?
            AND (status IN ('queued','running') OR lease_until>unixepoch()) LIMIT 1""", (episode_id,)).fetchone()
        if active:
            raise HTTPException(409, "Wait for analysis jobs to finish before deleting this episode")
        samples = conn.execute("SELECT v.id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE c.episode_id=?", (episode_id,)).fetchall()
        conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
    for sample in samples:
        media.voice_reference_path(episode['project_id'], sample['id']).unlink(missing_ok=True)
    shutil.rmtree(db.DATA_DIR / "projects" / episode["project_id"] / "episodes" / episode_id, ignore_errors=True)
    return {"ok": True}


@app.put("/api/cuts/{cut_id}")
def update_cut(cut_id: str, body: CutInput):
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    with db.connect() as conn:
        fields = (body.title, body.description, body.notes, json.dumps(body.results))
        changed = conn.execute('''UPDATE cuts SET title=?,description=?,notes=?,results_json=? WHERE id=?
            AND (title!=? OR description!=? OR notes!=? OR results_json!=?)''', (*fields, cut_id, *fields))
        if changed.rowcount:
            structured_results.invalidate_notes(conn, [cut_id], 'Cut notes or details changed')
    return serialize_cut(db.one("SELECT * FROM cuts WHERE id=?", (cut_id,)))


@app.put("/api/cuts/{cut_id}/analysis-enabled")
def set_cut_analysis_enabled(cut_id: str, body: CutAnalysisEnabledInput):
    with db.connect() as conn:
        if not conn.execute("UPDATE cuts SET analysis_enabled=? WHERE id=? AND active=1", (int(body.analysis_enabled), cut_id)).rowcount:
            raise HTTPException(404, "Active cut not found")
    return serialize_cut(db.one("SELECT * FROM cuts WHERE id=?", (cut_id,)))


@app.post('/api/cuts/{cut_id}/detect-hardcuts')
def detect_hardcuts_in_cut(cut_id: str, body: CutDetectionInput):
    cut = db.one('SELECT episode_id FROM cuts WHERE id=? AND active=1', (cut_id,))
    if not cut:
        raise HTTPException(404, 'Active cut not found')
    try:
        return {'job_id': worker.enqueue(cut['episode_id'], 'hardcuts', {'cut_id': cut_id, 'scene_threshold': body.scene_threshold})}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def edit_cut_timeline(cut_id: str, action: str, timing: CutTimingInput | None = None, split_time: float | None = None):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cut = conn.execute("SELECT * FROM cuts WHERE id=? AND active=1", (cut_id,)).fetchone()
        if not cut:
            raise HTTPException(404, "Active cut not found")
        episode = conn.execute("SELECT * FROM episodes WHERE id=?", (cut["episode_id"],)).fetchone()
        if cut_history.busy(conn, cut['episode_id']):
            raise HTTPException(423, 'Wait for media preparation or full hard-cut detection to finish')
        cuts = conn.execute("SELECT * FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index", (cut["episode_id"],)).fetchall()
        index = next(i for i, item in enumerate(cuts) if item["id"] == cut_id)
        before = cut_history.snapshot(conn, cut["episode_id"])
        created_ids = set()
        number = cut["cut_index"] + 1
        affected_ids, archived = set(), set()
        if action == "timing":
            assert timing is not None
            start, end = timing.start_time, timing.end_time
            if (timing.expected_start_time is not None and timing.expected_start_time != cut["start_time"]) or (timing.expected_end_time is not None and timing.expected_end_time != cut["end_time"]):
                raise HTTPException(409, "The cut changed while you were adjusting it; reload and try again")
            if not math.isfinite(start) or not math.isfinite(end):
                raise HTTPException(400, "Times must be finite")
            if (index == 0 and abs(start) > 0.000001) or (index == len(cuts) - 1 and abs(end - episode["duration"]) > 0.000001):
                raise HTTPException(400, "The first and last boundaries must cover the entire video")
            if index == 0:
                start = 0.0
            if index == len(cuts) - 1:
                end = episode["duration"]
            if start >= end or (index > 0 and start <= cuts[index - 1]["start_time"]) or (index + 1 < len(cuts) and end >= cuts[index + 1]["end_time"]):
                raise HTTPException(400, "Times must leave every neighboring cut with a positive duration")
            if index > 0 and start != cut["start_time"]:
                conn.execute("UPDATE cuts SET end_time=? WHERE id=?", (start, cuts[index - 1]["id"]))
                affected_ids.add(cuts[index - 1]["id"])
            if index + 1 < len(cuts) and end != cut["end_time"]:
                conn.execute("UPDATE cuts SET start_time=? WHERE id=?", (end, cuts[index + 1]["id"]))
                affected_ids.add(cuts[index + 1]["id"])
            if start != cut["start_time"] or end != cut["end_time"]:
                conn.execute("UPDATE cuts SET start_time=?,end_time=? WHERE id=?", (start, end, cut_id))
                affected_ids.add(cut_id)
        elif action == "split":
            assert split_time is not None
            if not math.isfinite(split_time) or split_time < cut["start_time"] + MIN_SPLIT_SECONDS or split_time > cut["end_time"] - MIN_SPLIT_SECONDS:
                raise HTTPException(400, f"Split inside the cut, at least {MIN_SPLIT_SECONDS}s from either edge")
            # Later cuts move up one index, highest first so UNIQUE(episode_id, cut_index) never collides.
            for item in reversed(cuts[index + 1:]):
                conn.execute("UPDATE cuts SET cut_index=? WHERE id=?", (item["cut_index"] + 1, item["id"]))
            new_id = str(uuid.uuid4())
            conn.execute("UPDATE cuts SET end_time=? WHERE id=?", (split_time, cut_id))
            conn.execute("INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,analysis_enabled,created_at) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                         (new_id, cut["episode_id"], cut["cut_index"] + 1, split_time, cut["end_time"], cut["analysis_enabled"]))
            affected_ids.update((cut_id, new_id))
            created_ids.add(new_id)
        else:
            if len(cuts) < 2:
                raise HTTPException(400, "The last cut must cover the video")
            if action == "merge":
                if index + 1 >= len(cuts):
                    raise HTTPException(400, "There is no next cut to merge")
                removed, kept = cuts[index + 1], cut
                conn.execute("UPDATE cuts SET end_time=? WHERE id=?", (removed["end_time"], kept["id"]))
            else:
                removed = cut
                kept = cuts[index - 1] if index else cuts[index + 1]
                if index:
                    conn.execute("UPDATE cuts SET end_time=? WHERE id=?", (cut["end_time"], kept["id"]))
                else:
                    conn.execute("UPDATE cuts SET start_time=? WHERE id=?", (cut["start_time"], kept["id"]))
            affected_ids.add(kept["id"])
            archived.add(removed['id'])
            archive_index = conn.execute("SELECT MIN(cut_index) FROM cuts WHERE episode_id=?", (cut["episode_id"],)).fetchone()[0]
            conn.execute("UPDATE cuts SET active=0,cut_index=? WHERE id=?", (min(-1, archive_index - 1), removed["id"]))
            for item in cuts:
                if item["cut_index"] > removed["cut_index"]:
                    conn.execute("UPDATE cuts SET cut_index=? WHERE id=?", (item["cut_index"] - 1, item["id"]))
        if affected_ids:
            label = {"timing": f"Changed the timing of cut {number}",
                     "split": f"Split cut {number} at {cut_history.timecode(split_time or 0)}",
                     "merge": f"Merged cut {number} with cut {number + 1}",
                     "delete": f"Deleted cut {number}"}[action]
            cut_history.record(conn, cut["episode_id"], action, label, cut_id, before, created_ids)
            for changed_id in affected_ids | archived:  # a person changed a cut an agent split: the threshold's training signal
                agent_log.note_override(conn, 'cuts', changed_id)
        cancelled = cut_previews.after_timeline_edit(conn, episode, affected_ids, archived)
    worker.after_cancel(cancelled)
    worker.wake_event.set()
    return {"selected_id": cut_id if action != "delete" else kept["id"]}


def reverse_cut_edit(episode_id: str, direction: Literal['undo', 'redo']):
    episode = require_episode(episode_id)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            outcome = (cut_history.undo if direction == 'undo' else cut_history.redo)(conn, episode_id)
        except cut_history.Busy as exc:
            raise HTTPException(423, str(exc)) from exc
        except cut_history.Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        current = conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        for changed_id in set(outcome['changed']) | set(outcome['removed']):
            agent_log.note_override(conn, 'cuts', changed_id)
        cancelled = cut_previews.after_timeline_edit(conn, current, outcome['changed'], outcome['removed'])
        active = {row["id"] for row in conn.execute("SELECT id FROM cuts WHERE episode_id=? AND active=1", (episode_id,))}
        first = conn.execute("SELECT id FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index LIMIT 1", (episode_id,)).fetchone()
    worker.after_cancel(cancelled)
    thumbnails = media.episode_dir(episode["project_id"], episode_id) / "thumbnails"
    for cut_id in outcome["removed"]:
        if db.one('SELECT id FROM cuts WHERE id=?', (cut_id,)):
            continue
        for position in ("first", "last"):
            (thumbnails / f"{cut_id}_{position}.jpg").unlink(missing_ok=True)
    worker.wake_event.set()
    selected = outcome["cut_id"] if outcome["cut_id"] in active else (first["id"] if first else None)
    return {"label": outcome["label"], "selected_id": selected, "cut_history": cut_history_state(episode_id)}


def cut_history_state(episode_id: str):
    with db.connect() as conn:
        return cut_history.state(conn, episode_id)


@app.post("/api/episodes/{episode_id}/cut-edits/undo")
def undo_cut_edit(episode_id: str):
    """Reverse the latest split, merge, delete, timing change or boundary move of this episode's cuts.

    409 when the cuts changed since (or there is nothing to undo); 423 while media preparation or full hard-cut detection is pending."""
    return reverse_cut_edit(episode_id, 'undo')


@app.post("/api/episodes/{episode_id}/cut-edits/redo")
def redo_cut_edit(episode_id: str):
    """Apply the cut edit that was undone last again."""
    return reverse_cut_edit(episode_id, 'redo')


MIN_SPLIT_SECONDS = 0.1


@app.post("/api/cuts/{cut_id}/split")
def split_cut(cut_id: str, body: CutSplitInput):
    return edit_cut_timeline(cut_id, "split", split_time=body.time)


@app.put("/api/cuts/{cut_id}/timing")
def update_cut_timing(cut_id: str, body: CutTimingInput):
    return edit_cut_timeline(cut_id, "timing", body)


def boundary_preview_source(cut_id: str):
    cut = db.one("SELECT * FROM cuts WHERE id=? AND active=1", (cut_id,))
    if not cut:
        raise HTTPException(404, "Active cut not found")
    episode = require_episode(cut['episode_id'])
    path = media.episode_dir(episode['project_id'], episode['id'])
    source = path / 'proxy.mp4'
    if not source.is_file():
        source = path / f"original{episode['original_ext']}"
    if not source.is_file():
        raise HTTPException(404, "The source video is unavailable")
    return cut, episode, source


@app.get('/api/cuts/{cut_id}/boundary-preview')
def boundary_preview_times(cut_id: str, position: Literal['first', 'last'], around: float | None = None):
    cut, episode, source = boundary_preview_source(cut_id)
    anchor = (cut['first_frame_time'] if position == 'first' else cut['last_frame_time'])
    if anchor is None:
        anchor = cut['start_time'] if position == 'first' else cut['end_time']
    if around is not None:
        if not math.isfinite(around) or not 0 <= around <= (episode['duration'] or 0):
            raise HTTPException(400, 'Invalid preview time')
        anchor = around
    try:
        return {'times': media.frame_timestamps_near(source, anchor), 'anchor': anchor}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/api/cuts/{cut_id}/boundary-preview-frame')
async def boundary_preview_frame(cut_id: str, time: float, request: Request):
    cut, episode, source = boundary_preview_source(cut_id)
    if not math.isfinite(time) or time < 0 or time > (episode['duration'] or 0):
        raise HTTPException(400, 'Frame is outside the boundary preview window')
    return await rendered_frame({'source': source, 'time': time, 'start': max(0, time - .000002), 'end': min(episode['duration'], time + 1), 'width': 480, 'last': False, 'seek_tolerance': .000002}, episode_id=episode['id'], request=request)


@app.post('/api/cuts/{cut_id}/boundary-frame')
def step_cut_boundary(cut_id: str, body: CutBoundaryStepInput):
    """Move this cut's end and the following cut's start by one video frame."""
    cut = db.one('SELECT episode_id FROM cuts WHERE id=? AND active=1', (cut_id,))
    if not cut:
        raise HTTPException(404, 'Active cut not found')
    try:
        return {'job_id': worker.enqueue(cut['episode_id'], 'cutframes', {
            'cut_id': cut_id, 'frame_step': -1 if body.direction == 'earlier' else 1,
            'frame_anchor': body.anchor, 'workflow': 'boundary_frame'})}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/cuts/{cut_id}/merge-next")
def merge_cut_next(cut_id: str):
    return edit_cut_timeline(cut_id, "merge")


@app.delete("/api/cuts/{cut_id}")
def delete_cut(cut_id: str):
    return edit_cut_timeline(cut_id, "delete")


@app.put("/api/chunks/{chunk_id}")
def update_chunk(chunk_id: str, body: ChunkInput):
    if not db.one("SELECT id FROM chunks WHERE id=?", (chunk_id,)):
        raise HTTPException(404, "Chunk not found")
    with db.connect() as conn:
        chunk = conn.execute('''UPDATE chunks SET title=?,notes=? WHERE id=? AND (title!=? OR notes!=?)
            RETURNING episode_id,start_time,end_time''', (body.title, body.notes, chunk_id, body.title, body.notes)).fetchone()
        if chunk:
            ids = [row['id'] for row in conn.execute('''SELECT id FROM cuts WHERE episode_id=? AND active=1
                AND start_time<? AND end_time>?''', (chunk['episode_id'], chunk['end_time'], chunk['start_time']))]
            structured_results.invalidate_notes(conn, ids, 'Chunk notes changed')
    return db.one("SELECT * FROM chunks WHERE id=?", (chunk_id,))


@app.get("/api/cuts/{cut_id}/analysis-results", responses={200: {"model": list[AnalysisResult]}})
def list_analysis_results(cut_id: str, history: bool = False, include_context: bool = True):
    """Return current result revisions, or all revisions when history is true.

    Set include_context=false to omit captured input evidence from each result.
    """
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    where = "cut_id=?" if history else "cut_id=? AND superseded_by IS NULL"
    columns = "*" if include_context else "id,cut_id,kind,title,content,origin,model_id,job_id,prompt,settings_json,'{}' AS context_json,parent_id,superseded_by,created_at,stale_reason"
    return [serialize_result(row) for row in db.all_rows(f"SELECT {columns} FROM analysis_results WHERE {where} ORDER BY rowid DESC", (cut_id,))]


@app.post("/api/cuts/{cut_id}/analysis-results", responses={200: {"model": AnalysisResult}})
def create_analysis_result(cut_id: str, body: ResultCreate):
    """Create an analysis result available to analysis context."""
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    result_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content) VALUES (?,?,?,?,?)",
                     (result_id, cut_id, body.kind, body.title, body.content))
        if body.title == 'Cut summary':
            structured_results.invalidate_sequence_reviews(conn, [cut_id], 'Cut summary changed')
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (result_id,)))


@app.put("/api/analysis-results/{result_id}", responses={200: {"model": AnalysisResult}, 409: {"description": "This result already has a newer revision."}})
def revise_analysis_result(result_id: str, body: ResultUpdate):
    """Append a human revision."""
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = conn.execute("SELECT * FROM analysis_results WHERE id=?", (result_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Analysis result not found")
        if old["superseded_by"]:
            raise HTTPException(409, "This result already has a newer revision")
        from . import structured_results
        result_type = structured_results.structured_type(old['settings_json'])
        if result_type:
            try:
                data = json.loads(body.content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid number')))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
            except ValueError as exc:
                raise HTTPException(422, 'Structured results must contain a valid JSON object') from exc
            conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content,origin,parent_id,settings_json,context_json) VALUES (?,?,?,?,?,'manual_edit',?,?,?)",
                         (new_id, old['cut_id'], old['kind'], body.title, body.content, result_id, old['settings_json'], old['context_json']))
            structured_results.invalidate_text(conn, old['cut_id'], result_type)
        else:
            if len(body.content) > 50000:
                raise HTTPException(422, 'Text results may contain at most 50000 characters')
            conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content,origin,parent_id) VALUES (?,?,?,?,?,?,?)",
                         (new_id, old["cut_id"], old["kind"], body.title, body.content, "manual_edit", result_id))
        conn.execute("UPDATE analysis_results SET superseded_by=? WHERE id=?", (new_id, result_id))
        if 'Cut summary' in (old['title'], body.title):
            structured_results.invalidate_sequence_reviews(conn, [old['cut_id']], 'Cut summary changed')
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (new_id,)))


@app.get('/api/cuts/{cut_id}/structured-analysis', response_model=list[StructuredAnalysisResult])
def get_structured_analysis(cut_id: str, history: bool = False):
    """Return structured single-cut outputs as JSON; history includes superseded revisions."""
    from .structured_results import list_results
    if not db.one('SELECT id FROM cuts WHERE id=?', (cut_id,)):
        raise HTTPException(404, 'Cut not found')
    return list_results(cut_id, history)


@app.get("/api/episodes/{episode_id}/transcript", responses={200: {"model": list[Transcript]}})
def list_transcript(episode_id: str, history: bool = False):
    """Return current non-deleted transcript segments; history includes every revision."""
    require_episode(episode_id)
    where = "episode_id=?" if history else "episode_id=? AND superseded_by IS NULL AND deleted=0"
    order = "rowid DESC" if history else "start_time,id"
    return [serialize_transcript(segment) for segment in db.all_rows(f"SELECT * FROM transcript_segments WHERE {where} ORDER BY {order}", (episode_id,))]


@app.post("/api/episodes/{episode_id}/transcribe")
def transcribe_episode(episode_id: str):
    with db.connect() as conn:
        episode = audio_episode_for_queue(conn, episode_id)
        job_id = worker.enqueue_in(conn, episode['id'], 'transcribe')
    worker.wake_event.set()
    return {'job_id': job_id}


def audio_episode_for_queue(conn, episode_id):
    conn.execute('BEGIN IMMEDIATE')
    episode = db.one('SELECT * FROM episodes WHERE id=?', (episode_id,), conn=conn)
    if not episode:
        raise HTTPException(404, 'Episode not found')
    if (not episode['has_audio'] or not episode['duration']) and not worker.pending_jobs(conn, episode_id, ('prepare',)):
        raise HTTPException(400, 'Prepare a video with audio first')
    return episode


@app.post("/api/episodes/{episode_id}/transcribe-range")
def transcribe_episode_range(episode_id: str, body: TranscribeRangeInput):
    """Queue transcription and speaker identification again for one interval of the audio.

    Only the interval changes: its untouched model segments get a deletion revision and the
    new segments are appended, so the earlier text stays in history. Speakers in the interval
    are named after the labels the neighbouring speech already uses.
    """
    with db.connect() as conn:
        episode = audio_episode_for_queue(conn, episode_id)
        duration = episode['duration']
        if duration and body.end_time > duration + .2:
            raise HTTPException(400, 'The selected audio ends after the video')
        start, end = body.start_time, min(body.end_time, duration) if duration else body.end_time
        if end - start < audio_worker.MIN_RANGE_SECONDS:
            raise HTTPException(400, f'Select at least {audio_worker.MIN_RANGE_SECONDS:g} seconds of audio')
        job_id = worker.enqueue_in(conn, episode_id, 'transcribe',
                                  {'range': {'start_time': start, 'end_time': end}, 'replace_manual': body.replace_manual})
    worker.wake_event.set()
    return {"job_id": job_id, "start_time": start, "end_time": end}


class SubtitleRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str = Field(min_length=2, max_length=60, description="Language of the burned-in subtitles; one language applies to the whole episode.")


def episode_subtitles(episode_id: str, history: bool = False):
    """The saved language, the current run and its lines, plus every run when history is requested."""
    with db.connect() as conn:
        run = subtitle_store.current_run(episode_id, conn)
        payload = {"language": subtitle_store.episode_language(episode_id, conn), "run": run,
                   "lines": subtitle_store.lines(episode_id, run_id=run["id"], conn=conn) if run else []}
        if run:
            run["line_count"] = len(payload["lines"])
        if history:
            payload["runs"] = [subtitle_store.serialize_run(row, with_prompt=True) for row in conn.execute("SELECT * FROM subtitle_runs WHERE episode_id=? ORDER BY rowid DESC", (episode_id,))]
    return payload


@app.post("/api/episodes/{episode_id}/subtitles")
def extract_episode_subtitles(episode_id: str, body: SubtitleRunInput):
    """Queue one job that extracts the burned-in subtitles of the entire episode, independent of its cuts.

    The language is required: the whole video has one subtitle language and the model is told it as a fact.
    A pending transcription becomes a prerequisite. Speech times guide where the subtitle strip is located.
    """
    from . import subtitle_reader
    try:
        language = subtitle_reader.resolve_language(body.language)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        episode = db.one('SELECT duration,width FROM episodes WHERE id=?', (episode_id,), conn=conn)
        if not episode:
            raise HTTPException(404, 'Episode not found')
        if (not episode['duration'] or not episode['width']) and not worker.pending_jobs(conn, episode_id, ('prepare',)):
            raise HTTPException(400, 'Prepare the video before extracting subtitles')
        job_id = queue_subtitles(conn, episode_id, language['name'])
    worker.wake_event.set()
    return {"job_id": job_id, "language": language["name"]}


def queue_subtitles(conn, episode_id, language, depends_on=(), accept_unread=False):
    payload = {'language': language, 'depends_on': list(depends_on)}
    if accept_unread:
        payload['accept_unread'] = True  # subtitle_pipeline.finish: complete with at most UNREAD_SHARE of the images unread
    job_id = worker.enqueue_in(conn, episode_id, 'subtitles', payload)
    conn.execute('UPDATE episodes SET subtitle_language=? WHERE id=?', (language, episode_id))
    return job_id


@app.post('/api/episodes/{episode_id}/prepare')
def prepare_episode(episode_id: str, body: SubtitleRunInput):
    """Queue the episode's remaining setup steps in their input order."""
    from . import anime_bulk, subtitle_reader
    try:
        language = subtitle_reader.resolve_language(body.language)['name']
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        episode = db.one('SELECT * FROM episodes WHERE id=?', (episode_id,), conn=conn)
        if not episode:
            raise HTTPException(404, 'Episode not found')
        steps, depends_on = [], []

        def record(step, jobs):
            depends_on.extend(job for job in jobs if job not in depends_on)
            steps.append({'step': step, 'job_ids': jobs, 'status': 'queued' if jobs else 'skipped'})

        media_jobs = worker.pending_jobs(conn, episode_id, ('prepare',))
        folder = media.episode_dir(episode['project_id'], episode_id)
        if not media_jobs and (not episode['duration'] or not (folder / 'proxy.mp4').is_file()):
            media_jobs = [worker.enqueue_in(conn, episode_id, 'prepare')]
        record('media', media_jobs)

        cut_jobs = [row['id'] for row in conn.execute("""SELECT id FROM jobs WHERE episode_id=? AND pipeline='hardcuts'
            AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id') IS NULL ORDER BY rowid""", (episode_id,))]
        if not cut_jobs and not conn.execute('SELECT 1 FROM cuts WHERE episode_id=? AND active=1 LIMIT 1', (episode_id,)).fetchone():
            cut_jobs = [worker.enqueue_in(conn, episode_id, 'hardcuts', {'depends_on': list(depends_on)})]
        record('hardcuts', cut_jobs)

        full_audio = conn.execute("""SELECT id,status,payload_json FROM jobs WHERE episode_id=? AND pipeline='transcribe'
            AND status IN ('queued','running','done') AND json_extract(payload_json,'$.range') IS NULL
            ORDER BY CASE WHEN status='done' THEN 1 ELSE 0 END,rowid DESC LIMIT 1""", (episode_id,)).fetchone()
        audio_jobs = []
        if full_audio and full_audio['status'] != 'done':
            if full_audio['status'] == 'queued':
                worker.add_dependencies(conn, full_audio, depends_on)
            audio_jobs = [full_audio['id']]
        elif not full_audio and episode['has_audio'] != 0:
            audio_jobs = [worker.enqueue_in(conn, episode_id, 'transcribe', {'depends_on': list(depends_on)})]
        record('transcribe', audio_jobs)

        subtitles = subtitle_store.current_run(episode_id, conn)
        subtitle_jobs = []
        if not subtitles or subtitles['language'] != language or subtitles.get('stale_reason') or audio_jobs:
            subtitle_jobs = [queue_subtitles(conn, episode_id, language, depends_on)]
        record('subtitles', subtitle_jobs)

        for phase in ('boxes', 'recognize'):
            # Pending boundary replacement may produce additional cuts. The dispatcher
            # checks the new cuts after the prerequisites and skips any completed work.
            queued = []
            if cut_jobs or (phase == 'recognize' and steps[-1]['job_ids']) or anime_bulk.eligible_cuts(conn, episode_id, phase):
                queued = anime_bulk.enqueue(conn, episode_id, phase, depends_on=depends_on)
            record(phase, queued)
        worker.sync_episode_status(conn, episode_id)
    worker.wake_event.set()
    return {'steps': steps, 'job_ids': depends_on}


@app.get("/api/episodes/{episode_id}/subtitles", responses={200: {"model": EpisodeSubtitles}})
def list_episode_subtitles(episode_id: str, history: bool = False):
    """Return the subtitle language, the current run and its lines. Lines belong to the episode, not to a cut."""
    require_episode(episode_id)
    return episode_subtitles(episode_id, history)


@app.get("/api/projects/{project_id}/voice-samples", responses={200: {"model": list[VoiceSample]}})
def list_voice_samples(project_id: str, character_context_id: str | None = None):
    """List character voice references in the project and whether their audio is available."""
    require_project(project_id)
    character_filter = ' AND v.character_context_id=?' if character_context_id else ''
    args = (project_id, character_context_id) if character_context_id else (project_id,)
    samples = db.all_rows(f"""SELECT v.id,v.character_context_id,v.episode_id,v.start_time,v.end_time,v.ranges_json,v.source_filename,v.model_id,v.created_at,
        e.title AS episode_title FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
        LEFT JOIN episodes e ON e.id=v.episode_id WHERE c.project_id=?{character_filter} ORDER BY v.created_at DESC""", args)
    for sample in samples:
        sample['ranges'] = json.loads(sample.pop('ranges_json')) if sample['ranges_json'] else ([] if sample['source_filename'] else [{'start_time': sample['start_time'], 'end_time': sample['end_time']}])
        sample['preview_available'] = media.voice_reference_path(project_id, sample['id']).is_file() or bool(sample['episode_id'] and sample['ranges'] and (media.episode_dir(project_id, sample['episode_id']) / 'audio.wav').is_file())
    return samples


def validate_voice_character(episode, character_id, conn=None):
    character = db.one("SELECT id,unregistered FROM context_items WHERE id=? AND project_id=? AND kind='character'", (character_id, episode["project_id"]), conn=conn)
    if not character:
        raise HTTPException(400, "Choose a character from this project")
    if unregistered.is_card(character):
        raise HTTPException(400, "An Unregistered card stands for different people: it has no voice samples")


def validate_voice_ranges(episode, ranges, minimum=2):
    ordered = sorted(ranges, key=lambda r: r['start_time'])
    if (any(r['end_time'] <= r['start_time'] or (episode['duration'] and r['end_time'] > episode['duration']) for r in ranges)
            or any(a['end_time'] > b['start_time'] for a, b in zip(ordered, ordered[1:]))
            or not minimum <= sum(r['end_time'] - r['start_time'] for r in ranges) <= 60):
        raise HTTPException(400, "Choose non-overlapping ranges within the video totaling at most 60 seconds" if minimum == 0 else "Choose non-overlapping ranges within the video totaling 2 to 60 seconds")


def preview_voice_ranges(audio, ranges):
    if not audio.is_file():
        raise HTTPException(404, 'Episode audio is not ready')
    output = io.BytesIO()
    try:
        media.write_wav_ranges(audio, ranges, output)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (wave.Error, EOFError) as exc:
        raise HTTPException(400, 'Episode audio could not be previewed') from exc
    return Response(content=output.getvalue(), media_type='audio/wav')


@app.post('/api/episodes/{episode_id}/voice-samples/preview')
def preview_voice_sample(episode_id: str, body: VoicePreviewInput):
    episode = require_episode(episode_id)
    if not episode['has_audio'] or not episode['duration']:
        raise HTTPException(400, 'Prepare a video with audio first')
    ranges = [r.model_dump() for r in body.ranges]
    validate_voice_ranges(episode, ranges, minimum=0)
    return preview_voice_ranges(media.episode_dir(episode['project_id'], episode_id) / 'audio.wav', ranges)


@app.post("/api/episodes/{episode_id}/voice-samples")
def create_voice_sample(episode_id: str, body: VoiceSampleInput):
    if body.ranges is not None:
        ranges = [r.model_dump() for r in body.ranges]
    elif body.start_time is not None and body.end_time is not None:
        ranges = [{'start_time': body.start_time, 'end_time': body.end_time}]
    else:
        raise HTTPException(400, "Choose at least one audio range")
    with db.connect() as conn:
        episode = audio_episode_for_queue(conn, episode_id)
        validate_voice_ranges(episode, ranges)
        validate_voice_character(episode, body.character_context_id, conn)
        job_id = worker.enqueue_in(conn, episode_id, 'voice_sample', {'character_context_id': body.character_context_id, 'ranges': ranges})
    worker.wake_event.set()
    return {'job_id': job_id}


@app.post("/api/episodes/{episode_id}/voice-samples/upload")
async def upload_voice_sample(episode_id: str, character_context_id: str = Form(...), file: UploadFile = File(...)):
    episode = require_episode(episode_id)
    validate_voice_character(episode, character_context_id)
    upload_id = str(uuid.uuid4())
    directory = media.episode_dir(episode['project_id'], episode_id) / 'voice-uploads'
    directory.mkdir(exist_ok=True)
    target = directory / upload_id
    try:
        total = 0
        digest = hashlib.sha256()
        with target.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 25 * 1024 * 1024:
                    raise HTTPException(413, 'Voice audio must be 25 MB or smaller')
                output.write(chunk)
                digest.update(chunk)
        if not total:
            raise HTTPException(400, 'Voice audio file is empty')
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            validate_voice_character(episode, character_context_id, conn)
            job_id = worker.enqueue_in(conn, episode_id, 'voice_sample', {'character_context_id': character_context_id,
                'upload_id': upload_id, 'upload_hash': digest.hexdigest(), 'source_filename': Path(file.filename or 'Audio upload').name[:200]})
            saved = db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,), conn=conn)
        if json.loads(saved['payload_json'])['upload_id'] != upload_id:
            target.unlink(missing_ok=True)
        worker.wake_event.set()
        return {'job_id': job_id}
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.delete("/api/voice-samples/{sample_id}")
def delete_voice_sample(sample_id: str):
    with db.connect() as conn:
        sample = conn.execute('SELECT c.project_id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE v.id=?', (sample_id,)).fetchone()
        if not conn.execute("DELETE FROM character_voice_samples WHERE id=?", (sample_id,)).rowcount:
            raise HTTPException(404, "Voice sample not found")
    media.voice_reference_path(sample['project_id'], sample_id).unlink(missing_ok=True)
    return {"ok": True}


@app.get('/api/voice-samples/{sample_id}/audio', response_class=Response,
         responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}, "description": "Reference voice WAV"},
                    404: {"description": "Reference or audio is unavailable"}})
def get_voice_sample_audio(sample_id: str):
    """Download the saved reference WAV or a reconstructed legacy reference, when available."""
    sample = db.one('SELECT v.*,c.project_id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE v.id=?', (sample_id,))
    if not sample:
        raise HTTPException(404, 'Voice sample not found')
    target = media.voice_reference_path(sample['project_id'], sample['id'])
    if target.is_file():
        return FileResponse(target, media_type='audio/wav')
    if not sample['episode_id'] or sample['source_filename']:
        raise HTTPException(404, 'Audio preview is not available for this reference')
    ranges = json.loads(sample['ranges_json']) if sample['ranges_json'] else [{'start_time': sample['start_time'], 'end_time': sample['end_time']}]
    return preview_voice_ranges(media.episode_dir(sample['project_id'], sample['episode_id']) / 'audio.wav', ranges)


class GlossaryInput(BaseModel):
    spelling: str = Field(min_length=1, max_length=120)
    reading: str = Field(min_length=1, max_length=120)
    character_context_id: str | None = None
    part: Literal['family', 'given'] | None = None


@app.get('/api/projects/{project_id}/name-glossary')
def get_name_glossary(project_id: str):
    require_project(project_id)
    with db.connect() as conn:
        return name_glossary.rows(conn, project_id)


def glossary_changed(conn, project_id):
    for episode in conn.execute('SELECT id FROM episodes WHERE project_id=?', (project_id,)).fetchall():
        speaker_attribution.mark_stale(conn, episode['id'], 'Name glossary changed')


def save_glossary_entry(project_id, body, entry_id=None):
    require_project(project_id)
    spelling, reading = body.spelling.strip(), body.reading.strip()
    if not spelling or not re.fullmatch(r'[ぁ-ゖァ-ヺー]+', reading):
        raise HTTPException(422, 'Enter a spelling and a kana reading.')
    if bool(body.character_context_id) != bool(body.part):
        raise HTTPException(422, 'Choose a family or given name for a character name.')
    values = (spelling, reading, body.character_context_id, body.part)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if body.character_context_id and not conn.execute("""SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character' AND unregistered=0""",
                                                        (body.character_context_id, project_id)).fetchone():
            raise HTTPException(422, 'Choose a character from this project.')
        try:
            if entry_id:
                cursor = conn.execute("""UPDATE project_name_glossary SET spelling=?,reading=?,character_context_id=?,part=?
                    WHERE id=? AND project_id=?""", (*values, entry_id, project_id))
                if not cursor.rowcount:
                    raise HTTPException(404, 'Glossary entry not found')
            else:
                conn.execute("""INSERT INTO project_name_glossary(spelling,reading,character_context_id,part,id,project_id)
                    VALUES (?,?,?,?,?,?)""", (*values, str(uuid.uuid4()), project_id))
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, 'This glossary entry already exists.') from exc
        glossary_changed(conn, project_id)
        return name_glossary.rows(conn, project_id)


@app.post('/api/projects/{project_id}/name-glossary')
def create_glossary_entry(project_id: str, body: GlossaryInput):
    return save_glossary_entry(project_id, body)


@app.put('/api/projects/{project_id}/name-glossary/{entry_id}')
def update_glossary_entry(project_id: str, entry_id: str, body: GlossaryInput):
    return save_glossary_entry(project_id, body, entry_id)


@app.delete('/api/projects/{project_id}/name-glossary/{entry_id}')
def delete_glossary_entry(project_id: str, entry_id: str):
    with db.connect() as conn:
        cursor = conn.execute('DELETE FROM project_name_glossary WHERE id=? AND project_id=?', (entry_id, project_id))
        if not cursor.rowcount:
            raise HTTPException(404, 'Glossary entry not found')
        glossary_changed(conn, project_id)
        return name_glossary.rows(conn, project_id)


@app.post('/api/episodes/{episode_id}/speaker-attribution')
def rerun_speaker_attribution(episode_id: str):
    require_episode(episode_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        dependencies = worker.pending_jobs(conn, episode_id, ('transcribe', 'subtitles'))
        return {'job_id': speaker_attribution.insert(conn, episode_id, dependencies=dependencies)}


class AudioRangeInput(BaseModel):
    start_time: float
    end_time: float
    kind: Literal['OP', 'ED', 'repeat', 'manual'] = 'manual'


@app.get('/api/episodes/{episode_id}/audio-ranges')
def get_audio_ranges(episode_id: str):
    require_episode(episode_id)
    with db.connect() as conn:
        return audio_ranges.public(conn, episode_id)


def save_audio_range(episode_id, body, range_id=None):
    episode = require_episode(episode_id)
    if not 0 <= body.start_time < body.end_time <= (episode['duration'] or 0):
        raise HTTPException(422, 'Set a range within the episode.')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if range_id and not audio_ranges.remove(conn, episode_id, range_id):
            raise HTTPException(404, 'Audio range not found')
        audio_ranges.add(conn, episode_id, body.start_time, body.end_time, body.kind)
        return audio_ranges.public(conn, episode_id)


@app.post('/api/episodes/{episode_id}/audio-ranges')
def create_audio_range(episode_id: str, body: AudioRangeInput):
    return save_audio_range(episode_id, body)


@app.put('/api/episodes/{episode_id}/audio-ranges/{range_id}')
def update_audio_range(episode_id: str, range_id: str, body: AudioRangeInput):
    return save_audio_range(episode_id, body, range_id)


@app.delete('/api/episodes/{episode_id}/audio-ranges/{range_id}')
def delete_audio_range(episode_id: str, range_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not audio_ranges.remove(conn, episode_id, range_id):
            raise HTTPException(404, 'Audio range not found')
        return audio_ranges.public(conn, episode_id)


@app.post("/api/episodes/{episode_id}/transcript")
def create_transcript_segment(episode_id: str, body: TranscriptInput):
    episode = require_episode(episode_id)
    validate_transcript(episode, body)
    segment_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,speaker_source) VALUES (?,?,?,?,?,?,?,'manual')",
                     (segment_id, episode_id, body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id))
        structured_results.invalidate_transcript(conn, episode_id, body.start_time, body.end_time)
        subtitle_store.invalidate_probes(conn, episode_id)
        agent_log.settle(conn, episode_id)
    return serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,)))


@app.put("/api/transcript-segments/{segment_id}")
def update_transcript_segment(segment_id: str, body: TranscriptInput):
    segment = db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,))
    if not segment:
        raise HTTPException(404, "Transcript segment not found")
    episode = require_episode(segment["episode_id"])
    validate_transcript(episode, body)
    new_id = str(uuid.uuid4())
    metadata = manual_metadata(segment, body.model_dump())
    if body.left_unknown:
        # The line editor's "Unknown – don't ask again": settled with or without a character (cut_readiness.speaker_open).
        metadata = (metadata[0], cut_readiness.LEFT_UNKNOWN, None, metadata[3])
    elif segment['character_context_id'] and not body.character_context_id:
        # The person cleared the speaker: their last word, never asked again (HARNESS.md D32)
        metadata = (metadata[0], cut_readiness.LEFT_UNKNOWN, None, metadata[3])
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = conn.execute("SELECT superseded_by,deleted FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if not current or current["superseded_by"] or current["deleted"]:
            raise HTTPException(409, "This transcript segment is deleted or has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id, segment["episode_id"], body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id, "manual_edit", segment["model_id"], segment["job_id"], segment_id, *metadata))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
        agent_log.note_override(conn, 'transcript_segments', segment_id)
        agent_log.settle(conn, episode['id'])  # a voice or line-split question this line settles closes (cut dialog, Audio tab)
        structured_results.invalidate_transcript(conn, episode['id'], segment['start_time'], segment['end_time'])
        if (segment['start_time'], segment['end_time']) != (body.start_time, body.end_time):
            structured_results.invalidate_transcript(conn, episode['id'], body.start_time, body.end_time)
        if metadata[0] != segment['words_json'] or (segment['start_time'], segment['end_time']) != (body.start_time, body.end_time):
            subtitle_store.invalidate_probes(conn, episode['id'])
    return serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (new_id,)))


@app.post("/api/episodes/{episode_id}/transcript/confirm-speakers")
def confirm_transcript_speakers(episode_id: str, body: SpeakerConfirmInput):
    """The person accepts the speaker the app chose: each line becomes a manual revision with the same text, times and character.

    Only the speaker's provenance changes (``speaker_source='manual'``, no confidence), so nothing derived from the words goes stale;
    ``cut_readiness`` and the review lists stop asking about the line. A line that is no longer current or has no character is skipped."""
    require_episode(episode_id)
    confirmed, skipped = [], []
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for segment_id in dict.fromkeys(body.segment_ids):
            old = conn.execute("SELECT * FROM transcript_segments WHERE id=? AND episode_id=?", (segment_id, episode_id)).fetchone()
            if not old or old["superseded_by"] or old["deleted"] or not old["character_context_id"]:
                skipped.append(segment_id)
                continue
            new_id = str(uuid.uuid4())
            conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (new_id, episode_id, old["start_time"], old["end_time"], old["text"], old["speaker_label"], old["character_context_id"], "manual_edit",
                          old["model_id"], old["job_id"], segment_id, old["words_json"], "manual", None, old["evidence_json"]))
            conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
            confirmed.append(new_id)
        agent_log.settle(conn, episode_id)
    return {"confirmed": len(confirmed), "skipped": skipped}


@app.delete("/api/transcript-segments/{segment_id}")
def delete_transcript_segment(segment_id: str):
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = conn.execute("SELECT * FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Transcript segment not found")
        if old["superseded_by"] or old["deleted"]:
            raise HTTPException(409, "This transcript segment is deleted or has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,deleted,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id, old["episode_id"], old["start_time"], old["end_time"], old["text"], old["speaker_label"], old["character_context_id"], 1, "manual_edit", old["model_id"], old["job_id"], segment_id, *manual_metadata(old, old)))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
        agent_log.note_override(conn, 'transcript_segments', segment_id)
        structured_results.invalidate_transcript(conn, old['episode_id'], old['start_time'], old['end_time'])
        subtitle_store.invalidate_probes(conn, old['episode_id'])
        agent_log.settle(conn, old['episode_id'])
    return {"ok": True}


@app.post("/api/episodes/{episode_id}/transcript/merge")
def merge_transcript_segments(episode_id: str, body: TranscriptMergeInput):
    """Join several current lines into one revision; the sources are superseded by it."""
    episode = require_episode(episode_id)
    require_character(episode, body.character_context_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            new_id = line_edit.merge(conn, episode_id, body.segment_ids, body.text, body.speaker_label, body.character_context_id)
        except line_edit.LineEditError as exc:
            raise HTTPException(exc.status, exc.message)
        for segment_id in body.segment_ids:
            agent_log.note_override(conn, 'transcript_segments', segment_id)
        agent_log.settle(conn, episode_id)
    return serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (new_id,)))


@app.post("/api/episodes/{episode_id}/transcript/corrections")
def correct_transcript(episode_id: str, body: TranscriptCorrections):
    """Respell current lines (and, rarely, name another character) from a checked source such as a real .srt.

    Times and aligner words are kept; lines changed since the plan was made come back under ``skipped``.
    """
    episode = require_episode(episode_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        outcome = transcript_corrections.apply(conn, episode, body.source.model_dump(), [item.model_dump() for item in body.corrections])
        agent_log.settle(conn, episode_id)
        return outcome


@app.post("/api/transcript-segments/{segment_id}/split")
def split_transcript_segment(segment_id: str, body: TranscriptSplitInput):
    """Divide one current line in two at a time; the original is superseded by the first part."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            left_id, right_id = line_edit.split(conn, segment_id, body.at, body.left.text, body.right.text)
        except line_edit.LineEditError as exc:
            raise HTTPException(exc.status, exc.message)
        agent_log.note_override(conn, 'transcript_segments', segment_id)
        agent_log.settle(conn, conn.execute('SELECT episode_id FROM transcript_segments WHERE id=?', (left_id,)).fetchone()[0])
    return [serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (item,))) for item in (left_id, right_id)]


@app.post("/api/episodes/{episode_id}/transcript/assign-voice")
def assign_voice(episode_id: str, body: AssignVoiceInput):
    """Give one diarization voice a character on every current line whose speaker is still open (none, or an unsure guess)."""
    episode = require_episode(episode_id)
    require_character(episode, body.character_context_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        lines = conn.execute(f"""SELECT * FROM transcript_segments
            WHERE episode_id=? AND superseded_by IS NULL AND deleted=0
            AND coalesce(speaker_label,'')=? AND {cut_readiness.open_speaker_sql()}
            ORDER BY start_time,id""", (episode_id, body.speaker_label)).fetchall()
        for line in lines:
            new_id = str(uuid.uuid4())
            conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (new_id, line["episode_id"], line["start_time"], line["end_time"], line["text"], line["speaker_label"], body.character_context_id, "manual_edit", line["model_id"], line["job_id"], line["id"], line["words_json"], "manual", None, line["evidence_json"]))
            conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, line["id"]))
        if lines:
            structured_results.invalidate_transcript(conn, episode_id)
            from . import voice_resolver
            voice_resolver.close_settled(conn, episode_id)
    return {"updated": len(lines)}


@app.get("/api/cuts/{cut_id}/analysis-context", responses={200: {"model": AnalysisContext}})
def get_analysis_context(cut_id: str):
    """Return cut context with current neighboring results and overlapping transcript segments."""
    with db.connect() as conn:
        conn.execute('BEGIN')
        context = analysis_context.load(conn, cut_id)
    if context is None:
        raise HTTPException(404, "Cut not found")
    return context


SINGLE_WINDOW_SECONDS = 6.0


def automatic_video_fps(duration: float) -> float:
    """Frame rate that fits the automatic frame budget.

    A cut of up to SINGLE_WINDOW_SECONDS is sampled as 30 frames (at least 5 per second), which the
    video model watches as one clip, so its motion is described once and no event is split between
    two windows. Longer cuts keep 56 frames across windows of at most 32.
    """
    return min(30.0, (30.0 if duration <= SINGLE_WINDOW_SECONDS else 56.0) / duration)


def analysis_config_for_cut(cut_id: str):
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    with db.connect() as conn:
        return cut, snapshot_analysis_config(conn, cut)


def snapshot_analysis_config(conn, cut):
    saved = db.one("SELECT config_json FROM cut_analysis_configs WHERE cut_id=?", (cut['id'],), conn=conn)
    duration = max(0.000001, cut["end_time"] - cut["start_time"])
    if saved:
        config = AnalysisConfig.model_validate_json(saved["config_json"])
        if config.video_frame_budget is not None:
            config.video_fps = automatic_video_fps(duration)
    else:
        config = AnalysisConfig(video_fps=automatic_video_fps(duration), video_frame_budget=60)
    return config.model_copy(update=text_generation_settings(conn))


def selected_analysis_context(cut_id: str, config: AnalysisConfig):
    with db.connect() as conn:
        conn.execute('BEGIN')
        try:
            context = analysis_context.selected(conn, cut_id, config.model_dump(), strict=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if context is None:
        raise HTTPException(404, "Cut not found")
    return context


@app.post('/api/cuts/{cut_id}/character-detection')
def queue_character_detection(cut_id: str):
    """Detect and recognize characters in one cut with the bulk pipeline.

    A cut with boxes is recognized again like Re-run Recognition: automatic
    labels are redone and manual labels and drawn boxes are kept. A cut
    without boxes gets the full pass. An explicit request ignores the
    "No characters" mark and the analysis switch, which only steer batches.
    """
    cut = db.one('SELECT * FROM cuts WHERE id=?', (cut_id,))
    if not cut or not cut['active']:
        raise HTTPException(404, 'Active cut not found')
    result = _queue_bulk_character_jobs(cut['episode_id'], 'recognize', rerun=True, cut_id=cut_id)
    if not result['job_ids']:
        has_boxes = db.one('''SELECT 1 FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id
            WHERE r.cut_id=? AND r.cut_start=? AND r.cut_end=? AND r.rowid=(SELECT max(rowid) FROM character_detection_runs WHERE cut_id=?)
            AND d.position='first' ''',
            (cut_id, cut['start_time'], cut['end_time'], cut_id))
        return {'job_id': None, 'queued': 0, 'detail': 'Every box in this cut already has your label.' if has_boxes
                else 'No people were found in this cut. Draw a box for anyone missing.'}
    return {'job_id': result['job_ids'][0], 'queued': 1}


class NoCharactersInput(BaseModel):
    no_characters: bool


@app.put('/api/cuts/{cut_id}/no-characters')
def set_cut_no_characters(cut_id: str, body: NoCharactersInput):
    """Mark a cut as having no characters so batch detection skips it."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        cut = conn.execute('SELECT c.episode_id,c.no_characters,e.project_id FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,)).fetchone()
        if not cut:
            raise HTTPException(404, 'Cut not found')
        conn.execute('UPDATE cuts SET no_characters=? WHERE id=?', (int(body.no_characters), cut_id))
        if bool(cut['no_characters']) != body.no_characters:
            structured_results.invalidate_detections(conn, cut_id, reason='The cut’s character setting changed')
        detect_pending = worker.pending_jobs(conn, cut['episode_id'], ('character_detect',), [cut_id])
    if body.no_characters and not detect_pending:
        from . import anime_bulk
        anime_bulk.clear_cut_frames({'project_id': cut['project_id'], 'id': cut['episode_id']}, cut_id)
    return {'cut_id': cut_id, 'no_characters': body.no_characters}


def _queue_bulk_character_jobs(episode_id: str, phase: str, rerun: bool = False, cut_id: str | None = None,
                               from_cut: int | None = None, to_cut: int | None = None):
    """Queue one bulk job per cut. The gallery of confirmed labels is read live by the worker.

    With rerun, recognition is redone for every cut that has boxes. The new
    labels replace old automatic labels and suggestions when the run publishes.
    Manual labels (and the boxes the user drew) are never touched, and a cut
    whose boxes are all manually labelled is skipped. Cuts without a current
    run get the full pass. With cut_id only that cut is considered, whatever
    its "No characters" mark or analysis switch. from_cut/to_cut limit the
    episode request to those displayed cut numbers (inclusive).
    """
    span = episode_requests.cut_range(from_cut, to_cut)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
        if not episode:
            raise HTTPException(404, 'Episode not found')
        if from_cut is not None or to_cut is not None:
            episode_requests.refuse_range_while_cutting(conn, episode_id, True)
        from . import anime_bulk
        job_ids = anime_bulk.enqueue(conn, episode_id, phase, rerun=rerun, cut_id=cut_id,
                                     first_index=span['first_index'], last_index=span.get('last_index'))
        if job_ids:
            worker.sync_episode_status(conn, episode_id)
    return {'queued': len(job_ids), 'job_ids': job_ids}


@app.post('/api/episodes/{episode_id}/character-boxes')
def queue_character_boxes(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    """Step 1: AniRef boxes on the first frame of cuts without a current detection run."""
    return _queue_bulk_character_jobs(episode_id, 'boxes', from_cut=from_cut, to_cut=to_cut)


@app.post('/api/episodes/{episode_id}/character-detection-all')
def queue_all_character_detection(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    """Steps 1 and 2 together: batched AniRef boxes, then gallery matching (optionally Qwen), sharing models across all cuts."""
    return _queue_bulk_character_jobs(episode_id, 'full', from_cut=from_cut, to_cut=to_cut)


@app.post('/api/episodes/{episode_id}/character-detection')
def queue_missing_character_detections(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    """Step 2: match boxes with the gallery of confirmed labels (optionally Qwen) for cuts whose boxes exist but are not complete."""
    return _queue_bulk_character_jobs(episode_id, 'recognize', from_cut=from_cut, to_cut=to_cut)


@app.post('/api/episodes/{episode_id}/character-detection-rerun')
def rerun_character_recognition(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    """Redo automatic recognition for the whole episode; manual labels and drawn boxes are kept as they are."""
    return _queue_bulk_character_jobs(episode_id, 'recognize', rerun=True, from_cut=from_cut, to_cut=to_cut)


@app.get('/api/cuts/{cut_id}/character-detections')
def get_character_detections(cut_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN')
        if not conn.execute('SELECT 1 FROM cuts WHERE id=?', (cut_id,)).fetchone():
            raise HTTPException(404, 'Cut not found')
        return analysis_context.character_detections(conn, cut_id)


@app.delete('/api/character-detection-runs/{run_id}/frame/{position}')
def remove_stale_character_frame(run_id: str, position: Literal['first', 'last']):
    """Remove an outdated boundary frame and its boxes from a stale detection run."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = conn.execute('''SELECT r.*,c.start_time AS current_start,c.end_time AS current_end,c.episode_id
                              FROM character_detection_runs r JOIN cuts c ON c.id=r.cut_id WHERE r.id=?''',
                           (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, 'Character detection run not found')
        boundary_changed = (run['cut_start'] != run['current_start'] if position == 'first'
                            else run['cut_end'] != run['current_end'])
        if not boundary_changed:
            raise HTTPException(409, 'This frame still matches the current cut boundary')
        try:
            frames = json.loads(run['frames_json'])
        except (TypeError, ValueError) as exc:
            raise HTTPException(409, 'The saved detection frames are invalid') from exc
        if not isinstance(frames, list) or not any(isinstance(frame, dict) and frame.get('position') == position for frame in frames):
            raise HTTPException(404, 'Frame not found in this detection run')
        if conn.execute('''SELECT 1 FROM character_image_references ref
                           JOIN character_detections d ON d.id=ref.detection_id
                           WHERE d.run_id=? AND d.position=? LIMIT 1''', (run_id, position)).fetchone():
            raise HTTPException(409, 'Remove the saved project reference before removing this frame')
        removed = conn.execute('DELETE FROM character_detections WHERE run_id=? AND position=?', (run_id, position)).rowcount
        frames = [frame for frame in frames if not (isinstance(frame, dict) and frame.get('position') == position)]
        conn.execute("UPDATE character_detection_runs SET frames_json=?,phase='boxes' WHERE id=?",
                     (json.dumps(frames), run_id))
        episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (run['episode_id'],)).fetchone()
        pending = worker.pending_jobs(conn, run['episode_id'], ('character_detect',), [run['cut_id']])
        character_decisions.close_settled(conn, run['episode_id'])
    cache_dir = db.DATA_DIR / 'projects' / episode['project_id'] / 'episodes' / run['episode_id'] / 'character-frames'
    if not pending and not cache_dir.is_symlink():
        frame_path = cache_dir / f"{run['cut_id']}-{position}.jpg"
        try:
            frame_path.unlink(missing_ok=True)
            cache_dir.rmdir()
        except OSError:
            pass
    return {'ok': True, 'run_id': run_id, 'position': position, 'removed_detections': removed}


def row_episode(conn, run_id=None, detection_id=None):
    """The episode of a detection run (or of the run a detection belongs to)."""
    if detection_id:
        found = conn.execute('SELECT run_id FROM character_detections WHERE id=?', (detection_id,)).fetchone()
        run_id = found['run_id'] if found else None
    row = conn.execute('SELECT c.episode_id FROM character_detection_runs r JOIN cuts c ON c.id=r.cut_id WHERE r.id=?', (run_id,)).fetchone()
    return row['episode_id'] if row else None


@app.put('/api/character-detections/{detection_id}')
def label_character_detection(detection_id: str, body: CharacterDetectionLabelInput):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('''SELECT d.id,d.run_id,d.position,e.project_id FROM character_detections d
                              JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id
                              JOIN episodes e ON e.id=c.episode_id WHERE d.id=?''', (detection_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Detection not found')
        if body.character_context_id and not conn.execute("SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character'", (body.character_context_id, row['project_id'])).fetchone():
            raise HTTPException(400, 'Choose a character from this project')
        if conn.execute('SELECT 1 FROM character_image_references WHERE detection_id=?', (detection_id,)).fetchone():
            raise HTTPException(409, 'Remove the saved reference before changing this label')
        conn.execute("UPDATE character_detections SET character_context_id=?,label_source=?,match_reason=NULL,match_model_id=NULL,match_model_revision=NULL,gallery_status=NULL,gallery_note=NULL WHERE id=?", (body.character_context_id, 'manual', detection_id))
        structured_results.opening_box_edited(conn, row['run_id'], row['position'])
        agent_log.note_override(conn, 'character_detections', detection_id, now={'character_context_id': body.character_context_id})
        character_decisions.close_settled(conn, row_episode(conn, row['run_id']))
    return {'ok': True}


@app.post('/api/character-detections/{detection_id}/new-character')
def create_character_from_detection(detection_id: str, body: NewCharacterInput):
    """Create a project character, label this box with it and queue a description of its features.

    The ``character_profile`` job appends a character sheet and an outfit sheet, read from this box,
    to the character's Details. The payload holds ids only: the worker reads the box when it starts.
    """
    title = body.title.strip()
    if not title:
        raise HTTPException(400, 'Enter a character name')
    request = {'title': title, 'content': body.content, 'is_main': body.is_main, 'episode_only': body.episode_only}
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            created = character_decisions.create_character(conn, detection_id, request)
        except character_decisions.Refused as exc:
            raise HTTPException(exc.args[1] if len(exc.args) > 1 else 409, str(exc.args[0])) from exc
        agent_log.note_override(conn, 'character_detections', detection_id)
        character_decisions.close_settled(conn, created['character'].get('episode_id') or row_episode(conn, detection_id=detection_id))
    return created


@app.put('/api/character-detections/{detection_id}/recognition-use')
def set_recognition_use(detection_id: str, body: RecognitionUseInput):
    """Override whether a confirmed box is used as a matching example (null restores the automatic choice)."""
    with db.connect() as conn:
        value = None if body.use is None else int(body.use)
        if not conn.execute('UPDATE character_detections SET recognition_use=?,gallery_status=NULL,gallery_note=NULL WHERE id=?', (value, detection_id)).rowcount:
            raise HTTPException(404, 'Detection not found')
    return {'ok': True, 'use': body.use}


@app.delete('/api/character-detections/{detection_id}')
def delete_character_detection(detection_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM character_image_references WHERE detection_id=?', (detection_id,)).fetchone():
            raise HTTPException(409, 'Remove the saved reference before deleting this detection')
        episode_id = row_episode(conn, detection_id=detection_id)
        removed = conn.execute('DELETE FROM character_detections WHERE id=? RETURNING run_id,position', (detection_id,)).fetchone()
        if not removed:
            raise HTTPException(404, 'Detection not found')
        structured_results.opening_box_edited(conn, removed['run_id'], removed['position'])
        agent_log.note_override(conn, 'character_detections', detection_id)
        character_decisions.close_settled(conn, episode_id)
    return {'ok': True}


@app.post('/api/character-detections/{detection_id}/reference')
def save_character_reference(detection_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        detection = conn.execute('SELECT character_context_id FROM character_detections WHERE id=?', (detection_id,)).fetchone()
        if not detection:
            raise HTTPException(404, 'Detection not found')
        if not detection['character_context_id']:
            raise HTTPException(400, 'Assign a character before saving a reference')
        if unregistered.card_ids_for(conn, [detection['character_context_id']]):
            raise HTTPException(400, 'An Unregistered card stands for different people: it has no references')
        ref = conn.execute('SELECT id FROM character_image_references WHERE detection_id=?', (detection_id,)).fetchone()
        if ref:
            return {'id': ref['id']}
        ref_id = str(uuid.uuid4())
        conn.execute('INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES (?,?,?)', (ref_id, detection['character_context_id'], detection_id))
        # Only the user's explicit Save as reference action promotes vectors
        # from an unverified detection into the reusable reference index.
        conn.execute('''INSERT INTO character_reference_embeddings(reference_id,model_id,model_revision,preprocessing,dimensions,embedding)
                        SELECT ?,model_id,model_revision,preprocessing,dimensions,embedding
                        FROM character_detection_embeddings WHERE detection_id=?''', (ref_id, detection_id))
        conn.execute("UPDATE character_detections SET label_source='manual' WHERE id=?", (detection_id,))
        character_decisions.close_settled(conn, row_episode(conn, detection_id=detection_id))
    return {'id': ref_id}


@app.delete('/api/character-detections/{detection_id}/reference')
def delete_character_reference(detection_id: str):
    with db.connect() as conn:
        if not conn.execute('DELETE FROM character_image_references WHERE detection_id=?', (detection_id,)).rowcount:
            raise HTTPException(404, 'Reference not found')
    return {'ok': True}


@app.get('/api/jobs/{job_id}/character-detection-status')
def character_detection_status(job_id: str):
    job = db.one("SELECT id,status,stage,error FROM jobs WHERE id=? AND pipeline='character_detect'", (job_id,))
    if not job:
        raise HTTPException(404, 'Character detection job not found')
    return job


@app.get('/api/jobs/{job_id}/character-profile-status')
def character_profile_status(job_id: str):
    job = db.one("SELECT id,status,stage,error,json_extract(payload_json,'$.character_context_id') AS character_context_id FROM jobs WHERE id=? AND pipeline='character_profile'", (job_id,))
    if not job:
        raise HTTPException(404, 'Character description job not found')
    return job


@app.get('/api/character-detection-runs/{run_id}/frame/{position}')
async def character_detection_frame(run_id: str, position: Literal['first', 'last'], request: Request):
    run = db.one('''SELECT r.*,e.id AS episode_id,e.project_id,e.original_ext FROM character_detection_runs r
                    JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id WHERE r.id=?''', (run_id,))
    if not run:
        raise HTTPException(404, 'Detection run not found')
    frames = json.loads(run['frames_json'])
    frame = next((frame for frame in frames if frame['position'] == position), None)
    source = media.episode_dir(run['project_id'], run['episode_id']) / f"original{run['original_ext']}"
    if not frame or not source.is_file():
        raise HTTPException(404, 'Frame unavailable')
    spec = first_frame_spec(source, run, frame)
    try:
        headers = frame_cache.http_headers(spec, request)
        if frame_cache.unchanged(request, headers):
            return Response(status_code=304, headers=headers)
        image = await frame_cache.get(spec)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(content=image, media_type='image/jpeg', headers=headers)


@app.get('/api/projects/{project_id}/character-references')
def list_character_references(project_id: str):
    require_project(project_id)
    rows = db.all_rows('''SELECT ref.id,ref.character_context_id,ci.title AS character_name,d.id AS detection_id,d.position,d.box_json,
                                 r.id AS run_id,r.frames_json FROM character_image_references ref
                           JOIN context_items ci ON ci.id=ref.character_context_id JOIN character_detections d ON d.id=ref.detection_id
                           JOIN character_detection_runs r ON r.id=d.run_id WHERE ci.project_id=? AND ci.kind='character' AND COALESCE(ci.unregistered,0)=0 ORDER BY ci.title,ref.created_at DESC''', (project_id,))
    for row in rows:
        frame = next((frame for frame in json.loads(row.pop('frames_json')) if frame['position'] == row['position']), None)
        row['image_size'] = frame['image_size'] if frame else None
        row['box'] = json.loads(row.pop('box_json'))
    return rows


class MatcherUpdate(BaseModel):
    ids: list[str] = Field(min_length=1)
    use: bool | None = None


@app.get('/api/projects/{project_id}/character-matcher')
def character_matcher(project_id: str):
    from . import anime_gallery
    require_project(project_id)
    rows = anime_gallery.exemplar_rows(project_id)
    episodes = {r['id']: r['title'] for r in db.all_rows('SELECT id,title FROM episodes WHERE project_id=?', (project_id,))}
    cuts = {r['id']: r['cut_index'] for r in db.all_rows('SELECT c.id,c.cut_index FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE e.project_id=?', (project_id,))}
    for row in rows:
        frame = next((f for f in json.loads(row.pop('frames_json')) if f['position'] == row['position']), None)
        box = json.loads(row.pop('box_json'))
        automatic = row['gallery_status'] in ('small', 'atypical') or (row['recognition_use'] is None and not row['is_reference'] and frame and anime_gallery.box_problem(box, frame['image_size']) == 'small')
        row['state'] = 'disabled' if row['recognition_use'] == 0 else 'automatic' if automatic and row['recognition_use'] != 1 else 'active'
        row['episode_title'] = episodes.get(row['episode_id'], '')
        row['cut_index'] = cuts.get(row['cut_id'])
    return rows


@app.put('/api/projects/{project_id}/character-matcher')
def update_character_matcher(project_id: str, body: MatcherUpdate):
    from . import anime_gallery
    require_project(project_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        allowed = {row['id'] for row in anime_gallery.exemplar_rows(project_id, conn)}
        if not set(body.ids) <= allowed:
            raise HTTPException(404, 'Matcher image not found in this project')
        value = None if body.use is None else int(body.use)
        conn.executemany('UPDATE character_detections SET recognition_use=?,gallery_status=NULL,gallery_note=NULL WHERE id=?', [(value, id) for id in set(body.ids)])
    return {'ok': True}


@app.get('/api/character-matcher/{reference_id}/image', response_class=Response)
@app.get('/api/character-references/{reference_id}/image', response_class=Response)
async def character_reference_image(reference_id: str, request: Request):
    """The reference's box alone, at most 240 px high: the references list shows these instead of whole 1280 px frames."""
    row = db.one('''SELECT d.position,d.box_json,r.frames_json,r.cut_start,r.cut_end,e.id AS episode_id,e.project_id,e.original_ext
                    FROM character_detections d
                    JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id
                    WHERE d.id=? OR d.id=(SELECT detection_id FROM character_image_references WHERE id=?)''', (reference_id, reference_id))
    if not row:
        raise HTTPException(404, 'Reference not found')
    frame = next((frame for frame in json.loads(row['frames_json']) if frame['position'] == row['position']), None)
    size = frame.get('image_size') if frame else None
    if not size or not size[0] or not size[1]:
        raise HTTPException(404, 'Frame unavailable')
    box = json.loads(row['box_json'])
    crop = tuple(round(min(1.0, max(0.0, value / scale)), 6) for value, scale in zip(box, (size[0], size[1], size[0], size[1])))
    source = media.episode_dir(row['project_id'], row['episode_id']) / f"original{row['original_ext']}"
    if not source.is_file():
        raise HTTPException(404, 'Frame unavailable')
    spec = {'source': source, 'time': frame['time'], 'start': row['cut_start'], 'end': row['cut_end'], 'width': 1280, 'last': False,
            'seek_tolerance': .000002, 'crop': crop, 'crop_height': 240}
    try:
        headers = frame_cache.http_headers(spec, request)
        if frame_cache.unchanged(request, headers):
            return Response(status_code=304, headers=headers)
        image = await frame_cache.get(spec)
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(content=image, media_type='image/jpeg', headers=headers)


@app.delete('/api/character-references/{reference_id}')
def delete_character_reference_by_id(reference_id: str):
    with db.connect() as conn:
        if not conn.execute('DELETE FROM character_image_references WHERE id=?', (reference_id,)).rowcount:
            raise HTTPException(404, 'Reference not found')
    return {'ok': True}


@app.post('/api/character-detection-runs/{run_id}/boxes')
def add_character_detection_box(run_id: str, body: CharacterDetectionBoxInput):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = conn.execute('SELECT * FROM character_detection_runs WHERE id=?', (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, 'Detection run not found')
        frame = next((f for f in json.loads(run['frames_json']) if f['position'] == body.position), None)
        if not frame:
            raise HTTPException(400, 'Frame not found')
        x1, y1, x2, y2 = body.box
        width, height = frame['image_size']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise HTTPException(400, 'Box must fit inside the frame')
        box_id = str(uuid.uuid4())
        conn.execute('INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,?,?)', (box_id, run_id, body.position, json.dumps(body.box)))
        structured_results.opening_box_edited(conn, run_id, body.position)
    return {'id': box_id}


class CharacterDetectionBoxUpdate(BaseModel):
    box: tuple[float, float, float, float]


@app.put('/api/character-detections/{detection_id}/box')
def resize_character_detection(detection_id: str, body: CharacterDetectionBoxUpdate):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('''SELECT d.position,d.run_id,r.frames_json FROM character_detections d
                              JOIN character_detection_runs r ON r.id=d.run_id WHERE d.id=?''', (detection_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Detection not found')
        if conn.execute('SELECT 1 FROM character_image_references WHERE detection_id=?', (detection_id,)).fetchone():
            raise HTTPException(409, 'Remove the saved reference before resizing this box')
        frame = next((f for f in json.loads(row['frames_json']) if f['position'] == row['position']), None)
        if not frame:
            raise HTTPException(400, 'Frame not found')
        x1, y1, x2, y2 = body.box
        width, height = frame['image_size']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise HTTPException(400, 'Box must fit inside the frame')
        conn.execute('UPDATE character_detections SET box_json=?,gallery_status=NULL,gallery_note=NULL WHERE id=?', (json.dumps(body.box), detection_id))
        conn.execute('DELETE FROM character_detection_embeddings WHERE detection_id=?', (detection_id,))
        structured_results.opening_box_edited(conn, row['run_id'], row['position'])
    return {'ok': True}


@app.get("/api/cuts/{cut_id}/analysis-config")
def get_cut_analysis_config(cut_id: str):
    _, config = analysis_config_for_cut(cut_id)
    return config


@app.get('/api/cuts/{cut_id}/jobs')
def get_cut_jobs(cut_id: str):
    """Every job that works on this cut, newest first, with the same fields as the episode's job list (which is bounded)."""
    cut = db.one('SELECT id,episode_id FROM cuts WHERE id=?', (cut_id,))
    if not cut:
        raise HTTPException(404, 'Cut not found')
    with db.connect() as conn:
        return episode_jobs.for_cut(conn, cut['episode_id'], cut_id)


@app.get('/api/cuts/{cut_id}/analysis-stages')
def get_cut_analysis_stages(cut_id: str):
    """Return per-attempt timing and state for analysis stages of a cut."""
    analysis_config_for_cut(cut_id)
    import time
    now = time.time()
    rows = db.all_rows('''SELECT s.*,j.status AS job_status,j.lease_until FROM analysis_job_stages s
                          JOIN jobs j ON j.id=s.job_id WHERE s.cut_id=? ORDER BY s.started_at,s.rowid''', (cut_id,))
    for row in rows:
        if row['status'] == 'running' and (row['job_status'] != 'running' or not row['lease_until'] or row['lease_until'] <= now):
            row['status'] = 'interrupted'
            end = row['last_seen_at']
        else:
            end = row['finished_at'] or now
        row['elapsed_seconds'] = max(0, end - row['started_at'])
        row.pop('lease_until')
    return rows


@app.get('/api/cuts/{cut_id}/timeline')
def get_cut_timeline(cut_id: str):
    """Return the chronological processing history of one cut with execution-only durations."""
    from . import cut_timeline
    result = cut_timeline.build(cut_id)
    if result is None:
        raise HTTPException(404, 'Cut not found')
    return result


@app.get('/api/jobs/{job_id}/cuts/{cut_id}/run-details')
def get_cut_run_details(job_id: str, cut_id: str):
    """Return saved run inputs and outputs for one cut, including reused checkpoints."""
    job = db.one('SELECT id,pipeline,status,error,payload_json FROM jobs WHERE id=?', (job_id,))
    if not job:
        raise HTTPException(404, 'Run not found')
    payload = json_object_or_empty(job['payload_json'])
    if job['pipeline'] == 'analyze':
        run_input = next((item for item in payload.get('cuts', []) if isinstance(item, dict) and item.get('cut_id') == cut_id), None)
    else:
        run_input = payload if payload.get('cut_id') == cut_id and job['pipeline'] in ('regen', 'consistency_summary') else None
    if run_input is None:
        raise HTTPException(404, 'Cut not found in this run')
    checkpoint = db.one('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job_id, cut_id))
    saved = json_object_or_empty(checkpoint['evidence_json']) if checkpoint else {}
    checkpoint_ids = [value for value in saved.values() if isinstance(value, str)]
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute('SELECT * FROM analysis_results WHERE job_id=? AND cut_id=? ORDER BY rowid', (job_id, cut_id))]
        seen = {row['id'] for row in rows}
        for result_id in checkpoint_ids:
            if result_id not in seen:
                row = conn.execute('SELECT * FROM analysis_results WHERE id=? AND cut_id=?', (result_id, cut_id)).fetchone()
                if row:
                    rows.append(dict(row))
                    seen.add(result_id)
        windows = [dict(row) for row in conn.execute('SELECT window_index,start_time,end_time,model_revision,content_json FROM analysis_video_windows WHERE job_id=? AND cut_id=? ORDER BY window_index', (job_id, cut_id))]
        rejected = [dict(row) for row in conn.execute('SELECT window_index,start_time,end_time,error,response FROM rejected_video_responses WHERE job_id=? AND cut_id=? ORDER BY id', (job_id, cut_id))]
    for row in windows:
        row['content'] = json_object_or_empty(row.pop('content_json'))
    return {'pipeline': job['pipeline'], 'status': job['status'], 'error': job['error'], 'input': run_input,
            'results': [serialize_result(row) for row in rows], 'video_windows': windows, 'rejected_responses': rejected}


@app.get('/api/jobs/{job_id}/analysis-windows')
def get_analysis_windows(job_id: str):
    """Return saved video-window analyses and source metadata for a run."""
    if not db.one('SELECT id FROM jobs WHERE id=?', (job_id,)):
        raise HTTPException(404, 'Job not found')
    rows = db.all_rows('SELECT cut_id,window_index,start_time,end_time,model_revision,content_json FROM analysis_video_windows WHERE job_id=? ORDER BY cut_id,window_index', (job_id,))
    for row in rows:
        row['content'] = json_object_or_empty(row.pop('content_json'))
    return rows


@app.put("/api/cuts/{cut_id}/analysis-config")
def put_cut_analysis_config(cut_id: str, config: AnalysisConfig):
    analysis_config_for_cut(cut_id)
    selected_analysis_context(cut_id, config)
    with db.connect() as conn:
        conn.execute("INSERT INTO cut_analysis_configs (cut_id,config_json) VALUES (?,?) ON CONFLICT(cut_id) DO UPDATE SET config_json=excluded.config_json,updated_at=CURRENT_TIMESTAMP", (cut_id, config.model_dump_json()))
    return config


# The last frame is no longer analyzed; results from earlier runs stay readable as history.
VISUAL_STAGES = ('initial_frame', 'keyframe_analysis')
RETIRED_STAGES = {'video_analysis': 'Video analysis was replaced by keyframe analysis; run Keyframes instead'}


def keyframe_wait(conn, episode_id, cut_ids, stages):
    """Find or queue the selection jobs a visual job depends on."""
    from . import keyframes
    if 'keyframe_analysis' not in set(stages or VISUAL_STAGES):
        return {}
    waits = keyframes.ensure_selection(conn, episode_id, cut_ids)
    return {'depends_on': waits} if waits else {}


def enqueue_analysis(episode_id: str, cut_id: str, stages: list[str] | None = None):
    retired = [RETIRED_STAGES[stage] for stage in stages or () if stage in RETIRED_STAGES]
    if retired:
        raise HTTPException(410, retired[0])
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        cut = db.one('SELECT id,active,analysis_enabled,start_time,end_time FROM cuts WHERE id=? AND episode_id=?',
                     (cut_id, episode_id), conn=conn)
        if not cut or not cut['active'] or not cut['analysis_enabled']:
            raise HTTPException(409, 'The cut is archived or disabled for analysis')
        job_id = insert_visual_job(conn, episode_id, [cut], stages=stages)
    return {'job_id': job_id}


def insert_visual_job(conn, episode_id, cuts, *, stages=None, batch=False, only_missing=False,
                      characters_resolved=False, from_index=0, dynamic_cuts=False, beats='always'):
    from . import visual_models

    cut_ids = [cut['id'] for cut in cuts]
    items = [{'cut_id': cut['id'], 'config': snapshot_analysis_config(conn, cut).model_dump()} for cut in cuts]
    payload = {'workflow': 'single_cut_v2', 'auto_consistency': False, 'batch': batch, 'cuts': items,
               'models': visual_models.snapshot(conn, VISUAL_STAGES)}
    if stages:
        payload['stages'] = stages
    if only_missing:
        payload['only_missing'] = True
    if characters_resolved:
        payload['characters_resolved'] = True
    if beats != 'always':
        # keyframe_analysis reads it when it saves: 'when_ready' queues beats only for a ready cut (cut_readiness), 'off' never
        payload['beats'] = beats
    if dynamic_cuts:
        payload.update(dynamic_cuts=True, from_index=from_index,
                       default_config=AnalysisConfig(**text_generation_settings(conn)).model_dump())
    dependencies = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts', 'character_detect'),
                                       None if dynamic_cuts else cut_ids)
    if not dynamic_cuts:
        dependencies.extend(keyframe_wait(conn, episode_id, cut_ids, stages).get('depends_on', []))
    if dependencies:
        payload['depends_on'] = list(dict.fromkeys(dependencies))
    job_id = worker.insert_job(conn, episode_id, 'analyze', payload)
    worker.sync_episode_status(conn, episode_id)
    worker.wake_event.set()
    return job_id


@app.post('/api/episodes/{episode_id}/visual-analysis')
def queue_episode_visual_analysis(episode_id: str, from_cut: int | None = None, to_cut: int | None = None,
                                  only_missing: bool = True, characters_resolved: bool = False,
                                  beats: Literal['always', 'when_ready', 'off'] = 'always'):
    """Queue one model-major run. Check cut readiness and unfinished stages at start.

    ``beats``: whether saving a cut's keyframes queues its first performance beats (``always``, the editor's default),
    only when the cut's inputs are settled (``when_ready``, the autopilot) or never (``off``)."""
    span = episode_requests.cut_range(from_cut, to_cut)
    first_index, last_index = span['first_index'], span.get('last_index')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        episode_requests.refuse_range_while_cutting(conn, episode_id, last_index is not None)
        cuts = db.all_rows("""SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1
            AND analysis_enabled=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?) ORDER BY cut_index""",
            (episode_id, first_index, last_index, last_index), conn=conn)
        dynamic_cuts = bool(worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts')))
        job_ids = [insert_visual_job(conn, episode_id, cuts, batch=True, only_missing=only_missing,
                                     characters_resolved=characters_resolved, from_index=first_index,
                                     dynamic_cuts=dynamic_cuts, beats=beats)] if cuts or dynamic_cuts else []
    return {'queued': len(cuts), 'skipped': 0, 'blocked': {'unknown_characters': [], 'not_detected': []}, 'job_ids': job_ids}


class BatchTextInput(BaseModel):
    text_provider: Literal['local', 'openrouter'] = 'openrouter'
    local_thinking: bool = False
    openrouter_model: str = Field(default='openrouter/free', min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_./:-]+$')


@app.post('/api/episodes/{episode_id}/text-processing')
def queue_episode_text_processing(episode_id: str, body: BatchTextInput):
    """Queue consistency followed by a dependent production brief for each prepared cut."""
    return queue_text_batch(episode_id, 'text_processing', {'first_index': 0, 'text_choices': body.model_dump()})


def _batch_text_processing(conn, episode_id, options, cuts, configuration):
    from . import structured_results, summary_queue
    job_ids, skipped = [], 0
    for cut in cuts:
        cut_id = cut['id']
        config = {**configuration(cut), **options['text_choices']}
        results = structured_results.list_results(cut_id, conn=conn)
        pending = worker.pending_jobs(conn, episode_id, ('analyze', 'performance_beats'), [cut_id])
        if not pending and summary_queue.missing_evidence({item['type']: item for item in results}):
            skipped += 1
            continue
        consistency = {'cut_id': cut_id, 'config': config, 'batch': True}
        first = summary_queue.insert(conn, episode_id, consistency)
        second = summary_jobs.insert(conn, episode_id, summary_jobs.choices(cut_id, config), parents=[first], batch=True)
        job_ids.extend((first, second))
    return {'cuts_queued': len(job_ids) // 2, 'skipped': skipped, 'job_ids': job_ids}


def validate_summary_request(value, full_analysis=False):
    if set(value) != ({"model", "system", "input", "summary_system"} if full_analysis else {"model", "system", "input"}):
        raise HTTPException(400, "The request has missing or unexpected fields")
    if not isinstance(value["model"], str) or not re.fullmatch(r"[a-zA-Z0-9_./:-]{1,200}", value["model"]):
        raise HTTPException(400, "Enter a valid OpenRouter model")
    if not isinstance(value["system"], str) or not value["system"].strip() or len(value["system"]) > 50000:
        raise HTTPException(400, "The system instruction must contain 1–50000 characters")
    if full_analysis and (not isinstance(value["summary_system"], str) or not value["summary_system"].strip() or len(value["summary_system"]) > 50000):
        raise HTTPException(400, "The summary instruction must contain 1–50000 characters")
    if not isinstance(value["input"], dict):
        raise HTTPException(400, "Request input must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise HTTPException(400, "The request contains invalid JSON values") from exc
    if len(encoded.encode('utf-8')) > 4_000_000:
        raise HTTPException(400, "The request is larger than 4 MB")
    return value


@app.get("/api/cuts/{cut_id}/summary-request-preview")
def summary_request_preview(cut_id: str, source: Literal["current", "failed", "failed_summary"] = "current"):
    cut, config = analysis_config_for_cut(cut_id)
    with db.connect() as conn:
        conn.execute('BEGIN')
        choices = summary_jobs.choices(cut_id, config.model_dump())
        payload = summary_jobs.gather(conn, {'payload_json': json.dumps(choices)}, preview=True)
        request_input = {"mode": "regenerate", "instruction": "", "current_summary": "", "context": payload["context"],
                         "current_results": payload["results"], "sampled_frames": payload["sampled_frames"], "human_instructions": payload["human_instructions"]}
        model = choices["model"]
        source_job = None
        previous_request = None
        if source == "failed":
            rows = conn.execute("""SELECT j.id,j.stage,j.error,j.finished_at,j.payload_json,a.evidence_json
                FROM jobs j JOIN analysis_job_cuts a ON a.job_id=j.id
                WHERE j.episode_id=? AND j.pipeline='analyze' AND j.status IN ('failed','cancelled') AND a.cut_id=? AND a.completed=0
                ORDER BY j.created_at DESC,j.rowid DESC""", (cut["episode_id"], cut_id)).fetchall()
            for row in rows:
                snapshot = json_object_or_empty(row["payload_json"])
                item = next((item for item in snapshot.get("cuts", []) if isinstance(item, dict) and item.get("cut_id") == cut_id), None)
                try:
                    evidence = json.loads(row["evidence_json"])
                except (TypeError, ValueError):
                    continue
                if item and isinstance(evidence, list) and evidence:
                    request_input["context"] = prompt_context.slim_context(item.get("context", {}))
                    request_input["sampled_frames"] = [{key: frame[key] for key in ("time", "description", "detections", "image_size", "ocr_response", "ocr") if key in frame} for frame in evidence if isinstance(frame, dict)]
                    model = item.get("config", {}).get("openrouter_model", model)
                    source_job = {key: row[key] for key in ("id", "stage", "error", "finished_at")}
                    source_job["kind"] = "analysis"
                    previous_request = item.get("request_override")
                    break
            if not source_job:
                raise HTTPException(404, "No failed analysis with saved frame evidence is available for this cut")
        elif source == "failed_summary":
            row = conn.execute("""SELECT id,stage,error,finished_at,payload_json,input_json FROM jobs
                WHERE episode_id=? AND pipeline='regen' AND status='failed'
                AND json_extract(payload_json,'$.cut_id')=? AND json_extract(payload_json,'$.result_type')='summary'
                ORDER BY created_at DESC,rowid DESC LIMIT 1""", (cut['episode_id'], cut_id)).fetchone()
            if not row:
                raise HTTPException(404, "No failed summary request is available for this cut")
            old = json_object_or_empty(row['payload_json'])
            source_job = {key: row[key] for key in ("id", "stage", "error", "finished_at")}
            source_job["kind"] = "summary"
            previous_request = old.get('request_override') or json_object_or_empty(row['input_json']).get('request')
            if not previous_request:
                available = [frame for frame in old.get('sampled_frames', []) if isinstance(frame, dict) and isinstance(frame.get('frame_id'), str) and frame_source.result_frame_spec(frame['frame_id'], conn)]
                previous_request = {"model": old.get('model', model), "system": PRODUCTION_BRIEF_SYSTEM if old.get('production_brief') else SUMMARY_TEXT_SYSTEM,
                                    "input": {"mode": old.get('mode'), "instruction": old.get('instruction'), "current_summary": old.get('current_prompt'), "context": old.get('context'), "current_results": old.get('results'), "sampled_frames": available, **({'human_instructions': old.get('human_instructions', '')} if old.get('production_brief') else {})}}
                source_job['reconstructed'] = True
    if source_job and source_job['kind'] == 'analysis':
        request = {"model": model, "system": SYSTEM_PROMPT, "summary_system": SUMMARY_SYSTEM,
                   "input": {"context": request_input["context"], "sampled_frames": request_input["sampled_frames"]}}
    else:
        request = {"model": model, "system": PRODUCTION_BRIEF_SYSTEM, "input": request_input}
    if source_job and previous_request:
        try:
            request = validate_summary_request(previous_request, full_analysis=source_job['kind'] == 'analysis')
            source_job['reconstructed'] = source_job.get('reconstructed', False)
        except HTTPException:
            source_job['reconstructed'] = True
    elif source_job:
        source_job['reconstructed'] = True
    return {"request": request, "base_id": payload["base"].get("id"), "source_job": source_job}


@app.post("/api/cuts/{cut_id}/retry-analysis-request")
def retry_analysis_request(cut_id: str, body: AnalysisRequestRetry):
    cut, current_config = analysis_config_for_cut(cut_id)
    edited = validate_summary_request(body.request_override, full_analysis=True)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT active,analysis_enabled FROM cuts WHERE id=?", (cut_id,)).fetchone()
        if not current or not current['active'] or not current['analysis_enabled']:
            raise HTTPException(409, "The cut is archived or disabled for analysis")
        old = conn.execute("SELECT * FROM jobs WHERE id=? AND episode_id=? AND pipeline='analyze' AND status IN ('failed','cancelled')", (body.source_job_id, cut['episode_id'])).fetchone()
        saved = conn.execute("""SELECT completed,source_cut_json,
            CASE WHEN json_valid(evidence_json) AND json_type(evidence_json)='array' THEN json_array_length(evidence_json) ELSE 0 END AS frames
            FROM analysis_job_cuts WHERE job_id=? AND cut_id=?""", (body.source_job_id, cut_id)).fetchone()
        if not old or not saved or saved['completed'] or not saved['frames']:
            raise HTTPException(409, "The failed analysis frame evidence is no longer available")
        snapshot = json_object_or_empty(old['payload_json'])
        item = next((item for item in snapshot.get('cuts', []) if isinstance(item, dict) and item.get('cut_id') == cut_id), None)
        if not item:
            raise HTTPException(409, "The saved analysis context is unavailable")
        original_cut = (item.get('context') or {}).get('cut') or {}
        bounds = json.loads(saved['source_cut_json']) if saved['source_cut_json'] else item.get('source_bounds')
        if not bounds and not {'start_time', 'end_time'} <= original_cut.keys():
            raise HTTPException(409, "The saved analysis context is unavailable")
        bounds = bounds or [original_cut['start_time'], original_cut['end_time']]
        options = {**item['config'], 'text_provider': current_config.text_provider, 'local_thinking': current_config.local_thinking}
        choices = {'cuts': [{'cut_id': cut_id, 'source_bounds': bounds, 'request_override': edited, 'config': options}],
                   'workflow': 'legacy_text_retry', 'text_only': True,
                   'depends_on': worker.pending_jobs(conn, cut['episode_id'], ('prepare', 'hardcuts'), [cut_id])}
        job_id = worker.insert_job(conn, cut['episode_id'], 'analyze', choices, retry_of=old['id'])
        conn.execute("""INSERT OR IGNORE INTO analysis_job_cuts (job_id,cut_id,evidence_json,source_cut_json)
            SELECT ?,cut_id,evidence_json,? FROM analysis_job_cuts WHERE job_id=? AND cut_id=?""",
            (job_id, json.dumps(bounds), old['id'], cut_id))
        worker.sync_episode_status(conn, cut['episode_id'])
    return {"job_id": job_id}


@app.post('/api/cuts/{cut_id}/performance-beats')
def queue_performance_beats(cut_id: str):
    """Queue the performance beats of one cut: a text model compiles its keyframe analysis into timed acting (performance_beats.py)."""
    from . import performance_beats
    cut, config = analysis_config_for_cut(cut_id)
    if not cut['active']:
        raise HTTPException(400, 'Only active cuts can have performance beats')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        job_id = performance_beats.insert(conn, cut['episode_id'], cut_id, {key: getattr(config, key) for key in ('text_provider', 'local_thinking', 'openrouter_model')})
        worker.sync_episode_status(conn, cut['episode_id'])
    return {'job_id': job_id}


@app.post('/api/cuts/{cut_id}/consistency-summary')
def queue_consistency_summary(cut_id: str, with_beats: bool = False):
    """Queue a summary of the evidence available when work starts.

    With ``with_beats`` the cut's performance beats are queued first (an already waiting or running job is reused) and the
    summary waits for them, so the review reads the beats written from the current keyframes. A cut without keyframe
    analysis, or one that no visual job will analyse, has no beats to write and only gets the summary.
    """
    from . import performance_beats, summary_queue
    cut, config = analysis_config_for_cut(cut_id)
    if not cut['active']:
        raise HTTPException(400, 'Only active cuts can be summarized')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        parents = []
        if with_beats:
            by_type = {item['type']: item for item in structured_results.list_results(cut_id, conn=conn)}
            visual = worker.pending_jobs(conn, cut['episode_id'], ('analyze',), [cut_id])
            beats = performance_beats.pending_job(conn, cut_id)
            if beats or visual or 'keyframe_analysis' in by_type:
                beats = beats or performance_beats.insert(conn, cut['episode_id'], cut_id,
                    {key: getattr(config, key) for key in ('text_provider', 'local_thinking', 'openrouter_model')})
                parents = [beats]
        job_id = summary_queue.insert(conn, cut['episode_id'], {'cut_id': cut_id, 'config': config.model_dump(), 'depends_on': parents})
        worker.sync_episode_status(conn, cut['episode_id'])
    return {'job_id': job_id, 'beats_job_id': parents[0] if parents else None}


@app.post('/api/episodes/{episode_id}/consistency-summaries')
def queue_episode_consistency_summaries(episode_id: str, from_cut: int | None = None, to_cut: int | None = None,
                                        only_missing: bool = False):
    """Queue consistency summaries for the displayed cut numbers from_cut..to_cut (inclusive)."""
    options = episode_requests.cut_range(from_cut, to_cut)
    if only_missing:
        options['only_missing'] = True
    return queue_text_batch(episode_id, 'consistency', options)


def _batch_consistency(conn, episode_id, options, cuts, configuration):
    from . import summary_queue
    if options.get('only_missing'):
        done = {row['cut_id'] for row in conn.execute("""SELECT a.cut_id FROM analysis_results a JOIN cuts c ON c.id=a.cut_id
            WHERE c.episode_id=? AND a.superseded_by IS NULL
            AND json_extract(a.settings_json,'$.structured_type')='consistency_summary'""", (episode_id,))}
        kept = [cut for cut in cuts if cut['id'] not in done]
    else:
        kept = cuts
    job_ids = [summary_queue.insert(conn, episode_id, {'cut_id': cut['id'], 'batch': True,
                'config': configuration(cut)}) for cut in kept]
    return {'queued': len(job_ids), 'skipped': len(cuts) - len(kept), 'job_ids': job_ids}


@app.post('/api/cuts/{cut_id}/review-answers')
def queue_review_answers(cut_id: str):
    """Queue a draft of production decisions from the current consistency summary."""
    from . import review_answers
    cut, config = analysis_config_for_cut(cut_id)
    if not cut['active']:
        raise HTTPException(400, 'Only active cuts can be reviewed')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        job_id = review_answers.insert(conn, cut['episode_id'], {'cut_id': cut_id, 'config': config.model_dump()})
        worker.sync_episode_status(conn, cut['episode_id'])
    return {'job_id': job_id}


@app.post('/api/episodes/{episode_id}/review-answers')
def queue_episode_review_answers(episode_id: str, from_cut: int | None = None, to_cut: int | None = None,
                                 auto_save: bool = True):
    """Queue production-decision drafts for the enabled cuts in the requested range.

    Without ``auto_save`` each draft is kept as a suggestion, like the single-cut button, and the saved answers stay as they are.
    """
    options = episode_requests.cut_range(from_cut, to_cut)
    if not auto_save:
        options['auto_save'] = False
    return queue_text_batch(episode_id, 'review_answers', options)


def _batch_review_answers(conn, episode_id, options, cuts, configuration):
    from . import review_answers
    blocked = {'no_consistency': [], 'busy': [], 'nothing_to_review': []}
    job_ids = []
    for cut in cuts:
        pending = worker.pending_jobs(conn, episode_id, ('consistency_summary',), [cut['id']])
        if not pending and (reason := review_answers.readiness(conn, cut['id'])):
            blocked[reason].append(cut['cut_index'] + 1)
            continue
        config = configuration(cut)
        payload = {'cut_id': cut['id'], 'config': config}
        if options.get('auto_save', True):
            payload['auto_save'] = True
        job_ids.append(review_answers.insert(conn, episode_id, payload))
    return {'queued': len(job_ids), 'blocked': blocked, 'job_ids': job_ids}


@app.post('/api/episodes/{episode_id}/text-chain')
def queue_episode_text_chain(episode_id: str, from_cut: int | None = None, to_cut: int | None = None, only_missing: bool = True):
    """Queue related text steps and reuse current requests with the same choices."""
    return queue_text_batch(episode_id, 'text_chain', {**episode_requests.cut_range(from_cut, to_cut), 'only_missing': only_missing})


def _batch_text_chain(conn, episode_id, options, cuts, configuration):
    from . import performance_beats, review_answers, summary_queue
    only_missing = options['only_missing']
    blocked = {'missing_results': [], 'busy': []}
    job_ids, queued, kept_summaries, attached = [], 0, 0, 0
    for cut in cuts:
        cut_id, number = cut['id'], cut['cut_index'] + 1
        config = configuration(cut)
        prior = []
        for row in conn.execute("""SELECT j.id,j.payload_json FROM jobs j WHERE episode_id=? AND pipeline='review_answers'
            AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?
            AND json_extract(payload_json,'$.auto_save')=1
            AND EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.depends_on') p JOIN jobs parent ON parent.id=p.value
                WHERE parent.pipeline='consistency_summary')
            ORDER BY rowid DESC""", (episode_id, cut_id)):
            if json.loads(row['payload_json']).get('config') == config:
                prior.append(row)
        if prior:
            answer_id = prior[0]['id']
            parents = json.loads(prior[0]['payload_json']).get('depends_on', [])
            job_ids.extend([*parents, answer_id])
        else:
            results = structured_results.list_results(cut_id, conn=conn)
            by_type = {item['type']: item for item in results}
            visual = worker.pending_jobs(conn, episode_id, ('analyze',), [cut_id])
            beats = performance_beats.pending_job(conn, cut_id)
            if not beats and 'keyframe_analysis' in by_type and not prompt_context.beats_current(by_type) and by_type.get('initial_frame', {}).get('content'):
                beats = performance_beats.insert(conn, episode_id, cut_id,
                                               {key: config[key] for key in ('text_provider', 'local_thinking', 'openrouter_model')})
            if not visual and summary_queue.missing_evidence(by_type, pending_beats=bool(beats)):
                blocked['missing_results'].append(number)
                continue
            parents = [beats] if beats else []
            consistency_id = summary_queue.insert(conn, episode_id, {'cut_id': cut_id, 'config': config,
                                                  'batch': True, 'depends_on': parents})
            answer_id = review_answers.insert(conn, episode_id, {'cut_id': cut_id, 'config': config,
                                             'auto_save': True, 'depends_on': [consistency_id]})
            job_ids.extend([*parents, consistency_id, answer_id])
            queued += 1
        if only_missing and summary_jobs.current(conn, cut_id):
            kept_summaries += 1
            continue
        existing = worker.pending_jobs(conn, episode_id, ('regen',), [cut_id])
        summary_id = summary_jobs.insert(conn, episode_id, summary_jobs.choices(cut_id, config), parents=[answer_id], batch=True)
        if prior and summary_id not in existing:
            attached += 1
        job_ids.append(summary_id)
    return {'queued': queued, 'attached': attached, 'skipped': kept_summaries, 'blocked': blocked,
            'job_ids': list(dict.fromkeys(job_ids))}


@app.post('/api/episodes/{episode_id}/cut-summaries')
def queue_episode_cut_summaries(episode_id: str, from_cut: int | None = None, to_cut: int | None = None, only_missing: bool = True):
    """Queue Cut summaries for enabled cuts with displayed numbers from_cut..to_cut (inclusive)."""
    return queue_text_batch(episode_id, 'cut_summaries', {**episode_requests.cut_range(from_cut, to_cut), 'only_missing': only_missing})


def _batch_cut_summaries(conn, episode_id, options, cuts, configuration):
    only_missing = options['only_missing']
    blocked = {'no_consistency': [], 'answers_pending': [], 'busy': []}
    job_ids, skipped = [], 0
    for cut in cuts:
        cut_id = cut['id']
        if only_missing and summary_jobs.current(conn, cut_id):
            skipped += 1
            continue
        parents = worker.pending_jobs(conn, episode_id, ('consistency_summary',), [cut_id])
        if not parents and not conn.execute("""SELECT 1 FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
            AND json_extract(settings_json,'$.structured_type')='consistency_summary'""", (cut_id,)).fetchone():
            blocked['no_consistency'].append(cut['cut_index'] + 1)
            continue
        choices = summary_jobs.choices(cut_id, configuration(cut))
        job_ids.append(summary_jobs.insert(conn, episode_id, choices, parents=parents, batch=True))
    return {'queued': len(job_ids), 'skipped': skipped, 'blocked': blocked, 'job_ids': job_ids}


TEXT_BATCHES = {'text_processing': _batch_text_processing, 'consistency': _batch_consistency,
                'review_answers': _batch_review_answers, 'text_chain': _batch_text_chain, 'cut_summaries': _batch_cut_summaries}


def snapshot_batch_config(conn, episode_id):
    settings = text_generation_settings(conn)
    default = AnalysisConfig(video_frame_budget=60).model_copy(update=settings).model_dump()
    saved = {row['id']: AnalysisConfig.model_validate_json(row['config_json']).model_copy(update=settings).model_dump()
             for row in conn.execute("""SELECT c.id,ac.config_json FROM cuts c JOIN cut_analysis_configs ac ON ac.cut_id=c.id
                 WHERE c.episode_id=? AND c.active=1""", (episode_id,))}
    return {'default': default, 'cuts': saved}


def run_text_batch(conn, episode_id, action, options, cuts, configs=None):
    def configuration(cut):
        if configs is None:
            return snapshot_analysis_config(conn, cut).model_dump()
        chosen = dict(configs['cuts'].get(cut['id'], configs['default']))
        if chosen.get('video_frame_budget') is not None:
            chosen['video_fps'] = automatic_video_fps(max(0.000001, cut['end_time'] - cut['start_time']))
        return chosen

    return TEXT_BATCHES[action](conn, episode_id, options, [cut for cut in cuts if cut['analysis_enabled']], configuration)


def queue_text_batch(episode_id, action, options):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        episode_requests.refuse_range_while_cutting(conn, episode_id, 'last_index' in options)
        dispatch = episode_requests.defer(conn, episode_id, action, options, config=lambda: snapshot_batch_config(conn, episode_id))
        if dispatch:
            result = {'queued': 0, 'cuts_queued': 0, 'skipped': 0, 'blocked': {}, 'job_ids': [dispatch], 'deferred': True}
        else:
            cuts = episode_requests.cuts(conn, episode_id, options['first_index'], last_index=options.get('last_index'))
            result = run_text_batch(conn, episode_id, action, options, cuts)
        if result['job_ids']:
            worker.sync_episode_status(conn, episode_id)
    worker.wake_event.set()
    return result


@app.get('/api/cuts/{cut_id}/review-answers')
def get_review_answers(cut_id: str):
    from . import review_answers
    analysis_config_for_cut(cut_id)
    return review_answers.get_draft(cut_id)


class BriefInstructionsInput(BaseModel):
    content: str = Field(max_length=50000)


@app.get("/api/cuts/{cut_id}/brief-instructions")
def get_brief_instructions(cut_id: str):
    analysis_config_for_cut(cut_id)
    row = db.one("SELECT content, updated_at FROM cut_brief_instructions WHERE cut_id=?", (cut_id,))
    return {"content": row["content"] if row else "", "updated_at": row["updated_at"] if row else None}


@app.put("/api/cuts/{cut_id}/brief-instructions")
def put_brief_instructions(cut_id: str, body: BriefInstructionsInput):
    from . import agent_log, review_answers, review_decisions
    analysis_config_for_cut(cut_id)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if review_answers.saved_answers(conn, cut_id) != body.content:
            agent_log.note_override(conn, 'cut_brief_instructions', cut_id)
        review_answers.save_hand_answers(conn, cut_id, body.content)
        cut = conn.execute('SELECT episode_id FROM cuts WHERE id=?', (cut_id,)).fetchone()
        if cut:
            review_decisions.close_settled(conn, cut['episode_id'], [cut_id])
        row = db.one("SELECT updated_at FROM cut_brief_instructions WHERE cut_id=?", (cut_id,), conn=conn)
    return {"content": body.content, "updated_at": row["updated_at"] if row else None}


class TextGenerationSettings(BaseModel):
    text_provider: Literal['local', 'openrouter'] = 'openrouter'
    local_thinking: bool = False
    openrouter_model: str = Field(default='openrouter/free', min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_./:-]+$')
    local_model: Literal['standard', 'mtp', 'inkling'] | None = None  # omitted keeps the current choice
    openrouter_audio_model: Annotated[str | None, Field(min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_./:-]+$')] = None  # omitted keeps the current choice; the audio a cut is re-transcribed with


def text_generation_settings(conn=None):
    row = db.one('SELECT text_provider,local_thinking,openrouter_model,local_model,openrouter_audio_model FROM text_generation_settings WHERE id=1', conn=conn)
    if not row:
        return {**TextGenerationSettings().model_dump(), 'local_model': 'mtp', 'openrouter_audio_model': DEFAULT_AUDIO_MODEL}
    row['local_thinking'] = bool(row['local_thinking'])
    return row


@app.get('/api/text-generation-settings')
def get_text_generation_settings():
    return text_generation_settings()


@app.put('/api/text-generation-settings')
def put_text_generation_settings(body: TextGenerationSettings):
    current = text_generation_settings()
    local_model = body.local_model or current['local_model']
    audio_model = body.openrouter_audio_model or current['openrouter_audio_model']
    if local_model != current['local_model'] and local_model != 'standard':
        # The worker reports which model files exist; selecting one that is missing would fail every local text job.
        info = local_model_variants().get(local_model) or {}
        if not info.get('available'):
            raise HTTPException(409, f"That text model is not available yet ({info.get('detail') or 'the inference worker has not reported it'})")
    with db.connect() as conn:
        conn.execute('UPDATE text_generation_settings SET text_provider=?,local_thinking=?,openrouter_model=?,local_model=?,openrouter_audio_model=? WHERE id=1',
                     (body.text_provider, int(body.local_thinking), body.openrouter_model, local_model, audio_model))
    return text_generation_settings()


def local_model_variants():
    row = db.one('SELECT variants_json FROM local_text_model_state WHERE id=1')
    try:
        value = json.loads(row['variants_json']) if row else {}
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


@app.get('/api/text-model-comparison')
def text_model_comparison():
    """Speed and size of local text results per model variant, read from each result's saved usage.

    ``decode_tokens_per_second`` comes from llama-server's own timings (only results written since
    they were recorded); ``overall_tokens_per_second`` is completion tokens over the whole
    generation time, so it also covers older results, prompt processing included.
    """
    groups = {}
    for row in db.all_rows("""SELECT kind, settings_json FROM analysis_results
                           WHERE settings_json LIKE '%"usage"%' AND settings_json LIKE '%"provider": "local"%'"""):
        try:
            settings = json.loads(row['settings_json'])
        except (ValueError, TypeError):
            continue
        usage = settings.get('usage') if isinstance(settings, dict) else None
        seconds = settings.get('summary_generation_seconds')
        if not isinstance(usage, dict) or not isinstance(usage.get('completion_tokens'), (int, float)):
            continue
        variant = usage.get('model_variant') or 'standard'
        group = groups.setdefault(variant, {'results': 0, 'completion_tokens': 0, 'reasoning_tokens': 0, 'generation_seconds': 0.0, 'timed_results': 0,
                                            'predicted_n': 0, 'predicted_ms': 0.0, 'draft_n': 0, 'draft_n_accepted': 0})
        group['results'] += 1
        group['completion_tokens'] += usage['completion_tokens']
        group['reasoning_tokens'] += usage.get('reasoning_tokens') or 0
        if isinstance(seconds, (int, float)) and seconds > 0:
            group['generation_seconds'] += seconds
            group['timed_results'] += 1
        for name in ('predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted'):
            if isinstance(usage.get(name), (int, float)):
                group[name] += usage[name]
    out = []
    for variant, g in groups.items():
        out.append({'variant': variant, 'results': g['results'],
                    'avg_completion_tokens': round(g['completion_tokens'] / g['results']),
                    'avg_generation_seconds': round(g['generation_seconds'] / g['timed_results'], 1) if g['timed_results'] else None,
                    'overall_tokens_per_second': round(g['completion_tokens'] / g['generation_seconds'], 1) if g['generation_seconds'] else None,
                    'decode_tokens_per_second': round(g['predicted_n'] / g['predicted_ms'] * 1000, 1) if g['predicted_ms'] else None,
                    'draft_acceptance': round(g['draft_n_accepted'] / g['draft_n'], 3) if g['draft_n'] else None})
    return {'variants': sorted(out, key=lambda item: item['variant'])}


@app.get('/api/local-text-model')
def local_text_model_status():
    import time
    row = db.one('SELECT * FROM local_text_model_state WHERE id=1')
    if not row:
        raise HTTPException(503, 'Model status is unavailable')
    row['desired_loaded'] = bool(row['desired_loaded'])
    row['variants'] = local_model_variants()
    row.pop('variants_json', None)
    if time.time() - row['heartbeat'] > 15:
        row['state'] = 'worker_offline'
        row['detail'] = 'Inference worker is not responding'
    return row


class LocalModelLoadInput(BaseModel):
    loaded: bool


@app.put('/api/local-text-model')
def set_local_text_model(body: LocalModelLoadInput):
    with db.connect() as conn:
        conn.execute('UPDATE local_text_model_state SET desired_loaded=? WHERE id=1', (int(body.loaded),))
    return local_text_model_status()


class RemoteTextRoute(BaseModel):
    mode: Literal['local', 'remote', 'any'] = 'local'
    model: str = Field(default='', max_length=300)


class RemoteTextServerInput(BaseModel):
    enabled: bool = False
    base_url: str = Field(default='', max_length=500)
    token: str | None = Field(default=None, max_length=500, description='Bearer token (llama-server --api-key). Omit to keep the saved token; an empty string clears it.')
    parallel: int = Field(default=1, ge=1, le=8)
    routes: dict[str, RemoteTextRoute] = Field(default_factory=dict)
    assistant_fallback: str = Field(default='', max_length=300)
    prefer: Literal['local', 'remote'] = 'local'
    chat_reserve: int = Field(default=1, ge=0, le=8, description='Slots of each server kept free for chat; text jobs never use them.')


def remote_text_input(body: RemoteTextServerInput):
    """Validated values to save or probe; the saved token fills in an omitted one."""
    from . import remote_text
    base_url = body.base_url.strip().rstrip('/')
    if base_url and not remote_text.URL_PATTERN.match(base_url):
        raise HTTPException(400, 'The address must look like http://192.168.0.20:8080')
    if body.enabled and not base_url:
        raise HTTPException(400, 'Enter the address of the remote llama.cpp server')
    unknown = sorted(set(body.routes) - set(worker.TEXT_PIPELINES))
    if unknown:
        raise HTTPException(400, f"Unknown job types: {', '.join(unknown)}")
    routes = {}
    for pipeline, route in body.routes.items():
        model = route.model.strip()
        if route.mode != 'local' and not model:
            raise HTTPException(400, f'Choose a remote model for {pipeline}')
        routes[pipeline] = {'mode': route.mode, 'model': model if route.mode != 'local' else ''}
    token = body.token.strip() if body.token is not None else None
    if token is None:
        token = (db.one('SELECT token FROM remote_text_server WHERE id=1') or {}).get('token') or ''
    return {'enabled': body.enabled, 'base_url': base_url, 'token': token,
            'parallel': body.parallel, 'routes': routes, 'assistant_fallback': body.assistant_fallback.strip(),
            'prefer': body.prefer, 'chat_reserve': body.chat_reserve}


@app.get('/api/remote-text-server')
def get_remote_text_server():
    """The remote llama.cpp settings (never the token) and the state the inference worker last saw."""
    import time
    from . import remote_text
    cfg = remote_text.config()
    state = remote_text.state()
    if time.time() - state['heartbeat'] > 30:
        state.update(state='worker_offline', detail='Inference worker is not responding')
    token = cfg.pop('token')
    from . import local_text_runtime
    return {**cfg, 'has_token': bool(token), 'state': state, 'effective_routes': remote_text.effective_routes({**cfg, 'enabled': cfg['enabled']}),
            'local_slots': local_text_runtime.text_job_concurrency()}


@app.put('/api/remote-text-server')
def put_remote_text_server(body: RemoteTextServerInput):
    import time
    values = remote_text_input(body)
    with db.connect() as conn:
        conn.execute('UPDATE remote_text_server SET enabled=?,base_url=?,token=?,parallel=?,routes_json=?,assistant_fallback=?,prefer=?,chat_reserve=?,updated_at=? WHERE id=1',
                     (int(values['enabled']), values['base_url'], values['token'], values['parallel'],
                      json.dumps(values['routes']), values['assistant_fallback'], values['prefer'], values['chat_reserve'], time.time()))
    return get_remote_text_server()


@app.post('/api/remote-text-server/test')
def test_remote_text_server(body: RemoteTextServerInput):
    """Ask the (unsaved) address for its models from the API process, so the user can check it before saving."""
    from . import remote_text
    values = remote_text_input(body)
    if not values['base_url']:
        raise HTTPException(400, 'Enter the address of the remote llama.cpp server')
    return remote_text.probe(values, timeout=5)


@app.post("/api/cuts/{cut_id}/regenerate-summary")
def regenerate_cut_summary(cut_id: str, body: PromptRegenInput):
    cut, config = analysis_config_for_cut(cut_id)
    if not cut['active'] or not cut['analysis_enabled']:
        raise HTTPException(400, 'Only active cuts with analysis enabled can be analyzed')
    if body.mode == 'edit' and not body.instruction.strip():
        raise HTTPException(400, 'Describe the change you want to make')
    edited = validate_summary_request(body.request_override) if body.request_override is not None else None
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = summary_jobs.current(conn, cut_id)
        if body.preview_checked and (current or {}).get('id') != body.preview_base_id:
            raise HTTPException(409, 'The saved summary changed since the request was prepared; refresh the preview')
        if body.mode == 'edit' and not current and not (body.current_prompt or '').strip():
            pending = worker.pending_jobs(conn, cut['episode_id'], ('regen',), [cut_id])
            if not pending:
                raise HTTPException(400, 'There is no cut summary to edit; regenerate one first')
        if body.source_job_id and not conn.execute("""SELECT 1 FROM jobs WHERE id=? AND episode_id=? AND pipeline='regen'
            AND status IN ('failed','cancelled') AND json_extract(payload_json,'$.cut_id')=?""",
            (body.source_job_id, cut['episode_id'], cut_id)).fetchone():
            raise HTTPException(409, 'The failed summary run is no longer available')
        choices = summary_jobs.choices(cut_id, config.model_dump(), body.mode, body.instruction, body.current_prompt, edited)
        parents = worker.pending_jobs(conn, cut['episode_id'], ('regen',), [cut_id]) if body.mode == 'edit' else []
        job_id = summary_jobs.insert(conn, cut['episode_id'], choices, parents=parents, retry_of=body.source_job_id)
        worker.sync_episode_status(conn, cut['episode_id'])
    return {'job_id': job_id}


def delete_current_results(conn, ids: list[str]):
    """Hard-delete current revisions; a deleted revision's parent becomes current again."""
    for result_id in ids:
        row = conn.execute('''SELECT a.parent_id,a.cut_id,a.title,a.settings_json,c.episode_id
            FROM analysis_results a JOIN cuts c ON c.id=a.cut_id WHERE a.id=?''', (result_id,)).fetchone()
        if not row:
            continue
        if row["parent_id"]:
            conn.execute("UPDATE analysis_results SET superseded_by=NULL WHERE id=? AND superseded_by=?", (row["parent_id"], result_id))
        conn.execute("DELETE FROM analysis_results WHERE id=?", (result_id,))
        if result_type := structured_results.structured_type(row['settings_json']):
            structured_results.invalidate_text(conn, row['cut_id'], result_type)
        if row['title'] == 'Cut summary':
            structured_results.invalidate_sequence_reviews(conn, [row['cut_id']], 'Cut summary changed')
        episode = conn.execute("SELECT project_id FROM episodes WHERE id=?", (row["episode_id"],)).fetchone()
        (media.episode_dir(episode["project_id"], row["episode_id"]) / "analysis" / f"{result_id}.jpg").unlink(missing_ok=True)


@app.delete("/api/analysis-results/{result_id}")
def delete_analysis_result(result_id: str):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT superseded_by FROM analysis_results WHERE id=?", (result_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Analysis result not found")
        if row["superseded_by"]:
            raise HTTPException(409, "Only the current revision can be deleted")
        delete_current_results(conn, [result_id])
    return {"ok": True}


def remove_regenerated_legacy(path, episode_id, fingerprint):
    # Do not remove a file if a job could still be using it or it changed while
    # the replacement was generated. SQLite serializes this with job creation.
    try:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if storage.active_frame_reference(conn, episode_id, path.name):
                return
            stat = path.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) == fingerprint:
                path.unlink(missing_ok=True)
    except OSError:
        pass


async def rendered_frame(spec, legacy=None, episode_id=None, request=None):
    """A frame rendered from its source. A URL with ``v`` names every input of its picture (the caller's promise), so
    the browser keeps it for good; any other is revalidated by ETag and answered 304 without rendering."""
    fingerprint = None
    if legacy and legacy.is_file():
        stat = legacy.stat()
        fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    try:
        headers = frame_cache.http_headers(spec, request, versioned=request is not None and 'v' in request.query_params)
        # A legacy JPEG that differs from the render is served instead, so only skip the work without one.
        if not fingerprint and frame_cache.unchanged(request, headers):
            return Response(status_code=304, headers=headers)
        data = await frame_cache.get(spec)
        # Preserve legacy evidence if timestamp reconstruction is not byte-for-byte
        # identical; overlays must continue to match the image that produced them.
        if fingerprint and legacy.read_bytes() != data:
            return FileResponse(legacy, media_type='image/jpeg')
        task = BackgroundTask(remove_regenerated_legacy, legacy, episode_id, fingerprint) if fingerprint and episode_id else None
        return Response(data, media_type='image/jpeg', headers=headers, background=task)
    except (OSError, RuntimeError, ValueError) as exc:
        if fingerprint:
            return FileResponse(legacy, media_type='image/jpeg')
        raise HTTPException(503, f'Frame could not be recreated: {exc}') from exc


@app.get("/api/analysis-results/{result_id}/frame", response_class=Response,
         summary="Download a sampled analysis frame",
         responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}, "description": "Sampled JPEG for the result or an ancestor"},
                    404: {"description": "No frame can be reconstructed for this result or its ancestors"}})
async def get_analysis_result_frame(result_id: str, request: Request):
    """Retrieve a stored legacy frame or recreate it from its recorded source and time."""
    spec = frame_source.result_frame_spec(result_id)
    if not spec:
        raise HTTPException(404, "No source frame is available for this result")
    if 'source' not in spec:
        return FileResponse(spec['legacy'], media_type='image/jpeg')
    return await rendered_frame(spec, spec['legacy'], spec['episode_id'], request)


@app.post("/api/cuts/{cut_id}/analyze")
def analyze_cut(cut_id: str):
    cut, _ = analysis_config_for_cut(cut_id)
    if not cut["active"]:
        raise HTTPException(400, "Only active cuts can be analyzed")
    return enqueue_analysis(cut["episode_id"], cut_id)


class VisualStageInput(BaseModel):
    stage: Literal['initial_frame', 'keyframe_analysis', 'video_analysis']  # video_analysis is answered with 410


@app.post('/api/cuts/{cut_id}/analysis-stages/run')
def analyze_cut_stage(cut_id: str, body: VisualStageInput):
    cut, _ = analysis_config_for_cut(cut_id)
    if not cut['active']:
        raise HTTPException(400, 'Only active cuts can be analyzed')
    return enqueue_analysis(cut['episode_id'], cut_id, [body.stage])


@app.post("/api/episodes/{episode_id}/analyze")
def analyze_episode(episode_id: str):
    # Keep the older route as a visual-only alias.  New jobs must not trigger
    # text generation implicitly; the worker still handles old snapshots.
    result = queue_episode_visual_analysis(episode_id)
    return {**result, 'job_id': result['job_ids'][0] if result['job_ids'] else None}


@app.post("/api/jobs/cancel-queued")
def cancel_all_queued_jobs():
    """Cancel queued processing jobs across episodes; exports and running jobs continue."""
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        jobs = conn.execute("SELECT * FROM jobs WHERE status='queued' AND pipeline!='brief_export'").fetchall()
        changed = worker.cancel_jobs(conn, jobs)
    worker.after_cancel(changed)
    return {"cancelled": len(changed)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_inference_job(job_id: str):
    """Cancel a queued job, or a running one whose kind can stop at its next checkpoint (``job_kinds``)."""
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job, payload = runs.job_payload(conn, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job['status'] not in ('queued', 'running'):
            raise HTTPException(409, "This job has already finished")
        if not job_kinds.kind_of(job['pipeline']).can_cancel(job['status'], payload):
            raise HTTPException(409, "This job cannot be cancelled while running" if job['status'] == 'running' else "This job cannot be cancelled")
        changed = worker.cancel_jobs(conn, [job])
    worker.after_cancel(changed)
    return {"ok": True}


def retry_visual_job(conn, job, payload):
    existing = conn.execute("SELECT id FROM jobs WHERE retry_of=? AND status IN ('queued','running')", (job['id'],)).fetchone()
    if existing:
        return existing['id']
    cut_ids = [item['cut_id'] for item in payload['cuts']]
    dependencies = worker.pending_jobs(conn, job['episode_id'], ('prepare', 'hardcuts', 'character_detect'),
                                       None if payload.get('dynamic_cuts') else cut_ids)
    if not payload.get('dynamic_cuts'):
        dependencies.extend(keyframe_wait(conn, job['episode_id'], cut_ids, payload.get('stages')).get('depends_on', []))
    payload['depends_on'] = list(dict.fromkeys(dependencies))
    retry_id = worker.insert_job(conn, job['episode_id'], 'analyze', payload, retry_of=job['id'],
                                 priority_class=job['priority_class'])
    conn.execute("""INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed)
        SELECT ?,cut_id,evidence_json,completed FROM analysis_job_cuts WHERE job_id=?""", (retry_id, job['id']))
    now = time.time()
    labels = job_kinds.STAGE_LABELS
    for row in conn.execute('SELECT cut_id,evidence_json FROM analysis_job_cuts WHERE job_id=?', (retry_id,)):
        for stage in json_object_or_empty(row['evidence_json']).keys() & labels.keys():
            conn.execute("""INSERT INTO analysis_job_stages(id,job_id,cut_id,stage_key,label,status,started_at,last_seen_at,finished_at)
                VALUES (?,?,?,?,?,'reused',?,?,?)""", (str(uuid.uuid4()), retry_id, row['cut_id'], stage, labels[stage], now, now, now))
    worker.sync_episode_status(conn, job['episode_id'])
    worker.wake_event.set()
    return retry_id


@app.post("/api/jobs/{job_id}/retry")
def retry_analysis(job_id: str):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        retry_id = retry_job(conn, job_id)
        agent_log.settle(conn, conn.execute('SELECT episode_id FROM jobs WHERE id=?', (job_id,)).fetchone()[0])  # its failed-run question
        return {'job_id': retry_id}


def retry_job(conn, job_id):
    """Queue a failed or cancelled job again inside the caller's write transaction; returns the new job id.

    Used by the retry route and the autopilot's triage (``triage.py``). Raises HTTPException when the job cannot be retried.
    """
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    if job['pipeline'] == 'composition_compact' and job['status'] in ('failed', 'cancelled'):
        from . import composition_compaction
        payload = json_object_or_empty(job['payload_json'])
        choices = {key: payload[key] for key in ('text_provider', 'model', 'local_thinking') if key in payload}
        retry_id = composition_compaction.insert(conn, job['episode_id'], payload['cut_id'], choices, retry_of=job_id)
        return retry_id
    if job['pipeline'] == 'keyframes' and job['status'] in ('failed', 'cancelled'):
        from . import job_kinds
        payload = json_object_or_empty(job['payload_json'])
        choices = {'keyframe_settings': payload['keyframe_settings']}
        if payload.get('all_cuts'):
            choices['all_cuts'] = True
        else:
            choices['cut_ids'] = [ref['id'] for ref in job_kinds.KINDS['keyframes'].cuts(payload)]
        retry_id = worker.enqueue_in(conn, job['episode_id'], 'keyframes', choices, retry_of=job_id)
        worker.wake_event.set()
        return retry_id
    if job['pipeline'] == 'performance_beats' and job['status'] in ('failed', 'cancelled'):
        from . import performance_beats
        payload = json_object_or_empty(job['payload_json'])
        retry_id = performance_beats.insert(conn, job['episode_id'], payload['cut_id'], payload.get('config'), retry_of=job_id)
        worker.sync_episode_status(conn, job['episode_id'])
        return retry_id
    if job['pipeline'] == 'character_profile' and job['status'] in ('failed', 'cancelled'):
        payload = json_object_or_empty(job['payload_json'])
        if not conn.execute("SELECT 1 FROM context_items WHERE id=? AND kind='character'", (payload.get('character_context_id'),)).fetchone():
            raise HTTPException(409, 'The character was deleted')
        for key in ('depends_on', 'result', 'cut'):
            payload.pop(key, None)
        retry_id = worker.insert_job(conn, job['episode_id'], 'character_profile', payload, retry_of=job_id)
        worker.sync_episode_status(conn, job['episode_id'])
        return retry_id
    if job["pipeline"] != "analyze" or job["status"] not in ("failed", "cancelled"):
        raise HTTPException(409, "Only failed or cancelled analysis jobs can be retried")
    retry_payload = json_object_or_empty(job['payload_json'])
    if retry_payload.get('workflow') == 'single_cut_v2':
        return retry_visual_job(conn, dict(job), retry_payload)
    cuts = retry_payload.get('cuts') or []
    for item in cuts:
        saved = (item.pop('context', None) or {}).get('cut') or {}
        if {'start_time', 'end_time'} <= saved.keys():
            item.setdefault('source_bounds', [saved['start_time'], saved['end_time']])
    retry_payload['depends_on'] = worker.pending_jobs(conn, job['episode_id'], ('prepare', 'hardcuts', 'character_detect'),
                                                     [item['cut_id'] for item in cuts])
    retry_id = worker.insert_job(conn, job['episode_id'], 'analyze', retry_payload, retry_of=job_id)
    conn.execute("""INSERT OR IGNORE INTO analysis_job_cuts (job_id,cut_id,evidence_json,completed,source_cut_json)
        SELECT ?,cut_id,evidence_json,completed,source_cut_json FROM analysis_job_cuts WHERE job_id=?""", (retry_id, job_id))
    conn.execute("""INSERT OR IGNORE INTO analysis_video_windows (job_id,cut_id,window_index,start_time,end_time,model_revision,content_json)
        SELECT ?,cut_id,window_index,start_time,end_time,model_revision,content_json FROM analysis_video_windows WHERE job_id=?""", (retry_id, job_id))
    worker.sync_episode_status(conn, job['episode_id'])
    return retry_id


@app.get("/api/episodes/{episode_id}/waveform-range")
def get_waveform_range(episode_id: str, start: float, end: float, bins: int = 500):
    """Peaks of one stretch of the audio (at most 300 s), fine enough to place a line's edges."""
    episode = require_episode(episode_id)
    try:
        return waveform_range.peaks(media.episode_dir(episode["project_id"], episode_id) / "audio.wav", start, end, bins)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/episodes/{episode_id}/waveform")
def get_waveform(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id) / "waveform.json"
    if not path.exists():
        return {"peaks": [], "duration": episode["duration"] or 0}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("peaks"), list):
            return data
    except (OSError, ValueError, UnicodeError):
        pass
    return {"peaks": [], "duration": episode["duration"] or 0}


@app.get("/api/episodes/{episode_id}/frames")
def get_frames(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id) / "frames" / "manifest.json"
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("frames"), list) and isinstance(data.get('interval'), (float, int)) and data['interval'] > 0:
            return data
    except (OSError, ValueError, UnicodeError):
        pass
    job = db.one("SELECT payload_json FROM jobs WHERE episode_id=? AND pipeline='prepare' AND status='done' ORDER BY created_at DESC,rowid DESC LIMIT 1", (episode_id,))
    settings = json_object_or_empty(job['payload_json']).get('settings', {}) if job else {}
    interval = settings.get('frame_interval') if isinstance(settings, dict) else None
    if not isinstance(interval, (float, int)) or not math.isfinite(interval) or interval <= 0:
        interval = episode['settings']['frame_interval']
    return media.frame_manifest(interval, episode['duration'] or 0)


@app.get("/api/episodes/{episode_id}/media/{name}")
async def get_media(episode_id: str, name: str, request: Request):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id)
    if name == "proxy":
        target, mime = path / "proxy.mp4", "video/mp4"
    elif name == "original":
        target = path / f"original{episode['original_ext']}"
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    elif name == "audio":
        target, mime = path / "audio.wav", "audio/wav"
    elif re.fullmatch(r"frame_\d{6}\.jpg", name):
        target, mime = path / "frames" / name, "image/jpeg"
        manifest = get_frames(episode_id)
        frame = next((item for item in manifest['frames'] if item.get('name') == name), None)
        proxy = path / 'proxy.mp4'
        if frame and proxy.is_file() and episode['duration']:
            # The historic fps filter selects the last decoded frame before
            # the midpoint of each sampling interval.
            index = int(name[6:12]) - 1
            boundary = min(episode['duration'], (index + .5) * manifest['interval'])
            if boundary > 0:
                return await rendered_frame({'source': proxy, 'time': boundary, 'start': max(0, boundary - 5), 'end': boundary, 'width': 480, 'last': True, 'quality': 4, 'seek_tolerance': .000002, 'probe_padding': 1}, target, episode['id'], request)
    elif re.fullmatch(r"cut_\d{6}\.jpg", name):
        target, mime = path / "thumbnails" / name, "image/jpeg"
        index = int(name[4:10])
        cut = db.one('SELECT id FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (episode_id, index))
        if cut:
            return await get_cut_frame(cut['id'], 'first', request)
    else:
        raise HTTPException(404, "Media not found")
    if not target.is_file():
        raise HTTPException(404, "Media not ready")
    return FileResponse(target, media_type=mime)


@app.get("/api/cuts/{cut_id}/frame/{position}", response_class=Response,
         responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}, "description": "Cut boundary frame"},
                    404: {"description": "Cut or source video not found"}})
async def get_cut_frame(cut_id: str, position: Literal["first", "last"], request: Request):
    """Download a cut boundary, recreating it from video when necessary."""
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    episode = require_episode(cut["episode_id"])
    path = media.episode_dir(episode["project_id"], episode["id"])
    target = path / "thumbnails" / f"{cut_id}_{position}.jpg"
    source = path / 'proxy.mp4'
    if not source.is_file():
        source = path / f"original{episode['original_ext']}"
    if not source.is_file():
        if target.is_file():
            return FileResponse(target, media_type='image/jpeg')
        raise HTTPException(404, "The source video is unavailable")
    saved = cut['first_frame_time'] if position == 'first' else cut['last_frame_time']
    first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
    timestamp = saved if saved is not None else first if position == 'first' else last
    return await rendered_frame({'source': source, 'time': timestamp, 'start': cut['start_time'], 'end': cut['end_time'], 'width': 480, 'last': position == 'last' and saved is None, 'seek_tolerance': .000002, 'probe_padding': 1 if position == 'last' and saved is None else 0}, target, episode['id'], request)


app.include_router(external_router)


static_dir = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parent.parent / "static"))
if static_dir.is_dir():
    @app.get("/project/{project_id}", include_in_schema=False)
    def project_page(project_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/episode/{episode_id}", include_in_schema=False)
    def episode_page(episode_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    # /episode/<id>/cut/<cut id>[/analysis[/<place>]]: the page moves the path into the hash route (frontend/src/cutRoute.ts).
    @app.get("/episode/{episode_id}/cut/{rest:path}", include_in_schema=False)
    def cut_page(episode_id: str, rest: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
