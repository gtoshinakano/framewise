import json
import asyncio
import io
import math
import mimetypes
import os
import re
import shutil
import uuid
import sqlite3
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field

from . import db, media, worker, storage, frame_source, frame_cache
from .external_api import AnalysisContext, AnalysisResult, ContextItem, Transcript, VoiceSample, router as external_router
from .inference_worker import SUMMARY_TITLE, SUMMARY_TEXT_SYSTEM, SYSTEM_PROMPT, SUMMARY_SYSTEM

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_GB", "40")) * 1024 ** 3
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_SETTINGS = {"proxy_width": 960, "proxy_crf": 28, "proxy_preset": "veryfast", "frame_interval": 5.0, "scene_threshold": 0.32, "chunk_duration": 60}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    worker.start_worker()
    frame_cache.clear_abandoned_temporary_files()
    async def clear_cache_periodically():
        while True:
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
app.include_router(storage.router)
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


class AnalysisConfig(BaseModel):
    frame_mode: Literal["endpoints", "uniform"] = "uniform"
    frame_count: int = Field(default=4, ge=2, le=32)
    context_ids: list[str] = Field(default_factory=list, max_length=200)
    detection_queries: list[Annotated[str, Field(min_length=1, max_length=100, pattern=r".*\S.*")]] = Field(default_factory=lambda: ["person", "clothing", "animal", "vehicle", "chair", "table", "subtitles"], min_length=1, max_length=20)
    additional_context: str = Field(default="", max_length=20000)
    openrouter_model: str = Field(default="openrouter/free", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_./:-]+$")


class CutInput(BaseModel):
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=20000)
    notes: str = Field(default="", max_length=20000)
    results: dict = Field(default_factory=dict)


class CutAnalysisEnabledInput(BaseModel):
    analysis_enabled: bool


class CutTimingInput(BaseModel):
    start_time: float
    end_time: float


class ChunkInput(BaseModel):
    title: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=20000)


ResultKind = Literal["first_frame", "camera_movement", "composition", "last_frame", "cut_description", "audio_visual_match", "generation_prompt", "other"]


class ResultCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ResultKind
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=50000)


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
    content: str = Field(min_length=1, max_length=50000)


class TranscriptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=20000)
    speaker_label: str = Field(default="", max_length=200)
    character_context_id: str | None = None


class VoiceRange(BaseModel):
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)


class VoiceSampleInput(BaseModel):
    character_context_id: str
    ranges: list[VoiceRange] | None = Field(default=None, min_length=1, max_length=50)
    # Preserve requests from clients using the original single-range API.
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class VoicePreviewInput(BaseModel):
    ranges: list[VoiceRange] = Field(min_length=1, max_length=50)


class ContextInput(BaseModel):
    kind: Literal["chapter_synopsis", "chapter_happenings", "character", "world", "scenario", "object", "general"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=50000)
    episode_id: str | None = None


def json_object_or_empty(value):
    """Keep damaged optional metadata from making the episode editor inaccessible."""
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


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
    episode["cuts"] = [serialize_cut(cut) for cut in db.all_rows("SELECT c.*, EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL) AS has_analysis_results FROM cuts c WHERE c.episode_id=? AND c.active=1 ORDER BY c.cut_index", (episode_id,))]
    episode["archived_cuts"] = [serialize_cut(cut) for cut in db.all_rows("SELECT * FROM cuts WHERE episode_id=? AND active=0 ORDER BY start_time,cut_index", (episode_id,))]
    episode["chunks"] = db.all_rows("SELECT * FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,))
    # The queued payload can contain every cut's context and may be very large.
    # Keep it in SQLite for the worker and return only job metadata to the UI.
    episode["jobs"] = db.all_rows("SELECT id,episode_id,pipeline,status,stage,error,attempt_count,created_at,updated_at,started_at,finished_at,retry_of,payload_json FROM jobs WHERE episode_id=? ORDER BY created_at DESC, rowid DESC", (episode_id,))
    completed = {(row["job_id"], row["cut_id"]) for row in db.all_rows(
        "SELECT a.job_id,a.cut_id FROM analysis_job_cuts a JOIN jobs j ON j.id=a.job_id WHERE j.episode_id=? AND a.completed=1", (episode_id,))}
    for job in episode["jobs"]:
        payload = json_object_or_empty(job.pop("payload_json"))
        if job["pipeline"] == "analyze":
            items = payload.get("cuts", [])
        elif job["pipeline"] == "regen":
            items = [{"cut_id": payload.get("cut_id"), "context": payload.get("context", {})}]
        else:
            items = []
        job["cuts"] = []
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            context = item.get("context")
            cut = context.get("cut") if isinstance(context, dict) else {}
            if not isinstance(cut, dict):
                cut = {}
            cut_id = item.get("cut_id")
            if isinstance(cut_id, str) and cut_id:
                index = cut.get("cut_index")
                title = cut.get("title")
                def time_value(value):
                    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
                job["cuts"].append({"id": cut_id, "cut_index": index if isinstance(index, int) and not isinstance(index, bool) else None,
                                    "title": title if isinstance(title, str) else "", "start_time": time_value(cut.get("start_time")),
                                    "end_time": time_value(cut.get("end_time")),
                                    "completed": (job["id"], cut_id) in completed if job["pipeline"] == "analyze" else job["status"] == "done"})
    return episode


def serialize_cut(cut: dict):
    cut["results"] = json_object_or_empty(cut.pop("results_json"))
    cut["active"] = bool(cut["active"])
    cut["analysis_enabled"] = bool(cut["analysis_enabled"])
    if "has_analysis_results" in cut:
        cut["has_analysis_results"] = bool(cut["has_analysis_results"])
    fallback_first, fallback_last = media.cut_frame_times(cut["start_time"], cut["end_time"])
    cut["first_frame_time"] = cut.get("first_frame_time") if cut.get("first_frame_time") is not None else fallback_first
    cut["last_frame_time"] = cut.get("last_frame_time") if cut.get("last_frame_time") is not None else fallback_last
    return cut


def serialize_result(result: dict):
    result["settings"] = json_object_or_empty(result.pop("settings_json"))
    result["context"] = json_object_or_empty(result.pop("context_json"))
    return result


def serialize_transcript(segment: dict):
    segment["deleted"] = bool(segment["deleted"])
    return segment


def validate_transcript(episode: dict, body: TranscriptInput):
    if body.end_time <= body.start_time:
        raise HTTPException(400, "Transcript end must be after its start")
    if episode["duration"] is not None and body.end_time > episode["duration"] + 0.2:
        raise HTTPException(400, "Transcript segment ends after the video")
    if body.character_context_id:
        item = db.one("SELECT project_id,kind FROM context_items WHERE id=?", (body.character_context_id,))
        if not item or item["project_id"] != episode["project_id"] or item["kind"] != "character":
            raise HTTPException(400, "Choose a character from this project")


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
        conn.execute("UPDATE projects SET title=?, synopsis=? WHERE id=?", (body.title.strip(), body.synopsis, project_id))
    return require_project(project_id)


@app.get("/api/projects/{project_id}/context", responses={200: {"model": list[ContextItem]}})
def list_context(project_id: str):
    """List project-wide and all episode-specific context items for this project."""
    require_project(project_id)
    return db.all_rows("SELECT * FROM context_items WHERE project_id=? ORDER BY kind,title", (project_id,))


@app.post("/api/projects/{project_id}/context")
def create_context(project_id: str, body: ContextInput):
    require_project(project_id)
    if body.episode_id and require_episode(body.episode_id)["project_id"] != project_id:
        raise HTTPException(400, "Episode belongs to another project")
    context_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO context_items (id,project_id,episode_id,kind,title,content) VALUES (?,?,?,?,?,?)",
                     (context_id, project_id, body.episode_id, body.kind, body.title.strip(), body.content))
    return db.one("SELECT * FROM context_items WHERE id=?", (context_id,))


@app.put("/api/context/{context_id}")
def update_context(context_id: str, body: ContextInput):
    item = db.one("SELECT * FROM context_items WHERE id=?", (context_id,))
    if not item:
        raise HTTPException(404, "Context item not found")
    if body.episode_id and require_episode(body.episode_id)["project_id"] != item["project_id"]:
        raise HTTPException(400, "Episode belongs to another project")
    with db.connect() as conn:
        conn.execute("UPDATE context_items SET episode_id=?,kind=?,title=?,content=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (body.episode_id, body.kind, body.title.strip(), body.content, context_id))
    return db.one("SELECT * FROM context_items WHERE id=?", (context_id,))


@app.delete("/api/context/{context_id}")
def delete_context(context_id: str):
    with db.connect() as conn:
        item = conn.execute("SELECT project_id FROM context_items WHERE id=?", (context_id,)).fetchone()
        if not item:
            raise HTTPException(404, "Context item not found")
        samples = conn.execute("SELECT id FROM character_voice_samples WHERE character_context_id=?", (context_id,)).fetchall()
        conn.execute("DELETE FROM context_items WHERE id=?", (context_id,))
    for sample in samples:
        media.voice_reference_path(item['project_id'], sample['id']).unlink(missing_ok=True)
    return {"ok": True}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    require_project(project_id)
    with db.connect() as conn:
        active = conn.execute("SELECT 1 FROM jobs j JOIN episodes e ON e.id=j.episode_id WHERE e.project_id=? AND j.status IN ('queued','running')", (project_id,)).fetchone()
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
    latest = db.one("SELECT id,status FROM jobs WHERE episode_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1", (episode_id,))
    return {**episode, "job_id": latest["id"] if latest else None, "job_status": latest["status"] if latest else None}


@app.patch("/api/episodes/{episode_id}")
def update_episode(episode_id: str, body: EpisodeInput):
    require_episode(episode_id)
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET title=? WHERE id=?", (body.title.strip(), episode_id))
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
        active = conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (episode_id,)).fetchone()
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
        conn.execute("UPDATE cuts SET title=?, description=?, notes=?, results_json=? WHERE id=?",
                     (body.title, body.description, body.notes, json.dumps(body.results), cut_id))
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


def edit_cut_timeline(cut_id: str, action: str, timing: CutTimingInput | None = None):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cut = conn.execute("SELECT * FROM cuts WHERE id=? AND active=1", (cut_id,)).fetchone()
        if not cut:
            raise HTTPException(404, "Active cut not found")
        episode = conn.execute("SELECT * FROM episodes WHERE id=?", (cut["episode_id"],)).fetchone()
        if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts','cutframes') AND status IN ('queued','running')", (cut["episode_id"],)).fetchone():
            raise HTTPException(409, "Wait for the current media job to finish")
        cuts = conn.execute("SELECT * FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index", (cut["episode_id"],)).fetchall()
        index = next(i for i, item in enumerate(cuts) if item["id"] == cut_id)
        affected_ids = set()
        if action == "timing":
            assert timing is not None
            start, end = timing.start_time, timing.end_time
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
            archive_index = conn.execute("SELECT MIN(cut_index) FROM cuts WHERE episode_id=?", (cut["episode_id"],)).fetchone()[0]
            conn.execute("UPDATE cuts SET active=0,cut_index=? WHERE id=?", (min(-1, archive_index - 1), removed["id"]))
            for item in cuts:
                if item["cut_index"] > removed["cut_index"]:
                    conn.execute("UPDATE cuts SET cut_index=? WHERE id=?", (item["cut_index"] - 1, item["id"]))
        if affected_ids:
            for affected_id in affected_ids:
                conn.execute("UPDATE cuts SET first_frame_time=NULL,last_frame_time=NULL WHERE id=?", (affected_id,))
            job_id = str(uuid.uuid4())
            conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, cut["episode_id"], "cutframes", json.dumps({"settings": json.loads(episode["settings_json"]), "cut_ids": sorted(affected_ids)})))
            worker.sync_episode_status(conn, cut["episode_id"])
    worker.wake_event.set()
    return {"selected_id": cut_id if action != "delete" else kept["id"]}


@app.put("/api/cuts/{cut_id}/timing")
def update_cut_timing(cut_id: str, body: CutTimingInput):
    return edit_cut_timeline(cut_id, "timing", body)


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
        conn.execute("UPDATE chunks SET title=?, notes=? WHERE id=?", (body.title, body.notes, chunk_id))
    return db.one("SELECT * FROM chunks WHERE id=?", (chunk_id,))


@app.get("/api/cuts/{cut_id}/analysis-results", responses={200: {"model": list[AnalysisResult]}})
def list_analysis_results(cut_id: str, history: bool = False, include_context: bool = True):
    """Return current result revisions, or all revisions when history is true.

    Set include_context=false to omit captured input evidence from each result.
    """
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    where = "cut_id=?" if history else "cut_id=? AND superseded_by IS NULL"
    columns = "*" if include_context else "id,cut_id,kind,title,content,origin,model_id,job_id,prompt,settings_json,'{}' AS context_json,parent_id,superseded_by,created_at"
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
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (result_id,)))


@app.put("/api/analysis-results/{result_id}", responses={200: {"model": AnalysisResult}, 409: {"description": "This result already has a newer revision."}})
def revise_analysis_result(result_id: str, body: ResultUpdate):
    """Append a human revision."""
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        old = conn.execute("SELECT * FROM analysis_results WHERE id=?", (result_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Analysis result not found")
        if old["superseded_by"]:
            raise HTTPException(409, "This result already has a newer revision")
        conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content,origin,parent_id) VALUES (?,?,?,?,?,?,?)",
                     (new_id, old["cut_id"], old["kind"], body.title, body.content, "manual_edit", result_id))
        conn.execute("UPDATE analysis_results SET superseded_by=? WHERE id=?", (new_id, result_id))
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (new_id,)))


@app.get("/api/episodes/{episode_id}/transcript", responses={200: {"model": list[Transcript]}})
def list_transcript(episode_id: str, history: bool = False):
    """Return current non-deleted transcript segments; history includes every revision."""
    require_episode(episode_id)
    where = "episode_id=?" if history else "episode_id=? AND superseded_by IS NULL AND deleted=0"
    order = "rowid DESC" if history else "start_time,id"
    return [serialize_transcript(segment) for segment in db.all_rows(f"SELECT * FROM transcript_segments WHERE {where} ORDER BY {order}", (episode_id,))]


@app.post("/api/episodes/{episode_id}/transcribe")
def transcribe_episode(episode_id: str):
    episode = require_episode(episode_id)
    if not episode["has_audio"] or not episode["duration"]:
        raise HTTPException(400, "Prepare a video with audio first")
    try:
        return {"job_id": worker.enqueue(episode_id, "transcribe")}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/projects/{project_id}/voice-samples", responses={200: {"model": list[VoiceSample]}})
def list_voice_samples(project_id: str):
    """List character voice references in the project and whether their audio is available."""
    require_project(project_id)
    samples = db.all_rows("""SELECT v.id,v.character_context_id,v.episode_id,v.start_time,v.end_time,v.ranges_json,v.source_filename,v.model_id,v.created_at,
        e.title AS episode_title FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id
        LEFT JOIN episodes e ON e.id=v.episode_id WHERE c.project_id=? ORDER BY v.created_at DESC""", (project_id,))
    for sample in samples:
        sample['ranges'] = json.loads(sample.pop('ranges_json')) if sample['ranges_json'] else ([] if sample['source_filename'] else [{'start_time': sample['start_time'], 'end_time': sample['end_time']}])
        sample['preview_available'] = media.voice_reference_path(project_id, sample['id']).is_file() or bool(sample['episode_id'] and sample['ranges'] and (media.episode_dir(project_id, sample['episode_id']) / 'audio.wav').is_file())
    return samples


def validate_voice_character(episode, character_id):
    character = db.one("SELECT id FROM context_items WHERE id=? AND project_id=? AND kind='character'", (character_id, episode["project_id"]))
    if not character:
        raise HTTPException(400, "Choose a character from this project")


def validate_voice_ranges(episode, ranges, minimum=2):
    ordered = sorted(ranges, key=lambda r: r['start_time'])
    if (any(r['end_time'] <= r['start_time'] or r['end_time'] > episode['duration'] for r in ranges)
            or any(a['end_time'] > b['start_time'] for a, b in zip(ordered, ordered[1:]))
            or not minimum <= sum(r['end_time'] - r['start_time'] for r in ranges) <= 60):
        raise HTTPException(400, "Choose non-overlapping ranges within the video totaling at most 60 seconds" if minimum == 0 else "Choose non-overlapping ranges within the video totaling 2 to 60 seconds")


def preview_voice_ranges(audio, ranges):
    if not audio.is_file():
        raise HTTPException(404, 'Episode audio is not ready')
    output = io.BytesIO()
    try:
        with wave.open(str(audio), 'rb') as source, wave.open(output, 'wb') as target:
            target.setparams(source.getparams())
            for item in ranges:
                start = round(item['start_time'] * source.getframerate())
                end = min(source.getnframes(), round(item['end_time'] * source.getframerate()))
                if start >= source.getnframes() or end <= start:
                    raise HTTPException(400, 'Excerpt is outside the available audio')
                source.setpos(start)
                target.writeframes(source.readframes(end - start))
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
    episode = require_episode(episode_id)
    if not episode["has_audio"] or not episode["duration"]:
        raise HTTPException(400, "Prepare a video with audio first")
    if body.ranges is not None:
        ranges = [r.model_dump() for r in body.ranges]
    elif body.start_time is not None and body.end_time is not None:
        ranges = [{'start_time': body.start_time, 'end_time': body.end_time}]
    else:
        raise HTTPException(400, "Choose at least one audio range")
    validate_voice_ranges(episode, ranges)
    validate_voice_character(episode, body.character_context_id)
    try:
        return {"job_id": worker.enqueue(episode_id, "voice_sample", {'character_context_id': body.character_context_id, 'ranges': ranges})}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


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
        with target.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 25 * 1024 * 1024:
                    raise HTTPException(413, 'Voice audio must be 25 MB or smaller')
                output.write(chunk)
        if not total:
            raise HTTPException(400, 'Voice audio file is empty')
        try:
            job_id = worker.enqueue(episode_id, 'voice_sample', {'character_context_id': character_context_id,
                'upload_id': upload_id, 'source_filename': Path(file.filename or 'Audio upload').name[:200]})
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
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


@app.post("/api/episodes/{episode_id}/transcript")
def create_transcript_segment(episode_id: str, body: TranscriptInput):
    episode = require_episode(episode_id)
    validate_transcript(episode, body)
    segment_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id) VALUES (?,?,?,?,?,?,?)",
                     (segment_id, episode_id, body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id))
    return serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,)))


@app.put("/api/transcript-segments/{segment_id}")
def update_transcript_segment(segment_id: str, body: TranscriptInput):
    segment = db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,))
    if not segment:
        raise HTTPException(404, "Transcript segment not found")
    episode = require_episode(segment["episode_id"])
    validate_transcript(episode, body)
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        current = conn.execute("SELECT superseded_by,deleted FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if current["superseded_by"] or current["deleted"]:
            raise HTTPException(409, "This transcript segment is deleted or has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id, segment["episode_id"], body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id, "manual_edit", segment["model_id"], segment["job_id"], segment_id))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
    return serialize_transcript(db.one("SELECT * FROM transcript_segments WHERE id=?", (new_id,)))


@app.delete("/api/transcript-segments/{segment_id}")
def delete_transcript_segment(segment_id: str):
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        old = conn.execute("SELECT * FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Transcript segment not found")
        if old["superseded_by"] or old["deleted"]:
            raise HTTPException(409, "This transcript segment is deleted or has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,deleted,origin,model_id,job_id,parent_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id, old["episode_id"], old["start_time"], old["end_time"], old["text"], old["speaker_label"], old["character_context_id"], 1, "manual_edit", old["model_id"], old["job_id"], segment_id))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
    return {"ok": True}


@app.get("/api/cuts/{cut_id}/analysis-context", responses={200: {"model": AnalysisContext}})
def get_analysis_context(cut_id: str):
    """Return cut context with current neighboring results and overlapping transcript segments."""
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    episode = require_episode(cut["episode_id"])
    project = require_project(episode["project_id"])
    previous = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] - 1)) if cut["active"] else None
    following = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] + 1)) if cut["active"] else None
    neighbors = {}
    for label, neighbor in (("previous", previous), ("next", following)):
        neighbors[label] = db.all_rows("SELECT kind,title,content,origin,model_id FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid DESC", (neighbor["id"],)) if neighbor else []
    return {
        "project": {"id": project["id"], "title": project["title"], "synopsis": project["synopsis"]},
        "episode": {"id": episode["id"], "title": episode["title"], "duration": episode["duration"]},
        "cut": serialize_cut(cut),
        "previous_cut": serialize_cut(previous) if previous else None,
        "next_cut": serialize_cut(following) if following else None,
        "neighbor_results": neighbors,
        "chunks": db.all_rows("SELECT * FROM chunks WHERE episode_id=? AND start_time<? AND end_time>? ORDER BY chunk_index", (episode["id"], cut["end_time"], cut["start_time"])),
        "context_items": db.all_rows("SELECT * FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title", (project["id"], episode["id"])),
        "transcript_segments": [serialize_transcript(segment) for segment in db.all_rows("SELECT t.*,c.title AS character_name FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? ORDER BY t.start_time,t.id", (episode["id"], cut["end_time"], cut["start_time"]))],
    }


def analysis_config_for_cut(cut_id: str):
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    saved = db.one("SELECT config_json FROM cut_analysis_configs WHERE cut_id=?", (cut_id,))
    return cut, AnalysisConfig.model_validate_json(saved["config_json"]) if saved else AnalysisConfig()


def selected_analysis_context(cut_id: str, config: AnalysisConfig):
    context = get_analysis_context(cut_id)
    valid = {item["id"] for item in context["context_items"]}
    if not set(config.context_ids).issubset(valid):
        raise HTTPException(400, "One or more selected context items no longer apply to this cut")
    context["context_items"] = [item for item in context["context_items"] if item["id"] in config.context_ids]
    context["additional_context"] = config.additional_context
    return context


@app.get("/api/cuts/{cut_id}/analysis-config")
def get_cut_analysis_config(cut_id: str):
    _, config = analysis_config_for_cut(cut_id)
    return config


@app.put("/api/cuts/{cut_id}/analysis-config")
def put_cut_analysis_config(cut_id: str, config: AnalysisConfig):
    analysis_config_for_cut(cut_id)
    selected_analysis_context(cut_id, config)
    with db.connect() as conn:
        conn.execute("INSERT INTO cut_analysis_configs (cut_id,config_json) VALUES (?,?) ON CONFLICT(cut_id) DO UPDATE SET config_json=excluded.config_json,updated_at=CURRENT_TIMESTAMP", (cut_id, config.model_dump_json()))
    return config


def enqueue_analysis(episode_id: str, cut_id: str | None = None):
    job_id = str(uuid.uuid4())
    try:
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("SELECT 1 FROM episodes WHERE id=?", (episode_id,)).fetchone():
                raise HTTPException(404, "Episode not found")
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts') AND status IN ('queued','running')", (episode_id,)).fetchone():
                raise HTTPException(409, "Wait for media preparation or cut detection to finish")
            if cut_id is None:
                cuts = conn.execute("SELECT id FROM cuts WHERE episode_id=? AND active=1 AND analysis_enabled=1 ORDER BY cut_index", (episode_id,)).fetchall()
                if not cuts:
                    raise HTTPException(400, "Detect and enable cuts before running analysis")
            else:
                cut = conn.execute("SELECT id FROM cuts WHERE id=? AND episode_id=? AND active=1 AND analysis_enabled=1", (cut_id, episode_id)).fetchone()
                if not cut:
                    raise HTTPException(409, "The cut is archived or disabled for analysis")
                cuts = [cut]
            # The write lock keeps cut boundaries, presets and context stable until
            # the complete job payload is committed. The helpers use read-only
            # connections so the UI never sees a partial queue snapshot.
            inputs = []
            for cut in cuts:
                _, config = analysis_config_for_cut(cut["id"])
                context = selected_analysis_context(cut["id"], config)
                context["current_results"] = [dict(row) for row in conn.execute("SELECT kind,title,content,origin FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid", (cut["id"],))]
                inputs.append({"cut_id": cut["id"], "config": config.model_dump(), "context": context})
            conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, "analyze", json.dumps({"cuts": inputs})))
            worker.sync_episode_status(conn, episode_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An analysis job is already running for this episode") from exc
    return {"job_id": job_id}


def summary_snapshot(conn, cut, config, mode, instruction, current_prompt=None):
    cut_id = cut["id"]
    rows = conn.execute("SELECT id,kind,title,content,origin,context_json FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid", (cut_id,)).fetchall()
    summary = next((row for row in reversed(rows) if row["title"] == SUMMARY_TITLE), None)
    if mode == "edit" and not summary and not (current_prompt or "").strip():
        raise HTTPException(400, "There is no cut summary to edit; regenerate one instead")
    current = current_prompt if current_prompt is not None else (summary["content"] if summary else "")
    results = [{"title": row["title"], "kind": row["kind"], "content": row["content"], "origin": row["origin"]} for row in rows if row["title"] != SUMMARY_TITLE]
    episode = require_episode(cut["episode_id"])
    sampled_frames = []
    for row in rows:
        match = re.fullmatch(r"(?:Frame|Object detections) at (\d+(?:\.\d+)?)s", row["title"] or "")
        if match and frame_source.result_frame_spec(row['id'], conn):
            timestamp = float(match.group(1))
            try:
                snapshot = json.loads(row["context_json"] or "{}")
                if isinstance(snapshot.get('frame_time'), (int, float)):
                    timestamp = float(snapshot['frame_time'])
                observation = next((item for item in snapshot.get("sampled_frames", []) if item.get("frame_id") == row["id"]), None)
                if observation:
                    timestamp = float(observation["time"])
            except (ValueError, TypeError, AttributeError):
                pass
            if cut["start_time"] <= timestamp < cut["end_time"]:
                sampled_frames.append({"frame_id": row["id"], "time": timestamp, "description": row["content"]})
    payload = {"result_type": "summary", "cut_id": cut_id, "mode": mode, "instruction": instruction.strip(), "current_prompt": current if mode == "edit" else "",
               "model": config.openrouter_model, "base_id": summary["id"] if summary else None, "results": results, "sampled_frames": sampled_frames, "context": selected_analysis_context(cut_id, config)}
    return payload


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
        payload = summary_snapshot(conn, cut, config, "regenerate", "")
        request_input = {"mode": "regenerate", "instruction": "", "current_summary": "", "context": payload["context"],
                         "current_results": payload["results"], "sampled_frames": payload["sampled_frames"]}
        model = payload["model"]
        source_job = None
        previous_request = None
        if source == "failed":
            rows = conn.execute("""SELECT j.id,j.stage,j.error,j.finished_at,j.payload_json,a.evidence_json
                FROM jobs j JOIN analysis_job_cuts a ON a.job_id=j.id
                WHERE j.episode_id=? AND j.pipeline='analyze' AND j.status='failed' AND a.cut_id=? AND a.completed=0
                ORDER BY j.created_at DESC,j.rowid DESC""", (cut["episode_id"], cut_id)).fetchall()
            for row in rows:
                snapshot = json_object_or_empty(row["payload_json"])
                item = next((item for item in snapshot.get("cuts", []) if isinstance(item, dict) and item.get("cut_id") == cut_id), None)
                try:
                    evidence = json.loads(row["evidence_json"])
                except (TypeError, ValueError):
                    continue
                if item and isinstance(evidence, list) and evidence:
                    request_input["context"] = item.get("context", {})
                    request_input["sampled_frames"] = [{key: frame[key] for key in ("time", "description", "detections", "image_size", "ocr_response", "ocr") if key in frame} for frame in evidence if isinstance(frame, dict)]
                    model = item.get("config", {}).get("openrouter_model", model)
                    source_job = {key: row[key] for key in ("id", "stage", "error", "finished_at")}
                    source_job["kind"] = "analysis"
                    previous_request = item.get("request_override")
                    break
            if not source_job:
                raise HTTPException(404, "No failed analysis with saved frame evidence is available for this cut")
        elif source == "failed_summary":
            row = conn.execute("""SELECT id,stage,error,finished_at,payload_json FROM jobs
                WHERE episode_id=? AND pipeline='regen' AND status='failed' AND stage!='Cancelled'
                AND json_extract(payload_json,'$.cut_id')=? AND json_extract(payload_json,'$.result_type')='summary'
                ORDER BY created_at DESC,rowid DESC LIMIT 1""", (cut['episode_id'], cut_id)).fetchone()
            if not row:
                raise HTTPException(404, "No failed summary request is available for this cut")
            old = json_object_or_empty(row['payload_json'])
            source_job = {key: row[key] for key in ("id", "stage", "error", "finished_at")}
            source_job["kind"] = "summary"
            previous_request = old.get('request_override')
            if not previous_request:
                available = [frame for frame in old.get('sampled_frames', []) if isinstance(frame, dict) and isinstance(frame.get('frame_id'), str) and frame_source.result_frame_spec(frame['frame_id'], conn)]
                previous_request = {"model": old.get('model', model), "system": SUMMARY_TEXT_SYSTEM,
                                    "input": {"mode": old.get('mode'), "instruction": old.get('instruction'), "current_summary": old.get('current_prompt'), "context": old.get('context'), "current_results": old.get('results'), "sampled_frames": available}}
                source_job['reconstructed'] = True
    if source_job and source_job['kind'] == 'analysis':
        request = {"model": model, "system": SYSTEM_PROMPT, "summary_system": SUMMARY_SYSTEM,
                   "input": {"context": request_input["context"], "sampled_frames": request_input["sampled_frames"]}}
    else:
        request = {"model": model, "system": SUMMARY_TEXT_SYSTEM, "input": request_input}
    if source_job and previous_request:
        try:
            request = validate_summary_request(previous_request, full_analysis=source_job['kind'] == 'analysis')
            source_job['reconstructed'] = source_job.get('reconstructed', False)
        except HTTPException:
            source_job['reconstructed'] = True
    elif source_job:
        source_job['reconstructed'] = True
    return {"request": request, "base_id": payload["base_id"], "source_job": source_job}


@app.post("/api/cuts/{cut_id}/retry-analysis-request")
def retry_analysis_request(cut_id: str, body: AnalysisRequestRetry):
    cut, _ = analysis_config_for_cut(cut_id)
    edited = validate_summary_request(body.request_override, full_analysis=True)
    job_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not cut['active'] or not cut['analysis_enabled']:
            raise HTTPException(409, "The cut is archived or disabled for analysis")
        if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts') AND status IN ('queued','running')", (cut['episode_id'],)).fetchone():
            raise HTTPException(409, "Wait for media preparation or cut detection to finish")
        old = conn.execute("SELECT * FROM jobs WHERE id=? AND episode_id=? AND pipeline='analyze' AND status='failed'", (body.source_job_id, cut['episode_id'])).fetchone()
        saved = conn.execute("SELECT evidence_json,completed FROM analysis_job_cuts WHERE job_id=? AND cut_id=?", (body.source_job_id, cut_id)).fetchone()
        if not old or not saved or saved['completed']:
            raise HTTPException(409, "The failed analysis evidence is no longer available")
        snapshot = json_object_or_empty(old['payload_json'])
        item = next((item for item in snapshot.get('cuts', []) if isinstance(item, dict) and item.get('cut_id') == cut_id), None)
        if not item or not isinstance(item.get('context', {}).get('cut'), dict):
            raise HTTPException(409, "The saved analysis context is unavailable")
        original_cut = item['context']['cut']
        current = conn.execute("SELECT active,analysis_enabled,start_time,end_time FROM cuts WHERE id=?", (cut_id,)).fetchone()
        if not current or not current['active'] or not current['analysis_enabled'] or any(current[key] != original_cut.get(key) for key in ('start_time', 'end_time')):
            raise HTTPException(409, "The cut changed since this analysis; start a new analysis")
        try:
            evidence = json.loads(saved['evidence_json'])
        except (ValueError, TypeError) as exc:
            raise HTTPException(409, "The saved frame evidence is unreadable") from exc
        if not isinstance(evidence, list) or not evidence:
            raise HTTPException(409, "The saved frame evidence is unavailable")
        item['request_override'] = edited
        conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json,retry_of) VALUES (?,?,?,?,?)", (job_id, cut['episode_id'], 'analyze', json.dumps({'cuts': [item]}), old['id']))
        conn.execute("INSERT INTO analysis_job_cuts (job_id,cut_id,evidence_json,completed) VALUES (?,?,?,0)", (job_id, cut_id, saved['evidence_json']))
        worker.sync_episode_status(conn, cut['episode_id'])
    return {"job_id": job_id}


@app.post("/api/cuts/{cut_id}/regenerate-summary")
def regenerate_cut_summary(cut_id: str, body: PromptRegenInput):
    cut, config = analysis_config_for_cut(cut_id)
    if not cut["active"]:
        raise HTTPException(400, "Only active cuts can be analyzed")
    if body.mode == "edit" and not body.instruction.strip():
        raise HTTPException(400, "Describe the change you want to make")
    job_id = str(uuid.uuid4())
    try:
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts') AND status IN ('queued','running')", (cut["episode_id"],)).fetchone():
                raise HTTPException(409, "Wait for media preparation or cut detection to finish")
            if not conn.execute("SELECT 1 FROM cuts WHERE id=? AND active=1 AND analysis_enabled=1", (cut_id,)).fetchone():
                raise HTTPException(409, "The cut is archived or disabled for analysis")
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='regen' AND status IN ('queued','running') AND json_extract(payload_json, '$.result_type')='summary' AND json_extract(payload_json, '$.cut_id')=?", (cut["episode_id"], cut_id)).fetchone():
                raise HTTPException(409, "A summary for this cut is already queued or running")
            payload = summary_snapshot(conn, cut, config, body.mode, body.instruction, body.current_prompt)
            if body.preview_checked and payload["base_id"] != body.preview_base_id:
                raise HTTPException(409, "The saved summary changed since the request was prepared; refresh the preview")
            if body.request_override is not None:
                payload["request_override"] = validate_summary_request(body.request_override)
            if body.source_job_id and not conn.execute("SELECT 1 FROM jobs WHERE id=? AND episode_id=? AND pipeline='regen' AND status='failed' AND json_extract(payload_json,'$.cut_id')=?", (body.source_job_id, cut['episode_id'], cut_id)).fetchone():
                raise HTTPException(409, "The failed summary run is no longer available")
            conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json,retry_of) VALUES (?,?,?,?,?)", (job_id, cut["episode_id"], "regen", json.dumps(payload), body.source_job_id))
            worker.sync_episode_status(conn, cut["episode_id"])
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An analysis job is already running for this episode") from exc
    return {"job_id": job_id}


def delete_current_results(conn, ids: list[str]):
    """Hard-delete current revisions; a deleted revision's parent becomes current again."""
    for result_id in ids:
        row = conn.execute("SELECT a.parent_id,c.episode_id FROM analysis_results a JOIN cuts c ON c.id=a.cut_id WHERE a.id=?", (result_id,)).fetchone()
        if not row:
            continue
        if row["parent_id"]:
            conn.execute("UPDATE analysis_results SET superseded_by=NULL WHERE id=? AND superseded_by=?", (row["parent_id"], result_id))
        conn.execute("DELETE FROM analysis_results WHERE id=?", (result_id,))
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
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (episode_id,)).fetchone():
                return
            stat = path.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) == fingerprint:
                path.unlink(missing_ok=True)
    except OSError:
        pass


async def rendered_frame(spec, legacy=None, episode_id=None):
    fingerprint = None
    if legacy and legacy.is_file():
        stat = legacy.stat()
        fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    try:
        data = await frame_cache.get(spec)
        # Preserve legacy evidence if timestamp reconstruction is not byte-for-byte
        # identical; overlays must continue to match the image that produced them.
        if fingerprint and legacy.read_bytes() != data:
            return FileResponse(legacy, media_type='image/jpeg')
        task = BackgroundTask(remove_regenerated_legacy, legacy, episode_id, fingerprint) if fingerprint and episode_id else None
        return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=120'}, background=task)
    except (OSError, RuntimeError, ValueError) as exc:
        if fingerprint:
            return FileResponse(legacy, media_type='image/jpeg')
        raise HTTPException(503, f'Frame could not be recreated: {exc}') from exc


@app.get("/api/analysis-results/{result_id}/frame", response_class=Response,
         summary="Download a sampled analysis frame",
         responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}, "description": "Sampled JPEG for the result or an ancestor"},
                    404: {"description": "No frame can be reconstructed for this result or its ancestors"}})
async def get_analysis_result_frame(result_id: str):
    """Retrieve a stored legacy frame or recreate it from its recorded source and time."""
    spec = frame_source.result_frame_spec(result_id)
    if not spec:
        raise HTTPException(404, "No source frame is available for this result")
    if 'source' not in spec:
        return FileResponse(spec['legacy'], media_type='image/jpeg')
    return await rendered_frame(spec, spec['legacy'], spec['episode_id'])


@app.post("/api/cuts/{cut_id}/analyze")
def analyze_cut(cut_id: str):
    cut, _ = analysis_config_for_cut(cut_id)
    if not cut["active"]:
        raise HTTPException(400, "Only active cuts can be analyzed")
    return enqueue_analysis(cut["episode_id"], cut_id)


@app.post("/api/episodes/{episode_id}/analyze")
def analyze_episode(episode_id: str):
    require_episode(episode_id)
    return enqueue_analysis(episode_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_inference_job(job_id: str):
    """Cancel a queued inference job or a running summary job."""
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        cancellable = (job["status"] == "queued" and job["pipeline"] in ("analyze", "regen", "transcribe", "voice_sample")) or (job["status"] == "running" and job["pipeline"] == "regen")
        if not cancellable:
            raise HTTPException(409, "Only queued inference jobs or running summary jobs can be cancelled")
        message = "Cancelled before processing" if job["status"] == "queued" else "Cancelled during processing"
        conn.execute("UPDATE jobs SET status='failed',stage='Cancelled',error=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (message, job_id))
        worker.sync_episode_status(conn, job["episode_id"])
    worker.cleanup_voice_upload(job)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_analysis(job_id: str):
    try:
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job:
                raise HTTPException(404, "Job not found")
            if job["pipeline"] != "analyze" or job["status"] != "failed":
                raise HTTPException(409, "Only failed analysis jobs can be retried")
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts') AND status IN ('queued','running')", (job["episode_id"],)).fetchone():
                raise HTTPException(409, "Wait for media preparation or cut detection to finish")
            for item in json.loads(job["payload_json"])["cuts"]:
                done = conn.execute("SELECT completed FROM analysis_job_cuts WHERE job_id=? AND cut_id=?", (job_id, item["cut_id"])).fetchone()
                if done and done["completed"]:
                    continue
                cut = conn.execute("SELECT active,analysis_enabled,start_time,end_time FROM cuts WHERE id=? AND episode_id=?", (item["cut_id"], job["episode_id"])).fetchone()
                saved = item["context"]["cut"]
                if not cut or not cut["active"] or not cut["analysis_enabled"] or cut["start_time"] != saved["start_time"] or cut["end_time"] != saved["end_time"]:
                    raise HTTPException(409, "An unfinished cut changed since this run; start a new analysis")
            retry_id = str(uuid.uuid4())
            conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json,retry_of) VALUES (?,?,?,?,?)", (retry_id, job["episode_id"], 'analyze', job["payload_json"], job_id))
            conn.execute("INSERT INTO analysis_job_cuts (job_id,cut_id,evidence_json,completed) SELECT ?,cut_id,evidence_json,completed FROM analysis_job_cuts WHERE job_id=?", (retry_id, job_id))
            worker.sync_episode_status(conn, job["episode_id"])
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An analysis job is already running for this episode") from exc
    return {"job_id": retry_id}


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
async def get_media(episode_id: str, name: str):
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
                return await rendered_frame({'source': proxy, 'time': boundary, 'start': max(0, boundary - 5), 'end': boundary, 'width': 480, 'last': True, 'quality': 4, 'seek_tolerance': .000002, 'probe_padding': 1}, target, episode['id'])
    elif re.fullmatch(r"cut_\d{6}\.jpg", name):
        target, mime = path / "thumbnails" / name, "image/jpeg"
        index = int(name[4:10])
        cut = db.one('SELECT id FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (episode_id, index))
        if cut:
            return await get_cut_frame(cut['id'], 'first')
    else:
        raise HTTPException(404, "Media not found")
    if not target.is_file():
        raise HTTPException(404, "Media not ready")
    return FileResponse(target, media_type=mime)


@app.get("/api/cuts/{cut_id}/frame/{position}", response_class=Response,
         responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}, "description": "Cut boundary frame"},
                    404: {"description": "Cut or source video not found"}})
async def get_cut_frame(cut_id: str, position: Literal["first", "last"]):
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
    return await rendered_frame({'source': source, 'time': timestamp, 'start': cut['start_time'], 'end': cut['end_time'], 'width': 480, 'last': position == 'last' and saved is None, 'seek_tolerance': .000002, 'probe_padding': 1 if position == 'last' and saved is None else 0}, target, episode['id'])


app.include_router(external_router)

static_dir = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parent.parent / "static"))
if static_dir.is_dir():
    @app.get("/project/{project_id}", include_in_schema=False)
    def project_page(project_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/episode/{episode_id}", include_in_schema=False)
    def episode_page(episode_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
