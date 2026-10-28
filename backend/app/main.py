import json
import math
import mimetypes
import os
import re
import shutil
import uuid
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, media, worker

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_GB", "40")) * 1024 ** 3
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_SETTINGS = {"proxy_width": 960, "proxy_crf": 28, "proxy_preset": "veryfast", "frame_interval": 5.0, "scene_threshold": 0.32, "chunk_duration": 60}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    worker.start_worker()
    try:
        yield
    finally:
        worker.stop_worker()


app = FastAPI(title="Video Analyzer", lifespan=lifespan)


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


class AnalysisConfig(BaseModel):
    frame_mode: Literal["endpoints", "uniform"] = "endpoints"
    frame_count: int = Field(default=4, ge=2, le=32)
    context_ids: list[str] = Field(default_factory=list, max_length=200)
    detection_queries: list[Annotated[str, Field(min_length=1, max_length=100, pattern=r".*\S.*")]] = Field(default_factory=lambda: ["person", "clothing", "animal", "vehicle", "chair", "table"], min_length=1, max_length=20)
    additional_context: str = Field(default="", max_length=20000)
    openrouter_model: str = Field(default="openrouter/free", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_./:-]+$")


class CutInput(BaseModel):
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=20000)
    notes: str = Field(default="", max_length=20000)
    results: dict = Field(default_factory=dict)


class CutTimingInput(BaseModel):
    start_time: float
    end_time: float


class ChunkInput(BaseModel):
    title: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=20000)


ResultKind = Literal["first_frame", "camera_movement", "composition", "last_frame", "cut_description", "audio_visual_match", "generation_prompt", "other"]
ResultStatus = Literal["draft", "approved", "rejected"]


class ResultCreate(BaseModel):
    kind: ResultKind
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=50000)
    status: ResultStatus = "draft"


class ResultUpdate(BaseModel):
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=50000)
    status: ResultStatus


class TranscriptInput(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=20000)
    speaker_label: str = Field(default="", max_length=200)
    character_context_id: str | None = None
    status: ResultStatus = "draft"


class ContextInput(BaseModel):
    kind: Literal["chapter_synopsis", "chapter_happenings", "character", "world", "scenario", "object", "general"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=50000)
    episode_id: str | None = None


def require_project(project_id: str):
    project = db.one("SELECT * FROM projects WHERE id=?", (project_id,))
    if not project:
        raise HTTPException(404, "Project not found")
    return project


def require_episode(episode_id: str):
    episode = db.one("SELECT * FROM episodes WHERE id=?", (episode_id,))
    if not episode:
        raise HTTPException(404, "Episode not found")
    episode["settings"] = {**DEFAULT_SETTINGS, **json.loads(episode.pop("settings_json"))}
    episode["has_audio"] = bool(episode["has_audio"])
    return episode


def episode_response(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id)
    episode["proxy_ready"] = (path / "proxy.mp4").is_file()
    episode["cuts"] = [serialize_cut(cut) for cut in db.all_rows("SELECT * FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index", (episode_id,))]
    episode["archived_cuts"] = [serialize_cut(cut) for cut in db.all_rows("SELECT * FROM cuts WHERE episode_id=? AND active=0 ORDER BY start_time,cut_index", (episode_id,))]
    episode["chunks"] = db.all_rows("SELECT * FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,))
    # The queued payload can contain every cut's context and may be very large.
    # Keep it in SQLite for the worker and return only job metadata to the UI.
    episode["jobs"] = db.all_rows("SELECT id,episode_id,pipeline,status,stage,error,attempt_count,created_at,updated_at FROM jobs WHERE episode_id=? ORDER BY created_at DESC, rowid DESC", (episode_id,))
    return episode


def serialize_cut(cut: dict):
    cut["results"] = json.loads(cut.pop("results_json"))
    cut["active"] = bool(cut["active"])
    fallback_first, fallback_last = media.cut_frame_times(cut["start_time"], cut["end_time"])
    cut["first_frame_time"] = cut.get("first_frame_time") if cut.get("first_frame_time") is not None else fallback_first
    cut["last_frame_time"] = cut.get("last_frame_time") if cut.get("last_frame_time") is not None else fallback_last
    return cut


def serialize_result(result: dict):
    result["settings"] = json.loads(result.pop("settings_json"))
    result["context"] = json.loads(result.pop("context_json"))
    return result


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


@app.get("/api/projects/{project_id}/context")
def list_context(project_id: str):
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
        if not conn.execute("SELECT 1 FROM context_items WHERE id=?", (context_id,)).fetchone():
            raise HTTPException(404, "Context item not found")
        conn.execute("DELETE FROM context_items WHERE id=?", (context_id,))
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
        conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
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


def edit_cut_timeline(cut_id: str, action: str, timing: CutTimingInput | None = None):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cut = conn.execute("SELECT * FROM cuts WHERE id=? AND active=1", (cut_id,)).fetchone()
        if not cut:
            raise HTTPException(404, "Active cut not found")
        episode = conn.execute("SELECT * FROM episodes WHERE id=?", (cut["episode_id"],)).fetchone()
        if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (cut["episode_id"],)).fetchone():
            raise HTTPException(409, "Wait for the current media job to finish")
        cuts = conn.execute("SELECT * FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index", (cut["episode_id"],)).fetchall()
        index = next(i for i, item in enumerate(cuts) if item["id"] == cut_id)
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
            if index > 0:
                conn.execute("UPDATE cuts SET end_time=? WHERE id=?", (start, cuts[index - 1]["id"]))
            if index + 1 < len(cuts):
                conn.execute("UPDATE cuts SET start_time=? WHERE id=?", (end, cuts[index + 1]["id"]))
            conn.execute("UPDATE cuts SET start_time=?,end_time=? WHERE id=?", (start, end, cut_id))
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
            archive_index = conn.execute("SELECT MIN(cut_index) FROM cuts WHERE episode_id=?", (cut["episode_id"],)).fetchone()[0]
            conn.execute("UPDATE cuts SET active=0,cut_index=? WHERE id=?", (min(-1, archive_index - 1), removed["id"]))
            for item in cuts:
                if item["cut_index"] > removed["cut_index"]:
                    conn.execute("UPDATE cuts SET cut_index=? WHERE id=?", (item["cut_index"] - 1, item["id"]))
        conn.execute("UPDATE cuts SET first_frame_time=NULL,last_frame_time=NULL WHERE episode_id=? AND active=1", (cut["episode_id"],))
        job_id = str(uuid.uuid4())
        conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, cut["episode_id"], "cutframes", json.dumps({"settings": json.loads(episode["settings_json"])})))
        conn.execute("UPDATE episodes SET status='queued',stage='Updating cut frames',error=NULL WHERE id=?", (cut["episode_id"],))
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


@app.get("/api/cuts/{cut_id}/analysis-results")
def list_analysis_results(cut_id: str, history: bool = False, include_context: bool = True):
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    where = "cut_id=?" if history else "cut_id=? AND superseded_by IS NULL"
    columns = "*" if include_context else "id,cut_id,kind,title,content,status,origin,model_id,job_id,prompt,settings_json,'{}' AS context_json,parent_id,superseded_by,created_at"
    return [serialize_result(row) for row in db.all_rows(f"SELECT {columns} FROM analysis_results WHERE {where} ORDER BY rowid DESC", (cut_id,))]


@app.post("/api/cuts/{cut_id}/analysis-results")
def create_analysis_result(cut_id: str, body: ResultCreate):
    if not db.one("SELECT id FROM cuts WHERE id=?", (cut_id,)):
        raise HTTPException(404, "Cut not found")
    result_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content,status) VALUES (?,?,?,?,?,?)",
                     (result_id, cut_id, body.kind, body.title, body.content, body.status))
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (result_id,)))


@app.put("/api/analysis-results/{result_id}")
def revise_analysis_result(result_id: str, body: ResultUpdate):
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        old = conn.execute("SELECT * FROM analysis_results WHERE id=?", (result_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Analysis result not found")
        if old["superseded_by"]:
            raise HTTPException(409, "This result already has a newer revision")
        conn.execute("INSERT INTO analysis_results (id,cut_id,kind,title,content,status,origin,parent_id) VALUES (?,?,?,?,?,?,?,?)",
                     (new_id, old["cut_id"], old["kind"], body.title, body.content, body.status, "manual_edit", result_id))
        conn.execute("UPDATE analysis_results SET superseded_by=? WHERE id=?", (new_id, result_id))
    return serialize_result(db.one("SELECT * FROM analysis_results WHERE id=?", (new_id,)))


@app.get("/api/episodes/{episode_id}/transcript")
def list_transcript(episode_id: str, history: bool = False):
    require_episode(episode_id)
    where = "episode_id=?" if history else "episode_id=? AND superseded_by IS NULL AND status!='deleted'"
    order = "rowid DESC" if history else "start_time,id"
    return db.all_rows(f"SELECT * FROM transcript_segments WHERE {where} ORDER BY {order}", (episode_id,))


@app.post("/api/episodes/{episode_id}/transcript")
def create_transcript_segment(episode_id: str, body: TranscriptInput):
    episode = require_episode(episode_id)
    validate_transcript(episode, body)
    segment_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,status) VALUES (?,?,?,?,?,?,?,?)",
                     (segment_id, episode_id, body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id, body.status))
    return db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,))


@app.put("/api/transcript-segments/{segment_id}")
def update_transcript_segment(segment_id: str, body: TranscriptInput):
    segment = db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,))
    if not segment:
        raise HTTPException(404, "Transcript segment not found")
    episode = require_episode(segment["episode_id"])
    validate_transcript(episode, body)
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        current = conn.execute("SELECT superseded_by FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if current["superseded_by"]:
            raise HTTPException(409, "This transcript segment has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,status,origin,parent_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (new_id, segment["episode_id"], body.start_time, body.end_time, body.text, body.speaker_label, body.character_context_id, body.status, "manual_edit", segment_id))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
    return db.one("SELECT * FROM transcript_segments WHERE id=?", (new_id,))


@app.delete("/api/transcript-segments/{segment_id}")
def delete_transcript_segment(segment_id: str):
    new_id = str(uuid.uuid4())
    with db.connect() as conn:
        old = conn.execute("SELECT * FROM transcript_segments WHERE id=?", (segment_id,)).fetchone()
        if not old:
            raise HTTPException(404, "Transcript segment not found")
        if old["superseded_by"]:
            raise HTTPException(409, "This transcript segment has a newer revision")
        conn.execute("INSERT INTO transcript_segments (id,episode_id,start_time,end_time,text,speaker_label,character_context_id,status,origin,parent_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (new_id, old["episode_id"], old["start_time"], old["end_time"], old["text"], old["speaker_label"], old["character_context_id"], "deleted", "manual_edit", segment_id))
        conn.execute("UPDATE transcript_segments SET superseded_by=? WHERE id=?", (new_id, segment_id))
    return {"ok": True}


@app.get("/api/cuts/{cut_id}/analysis-context")
def get_analysis_context(cut_id: str):
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    episode = require_episode(cut["episode_id"])
    project = require_project(episode["project_id"])
    previous = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] - 1)) if cut["active"] else None
    following = db.one("SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?", (episode["id"], cut["cut_index"] + 1)) if cut["active"] else None
    neighbors = {}
    for label, neighbor in (("previous", previous), ("next", following)):
        neighbors[label] = db.all_rows("SELECT kind,title,content,origin,model_id FROM analysis_results WHERE cut_id=? AND status='approved' AND superseded_by IS NULL ORDER BY rowid DESC", (neighbor["id"],)) if neighbor else []
    return {
        "project": {"id": project["id"], "title": project["title"], "synopsis": project["synopsis"]},
        "episode": {"id": episode["id"], "title": episode["title"], "duration": episode["duration"]},
        "cut": serialize_cut(cut),
        "previous_cut": serialize_cut(previous) if previous else None,
        "next_cut": serialize_cut(following) if following else None,
        "neighbor_results": neighbors,
        "chunks": db.all_rows("SELECT * FROM chunks WHERE episode_id=? AND start_time<? AND end_time>? ORDER BY chunk_index", (episode["id"], cut["end_time"], cut["start_time"])),
        "context_items": db.all_rows("SELECT * FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?) ORDER BY kind,title", (project["id"], episode["id"])),
        "transcript_segments": db.all_rows("SELECT * FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND status='approved' AND start_time<? AND end_time>? ORDER BY start_time", (episode["id"], cut["end_time"], cut["start_time"])),
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
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (episode_id,)).fetchone():
                raise HTTPException(409, "An analysis job is already running for this episode")
            if cut_id is None:
                cuts = conn.execute("SELECT id FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index", (episode_id,)).fetchall()
                if not cuts:
                    raise HTTPException(400, "Detect cuts before running analysis")
            else:
                cut = conn.execute("SELECT id FROM cuts WHERE id=? AND episode_id=? AND active=1", (cut_id, episode_id)).fetchone()
                if not cut:
                    raise HTTPException(409, "The cut is no longer active")
                cuts = [cut]
            # The write lock keeps cut boundaries, presets and context stable until
            # the complete job payload is committed. The helpers use read-only
            # connections so the UI never sees a partial queue snapshot.
            inputs = []
            for cut in cuts:
                _, config = analysis_config_for_cut(cut["id"])
                inputs.append({"cut_id": cut["id"], "config": config.model_dump(), "context": selected_analysis_context(cut["id"], config)})
            conn.execute("INSERT INTO jobs (id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, "analyze", json.dumps({"cuts": inputs})))
            conn.execute("UPDATE episodes SET status='queued',stage='Waiting for analysis worker',error=NULL WHERE id=?", (episode_id,))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An analysis job is already running for this episode") from exc
    return {"job_id": job_id}


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
def cancel_queued_analysis(job_id: str):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        if job["pipeline"] != "analyze" or job["status"] != "queued":
            raise HTTPException(409, "Only queued analysis jobs can be cancelled")
        conn.execute("UPDATE jobs SET status='failed',stage='Cancelled',error='Cancelled before processing',updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        conn.execute("UPDATE episodes SET status='ready',stage='Ready',error=NULL WHERE id=?", (job["episode_id"],))
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
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (job["episode_id"],)).fetchone():
                raise HTTPException(409, "An analysis job is already running for this episode")
            latest = conn.execute("SELECT id FROM jobs WHERE episode_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1", (job["episode_id"],)).fetchone()
            if not latest or latest["id"] != job_id:
                raise HTTPException(409, "A newer job exists; start a new analysis")
            for item in json.loads(job["payload_json"])["cuts"]:
                done = conn.execute("SELECT completed FROM analysis_job_cuts WHERE job_id=? AND cut_id=?", (job_id, item["cut_id"])).fetchone()
                if done and done["completed"]:
                    continue
                cut = conn.execute("SELECT active,start_time,end_time FROM cuts WHERE id=? AND episode_id=?", (item["cut_id"], job["episode_id"])).fetchone()
                saved = item["context"]["cut"]
                if not cut or not cut["active"] or cut["start_time"] != saved["start_time"] or cut["end_time"] != saved["end_time"]:
                    raise HTTPException(409, "An unfinished cut changed since this run; start a new analysis")
            conn.execute("UPDATE jobs SET status='queued',stage='Waiting for analysis worker',error=NULL,lease_owner=NULL,lease_until=NULL,attempt_count=0,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            conn.execute("UPDATE episodes SET status='queued',stage='Waiting for analysis worker',error=NULL WHERE id=?", (job["episode_id"],))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An analysis job is already running for this episode") from exc
    return {"job_id": job_id}


@app.get("/api/episodes/{episode_id}/waveform")
def get_waveform(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id) / "waveform.json"
    if not path.exists():
        return {"peaks": [], "duration": episode["duration"] or 0}
    return json.loads(path.read_text())


@app.get("/api/episodes/{episode_id}/frames")
def get_frames(episode_id: str):
    episode = require_episode(episode_id)
    path = media.episode_dir(episode["project_id"], episode_id) / "frames" / "manifest.json"
    if not path.exists():
        return {"source": "proxy", "interval": episode["settings"]["frame_interval"], "frames": []}
    return json.loads(path.read_text())


@app.get("/api/episodes/{episode_id}/media/{name}")
def get_media(episode_id: str, name: str):
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
    elif re.fullmatch(r"cut_\d{6}\.jpg", name):
        target, mime = path / "thumbnails" / name, "image/jpeg"
    else:
        raise HTTPException(404, "Media not found")
    if not target.is_file():
        raise HTTPException(404, "Media not ready")
    return FileResponse(target, media_type=mime)


@app.get("/api/cuts/{cut_id}/frame/{position}")
def get_cut_frame(cut_id: str, position: Literal["first", "last"]):
    cut = db.one("SELECT episode_id FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    episode = require_episode(cut["episode_id"])
    target = media.episode_dir(episode["project_id"], episode["id"]) / "thumbnails" / f"{cut_id}_{position}.jpg"
    if not target.is_file():
        raise HTTPException(404, "Cut frame not ready; rerun hard-cut detection")
    return FileResponse(target, media_type="image/jpeg")


static_dir = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parent.parent / "static"))
if static_dir.is_dir():
    @app.get("/project/{project_id}", include_in_schema=False)
    def project_page(project_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/episode/{episode_id}", include_in_schema=False)
    def episode_page(episode_id: str):
        return FileResponse(static_dir / "index.html", media_type="text/html")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
