"""Read-only, versioned integration API for external consumers."""
import json
import math
import struct
import wave
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import db, media

router = APIRouter(prefix="/api/v1", tags=["External integration v1"])


class Project(BaseModel):
    id: str = Field(description="Stable project UUID. Use this ID, not the title, to select a project.")
    title: str
    synopsis: str
    created_at: str


class Episode(BaseModel):
    id: str = Field(description="Stable episode UUID; belongs to project_id.")
    project_id: str
    title: str
    original_filename: str
    original_ext: str
    status: str = Field(description="Current processing state, such as queued, running, ready or failed.")
    stage: str
    error: str | None
    duration: float | None = Field(description="Video duration in seconds; null until preparation has probed it.")
    width: int | None
    height: int | None
    has_audio: bool | None = Field(description="Whether the original has audio, or null before probing.")
    settings: dict[str, Any] = Field(description="Effective media preparation and detection settings.")
    created_at: str


class Cut(BaseModel):
    id: str = Field(description="Stable cut UUID. Store this instead of the changing cut_index.")
    episode_id: str
    cut_index: int = Field(description="Zero-based current timeline order; archived cuts have negative indices. CUT1 is index 0.")
    start_time: float = Field(description="Inclusive start in seconds from the beginning of the episode.")
    end_time: float = Field(description="Exclusive end in seconds from the beginning of the episode.")
    first_frame_time: float
    last_frame_time: float
    active: bool = Field(description="False means this cut is archived and absent from the current timeline.")
    analysis_enabled: bool = Field(description="Whether this cut is included in future visual analysis runs.")
    title: str
    description: str
    notes: str
    results: dict[str, Any] = Field(description="Legacy editable cut results; distinct from analysis_results revisions.")


class AnalysisResult(BaseModel):
    id: str
    cut_id: str
    kind: str = Field(description="Result category, for example generation_prompt or cut_description.")
    title: str
    content: str
    origin: str
    model_id: str | None
    job_id: str | None
    prompt: str | None
    settings: dict[str, Any] = Field(description="Recorded model/settings provenance, when available.")
    context: dict[str, Any] = Field(description="Recorded analysis input and evidence, when available.")
    parent_id: str | None = Field(description="Previous revision ID, if this is a revision.")
    superseded_by: str | None = Field(description="Next revision ID, if this revision is no longer current.")
    created_at: str


class ContextItem(BaseModel):
    id: str
    project_id: str
    episode_id: str | None = Field(description="Null means project-wide; otherwise applies to this episode.")
    kind: str = Field(description="chapter_synopsis, chapter_happenings, character, world, scenario, object, or general.")
    title: str
    content: str
    created_at: str
    updated_at: str


class Transcript(BaseModel):
    id: str
    episode_id: str
    start_time: float
    end_time: float
    text: str
    speaker_label: str
    character_context_id: str | None = Field(description="Optional ID of the corresponding character context item.")
    character_name: str | None = None
    deleted: bool = Field(description="True for a deletion revision returned in transcript history.")
    origin: str
    model_id: str | None
    job_id: str | None
    parent_id: str | None
    superseded_by: str | None
    created_at: str
    updated_at: str


class Chunk(BaseModel):
    id: str
    episode_id: str
    chunk_index: int
    start_time: float
    end_time: float
    title: str
    notes: str


class VoiceRange(BaseModel):
    start_time: float
    end_time: float


class VoiceSample(BaseModel):
    id: str
    character_context_id: str
    episode_id: str | None
    start_time: float
    end_time: float
    ranges: list[VoiceRange]
    source_filename: str | None
    model_id: str
    created_at: str
    episode_title: str | None
    preview_available: bool = Field(description="Whether the WAV reference can currently be downloaded.")


class VoiceSampleDetail(VoiceSample):
    embedding: Any = Field(description="Model-specific stored voice embedding; interpret only with the recorded model_id.")
    audio_url: str = Field(description="Relative URL for the reference WAV, if available.")


class JobCutEvidence(BaseModel):
    cut_id: str
    completed: bool
    evidence: Any = Field(description="Model-specific intermediate frame evidence retained for this job and cut.")


class JobDetail(BaseModel):
    id: str
    episode_id: str
    pipeline: str
    status: str
    stage: str
    error: str | None
    attempt_count: int
    created_at: str
    updated_at: str
    started_at: str | None = Field(description="UTC time processing first started, or null if it has not started.")
    finished_at: str | None = Field(description="UTC time the run finished or was cancelled, or null if still active.")
    retry_of: str | None = Field(description="Failed job copied for this retry, if any.")
    payload: dict[str, Any] = Field(description="Queued input snapshot. May be large and may differ from current cut or context values.")
    cut_evidence: list[JobCutEvidence] = Field(description="Stored intermediate analysis checkpoints, if any; may be incomplete.")


class CutPage(BaseModel):
    items: list[Cut]
    total: int
    offset: int
    limit: int
    next_offset: int | None = Field(description="Pass as offset for the next page; null means no more results.")


class CutDetail(BaseModel):
    project: Project
    episode: Episode
    cut: Cut
    analysis_results: list[AnalysisResult] = Field(description="Current result revisions.")
    transcript_segments: list[Transcript] = Field(description="Current non-deleted segments overlapping this cut; may extend beyond its boundaries.")
    context_items: list[ContextItem] = Field(description="Project-wide and this episode's context items, including characters.")
    chunks: list[Chunk] = Field(description="Working chunks overlapping the cut; chunks are not cuts.")
    previous_cut: Cut | None
    next_cut: Cut | None
    voice_samples: list[VoiceSample] = Field(description="All project character voice references; match character_context_id to context or transcript.")
    links: dict[str, str] = Field(description="Relative API paths for related data and media. Resolve against the server origin.")


class ContextProject(BaseModel):
    id: str
    title: str
    synopsis: str


class ContextEpisode(BaseModel):
    id: str
    title: str
    duration: float | None


class NeighborResult(BaseModel):
    kind: str
    title: str
    content: str
    origin: str
    model_id: str | None


class NeighborResults(BaseModel):
    previous: list[NeighborResult]
    next: list[NeighborResult]


class AnalysisContext(BaseModel):
    project: ContextProject
    episode: ContextEpisode
    cut: Cut
    previous_cut: Cut | None
    next_cut: Cut | None
    neighbor_results: NeighborResults = Field(description="Current results from adjacent active cuts.")
    chunks: list[Chunk]
    context_items: list[ContextItem]
    transcript_segments: list[Transcript] = Field(description="Current non-deleted transcript segments overlapping the cut.")


def main_api():
    # Late import avoids a circular import when the router is registered by main.
    from . import main
    return main


@router.get("/projects", response_model=list[Project], summary="List projects")
def projects():
    return db.all_rows("SELECT * FROM projects ORDER BY created_at DESC,id")


@router.get("/projects/{project_id}", response_model=Project, summary="Get project metadata")
def project(project_id: str):
    return main_api().require_project(project_id)


@router.get("/projects/{project_id}/episodes", response_model=list[Episode], summary="List a project's episodes")
def episodes(project_id: str):
    main = main_api()
    main.require_project(project_id)
    rows = db.all_rows("SELECT id FROM episodes WHERE project_id=? ORDER BY created_at,id", (project_id,))
    return [episode_metadata(row["id"]) for row in rows]


@router.get("/episodes/{episode_id}", response_model=Episode, summary="Get episode metadata")
def episode_metadata(episode_id: str):
    episode = main_api().require_episode(episode_id)
    # The UI helper converts null (unknown) to false; retain the distinction for integrations.
    raw = db.one("SELECT has_audio FROM episodes WHERE id=?", (episode_id,))
    episode["has_audio"] = None if raw["has_audio"] is None else bool(raw["has_audio"])
    return episode


@router.get("/voice-samples/{sample_id}", response_model=VoiceSampleDetail, summary="Get a character voice reference and embedding")
def voice_sample_detail(sample_id: str):
    row = db.one("SELECT v.embedding_json,c.project_id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE v.id=?", (sample_id,))
    if not row:
        raise HTTPException(404, "Voice sample not found")
    sample = next((sample for sample in main_api().list_voice_samples(row["project_id"]) if sample["id"] == sample_id), None)
    if not sample:
        raise HTTPException(404, "Voice sample not found")
    return {**sample, "embedding": json.loads(row["embedding_json"]), "audio_url": f"/api/voice-samples/{sample_id}/audio"}


@router.get("/jobs/{job_id}", response_model=JobDetail, summary="Get job provenance and saved analysis evidence")
def job_detail(job_id: str):
    row = db.one("SELECT id,episode_id,pipeline,status,stage,error,attempt_count,created_at,updated_at,started_at,finished_at,retry_of,payload_json FROM jobs WHERE id=?", (job_id,))
    if not row:
        raise HTTPException(404, "Job not found")
    row["payload"] = json.loads(row.pop("payload_json"))
    row["cut_evidence"] = [{"cut_id": item["cut_id"], "completed": bool(item["completed"]), "evidence": json.loads(item["evidence_json"])}
                           for item in db.all_rows("SELECT cut_id,completed,evidence_json FROM analysis_job_cuts WHERE job_id=? ORDER BY cut_id", (job_id,))]
    return row


@router.get("/episodes/{episode_id}/cuts", response_model=CutPage, summary="List timeline cuts")
def cuts(episode_id: str,
         offset: Annotated[int, Query(ge=0, description="Number of cuts to skip.")] = 0,
         limit: Annotated[int, Query(ge=1, le=500, description="Maximum cuts per page.")] = 100,
         include_archived: Annotated[bool, Query(description="Include inactive archived cuts after active cuts.")] = False):
    main = main_api()
    main.require_episode(episode_id)
    where = "episode_id=?" if include_archived else "episode_id=? AND active=1"
    total = db.one(f"SELECT COUNT(*) AS n FROM cuts WHERE {where}", (episode_id,))["n"]
    rows = db.all_rows(f"SELECT * FROM cuts WHERE {where} ORDER BY active DESC, CASE WHEN active=1 THEN cut_index END, start_time, id LIMIT ? OFFSET ?", (episode_id, limit, offset))
    return {"items": [main.serialize_cut(row) for row in rows], "total": total, "offset": offset, "limit": limit,
            "next_offset": offset + len(rows) if offset + len(rows) < total else None}


@router.get("/cuts/{cut_id}", response_model=CutDetail, summary="Get a cut and its analysis context")
def cut_detail(cut_id: str):
    main = main_api()
    context = main.get_analysis_context(cut_id)
    episode_id = context["episode"]["id"]
    project_id = context["project"]["id"]
    return {"project": main.require_project(project_id), "episode": episode_metadata(episode_id), "cut": context["cut"],
            "analysis_results": main.list_analysis_results(cut_id), "transcript_segments": context["transcript_segments"],
            "context_items": context["context_items"], "chunks": context["chunks"],
            "previous_cut": context["previous_cut"], "next_cut": context["next_cut"],
            "voice_samples": main.list_voice_samples(project_id),
            "links": {"analysis_results": f"/api/cuts/{cut_id}/analysis-results", "analysis_context": f"/api/cuts/{cut_id}/analysis-context",
                      "audio": f"/api/v1/cuts/{cut_id}/audio", "first_frame": f"/api/cuts/{cut_id}/frame/first",
                      "last_frame": f"/api/cuts/{cut_id}/frame/last", "transcript": f"/api/episodes/{episode_id}/transcript",
                      "voice_samples": f"/api/projects/{project_id}/voice-samples", "original": f"/api/episodes/{episode_id}/media/original",
                      "proxy": f"/api/episodes/{episode_id}/media/proxy"}}


def _wav_stream(path: Path, start_frame: int, frame_count: int, header: bytes):
    yield header
    with wave.open(str(path), "rb") as source:
        source.setpos(start_frame)
        remaining = frame_count
        while remaining:
            block = source.readframes(min(remaining, 65536))
            if not block:
                break
            remaining -= len(block) // (source.getnchannels() * source.getsampwidth())
            yield block


@router.get("/cuts/{cut_id}/audio", summary="Download cut audio with optional margins", response_class=StreamingResponse,
            responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
                             "headers": {"X-Audio-Start-Seconds": {"description": "Actual sample-aligned episode start time in seconds.", "schema": {"type": "number"}},
                                         "X-Audio-End-Seconds": {"description": "Actual sample-aligned episode end time in seconds.", "schema": {"type": "number"}}},
                             "description": "PCM WAV; X-Audio-Start-Seconds and X-Audio-End-Seconds report the actual excerpt boundaries."},
                       404: {"description": "Cut, episode audio, or prepared audio not found."},
                       409: {"description": "The cut has no samples in the available prepared audio, or the prepared audio is invalid."},
                       413: {"description": "Excerpt exceeds the WAV format size limit."}})
def cut_audio(cut_id: str,
              before: Annotated[float, Query(ge=0, le=3600, allow_inf_nan=False, description="Seconds to include before the cut start; clipped to zero.")] = 0,
              after: Annotated[float, Query(ge=0, le=3600, allow_inf_nan=False, description="Seconds to include after the cut end; clipped to available audio and episode duration.")] = 0):
    cut = db.one("SELECT * FROM cuts WHERE id=?", (cut_id,))
    if not cut:
        raise HTTPException(404, "Cut not found")
    episode = main_api().require_episode(cut["episode_id"])
    if not episode["has_audio"]:
        raise HTTPException(404, "Episode has no prepared audio")
    path = media.episode_dir(episode["project_id"], episode["id"]) / "audio.wav"
    if not path.is_file():
        raise HTTPException(404, "Episode audio is not ready")
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            start = max(0, min(source.getnframes(), math.floor(max(0, cut["start_time"] - before) * rate)))
            end_time = min(cut["end_time"] + after, episode["duration"] if episode["duration"] is not None else float("inf"))
            end = min(source.getnframes(), math.ceil(end_time * rate))
            if end <= start:
                raise HTTPException(409, "Cut has no samples in the available audio")
            length = (end - start) * source.getnchannels() * source.getsampwidth()
            if length > 0xFFFFFFFF - 36:
                raise HTTPException(413, "Excerpt exceeds the WAV format size limit")
            channels, width = source.getnchannels(), source.getsampwidth()
            block_align = channels * width
            header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + length, b"WAVE", b"fmt ",
                                 16, 1, channels, rate, rate * block_align, block_align, width * 8, b"data", length)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(409, "Prepared audio is invalid") from exc
    return StreamingResponse(_wav_stream(path, start, end - start, header), media_type="audio/wav", headers={
        "Content-Disposition": f'attachment; filename="cut-{cut_id}.wav"',
        "Content-Length": str(len(header) + length),
        "X-Audio-Start-Seconds": str(start / rate), "X-Audio-End-Seconds": str(end / rate),
    })
