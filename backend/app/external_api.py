"""Read-only, versioned integration API for external consumers."""
import json
import math
import struct
import wave
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class BriefSection(BaseModel):
    content: str
    confidence: str


class BriefDialogue(BaseModel):
    id: str
    start_time: float
    end_time: float
    text: str
    speaker_label: str | None
    character_context_id: str | None


class BriefCut(BaseModel):
    id: str
    cut_number: int
    start_time: float
    end_time: float
    production_brief: dict[str, Any] | None = Field(description="Current Cut summary with content and provenance; older summaries may predate production briefs.")
    initial_frame: dict[str, Any] | None = Field(description='Current structured initial-frame result and provenance, when available.')
    suggested_frame_url: str | None = Field(description='Relative URL of the cut’s first frame when available; no model recommendation is made.')
    audio_url: str | None
    dialogue: list[BriefDialogue]
    visible_characters: dict[str, BriefSection | None]
    camera: dict[str, BriefSection | None]
    setting: BriefSection | None
    action_beats: BriefSection | None
    lighting: BriefSection | None
    composition: BriefSection | None
    unavailable: dict[str, str | None]


class EpisodeBrief(BaseModel):
    episode_id: str
    from_cut: int = Field(alias='from')
    to: int
    cuts: list[BriefCut]


class CompositionModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class CompositionShot(CompositionModel):
    size: Literal['ECU', 'CU', 'MCU', 'MS', 'MWS', 'WS', 'EWS'] | None = None
    angle: Literal['eye', 'low', 'high', 'overhead', 'dutch'] | None = None
    camera: Literal['static', 'pan', 'tilt', 'dolly', 'zoom', 'handheld'] | None = None
    direction: str | None = None
    speed: Literal['slow', 'medium', 'fast'] | None = None


class CompositionSubject(CompositionModel):
    character_context_id: str | None
    name: str
    identity: Literal['confirmed', 'likely', 'guess'] | None = None
    frame_pos: Literal['left fg','left mid','left bg','center fg','center mid','center bg','right fg','right mid','right bg'] | None = None
    facing: str | None = None
    pose: str | None = None
    hands: str | None = None
    gaze: str | None = None
    expression: str | None = None
    outfit_visible: str | None = None

    @field_validator('facing', 'pose', 'hands', 'gaze', 'expression', 'outfit_visible')
    @classmethod
    def short_phrase(cls, value):
        if value is not None and len(value.split()) > 12:
            raise ValueError('Subject phrases must contain at most 12 words')
        return value


class CompositionBeat(CompositionModel):
    t0: float
    t1: float
    subject: str | Annotated[int, Field(ge=0)] = Field(description='Opening-subject index when uniquely linked, otherwise a descriptive name.')
    action: str

    @model_validator(mode='after')
    def valid_interval(self):
        if not math.isfinite(self.t0) or not math.isfinite(self.t1) or not 0 <= self.t0 < self.t1:
            raise ValueError('Beat times must form a finite positive interval')
        return self


class CompositionUtterance(CompositionModel):
    speaker_label: str
    character_context_id: str | None
    character_name: str
    text: str = Field(description='Complete overlapping transcript text; never truncated to meet the response size target.')
    t0: float
    t1: float
    crosses_cut: bool
    subtitle: str | None = None

    @model_validator(mode='after')
    def valid_interval(self):
        if not math.isfinite(self.t0) or not math.isfinite(self.t1) or not 0 <= self.t0 <= self.t1:
            raise ValueError('Utterance times must form a finite ordered interval')
        return self


class CompositionSubtitle(CompositionModel):
    text: str
    lang: str
    t: float


class CompositionEndSubject(CompositionModel):
    subject: str | Annotated[int, Field(ge=0)] = Field(description='Opening-subject index when uniquely linked, otherwise a descriptive name.')
    change: Literal['entered frame', 'left frame'] | None = None
    frame_pos: Literal['left fg','left mid','left bg','center fg','center mid','center bg','right fg','right mid','right bg'] | None = None
    facing: str | None = None
    pose: str | None = None
    hands: str | None = None
    gaze: str | None = None
    expression: str | None = None

    @field_validator('facing', 'pose', 'hands', 'gaze', 'expression')
    @classmethod
    def short_phrase(cls, value):
        if value is not None and len(value.split()) > 12:
            raise ValueError('End-state phrases must contain at most 12 words')
        return value


class CompositionEndState(CompositionModel):
    subjects: list[CompositionEndSubject] | None = None
    size: Literal['ECU', 'CU', 'MCU', 'MS', 'MWS', 'WS', 'EWS'] | None = None


class CompositionLink(CompositionModel):
    subject: int | None = Field(None, ge=0, description='Zero-based index into this cut’s subjects; resolves its character_context_id.')
    character_context_id: str | None = Field(None, description='Character ID when no unique opening subject can be referenced; omitted for unknown detections.')
    label_source: str
    stale: bool
    first: list[list[float]] | None = Field(None, description='First-frame boxes grouped by character, source and stale flag; each box is [x1,y1,x2,y2].')
    last: list[list[float]] | None = Field(None, description='Last-frame boxes grouped independently; list order does not pair unnamed people between frames.')

    @field_validator('first', 'last')
    @classmethod
    def valid_boxes(cls, boxes):
        if boxes is not None and any(len(box) != 4 or not all(math.isfinite(n) for n in box) for box in boxes):
            raise ValueError('Each box must contain four finite coordinates')
        return boxes


class CompositionConfidence(CompositionModel):
    shot: Literal['high', 'medium', 'low'] | None = None
    setting: Literal['high', 'medium', 'low'] | None = None
    lighting: Literal['high', 'medium', 'low'] | None = None
    subjects: Literal['high', 'medium', 'low'] | None = None
    beats: Literal['high', 'medium', 'low'] | None = None
    end_state: Literal['high', 'medium', 'low'] | None = None


class CompositionUnavailable(CompositionModel):
    shot: Literal['vp', 'vs', 's', 'sf'] | None = None
    frame: Literal['f'] | None = None
    audio: Literal['a'] | None = None
    beats_partial: Literal['bp'] | None = None
    setting: Literal['sp'] | None = None
    lighting: Literal['lp'] | None = None
    subjects: Literal['up'] | None = None
    end_state: Literal['ep', 'ia'] | None = None
    subtitles: Literal['op'] | None = None
    issues: Literal['ip'] | None = None


class CompositionCut(CompositionModel):
    id: str
    cut_number: int
    ready: bool
    updated_at: str
    analysis_rev: str = Field(description='Opaque revision token; compare for equality rather than interpreting its encoding.')
    duration: float | None = Field(None, description='Cut duration in seconds; beat and utterance times are relative to its start.')
    start_time: float | None = None
    end_time: float | None = None
    initial_frame_id: str | None = None
    last_frame_id: str | None = None
    shot: CompositionShot | None = None
    setting: str | None = None
    lighting: str | None = None
    subjects: list[CompositionSubject] | None = None
    beats: list[CompositionBeat] | None = None
    end_state: CompositionEndState | None = None
    dialogue: list[CompositionUtterance] | None = None
    subtitles: list[CompositionSubtitle] | None = None
    issues: list[str] | None = Field(None, max_length=2)
    links: list[CompositionLink] | None = None
    confidence: CompositionConfidence | None = None
    unavailable: CompositionUnavailable | None = Field(None, description='Section names mapped to short codes explained by the range unavailable_reasons table.')
    initial_frame: dict[str, Any] | None = None
    last_frame: dict[str, Any] | None = None
    video_analysis: dict[str, Any] | None = None
    subtitle_ocr: dict[str, Any] | None = None
    consistency_summary: dict[str, Any] | None = None
    dialogue_segments: list[dict[str, Any]] | None = None

    @field_validator('setting', 'lighting')
    @classmethod
    def bounded_description(cls, value, info):
        if value is not None and len(value.split()) > (20 if info.field_name == 'setting' else 15):
            raise ValueError('Description exceeds its word limit')
        return value

    @model_validator(mode='after')
    def ordered_beats(self):
        beats = self.beats or []
        if any(a.t1 > b.t0 for a,b in zip(beats, beats[1:])):
            raise ValueError('Beats must be ordered and non-overlapping')
        duration = self.duration if self.duration is not None else (self.end_time-self.start_time if self.start_time is not None and self.end_time is not None else None)
        if duration is not None and any(b.t1 > round(duration, 3) for b in beats):
            raise ValueError('Beat extends outside the cut')
        if duration is not None and len(beats) > max(1, math.ceil(duration / 2)):
            raise ValueError('Too many beats for the cut duration')
        if duration is not None and any(u.t1 > round(duration, 3) for u in self.dialogue or []):
            raise ValueError('Utterance extends outside the cut')
        if any(link.subject is not None and (self.subjects is None or link.subject >= len(self.subjects)) for link in self.links or []):
            raise ValueError('Detection link references a missing subject')
        references = [beat.subject for beat in beats]
        if self.end_state:
            references.extend(item.subject for item in self.end_state.subjects or [])
        if any(isinstance(index, int) and (self.subjects is None or index >= len(self.subjects)) for index in references):
            raise ValueError('Beat or end state references a missing subject')
        return self


class CompositionBrief(CompositionModel):
    episode_id: str
    from_cut: int = Field(alias='from')
    to: int
    ready_count: int
    pending: list[int]
    unavailable_reasons: dict[str, str]
    cuts: list[CompositionCut]


class BriefExportStatus(BaseModel):
    job_id: str
    status: str
    stage: str
    error: str | None
    download_url: str | None


@router.get('/episodes/{episode_id}/brief', response_model=EpisodeBrief | CompositionBrief, summary='Get an agent-ready brief for a range of cuts')
def episode_brief(episode_id: str, from_cut: int = Query(alias='from', ge=1), to: int = Query(ge=1),
                  view: Literal['composition'] | None = None,
                  include: str = Query('', description='Composition view: comma-separated initial_frame,last_frame,video_analysis,subtitle_ocr,consistency_summary,dialogue_segments.'),
                  fields: str = Query('', description='Composition view: comma-separated cut-record keys; identity and readiness metadata are retained.'),
                  only_ready: bool = False, updated_since: str | None = Query(None, description='Inclusive ISO 8601 change watermark with timezone.')):
    """Return inclusive, one-based active cuts. Missing structured analysis requires a new visual analysis run."""
    if view == 'composition':
        from .composition_brief import make_composition_brief
        result = make_composition_brief(episode_id, from_cut, to, include=include, fields=fields, only_ready=only_ready, updated_since=updated_since)
        CompositionBrief.model_validate(result)
        return JSONResponse(result)
    if include or fields or only_ready or updated_since:
        raise HTTPException(422, 'Selection and incremental options require view=composition')
    from .brief import make_brief
    return make_brief(episode_id, from_cut, to)


@router.post('/episodes/{episode_id}/brief/exports', response_model=BriefExportStatus, summary='Queue a brief and asset ZIP')
def create_brief_export(episode_id: str, from_cut: int = Query(alias='from', ge=1), to: int = Query(ge=1),
                        view: Literal['composition'] | None = None,
                        include: str = Query('', description='Same includes as the composition brief; last_frame also adds the final JPEG.'),
                        fields: str = Query('', description='Same cut-record selection as the composition brief.'),
                        only_ready: bool = False, updated_since: str | None = Query(None, description='Inclusive ISO 8601 change watermark with timezone.')):
    """Snapshot the brief and queue an export; poll the returned job status before downloading."""
    import uuid
    from . import worker
    from .brief import make_brief
    if view != 'composition' and (include or fields or only_ready or updated_since):
        raise HTTPException(422, 'Selection and incremental options require view=composition')
    job_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if view == 'composition':
            from .composition_brief import make_composition_brief, selection, INCLUDES
            snapshot = make_composition_brief(episode_id, from_cut, to, include=include, fields=fields, only_ready=only_ready, updated_since=updated_since)
            CompositionBrief.model_validate(snapshot)
        else:
            snapshot = make_brief(episode_id, from_cut, to)
        first_frame_times = {cut['id']: conn.execute('SELECT first_frame_time FROM cuts WHERE id=?', (cut['id'],)).fetchone()['first_frame_time'] for cut in snapshot['cuts']}
        payload = {'brief': snapshot, 'first_frame_times': first_frame_times}
        if view == 'composition':
            assets = []
            for item in snapshot['cuts']:
                cut = dict(conn.execute('SELECT * FROM cuts WHERE id=?', (item['id'],)).fetchone())
                cut['cut_number'] = cut['cut_index'] + 1
                from . import structured_results, frame_source
                for result in structured_results.list_results(item['id'], conn=conn):
                    if result['type'] not in ('initial_frame', 'last_frame'): continue
                    key = 'analysis_' + result['type'] + '_time'
                    if key in cut: continue
                    spec = frame_source.result_frame_spec(result['id'], conn)
                    if spec and spec.get('time') is not None and cut['start_time'] <= spec['time'] < cut['end_time']:
                        cut[key] = spec['time']
                cut['has_speech'] = bool(conn.execute('SELECT 1 FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND start_time<? AND end_time>? LIMIT 1', (episode_id,cut['end_time'],cut['start_time'])).fetchone())
                assets.append(cut)
            payload.update(composition=True, asset_cuts=assets, include_last_frame='last_frame' in selection(include, INCLUDES, 'include'))
        conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, 'brief_export', json.dumps(payload)))
    worker.wake_event.set()
    return {'job_id': job_id, 'status': 'queued', 'stage': 'queued', 'error': None, 'download_url': None}


@router.get('/brief-exports/{job_id}', response_model=BriefExportStatus, summary='Get brief export status')
def brief_export_status(job_id: str):
    row = db.one("SELECT id,status,stage,error FROM jobs WHERE id=? AND pipeline='brief_export'", (job_id,))
    if not row:
        raise HTTPException(404, 'Export not found')
    return {'job_id': row['id'], 'status': row['status'], 'stage': row['stage'], 'error': row['error'], 'download_url': f'/api/v1/brief-exports/{job_id}/download' if row['status'] == 'done' else None}


@router.get('/brief-exports/{job_id}/download', summary='Download a completed brief ZIP', responses={200: {'content': {'application/zip': {'schema': {'type': 'string', 'format': 'binary'}}}}})
def download_brief_export(job_id: str):
    from fastapi.responses import FileResponse
    from .brief import export_path
    job = db.one("SELECT * FROM jobs WHERE id=? AND pipeline='brief_export'", (job_id,))
    if not job:
        raise HTTPException(404, 'Export not found')
    if job['status'] != 'done':
        raise HTTPException(409, 'Export is not complete')
    episode = main_api().require_episode(job['episode_id'])
    path = export_path(episode, job_id)
    if not path.is_file():
        raise HTTPException(404, 'Export file unavailable')
    return FileResponse(path, media_type='application/zip', filename=f'framewise-brief-{job_id}.zip')
