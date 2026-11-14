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

from . import db, media, prompt_context, structured_results

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


class CutMetadata(BaseModel):
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


class Cut(CutMetadata):
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


class SubtitleLine(BaseModel):
    id: str
    start_time: float = Field(description="Episode-relative start in seconds, when the subtitle appears on screen.")
    end_time: float = Field(description="Episode-relative end in seconds, when it disappears. Times resolve to about one eighth of a second.")
    text: str = Field(description="The burned-in subtitle exactly as read, in the episode's single subtitle language; displayed lines are separated by a newline.")
    confidence: Literal["high", "low"] = Field(description="high when every character was clear; low when some characters were hard to read.")
    crosses_cut: bool | None = Field(default=None, description="Only in cut context: true when the line starts before or ends after the cut.")


class SubtitleStrip(BaseModel):
    top: float = Field(description="Top of the subtitle strip as a fraction of the frame height.")
    bottom: float = Field(description="Bottom of the subtitle strip as a fraction of the frame height.")


class CutSubtitles(BaseModel):
    extracted: bool = Field(description="False until subtitles were extracted for the episode; then empty lines mean no subtitle overlaps this cut.")
    language: str | None = Field(description="The language of every burned-in subtitle in this episode, as stated by the user; null before any extraction.")
    strip: SubtitleStrip | None = Field(description="Where the subtitles sit in the frame; null before any extraction.")
    lines: list[SubtitleLine] = Field(description="Current subtitle lines overlapping this cut. They belong to the episode, not the cut, and may extend beyond its boundaries.")


class SubtitleRun(BaseModel):
    id: str
    language: str
    language_code: str | None
    status: str = Field(description="complete for the current run; failed or running for others in history.")
    band: dict[str, Any] = Field(description="Detected subtitle strip and how it was found.")
    diagnostics: dict[str, Any] = Field(description="Scan and reading counts, timings and rejected examples.")
    model_id: str | None
    model_revision: str | None
    error: str | None
    created_at: str
    finished_at: str | None
    line_count: int | None = None


class EpisodeSubtitles(BaseModel):
    language: str | None = Field(description="Saved subtitle language for this episode.")
    run: SubtitleRun | None = Field(description="The current completed extraction, or null.")
    lines: list[SubtitleLine] = Field(description="All lines of the current run in time order.")
    runs: list[SubtitleRun] | None = Field(default=None, description="Every run, newest first; only with history=true.")


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


class CutDetailProject(BaseModel):
    id: str
    title: str
    created_at: str


class CutDetail(BaseModel):
    project: CutDetailProject
    cut: Cut
    cut_summary: str | None = Field(description="Text of the current Cut summary, or null if unavailable.")
    first_frame: dict[str, Any] | None = Field(description="Content of the current structured initial-frame analysis, or null if unavailable.")
    last_frame: dict[str, Any] | None = Field(description="Content of the structured last-frame analysis. Last frames are no longer analyzed: this is null except for cuts analyzed before that change, and it is superseded by `motion` when that is present.")
    motion: dict[str, Any] | None = Field(description="Motion direction from the current video analysis: summary, camera, per-character activity, travel, motion, speaking, emotion and end, plus timed events (seconds from the first frame). Current runs detect characters only in the first frame; historical results may also include end_state.in_frame and in_last_frame from last-frame character boxes. Null for cuts whose video analysis predates motion directions.")
    transcript_segments: list[Transcript] = Field(description="Current non-deleted segments overlapping this cut; may extend beyond its boundaries.")
    subtitles: CutSubtitles = Field(description="Burned-in subtitle lines overlapping this cut, from the episode-wide extraction.")
    context_items: list[ContextItem] = Field(description="Project-wide and this episode's context items, including characters.")
    previous_cut: CutMetadata | None
    next_cut: CutMetadata | None


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
    subtitles: CutSubtitles = Field(description="Burned-in subtitle lines overlapping the cut, from the episode-wide extraction.")


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


@router.get("/episodes/{episode_id}/subtitles", response_model=EpisodeSubtitles, summary="Get the episode's burned-in subtitle track")
def episode_subtitles(episode_id: str, history: bool = False):
    """Subtitles extracted from the entire episode as time-ranged lines in one language. Not tied to any cut."""
    main = main_api()
    main.require_episode(episode_id)
    return main.episode_subtitles(episode_id, history)


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
    project_id = context["project"]["id"]
    summary = db.one("SELECT content FROM analysis_results WHERE cut_id=? AND title='Cut summary' AND superseded_by IS NULL ORDER BY rowid DESC LIMIT 1", (cut_id,))
    frames = structured_results.list_results(cut_id)
    def frame_content(kind):
        return next((item['content'] for item in frames if item['type'] == kind), None)
    return {"project": main.require_project(project_id), "cut": context["cut"],
            "cut_summary": summary['content'] if summary else None,
            "first_frame": frame_content('initial_frame'), "last_frame": frame_content('last_frame'),
            "motion": prompt_context.video_view(frame_content('video_analysis')) if prompt_context.is_direction(frame_content('video_analysis')) else None,
            "transcript_segments": context["transcript_segments"], "subtitles": context["subtitles"], "context_items": context["context_items"],
            "previous_cut": context["previous_cut"], "next_cut": context["next_cut"]}


def _wav_stream(path: Path, start_frame: int, frame_count: int, header: bytes, padding_frames: int, channels: int, width: int):
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
    silence = (b'\x80' if width == 1 else b'\x00' * width) * channels
    while padding_frames:
        count = min(padding_frames, 65536)
        yield silence * count
        padding_frames -= count


@router.get("/cuts/{cut_id}/audio", summary="Download cut audio with optional margins", response_class=StreamingResponse,
            responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
                             "headers": {"X-Audio-Start-Seconds": {"description": "Actual sample-aligned episode start time in seconds.", "schema": {"type": "number"}},
                                         "X-Audio-End-Seconds": {"description": "Actual sample-aligned episode end time in seconds.", "schema": {"type": "number"}}},
                             "description": "PCM WAV; X-Audio-Start-Seconds and X-Audio-End-Seconds report the source excerpt boundaries, excluding any silence padding."},
                       404: {"description": "Cut, episode audio, or prepared audio not found."},
                       409: {"description": "The cut has no samples in the available prepared audio, or the prepared audio is invalid."},
                       413: {"description": "Excerpt exceeds the WAV format size limit."}})
def cut_audio(cut_id: str,
              before: Annotated[float, Query(ge=0, le=3600, allow_inf_nan=False, description="Seconds to include before the cut start; clipped to zero.")] = 0,
              after: Annotated[float, Query(ge=0, le=3600, allow_inf_nan=False, description="Seconds to include after the cut end; clipped to available audio and episode duration.")] = 0,
              min_duration: Annotated[float, Query(ge=0, le=3600, allow_inf_nan=False, description="Minimum output duration in seconds. Extend after then before the excerpt; pad the end with silence if source audio is too short.")] = 0):
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
            minimum_frames = math.ceil(min_duration * rate)
            needed = max(0, minimum_frames - (end - start))
            available_end = min(source.getnframes(), math.ceil(episode['duration'] * rate)) if episode['duration'] is not None else source.getnframes()
            extra_after = min(needed, max(0, available_end - end))
            end += extra_after
            needed -= extra_after
            extra_before = min(needed, start)
            start -= extra_before
            padding_frames = needed - extra_before
            length = (end - start + padding_frames) * source.getnchannels() * source.getsampwidth()
            if length > 0xFFFFFFFF - 36:
                raise HTTPException(413, "Excerpt exceeds the WAV format size limit")
            channels, width = source.getnchannels(), source.getsampwidth()
            block_align = channels * width
            header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + length, b"WAVE", b"fmt ",
                                 16, 1, channels, rate, rate * block_align, block_align, width * 8, b"data", length)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(409, "Prepared audio is invalid") from exc
    return StreamingResponse(_wav_stream(path, start, end - start, header, padding_frames, channels, width), media_type="audio/wav", headers={
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
    camera: Literal['static', 'pan', 'tilt', 'dolly', 'zoom', 'handheld', 'tracking'] | None = None
    direction: str | None = None
    speed: Literal['slow', 'medium', 'fast'] | None = None
    camera_detail: str | None = Field(None, description='What the moving camera follows and how the framing ends, from the motion direction.')


class CompositionMotion(CompositionModel):
    """One character's part of the cut's motion direction (video analysis)."""
    activity: str | None = Field(None, description='What the body does: standing, sitting, walking, running ... Several values joined by " → " when it changes.')
    travel: str | None = Field(None, description='Movement through the picture (left, right, toward camera, exits right ...); omitted when the character stays in place.')
    motion: str | None = Field(None, description='Visible body movement, or "holds still".')
    speaking: Literal['yes', 'no', 'unclear'] | None = Field(None, description='Whether the mouth visibly moves as in speech at any point of the cut.')
    emotion: str | None = Field(None, description='Emotion over the cut and how it changes.')
    end: str | None = Field(None, description='Where and how the character is in the final frames.')


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
    emotion: str | None = Field(None, description='Emotion the opening frame shows, with its strength.')
    outfit_visible: str | None = None
    motion: CompositionMotion | None = Field(None, description='What this character does during the cut; present when the cut has a motion direction.')

    @field_validator('facing', 'pose', 'hands', 'gaze', 'expression', 'emotion', 'outfit_visible')
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
    motion: CompositionMotion | None = Field(None, description='Motion of someone who is not an opening subject, for example a character who enters during the cut.')

    @field_validator('facing', 'pose', 'hands', 'gaze', 'expression')
    @classmethod
    def short_phrase(cls, value):
        if value is not None and len(value.split()) > 12:
            raise ValueError('End-state phrases must contain at most 12 words')
        return value


class CompositionEndState(CompositionModel):
    subjects: list[CompositionEndSubject] | None = None
    size: Literal['ECU', 'CU', 'MCU', 'MS', 'MWS', 'WS', 'EWS'] | None = None


class CompositionCut(CompositionModel):
    id: str
    cut_number: int
    ready: bool
    duration: float | None = Field(None, description='Cut duration in seconds; beat and utterance times are relative to its start.')
    start_time: float | None = None
    end_time: float | None = None
    initial_frame_id: str | None = None
    last_frame_id: str | None = Field(None, description='Only for cuts analyzed before last-frame analysis was removed.')
    shot: CompositionShot | None = None
    setting: str | None = None
    lighting: str | None = None
    mood: str | None = Field(None, description='Emotional tone of the opening frame.')
    props: str | None = Field(None, description='Objects that matter in the opening frame and who holds them.')
    action: str | None = Field(None, description='What happens in the cut, from the motion direction.')
    background: str | None = Field(None, description='Unmarked background people in the opening frame: count, outfits, where they stand.')
    subjects: list[CompositionSubject] | None = None
    beats: list[CompositionBeat] | None = None
    end_state: CompositionEndState | None = None
    dialogue: list[CompositionUtterance] | None = None
    subtitles: list[CompositionSubtitle] | None = None
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
    kind = 'sequence' if db.one("SELECT json_extract(payload_json,'$.kind') AS kind FROM jobs WHERE id=?", (job_id,))['kind'] == 'sequence' else 'brief'
    return FileResponse(path, media_type='application/zip', filename=f'framewise-{kind}-{job_id}.zip')


# ------------------------------------------------------------------ sequences

class SequenceSummary(BaseModel):
    id: str
    episode_id: str
    title: str
    instruction: str = Field(description="The user's intent for this short video, given to the director model.")
    start_time: float = Field(description='Episode-relative start of the first cut, in seconds.')
    end_time: float
    duration: float = Field(description='Length in seconds; at most 180.')
    first_cut: int | None = Field(description='Number of the first cut (cut_index + 1) as the cuts are now.')
    last_cut: int | None
    cuts_changed: bool = Field(description='True when the cuts were re-detected or edited after the sequence was last directed; direct it again.')
    cuts_missing: bool = Field(description='True when the first or last cut was archived.')
    has_bible: bool
    has_plan: bool
    has_review: bool
    shots_planned: int
    shots_written: int
    pending_jobs: int
    created_at: str
    updated_at: str


class SequenceCut(BaseModel):
    id: str
    cut_number: int
    start_time: float
    end_time: float


class SequenceReference(BaseModel):
    label: str = Field(description='The label the prompt uses: <Picture N> or <Audio 1>.')
    role: Literal['initial_frame', 'character_sheet', 'scenario', 'original_audio'] = Field(description='What to attach under this label: the shot initial frame, the sheet of the named character, the picture of the scenario, or the shot audio.')
    character_context_id: str | None = Field(default=None, description='For character_sheet: the character context item.')
    context_id: str | None = Field(default=None, description='For scenario: the scenario context item.')
    name: str | None = None


class SequenceSubject(BaseModel):
    number: int
    label: str = Field(description='<Subject N> as written in the prompt.')
    name: str
    kind: Literal['character', 'person', 'scenario'] = Field(description='character has a record; person is someone without one; scenario is the place.')
    character_context_id: str | None = None
    context_id: str | None = None
    picture: int | None = Field(default=None, description='Number of the <Picture N> that carries this subject, or null when it is only seen in the initial frame.')
    anchors: str | None = None


class SequenceLine(BaseModel):
    line: str = Field(description='Line id on the sequence, for example L7; a line split by a shot boundary appears in both shots with its own words.')
    t0: float = Field(description='Start in seconds from the start of the shot.')
    t1: float
    text: str = Field(description='The transcript words of this shot, verbatim.')
    speaker_label: str | None = None
    character_context_id: str | None = None
    character_name: str | None = None
    attribution: Literal['reviewed', 'transcript', 'director', 'unknown'] | None = Field(default=None, description='reviewed: the user reviewed the dialogue; transcript: the user linked the line; director: the sequence bible decided the voice; unknown: nobody did.')
    speaker: str | None = Field(default=None, description='The speaker as the shot\'s voices name it: the character name, Narrator, or the voice label.')
    delivery: Literal['on_screen', 'off_screen', 'inner_voice', 'narrative'] | None = Field(default=None, description="How the shot plays the line: its speaker's voice range at the middle of the line (see the shot's voices). Null until the shot is written.")
    subject: int | None = Field(default=None, description='Subject number that carries the voice: lip-synced (on_screen), with the mouth still (inner_voice), or far away (narrative).')
    fixed_by_user: bool | None = Field(default=None, description='True when the user decided this delivery.')
    crosses_start: bool | None = None
    crosses_end: bool | None = None


class SequenceVoice(BaseModel):
    speaker: str = Field(description='The character name, Narrator, or the voice label of an unresolved voice.')
    start: float = Field(alias='from', description='Second of the shot from which this delivery holds.')
    to: float = Field(description='Second of the shot at which it ends (the next range of the speaker, or the end of the shot).')
    delivery: Literal['on_screen', 'off_screen', 'inner_voice', 'narrative'] = Field(
        description='on_screen: visible with the mouth visible, lip-synced. off_screen: not in the picture. inner_voice: visible, the mouth does not move for the words. narrative: too far for lip sync; the words play over the picture.')
    subject: str | None = Field(default=None, description='Name of the subject who carries the voice.')
    subject_number: int | None = Field(default=None, description='Its <Subject N> number in the prompt.')
    fixed_by_user: bool = Field(default=False, description='True when the user decided this range in the Shots tab.')


class SequenceFixedVoice(BaseModel):
    speaker: str
    start: float = Field(alias='from', description='Second of the shot.')
    to: float
    delivery: Literal['on_screen', 'off_screen', 'inner_voice', 'narrative']


class SequenceFrame(BaseModel):
    cut_id: str
    cut_number: int
    time: float = Field(description='Episode time of the frame, in seconds.')
    strategy: Literal['cut_first_frame', 'frame_at_time'] = Field(description='cut_first_frame: the analysed first frame of that cut. frame_at_time: a later moment of the cut (the next part of a cut longer than one take, or just after a phrase the previous shot finished across the hard cut); its description is derived, not observed.')
    continues_previous_shot: bool = Field(description='True for the next part of a cut longer than one take (the previous shot showed the same cut); the last generated frame of that shot may serve as the initial frame instead.')
    description: str | None = Field(description='Who is where in the opening still.')
    still: dict[str, Any] | None = Field(description='The opening still as shot, subjects, layout, setting and lighting.')
    still_prompt: str | None = Field(description='A generator-agnostic prompt for the still image.')
    frame_url: str
    file: str = Field(description='Name of the JPEG in the export ZIP.')


class SequenceAudio(BaseModel):
    start_time: float
    end_time: float
    url: str
    file: str = Field(description='Name of the WAV in the export ZIP; the prompt quotes this name for <Audio 1>.')
    has_speech: bool


class SequenceBeat(BaseModel):
    start: float
    end: float
    action: str
    hold: bool | None = Field(default=None, description='True for the hold the app adds to a shot shorter than the minimum generation length.')


class SequenceEdge(BaseModel):
    how: Literal['start', 'end', 'continuation', 'keep_cut', 'before_line', 'after_line'] = Field(
        description='start/end: the edge of the sequence. continuation: the next or previous part of the same long cut. keep_cut: exactly on the hard cut. before_line/after_line: moved off the hard cut by shift seconds into the silence before or after a phrase, so the phrase stays in one shot (the shot borrows that much of the neighbouring cut\'s audio, never its picture).')
    boundary: str | None = Field(default=None, description='Id of the hard cut (K1…).')
    shift: float | None = Field(default=None, description='Seconds the boundary moved off the hard cut (negative: earlier).')
    hard_cut: float | None = Field(default=None, description='The hard cut, in seconds from the start of the sequence.')
    keeps: list[str] | None = Field(default=None, description='Line ids the moved boundary keeps whole.')
    split: dict[str, Any] | None = Field(default=None, description='Speech the edge divides: lines, the words before it and the words after it.')
    reason: str | None = Field(default=None, description="The director's (or the user's) reason for this choice.")
    by: Literal['director', 'app', 'user'] | None = Field(default=None, description='Who chose it.')


class SequenceShot(BaseModel):
    index: int = Field(description='One-based position in the sequence.')
    key: str = Field(description='Stable id of the shot: its episode interval in milliseconds.')
    start_time: float = Field(description='Episode-relative start of the audio interval, in seconds.')
    end_time: float
    t0: float = Field(description='Start in seconds from the start of the sequence.')
    t1: float
    duration: float = Field(description='Length of the audio interval the shot covers in the finished video.')
    generate_seconds: float = Field(description='Length to generate with H3 (the last beat of the prompt ends here); longer than duration for very short shots.')
    trim_seconds: float | None = Field(description='When set, cut the generated video to this length.')
    cut: int = Field(description='Number of the cut this shot is: every cut is one shot, and its picture is the only one shown.')
    cut_id: str
    part: list[int] | None = Field(default=None, description='[part, of] for a cut longer than one take, which is divided into consecutive parts.')
    source_cuts: list[int] = Field(description='Cuts whose audio the shot plays: its own cut, plus a neighbour when a boundary moved to keep a phrase whole.')
    cut_ids: list[str]
    merged: bool = Field(description='Always false: shots are never merged.')
    starts_mid_cut: bool = Field(description='True when the shot starts after its cut starts (a later part, or after a phrase the previous shot finished).')
    ends_mid_cut: bool
    opens: SequenceEdge = Field(description='How the shot meets the previous one.')
    closes: SequenceEdge = Field(description='How the shot meets the next one.')
    scene: str | None
    intent: str
    staging: str = Field(description='Always empty; see opens and closes.')
    initial_frame: SequenceFrame
    audio: SequenceAudio
    dialogue: list[SequenceLine]
    voices: list[SequenceVoice] | None = Field(default=None, description='How each speaker is heard in this shot, as time ranges per speaker. Null until the shot is written.')
    fixed_voices: list[SequenceFixedVoice] | None = Field(default=None, description='The ranges the user decided for this shot; the director writes the shot around them.')
    subjects: list[SequenceSubject]
    references: list[SequenceReference] = Field(description='Every label of the prompt in the order the files must be attached.')
    camera: dict[str, Any] | None
    beats: list[SequenceBeat]
    soundscape: str | None
    continuity: dict[str, Any] | None
    open_questions: list[str]
    prompt: str | None = Field(description='The final MiniMax H3 prompt, from subject_definitions: to non_diegetic_music:. Null until the shot is written.')
    check: dict[str, Any] | None = Field(description='Format verdict of the prompt: ok, errors, warnings, stats.')
    written: bool
    stale: list[str] = Field(description='What changed since the shot was written: plan, bible or evidence. Empty when it is current.')
    result: dict[str, Any] | None = Field(description='Revision id, origin (model, manual_edit, assistant_edit), model and time of the shot direction.')


class SequencePackage(BaseModel):
    contract: str = Field(description='Version of this document: sequence-v1.')
    id: str
    episode_id: str
    title: str
    instruction: str
    start_time: float
    end_time: float
    duration: float
    first_cut: int | None
    last_cut: int | None
    cuts_changed: bool
    cuts_missing: bool
    created_at: str
    updated_at: str
    cuts: list[SequenceCut]
    audio_url: str = Field(description='The original audio of the whole sequence as WAV.')
    ready: bool = Field(description='True when every planned shot has a current prompt that passes the format check.')
    pending: list[int] = Field(description='Indices of shots that are not written, stale or failing the check.')
    plan_problems: list[str]
    bible: dict[str, Any] | None = Field(description='The sequence bible: logline, synopsis, style, spoken_language, scenes, characters, voices, arc, pacing, continuity, open_questions.')
    shots: list[SequenceShot]
    review: dict[str, Any] | None = Field(description='The continuity review: verdict and issues (shot, shot_key, severity, problem, suggestion).')


def _excerpt(episode_id, start, end, name, minimum=0.0):
    """Prepared episode audio between two times as a WAV download, padded with silence up to ``minimum`` seconds."""
    episode = main_api().require_episode(episode_id)
    path = media.episode_dir(episode['project_id'], episode['id']) / 'audio.wav'
    if not episode['has_audio'] or not path.is_file():
        raise HTTPException(404, 'Episode audio is not available')
    try:
        with wave.open(str(path), 'rb') as source:
            rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
            first = max(0, min(source.getnframes(), math.floor(start * rate)))
            last = min(source.getnframes(), math.ceil(end * rate))
    except (wave.Error, EOFError) as exc:
        raise HTTPException(409, 'Prepared audio is invalid') from exc
    if last <= first:
        raise HTTPException(409, 'The interval has no samples in the available audio')
    padding = max(0, math.ceil(minimum * rate) - (last - first))
    length = (last - first + padding) * channels * width
    header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + length, b'WAVE', b'fmt ', 16, 1, channels, rate, rate * channels * width, channels * width, width * 8, b'data', length)
    return StreamingResponse(_wav_stream(path, first, last - first, header, padding, channels, width), media_type='audio/wav', headers={
        'Content-Disposition': f'attachment; filename="{name}"', 'Content-Length': str(len(header) + length),
        'X-Audio-Start-Seconds': str(first / rate), 'X-Audio-End-Seconds': str(last / rate)})


def _package(sequence_id):
    from . import sequence_view
    package = sequence_view.package(sequence_id)
    if package is None:
        raise HTTPException(404, 'Sequence not found')
    return package


def _shot(package, index):
    shot = next((item for item in package['shots'] if item['index'] == index), None)
    if shot is None:
        raise HTTPException(404, 'That shot is not part of the current plan')
    return shot


@router.get('/episodes/{episode_id}/sequences', response_model=list[SequenceSummary], summary="List an episode's sequences")
def episode_sequences(episode_id: str):
    """Sequences are ranges of cuts that were directed as one short video on the original audio."""
    from . import sequences
    return sequences.list_sequences(episode_id)


@router.get('/sequences/{sequence_id}', response_model=SequencePackage, summary='Get a directed sequence: bible, shots and final H3 prompts')
def sequence_package(sequence_id: str):
    """The current state of the sequence. Check `ready` before generating; `pending` lists the shots that still need writing."""
    return _package(sequence_id)


@router.post('/sequences/{sequence_id}/exports', response_model=BriefExportStatus, summary='Queue a sequence ZIP with audio, initial frames and prompts')
def create_sequence_export(sequence_id: str):
    """Snapshot the package and queue the export; poll `/api/v1/brief-exports/{job_id}` and download from there."""
    import uuid
    from . import worker
    package = _package(sequence_id)
    if not package['shots']:
        raise HTTPException(409, 'This sequence has no shot plan yet')
    job_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,'brief_export',?)", (job_id, package['episode_id'], json.dumps({'kind': 'sequence', 'sequence': package}, ensure_ascii=False)))
    worker.wake_event.set()
    return {'job_id': job_id, 'status': 'queued', 'stage': 'queued', 'error': None, 'download_url': None}


AUDIO_RESPONSE = {200: {'content': {'audio/wav': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'PCM WAV of the prepared episode audio (mono 16 kHz).'},
                  404: {'description': 'Sequence, shot or prepared audio not found.'}}


@router.get('/sequences/{sequence_id}/audio', summary='Download the original audio of a sequence', response_class=StreamingResponse, responses=AUDIO_RESPONSE)
def sequence_audio(sequence_id: str):
    package = _package(sequence_id)
    return _excerpt(package['episode_id'], package['start_time'], package['end_time'], f'sequence-{sequence_id}.wav')


@router.get('/sequences/{sequence_id}/shots/{index}/audio', summary="Download a shot's slice of the original audio", response_class=StreamingResponse, responses=AUDIO_RESPONSE)
def sequence_shot_audio(sequence_id: str, index: int,
                        pad: Annotated[bool, Query(description='Pad the end with silence up to generate_seconds, for shots shorter than the minimum generation length.')] = False):
    package = _package(sequence_id)
    shot = _shot(package, index)
    return _excerpt(package['episode_id'], shot['start_time'], shot['end_time'], shot['audio']['file'], shot['generate_seconds'] if pad else 0.0)


@router.get('/sequences/{sequence_id}/shots/{index}/frame', summary="Download a shot's initial frame at the original resolution", response_class=StreamingResponse,
            responses={200: {'content': {'image/jpeg': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'The frame of the original video at initial_frame.time.'}})
async def sequence_shot_frame(sequence_id: str, index: int):
    from . import sequences
    spec, episode_id = sequences.shot_frame_spec(sequence_id, index=index, width=None)
    return await main_api().rendered_frame(spec, None, episode_id)
