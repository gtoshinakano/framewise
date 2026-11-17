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
    is_main: bool = Field(False, description="Characters only: the user marked this character as a main character.")
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
    status: str = Field(description='queued, running, done, failed or cancelled (a cancelled job is never a failure).')
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
    status: str = Field(description='queued, running, done, failed or cancelled.')
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


class SequenceModel(BaseModel):
    """Keys are left out when they have no value; the documented ones are the only ones."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class SequenceCut(SequenceModel):
    number: int = Field(description='Cut number (cut_index + 1) as the cuts are now.')
    id: str
    at: float = Field(description='Start on the sequence clock, in seconds.')
    duration: float


class SequenceSpeaker(SequenceModel):
    name: str = Field(description="The name the shots' voices use: the character, Narrator, the diarization label of an unresolved voice, or Unknown speaker.")
    kind: Literal['character', 'narrator', 'voice', 'unknown'] = Field(description='character has a record; voice is a diarization label nobody resolved; unknown has no label at all.')
    character_context_id: str | None = None
    speaker_labels: list[str] | None = Field(default=None, description="Diarization labels of this speaker's lines.")


class SequenceIssue(SequenceModel):
    shots: list[int] | None = Field(default=None, description='The shots it concerns (an issue that reads the same in several shots is listed once); absent for the whole sequence.')
    line: str | None = Field(default=None, description='The line it concerns (L7).')
    severity: Literal['error', 'warning', 'info'] = Field(description='error: fix it in Framewise before generating. warning: decide or look before generating. info: how to generate.')
    kind: Literal['review', 'check_error', 'check_warning', 'open_question', 'unknown_speaker', 'derived_frame', 'padded_shot', 'no_sheet', 'unlinked_subject',
                  'missing_reference', 'stale', 'not_written', 'cuts_changed', 'cuts_missing', 'plan_problem', 'template_unavailable']
    text: str


class SequenceRevision(SequenceModel):
    id: str = Field(description='Result id of this revision.')
    origin: str = Field(description='model, manual_edit, assistant_edit, or source (the dialogue used as transcribed).')
    model_id: str | None = None
    created_at: str
    revision: int = Field(description='Position in its revision chain; 1 is the first version.')
    generation_seconds: float | None = None


class SequenceProvenance(SequenceModel):
    bible: SequenceRevision | None = None
    plan: SequenceRevision | None = None
    compilation: SequenceRevision | None = None
    review: SequenceRevision | None = None
    shots: dict[str, SequenceRevision] | None = Field(default=None, description='Each written shot by its index.')


class SequenceSubtitleState(SequenceModel):
    language: str | None = Field(default=None, description="The burned-in subtitles' language, as the user stated it.")
    extracted: bool = Field(description='False when the episode has no extracted burned-in subtitles; shots then have no subtitles.')


class SequenceEdge(SequenceModel):
    how: Literal['start', 'continuation', 'keep_cut', 'before_line', 'after_line'] = Field(
        description="start: the sequence starts. continuation: the next part of the same long cut. keep_cut: exactly on the hard cut. before_line/after_line: moved off the hard cut by shift seconds "
                    "into the silence before or after a phrase, so it stays in one shot (the shot borrows that much of the neighbouring cut's audio, never its picture). "
                    "With placed, the user put the boundary there by hand (earlier or later than the hard cut, even inside speech: see splits).")
    boundary: str | None = Field(default=None, description='Id of the hard cut (K1…).')
    shift: float | None = Field(default=None, description='Seconds the boundary moved off the hard cut (negative: earlier); absent when it sits on the cut.')
    keeps: list[str] | None = Field(default=None, description='Line ids the moved boundary keeps whole.')
    splits: list[str] | None = Field(default=None, description='Line ids the edge divides: the words stop in one shot and go on in the next.')
    reason: str | None = Field(default=None, description="The director's or the user's reason for the choice.")
    by: Literal['director', 'app', 'user'] | None = None
    placed: bool | None = Field(default=None, description='True when the user placed this boundary by hand on the Dialogue timeline; it is final. Absent otherwise.')


class SequenceBeat(SequenceModel):
    start: float
    end: float
    action: str | None = None
    hold: bool | None = Field(default=None, description='The hold the app adds to a shot shorter than 5 s: everyone keeps the final pose, the camera stays.')


class SequenceContinuity(SequenceModel):
    from_previous: str | None = Field(default=None, description='What must match the end of the previous shot.')
    to_next: str | None = Field(default=None, description='Full view only: the state this shot hands over.')


class SequenceFrame(SequenceModel):
    description: str | None = Field(default=None, description="Who is where in the opening frame, as the prompt's <Picture 1> line says it.")
    description_source: Literal['observed', 'derived'] = Field(
        description="observed: the analysed first frame of the shot's cut. derived: a later moment of the cut that was never analysed; the director inferred it from the first frame and the motion.")
    derived_from: str | None = Field(default=None, description='For a derived description: what it was inferred from.')
    continues_previous_shot: bool | None = Field(default=None, description='The next part of a long cut: the last generated frame of the previous shot may serve as the opening image instead.')
    at: float | None = Field(default=None, description='Full view only: where the frame lies in the source, in seconds from the start of the shot.')


class SequenceVoice(SequenceModel):
    speaker: str = Field(description='Speaker id (SP1…) in speakers.')
    start: float = Field(alias='from', description='Second of the shot from which this delivery holds.')
    to: float
    delivery: Literal['on_screen', 'off_screen', 'inner_voice', 'narrative'] = Field(
        description='on_screen: visible with the mouth visible, lip-synced. off_screen: not in the picture. inner_voice: visible, the mouth does not move for the words. narrative: too far for lip sync; the words play over the picture.')
    subject: int | None = Field(default=None, description='<Subject N> that carries the voice.')
    fixed: bool | None = Field(default=None, description='The user decided this range in Framewise.')


class SequenceCandidate(SequenceModel):
    character_context_id: str
    name: str
    confidence: Literal['low', 'medium'] = Field(description='A heuristic, never high: Framewise keeps no voice-match score.')
    reason: str


class SequenceLine(SequenceModel):
    id: str = Field(description='Line id on the sequence (L7); a line divided by a shot edge appears in both shots with its own words.')
    speaker: str = Field(description='Speaker id (SP1…) in speakers.')
    t0: float
    t1: float
    text: str = Field(description='The words of this shot, verbatim from the reviewed dialogue or the transcript.')
    by: Literal['reviewed', 'transcript', 'director', 'unknown'] = Field(
        description='Who decided the speaker. reviewed: the user reviewed the dialogue. transcript: the user linked the line. director: the sequence bible. unknown: nobody.')
    crosses_start: bool | None = None
    crosses_end: bool | None = None
    candidate: SequenceCandidate | None = Field(default=None, description='For a line nobody attributed: advice only; the stored decision stays unknown.')


class SequenceSubject(SequenceModel):
    n: int = Field(description='<Subject n> in the prompt.')
    name: str = Field(description='The name the prompt uses.')
    kind: Literal['character', 'person', 'object', 'place', 'unknown', 'scenario'] = Field(description='character has a record; scenario is the place of a scenario note.')
    character_context_id: str | None = None
    context_id: str | None = Field(default=None, description='For scenario: the scenario context item.')
    picture: int | None = Field(default=None, description='<Picture N> that carries this subject; absent when it is only seen in the initial frame.')
    linked_by: Literal['prompt', 'name'] | None = Field(default=None, description='prompt: the prompt itself names the record. name: the app matched the name afterwards; the prompt gives it no sheet.')
    record_name: str | None = Field(default=None, description='The record name, when the prompt calls the character something shorter.')
    kind_guessed: bool | None = Field(default=None, description="The kind is the app's guess from the opening frame's analysis.")
    anchors: str | None = Field(default=None, description='Full view only.')
    keep: str | None = Field(default=None, description='Full view only.')


class SequencePicture(SequenceModel):
    role: Literal['initial_frame', 'character_sheet', 'scenario'] = Field(description='Entry N is <Picture N>.')
    subject: int | None = Field(default=None, description='The subject whose sheet or scenario picture this is.')


class SequenceShotSubtitle(SequenceModel):
    t0: float
    t1: float
    text: str
    crosses_shot: bool | None = Field(default=None, description='The subtitle starts before or ends after this shot.')


class SequenceCheck(SequenceModel):
    ok: bool
    errors: list[str] | None = None
    warnings: list[str] | None = None
    stats: dict[str, Any] | None = Field(default=None, description='Full view only.')


class SequenceShot(SequenceModel):
    index: int = Field(description='One-based position; files are named shot-NN.')
    key: str | None = Field(default=None, description='Full view only: stable id, the episode interval in milliseconds.')
    cut: int | None = Field(default=None, description='The cut this shot is; its picture is the only one shown.')
    part: list[int] | None = Field(default=None, description='[part, of] for a cut longer than one take.')
    at: float | None = Field(default=None, description='Start on the sequence clock; episode time = start_time + at + shot seconds.')
    duration: float | None = Field(default=None, description='Length in the finished video.')
    generate_seconds: float | None = Field(default=None, description='Only for shots shorter than 5 s: the length to generate (the prompt ends on a hold); trim the result to duration.')
    source_cuts: list[int] | None = Field(default=None, description='Only when the shot plays a neighbour\'s audio: every cut whose audio it plays.')
    opens: SequenceEdge | None = Field(default=None, description='Only in a one-shot slice: how the shot meets the previous one (the previous shot\'s closes).')
    closes: SequenceEdge | None = Field(default=None, description='How the shot meets the next one; absent for the last shot.')
    scene: str | None = None
    intent: str | None = None
    camera: dict[str, Any] | None = Field(default=None, description='move, and amplitude, speed and detail when it moves.')
    beats: list[SequenceBeat] | None = None
    soundscape: str | None = Field(default=None, description='Full view only, and only on shots written before 2026-11-15: newer prompts follow <Audio 1> for all sound.')
    continuity: SequenceContinuity | None = None
    initial_frame: SequenceFrame | None = None
    voices: list[SequenceVoice] | None = Field(default=None, description='How each speaker is heard, as ranges; a line takes its speaker\'s range at its midpoint.')
    lines: list[SequenceLine] | None = None
    line_map: dict[str, str] | None = Field(default=None, description='Line id to the template slot 1.M (M: order by start among the lines of shot-NN.wav).')
    subjects: list[SequenceSubject] | None = None
    pictures: list[SequencePicture] | None = None
    subtitles: list[SequenceShotSubtitle] | None = None
    check: SequenceCheck | None = None
    stale: list[str] | None = Field(default=None, description='What changed since the shot was written: plan, bible, evidence.')
    written: bool | None = Field(default=None, description='Present (false) only while the shot has no direction.')
    prompt: str | None = Field(default=None, description='Full view or include=prompts: the final H3 prompt (also shot-NN-prompt.txt).')
    prompt_template: str | None = Field(default=None, description='Full view or include=prompts: the prompt with {language}, {line 1.M}, {start 1.M}, {end 1.M} placeholders.')
    still_prompt: str | None = Field(default=None, description='Full view or include=prompts: a prompt for the opening still (also shot-NN-still.txt).')
    still: dict[str, Any] | None = Field(default=None, description='Full view only: the opening still as shot, subjects, layout, setting and lighting.')


class SequencePackage(SequenceModel):
    contract: Literal['sequence-v2']
    view: Literal['lean', 'full']
    id: str
    episode_id: str
    title: str
    instruction: str | None = None
    start_time: float = Field(description='Episode time of the sequence start; the only episode offset in the document.')
    duration: float
    created_at: str
    updated_at: str
    ready: bool = Field(description='True when every planned shot has a current prompt that passes the format check.')
    pending: list[int] | None = Field(default=None, description='Shots that are not written, stale or failing the check.')
    cuts_changed: bool | None = None
    cuts_missing: bool | None = None
    plan_problems: list[str] | None = None
    cuts: list[SequenceCut] | None = None
    speakers: dict[str, SequenceSpeaker] | None = Field(default=None, description='Speaker ids used by voices and lines.')
    bible: dict[str, Any] | None = Field(default=None, description='logline, synopsis, spoken_language, arc, pacing, scenes, characters, voices, continuity (its open questions are in issues).')
    subtitles: SequenceSubtitleState | None = None
    issues: list[SequenceIssue] | None = Field(default=None, description='Everything to look at or decide, most severe first.')
    review: dict[str, Any] | None = Field(default=None, description="The continuity review's verdict (its issues are in issues).")
    provenance: SequenceProvenance | None = None
    shots: list[SequenceShot] | None = None


class SequenceDigestIssue(SequenceModel):
    shots: list[int] | None = Field(default=None, description='The shots with this same issue; absent for the whole sequence.')
    line: str | None = None
    severity: Literal['error', 'warning', 'info']
    kind: str
    text: str = Field(description='Shortened; the lean view has the full text.')


class SequenceDigestShot(SequenceModel):
    n: int
    cut: str = Field(description='Cut number, with part/of for a long cut.')
    at: float
    dur: float
    gen: float | None = Field(default=None, description='Generated length when longer than dur.')
    closes: str = Field(description='How the shot meets the next one (the next shot opens with it): how, hard cut, shift, lines it splits; end for the last shot.')
    frame: Literal['observed', 'derived']
    voices: list[str] | None = Field(default=None, description='speaker, delivery (on, off, inner, over) and range; (user) when the user decided it.')
    lines: list[str] | None = Field(default=None, description='Line id, speaker (? candidate), first words.')
    refs: str | None = Field(default=None, description='Pictures the prompt attaches besides <Picture 1> (the initial frame) and <Audio 1> (shot-NN.wav).')
    state: str | None = Field(default=None, description='Present only when the shot is not ready: not written, stale: …, or check errors.')


class SequenceDigest(SequenceModel):
    contract: Literal['sequence-v2']
    view: Literal['digest']
    id: str
    title: str
    duration: float
    cuts: str | None = None
    ready: bool
    pending: list[int] | None = None
    cuts_changed: bool | None = None
    cuts_missing: bool | None = None
    plan_problems: list[str] | None = None
    bible: dict[str, Any] | None = None
    subtitles: SequenceSubtitleState | None = None
    issues: list[SequenceDigestIssue] | None = None
    shots: list[SequenceDigestShot] | None = None


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


def _sequences():
    from . import sequence_package
    return sequence_package


def _document(sequence_id):
    found = _sequences().document(sequence_id)
    if found is None:
        raise HTTPException(404, 'Sequence not found')
    return found


def _package_call(function, *args):
    packages = _sequences()
    try:
        return function(*args)
    except packages.PackageError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@router.get('/episodes/{episode_id}/sequences', response_model=list[SequenceSummary], summary="List an episode's sequences")
def episode_sequences(episode_id: str):
    """Sequences are ranges of cuts that were directed as one short video on the original audio."""
    from . import sequences
    return sequences.list_sequences(episode_id)


@router.get('/sequences/{sequence_id}', response_model=SequencePackage | SequenceDigest, summary='Get a directed sequence (contract sequence-v2): digest, lean or full view, or one shot',
            responses={404: {'description': 'Sequence, or the shot asked for, not found.'}, 422: {'description': 'Unknown view, include or field.'}})
def sequence_package(sequence_id: str,
                     view: Annotated[Literal['lean', 'full', 'digest'], Query(description='digest: about 5 KB to read first. lean (default): no prompt texts. full: everything.')] = 'lean',
                     shot: Annotated[int | None, Query(ge=1, description="One shot (by index) with its speakers, issues and opening edge; the bible is left out.")] = None,
                     fields: Annotated[str, Query(description='Comma-separated shot keys to keep (index is always kept), for example lines,voices,initial_frame.')] = '',
                     include: Annotated[str, Query(description='prompts: add prompt, prompt_template and still_prompt to the lean view.')] = ''):
    """Check `ready` and `issues` before generating. Keys without a value are left out; shot times are seconds from the start of the shot."""
    full, _ = _document(sequence_id)
    result = _package_call(_sequences().project, full, view, include, shot, fields)
    (SequenceDigest if view == 'digest' else SequencePackage).model_validate(result)
    return JSONResponse(result)


@router.post('/sequences/{sequence_id}/exports', response_model=BriefExportStatus, summary='Queue a sequence ZIP: digest, lean JSON, audio, frames, strips and prompts',
             responses={409: {'description': 'The sequence has no shot plan yet.'}, 422: {'description': 'Unknown include.'}})
def create_sequence_export(sequence_id: str,
                           include: Annotated[str, Query(description='Comma-separated optional files: sequence_audio (sequence.wav), references (character crops, scene frames, voice samples), clips (silent source clip per shot).')] = ''):
    """Snapshot the package and queue the export; poll `/api/v1/brief-exports/{job_id}` and download from there."""
    import uuid
    from datetime import datetime, timezone
    from . import worker
    packages = _sequences()
    includes = _package_call(packages.selection, include, packages.EXPORT_INCLUDES, 'include')
    full, assets = _document(sequence_id)
    if not full.get('shots'):
        raise HTTPException(409, 'This sequence has no shot plan yet')
    job_id = str(uuid.uuid4())
    payload = {'kind': 'sequence', 'contract': packages.CONTRACT, 'sequence': full, 'assets': assets, 'include': sorted(includes),
               'queued_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    with db.connect() as conn:
        conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,'brief_export',?)", (job_id, full['episode_id'], json.dumps(payload, ensure_ascii=False)))
    worker.wake_event.set()
    return {'job_id': job_id, 'status': 'queued', 'stage': 'queued', 'error': None, 'download_url': None}


AUDIO_RESPONSE = {200: {'content': {'audio/wav': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'PCM WAV of the prepared episode audio (mono 16 kHz).'},
                  404: {'description': 'Sequence, shot or prepared audio not found.'}}


@router.get('/sequences/{sequence_id}/audio', summary='Download the original audio of a sequence', response_class=StreamingResponse, responses=AUDIO_RESPONSE)
def sequence_audio(sequence_id: str):
    interval = _package_call(_sequences().sequence_interval, sequence_id)
    return _excerpt(interval['episode_id'], interval['start_time'], interval['end_time'], f'sequence-{sequence_id}.wav')


@router.get('/sequences/{sequence_id}/shots/{index}/audio', summary="Download a shot's slice of the original audio", response_class=StreamingResponse, responses=AUDIO_RESPONSE)
def sequence_shot_audio(sequence_id: str, index: int,
                        pad: Annotated[bool, Query(description="For shots shorter than the minimum generation length: continue with the original audio after the shot up to generate_seconds (silence only where the episode audio ends).")] = False):
    spec = _package_call(_sequences().shot_spec, sequence_id, index)
    end = max(spec['end_time'], spec['start_time'] + spec['generate_seconds']) if pad else spec['end_time']
    return _excerpt(spec['episode_id'], spec['start_time'], end, spec['audio_file'], spec['generate_seconds'] if pad else 0.0)


@router.get('/sequences/{sequence_id}/shots/{index}/frame', summary="Download a shot's initial frame at the original resolution", response_class=StreamingResponse,
            responses={200: {'content': {'image/jpeg': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'The frame of the original video at initial_frame.time.'}})
async def sequence_shot_frame(sequence_id: str, index: int):
    from . import sequences
    spec, episode_id = sequences.shot_frame_spec(sequence_id, index=index, width=None)
    return await main_api().rendered_frame(spec, None, episode_id)
