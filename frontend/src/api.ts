import type { QueueSummary } from './queueView'
export type AnalysisConfig = { frame_mode: 'per_second' | 'endpoints' | 'uniform'; frame_count: number; video_width: number; video_height: number; video_fps: number; video_frame_budget: number | null; context_ids: string[]; detection_queries: string[]; additional_context: string; text_provider: 'local' | 'openrouter'; local_thinking: boolean; openrouter_model: string }
export type Settings = { proxy_width: number; proxy_crf: number; proxy_preset: 'ultrafast' | 'superfast' | 'veryfast' | 'faster' | 'fast' | 'medium'; frame_interval: number; scene_threshold: number; chunk_duration: number }
export type Cut = { id: string; cut_index: number; active: boolean; analysis_enabled: boolean; no_characters?: boolean; has_analysis_results?: boolean; has_summary?: boolean; has_boxes?: boolean; has_characters?: boolean; has_unlabeled_boxes?: boolean; has_target_boxes?: boolean; has_unlabeled_target_boxes?: boolean; has_visual_results?: boolean; has_consistency_summary?: boolean; last_job_failed?: boolean; start_time: number; end_time: number; first_frame_time: number; last_frame_time: number; title: string; description: string; notes: string; results: Record<string, unknown> }
export type ResultKind = 'first_frame' | 'camera_movement' | 'composition' | 'last_frame' | 'cut_description' | 'audio_visual_match' | 'generation_prompt' | 'other'
export type AnalysisResult = { id: string; cut_id: string; kind: ResultKind; title: string; content: string; origin: string; model_id: string | null; job_id: string | null; prompt: string | null; settings: Record<string, unknown>; context: Record<string, unknown>; parent_id: string | null; superseded_by: string | null; created_at: string; stale_reason?: string | null }
export type TranscriptSegment = { id: string; episode_id: string; start_time: number; end_time: number; text: string; speaker_label: string; character_context_id: string | null; deleted: boolean; origin: string; model_id: string | null; parent_id: string | null; superseded_by: string | null; created_at: string; updated_at: string }
export type AnalysisContext = { project: { id: string; title: string; synopsis: string }; episode: { id: string; title: string; duration: number | null }; cut: Cut; previous_cut: Cut | null; next_cut: Cut | null; neighbor_results: { previous: Pick<AnalysisResult, 'kind' | 'title' | 'content' | 'origin' | 'model_id'>[]; next: Pick<AnalysisResult, 'kind' | 'title' | 'content' | 'origin' | 'model_id'>[] }; chunks: Chunk[]; context_items: ContextItem[]; transcript_segments: TranscriptSegment[]; subtitles: CutSubtitles }
export type SubtitleLine = { id: string; start_time: number; end_time: number; text: string; confidence: 'high' | 'low'; crosses_cut?: boolean }
export type SubtitleRun = { id: string; language: string; language_code: string | null; status: string; band: { top?: number; bottom?: number; source?: string; support?: number }; diagnostics: { frames?: number; events?: number; lines?: number; warning?: string; readings?: { text: number; empty: number; rejected: number }; timings?: Record<string, number> }; error: string | null; created_at: string; finished_at: string | null; line_count: number | null; stale_reason?: string | null }
export type CutSubtitles = { extracted: boolean; language: string | null; strip: { top: number; bottom: number } | null; lines: SubtitleLine[] }
export type EpisodeSubtitles = { language: string | null; run: SubtitleRun | null; lines: SubtitleLine[] }
export type Chunk = { id: string; chunk_index: number; start_time: number; end_time: number; title: string; notes: string }
export type JobCut = { id: string; cut_index: number | null; title: string; start_time: number | null; end_time: number | null; completed: boolean }
export type VisualStage = 'initial_frame' | 'keyframe_analysis'
// Jobs queued before subtitles became an episode-wide step, or before the last frame stopped being analyzed, may still list these stages in their history.
export type JobStage = VisualStage | 'video_analysis' | 'subtitle_ocr' | 'last_frame'  // the last three only on jobs queued before they were retired
// What the queue may do with a job (backend job_kinds): cancel it now, or retry it after it failed or was cancelled.
export type JobActions = { cancel: boolean; retry: boolean }
// `cancelled` is its own status since 2026-11-15; older cancels were `failed` with the stage `Cancelled`.
export type Job = { actions?: JobActions; cuts: JobCut[]; id: string; pipeline: string; workflow?: string | null; stages?: JobStage[] | null; sequence?: { id: string; step: string; shot_key: string | null } | null; status: string; stage: string; error: string | null; created_at: string; updated_at: string; started_at: string | null; finished_at: string | null; retry_of: string | null }
export type EpisodeStatus = { status: string; stage: string; error: string | null; job_id: string | null; job_status: string | null; queue: QueueSummary }
export type EpisodeSummary = { id: string; title: string; status: string; stage: string; error: string | null; duration: number | null; created_at: string }
export type Episode = EpisodeSummary & { project_id: string; original_filename: string; width: number | null; height: number | null; has_audio: boolean; proxy_ready: boolean; settings: Settings; subtitle_language: string | null; subtitles: { language: string | null; run: SubtitleRun | null }; cuts: Cut[]; archived_cuts: Cut[]; chunks: Chunk[]; jobs: Job[]; cut_history?: CutHistory }
// The latest cut edit Undo would reverse and the one Redo would apply again (backend cut_history.py).
export type CutEditEntry = { id: string; label: string; action: string; created_at: string }
export type CutHistory = { undo: CutEditEntry | null; redo: CutEditEntry | null }
export type Project = { id: string; title: string; synopsis: string; created_at: string; episodes: EpisodeSummary[] }
export type ContextItem = { id: string; project_id: string; episode_id: string | null; kind: 'chapter_synopsis' | 'chapter_happenings' | 'character' | 'world' | 'scenario' | 'object' | 'general'; title: string; content: string; is_main?: number; created_at: string; updated_at: string }
export type Waveform = { peaks: number[]; duration: number }
// A part of the audio picked on the timeline or in the Audio tab, in seconds.
export type AudioSelection = { start: number; end: number }

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, headers: { ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options?.headers } })
  if (!response.ok) {
    let detail = response.statusText
    try { detail = (await response.json()).detail || detail } catch { /* keep status text */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return response.json()
}

export function upload<T>(path: string, body: FormData, onProgress: (value: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', `/api${path}`)
    request.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded / event.total) }
    request.onload = () => {
      try {
        const result = JSON.parse(request.responseText)
        if (request.status < 200 || request.status >= 300) reject(new Error(result.detail || request.statusText))
        else resolve(result)
      } catch (error) { reject(error) }
    }
    request.onerror = () => reject(new Error('Upload failed. Check the server connection.'))
    request.send(body)
  })
}

export const mediaUrl = (episodeId: string, name: string) => `/api/episodes/${episodeId}/media/${name}`
export const jsonBody = (value: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(value) })

/** One character in one keyframe (keyframes-v1): the vision model's state and, when DWPose read the body, the measured one. */
export type KeyframeCharacter = { name: string; character_context_id: string | null; state: string | null; pose: string | null }
export type KeyframeChange = { part: string; description: string; source: 'dwpose' | 'qwen' }
export type KeyframeSegment = {
  from: string; to: string; start: number; end: number; camera: { move: string; detail: string }
  characters: { name: string; character_context_id: string | null; walking: string; speaking: string; blinking: string; changes: KeyframeChange[] }[]
  other_changes: { kind: string; description: string }[]
}
export type KeyframeAnalysis = {
  id: string; contract: 'keyframes-v1'; method: 'hybrid' | 'qwen'; selection_key: string; summary: string
  frames: { label: string; ordinal: number; t: number; reason: string | null; characters: KeyframeCharacter[] }[]
  segments: KeyframeSegment[]
}
/** What the Target frames tab reads (GET /cuts/{id}/keyframe-analysis). */
export type KeyframeAnalysisState = {
  status: 'missing' | 'current' | 'stale'; selection_key: string | null; result: KeyframeAnalysis | null
  run: { id: string; phase: string; method: string; model?: string | null; roster_source: string; target_run_id: string | null; roster: { name: string }[]; timings: Record<string, number> } | null
  job: { id: string; status: string; stage: string; error: string | null } | null
}
export type BeatSpeech = { line: string; speaker: string; start: number; end: number; delivery: 'on_screen' | 'off_screen' | 'inner_voice' | 'narrative'; suggested: string | null }
export type PerformanceBeats = {
  contract: 'beats-v1'; duration: number; summary: string; based_on: { keyframe_analysis: string; initial_frame: string | null }
  beats: { start: number; end: number; camera: { move: string; detail: string }; intent: string
           characters: { character: string; body: string; face: string; gaze: string; mouth: string }[]; speech: BeatSpeech[] }[]
  speech: BeatSpeech[]; checks: { kind: string; [key: string]: unknown }[]
}
