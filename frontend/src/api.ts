export type AnalysisConfig = { frame_mode: 'endpoints' | 'uniform'; frame_count: number; context_ids: string[]; detection_queries: string[]; additional_context: string; openrouter_model: string }
export type Settings = { proxy_width: number; proxy_crf: number; proxy_preset: 'ultrafast' | 'superfast' | 'veryfast' | 'faster' | 'fast' | 'medium'; frame_interval: number; scene_threshold: number; chunk_duration: number }
export type Cut = { id: string; cut_index: number; active: boolean; analysis_enabled: boolean; start_time: number; end_time: number; first_frame_time: number; last_frame_time: number; title: string; description: string; notes: string; results: Record<string, unknown> }
export type ResultKind = 'first_frame' | 'camera_movement' | 'composition' | 'last_frame' | 'cut_description' | 'audio_visual_match' | 'generation_prompt' | 'other'
export type AnalysisResult = { id: string; cut_id: string; kind: ResultKind; title: string; content: string; origin: string; model_id: string | null; job_id: string | null; prompt: string | null; settings: Record<string, unknown>; context: Record<string, unknown>; parent_id: string | null; superseded_by: string | null; created_at: string }
export type TranscriptSegment = { id: string; episode_id: string; start_time: number; end_time: number; text: string; speaker_label: string; character_context_id: string | null; deleted: boolean; origin: string; model_id: string | null; parent_id: string | null; superseded_by: string | null; created_at: string; updated_at: string }
export type AnalysisContext = { project: { id: string; title: string; synopsis: string }; episode: { id: string; title: string; duration: number | null }; cut: Cut; previous_cut: Cut | null; next_cut: Cut | null; neighbor_results: { previous: Pick<AnalysisResult, 'kind' | 'title' | 'content' | 'origin' | 'model_id'>[]; next: Pick<AnalysisResult, 'kind' | 'title' | 'content' | 'origin' | 'model_id'>[] }; chunks: Chunk[]; context_items: ContextItem[]; transcript_segments: TranscriptSegment[] }
export type Chunk = { id: string; chunk_index: number; start_time: number; end_time: number; title: string; notes: string }
export type Job = { id: string; pipeline: string; status: string; stage: string; error: string | null; created_at: string; updated_at: string; started_at: string | null; finished_at: string | null; retry_of: string | null }
export type EpisodeStatus = { status: string; stage: string; error: string | null; job_id: string | null; job_status: string | null }
export type EpisodeSummary = { id: string; title: string; status: string; stage: string; error: string | null; duration: number | null; created_at: string }
export type Episode = EpisodeSummary & { project_id: string; original_filename: string; width: number | null; height: number | null; has_audio: boolean; proxy_ready: boolean; settings: Settings; cuts: Cut[]; archived_cuts: Cut[]; chunks: Chunk[]; jobs: Job[] }
export type Project = { id: string; title: string; synopsis: string; created_at: string; episodes: EpisodeSummary[] }
export type ContextItem = { id: string; project_id: string; episode_id: string | null; kind: 'chapter_synopsis' | 'chapter_happenings' | 'character' | 'world' | 'scenario' | 'object' | 'general'; title: string; content: string; created_at: string; updated_at: string }
export type Waveform = { peaks: number[]; duration: number }

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
