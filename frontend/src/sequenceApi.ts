import { api, jsonBody } from './api'

/** A range of cuts directed as one short video on the original audio. */
export type SequenceInfo = {
  id: string; episode_id: string; title: string; instruction: string; start_time: number; end_time: number; duration: number
  first_cut: number | null; last_cut: number | null; cuts_changed: boolean; cuts_missing: boolean; created_at: string; updated_at: string
}
export type SequenceSummary = SequenceInfo & { has_bible: boolean; has_plan: boolean; has_review: boolean; shots_planned: number; shots_written: number; pending_jobs: number }
export type SequenceCut = { id: string; number: number; t0: number; t1: number; start_time: number; end_time: number; ready: boolean }
/** A place where a shot may begin or end; `t` is in seconds from the start of the sequence, `time` in episode seconds. */
export type SequenceCandidate = { id: string; t: number; time: number; kind: 'start' | 'cut' | 'pause' | 'event' | 'quiet' | 'end'; cut: number | null; pause?: number; crosses_line?: string }
export type SequenceLine = { id: string; t0: number; t1: number; text: string; speaker_label: string; character_name: string | null; character_context_id: string | null }
export type ShotLine = { line: string; t0: number; t1: number; text: string; speaker_label: string; character_context_id: string | null; character_name: string | null; crosses_start: boolean; crosses_end: boolean }
export type Delivery = 'lip_sync' | 'off_screen' | 'inner_voice' | 'narration'
export type ShotDialogue = ShotLine & { attribution?: 'transcript' | 'director' | 'unknown'; delivery?: Delivery | null; subject?: number | null }
export type ShotCheck = { ok: boolean; errors: string[]; warnings: string[]; stats: { seconds?: number | null; chars?: number; description_words?: number; beats?: number } }
/** What the director writes for a shot; the app renders the H3 prompt from it. */
export type ShotDirection = {
  subjects: { name: string; anchors: string; keep: string }[]; picture_line: string; keep_picture: string
  still: { shot: string; subjects: string; layout: string; setting: string; lighting: string }
  summary: string; shot_line: string; framing_lock: string; light: string
  camera: { move: string; amplitude: string; speed: string; detail: string }
  beats: { end: number; action: string }[]; speech: { line: string; delivery: Delivery; subject: string | null }[]
  scenario_anchors: string; scenario_keep: string; soundscape: string; continuity: { from_previous: string; to_next: string }; open_questions: string[]
}
export type ShotReference = { label: string; role: 'initial_frame' | 'character_sheet' | 'scenario' | 'original_audio'; character_context_id?: string | null; context_id?: string | null; name?: string | null }
export type ShotResult = {
  id: string; content: ShotDirection; prompt: string | null; check: ShotCheck | null
  derived: { references?: ShotReference[]; dialogue?: ShotDialogue[]; still_prompt?: string; audio_file?: string }
  origin: string; model_id: string | null; job_id: string | null; parent_id: string | null; created_at: string; instruction: string | null; completed_by_app: string | null
}
export type StaleReason = 'plan' | 'bible' | 'evidence'
export type SequenceShot = {
  index: number; key: string; t0: number; t1: number; start_time: number; end_time: number; duration: number; generate_seconds: number; trim_seconds: number | null
  source_cuts: number[]; cut_ids: string[]; merged: boolean; starts_mid_cut: boolean; ends_mid_cut: boolean
  frame: { cut: number; cut_id: string; time: number; strategy: 'cut_first_frame' | 'frame_at_time'; continues_previous_shot: boolean }
  scene: string | null; intent: string; staging: string; auto_fixed: boolean; lines: ShotLine[]
  result: ShotResult | null; stale: StaleReason[]; job: 'queued' | 'running' | null
}
export type SequenceResult<T> = { id: string; content: T; origin: string; model_id: string | null; job_id: string | null; parent_id: string | null; created_at: string; completed_by_app: string | null; generation_seconds?: number | null }
export type Bible = {
  logline: string; synopsis: string; style: string; spoken_language: string; arc: string; pacing: string
  scenes: { id: string; from_cut: number; to_cut: number; location: string; time_of_day: string; lighting: string; mood: string; scenario: string | null }[]
  characters: { name: string; character_context_id: string; role: string; anchors: string; outfit: string; arc: string }[]
  voices: { speaker_label: string; character: string | null; kind: 'character' | 'narrator' | 'unknown'; confidence: 'high' | 'medium' | 'low'; source: 'transcript' | 'director'; reason: string }[]
  continuity: string[]; open_questions: string[]
}
export type Plan = { shots: { start_time: number; end_time: number; frame_cut_id: string | null; scene: string | null; intent: string; staging: string; auto_fixed?: boolean }[]; notes: string }
export type ReviewIssue = { shot: number; shot_key: string; severity: 'fix' | 'check'; problem: string; suggestion: string }
export type Review = { verdict: string; issues: ReviewIssue[] }
export type SequenceStep = 'bible' | 'plan' | 'shot' | 'review'
export type SequenceJob = { id: string; status: string; stage: string; error: string | null; created_at: string; started_at: string | null; finished_at: string | null; step: SequenceStep; shot_key: string | null }
export type SequenceDetail = {
  sequence: SequenceInfo; cuts: SequenceCut[]; missing: number[]
  timeline: { duration: number; pauses: [number, number][]; candidates: SequenceCandidate[]; lines: SequenceLine[] } | null
  bible: SequenceResult<Bible> | null; plan: SequenceResult<Plan> | null; plan_problems: string[]
  shots: SequenceShot[]; review: SequenceResult<Review> | null; ready: boolean; jobs: SequenceJob[]
  limits: { sequence_seconds: number; shot_seconds: number; min_generate_seconds: number }
}
export type RunStep = 'direct' | 'bible' | 'plan' | 'shots' | 'review'
type ExportStatus = { job_id: string; status: string; stage: string; error: string | null; download_url: string | null }

export const sequenceApi = {
  list: (episodeId: string) => api<SequenceSummary[]>(`/episodes/${episodeId}/sequences`),
  create: (episodeId: string, body: { title: string; from_cut: number; to_cut: number; instruction: string }) => api<SequenceDetail>(`/episodes/${episodeId}/sequences`, jsonBody(body)),
  get: (id: string) => api<SequenceDetail>(`/sequences/${id}`),
  update: (id: string, body: { title?: string; instruction?: string }) => api<SequenceDetail>(`/sequences/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  remove: (id: string) => api<{ ok: boolean }>(`/sequences/${id}`, { method: 'DELETE' }),
  run: (id: string, body: { step: RunStep; shot_keys?: string[]; only_missing?: boolean; instruction?: string }) => api<{ queued: number; job_ids: string[] }>(`/sequences/${id}/run`, jsonBody(body)),
  cancel: (id: string) => api<{ cancelled: number }>(`/sequences/${id}/cancel`, { method: 'POST' }),
  revise: (resultId: string, content: unknown) => api<SequenceDetail>(`/sequence-results/${resultId}`, { method: 'PUT', body: JSON.stringify({ content }) }),
  merge: (id: string, shotKey: string) => api<SequenceDetail>(`/sequences/${id}/plan/merge`, jsonBody({ shot_key: shotKey })),
  split: (id: string, shotKey: string, time: number) => api<SequenceDetail>(`/sequences/${id}/plan/split`, jsonBody({ shot_key: shotKey, time })),
  /** Queue the ZIP (package, audio, initial frames, prompts) and wait until it can be downloaded; returns its URL. */
  async exportZip(id: string, onStage?: (stage: string) => void) {
    let status = await api<ExportStatus>(`/v1/sequences/${id}/exports`, { method: 'POST' })
    while (status.status === 'queued' || status.status === 'running') {
      onStage?.(status.stage)
      await new Promise(resolve => setTimeout(resolve, 1000))
      status = await api<ExportStatus>(`/v1/brief-exports/${status.job_id}`)
    }
    if (status.status !== 'done' || !status.download_url) throw new Error(status.error || 'The export failed')
    return status.download_url
  },
}

export const shotFrameUrl = (sequenceId: string, shotKey: string) => `/api/sequences/${sequenceId}/shots/${shotKey}/frame`
