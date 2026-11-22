import { api, jsonBody } from './api'
import type { DiffSegment } from './assistantApi'
import type { BibleOutfit } from './outfitView'
import { subscribeEpisodeEvents } from './episodeEvents'

/** A shot boundary the user placed on the Dialogue timeline: the hard cut after `after_cut_id`, at `time` (episode seconds). */
export type ShotPoint = { after_cut_id: string; time: number }
/** A range of cuts directed as one short video on the original audio. */
export type SequenceInfo = {
  id: string; episode_id: string; title: string; instruction: string; start_time: number; end_time: number; duration: number
  first_cut: number | null; last_cut: number | null; cuts_changed: boolean; cuts_missing: boolean; created_at: string; updated_at: string
  /** Boundaries the user placed; they are final for the director. */
  shot_points?: ShotPoint[]
  /** Cuts added at the end that still wait for the direction to be redone; lines from `from_time` (episode seconds) on are new. */
  extension?: SequenceExtension | null
}
export type SequenceExtension = { cut_ids: string[]; cuts: number[]; from_time: number; rewrite_shots: boolean }
/** The cut after the sequence: whether it can be added and the words that run on into it. */
export type NextCut = {
  id: string; number: number; start_time: number; end_time: number; seconds: number; fits: boolean; ready: boolean
  speech: { text_before: string; words_after: string } | null
}
export type DialogueState = 'none' | 'running' | 'review' | 'confirmed'
export type SequenceSummary = SequenceInfo & { dialogue?: DialogueState; has_bible: boolean; has_plan: boolean; has_review: boolean; shots_planned: number; shots_written: number; pending_jobs: number }
export type SequenceCut = { id: string; number: number; t0: number; t1: number; start_time: number; end_time: number; ready: boolean }
/** Where a shot boundary sits at a hard cut: on the cut, or in the silence just before or after the phrase the cut falls into. */
export type BoundaryChoice = 'keep_cut' | 'before_line' | 'after_line'
export type Decider = 'director' | 'app' | 'user'
/** One possible position; `t` is in seconds from the start of the sequence, `time` in episode seconds, `shift` relative to the hard cut. */
export type BoundaryOption = { t: number; time: number; shift: number; drops?: { cut: number; from: number; to: number } }
export type BoundarySpeech = {
  kind: 'inside_line' | 'between_sentence_halves'; lines: string[]; words_before: string; words_after: string; keeps: string[]; phrase: [number, number]
  speaker_label: string; character_name: string | null; character_context_id: string | null
}
/** A hard cut of the sequence: its options and, with a usable plan, the choice made there. `fixed` is where the user placed it, if they did. */
export type SequenceBoundary = {
  id: string; after_cut: number; before_cut: number; after_cut_id: string; before_cut_id: string; t: number; time: number
  speech: BoundarySpeech | null; options: Partial<Record<BoundaryChoice, BoundaryOption>>; unavailable: Partial<Record<BoundaryChoice, string>>
  fixed?: BoundaryOption | null
  choice: BoundaryChoice | null; reason: string | null; by: Decider | null
}
/** How a shot meets its neighbour on one side. */
export type ShotEdge = {
  how: 'start' | 'end' | 'continuation' | BoundaryChoice; boundary?: string; shift?: number; hard_cut?: number; keeps?: string[]
  split?: { lines: string[]; before: string; after: string }; reason?: string; by?: Decider
  /** True when the user placed this boundary by hand on the Dialogue timeline. */
  placed?: boolean
}
export type SequenceLine = { id: string; t0: number; t1: number; text: string; speaker_label: string; character_name: string | null; character_context_id: string | null }
export type ShotLine = { line: string; t0: number; t1: number; text: string; speaker_label: string; character_context_id: string | null; character_name: string | null; crosses_start: boolean; crosses_end: boolean }
/** How a speaker is heard during part of a shot (decided per shot, never per line). */
export type Delivery = 'on_screen' | 'off_screen' | 'inner_voice' | 'narrative'
export type ShotDialogue = ShotLine & { attribution?: 'reviewed' | 'transcript' | 'director' | 'unknown'; speaker?: string; delivery?: Delivery | null; subject?: number | null; fixed_by_user?: boolean }
/** One range of a speaker's voice in a shot, in seconds from the start of the shot. */
export type ShotVoice = { speaker: string; from: number; to: number; delivery: Delivery; subject: string | null; subject_number?: number | null; fixed_by_user?: boolean; fixed?: boolean }
/** A range the user decided for a speaker in a shot (shot seconds). */
export type FixedVoice = { speaker: string; from: number; to: number; delivery: Delivery }
export type ShotCheck = { ok: boolean; errors: string[]; warnings: string[]; stats: { seconds?: number | null; chars?: number; description_words?: number; beats?: number } }
/** What the director writes for a shot; the app renders the H3 prompt from it. */
export type ShotDirection = {
  subjects: { name: string; anchors: string; keep: string }[]; picture_line: string; keep_picture: string
  still: { shot: string; subjects: string; layout: string; setting: string; lighting: string }
  summary: string; shot_line: string; framing_lock: string; light: string
  camera: { move: string; amplitude: string; speed: string; detail: string }
  beats: { end: number; action: string }[]; voices?: ShotVoice[]
  scenario_anchors: string; scenario_keep: string; soundscape?: string; continuity: { from_previous: string; to_next: string }; open_questions: string[]
  /** The keyframe the user chose as the staging reference (set by the app, kept by every later rewrite until cleared). */
  reference_frame?: ShotReferenceFrame | null
  /** The director's words for what that frame shows (who is where, poses, objects). */
  reference_line?: string
}
/** A frame of the shot's own cut: `ordinal` is the source frame index, `time` episode seconds, `at` seconds on the shot's clock, `frame` its keyframe label. */
export type ShotReferenceFrame = { cut_id: string; ordinal: number; time: number; at: number; frame?: string | null }
/** A keyframe the shot can take as its reference; `analysed` when the keyframe analysis has states for it. */
export type ShotKeyframe = { frame: string; ordinal: number; time: number; at: number; analysed: boolean }
export type ShotKeyframes = { cut_id: string; revision: string | null; status: 'current' | 'missing'; frames: ShotKeyframe[]; current: ShotReferenceFrame | null }
export type ShotReference = {
  label: string; role: 'initial_frame' | 'character_sheet' | 'scenario' | 'reference_frame' | 'original_audio'; character_context_id?: string | null; context_id?: string | null; name?: string | null
  /** reference_frame only. */
  at?: number; frame?: string | null
}
export type ShotResult = {
  id: string; content: ShotDirection; prompt: string | null; check: ShotCheck | null
  derived: { references?: ShotReference[]; dialogue?: ShotDialogue[]; voices?: ShotVoice[]; still_prompt?: string; audio_file?: string }
  origin: string; model_id: string | null; job_id: string | null; parent_id: string | null; created_at: string; instruction: string | null; completed_by_app: string | null
  generation_seconds: number | null; model_variant: string | null; decode_tokens_per_second: number | null
  /** Whose words `instruction` are: the user's (remembered by every later rewrite) or a suggestion of the automatic review (used once). Absent on older versions. */
  instruction_source?: 'user' | 'review' | null
  /** Why the director rewrote it without an instruction (`voices`: the user changed how a speaker is heard). */
  trigger?: string | null
  /** The assistant's one-line description when the version was applied from the Assistant tab. */
  assistant_summary?: string | null
  /** Where this version's reference keyframe came from: chosen for it (`user`), `kept` from the version before, `cleared`, or `dropped: why`. */
  reference_frame_source?: string | null
  /** `attached` when the director saw the reference keyframe, else why not. */
  reference_picture?: string | null
}
/** One version of a shot's prompt and what caused it; `diff` compares its prompt with the version it replaced (null for the first). */
export type ShotRevision = {
  id: string; number: number; origin: string; model_id: string | null; created_at: string; instruction: string | null; trigger: string | null
  instruction_source?: 'user' | 'review' | null
  /** How many of the user's earlier instructions this rewrite also followed. */
  earlier_instructions?: number | null
  /** Set for a change applied from the Assistant tab: what the user asked in the chat and the assistant's summary. */
  assistant: { request: string | null; summary: string; provider: string | null; model: string | null; session_id: string | null } | null
  model_variant: string | null; generation_seconds: number | null; check_ok: boolean | null; prompt: string; diff: DiffSegment[] | null
  /** The reference keyframe this version was written with, and where the choice came from (see `ShotResult.reference_frame_source`). */
  reference?: ShotReferenceFrame | null; reference_source?: string | null; reference_picture?: string | null
}
/** A change to a shot proposed in the Assistant tab that waits to be applied or discarded there. */
export type ShotProposal = { id: string; created_at: string; summary: string; request: string | null; provider: string | null; model: string | null; stale: boolean; prompt: string; diff: DiffSegment[] }
/** `remembered` are the user's instructions of earlier rewrites of the shot, oldest first: every later rewrite keeps following them, the later one winning where two differ. */
export type ShotHistory = {
  revisions: ShotRevision[]; remembered: string[]; proposals: ShotProposal[]
  /** `reference` is the keyframe chosen with the queued rewrite (`{clear: true}` when it clears it); absent when it keeps the current one. */
  pending: { status: 'queued' | 'running'; instruction: string | null; trigger: string | null; reference?: { cut_id?: string; ordinal?: number; time?: number; clear?: boolean } | null } | null
}
export type StaleReason = 'plan' | 'bible' | 'evidence'
export type SequenceShot = {
  index: number; key: string; t0: number; t1: number; start_time: number; end_time: number; duration: number; generate_seconds: number; trim_seconds: number | null
  /** The cut this shot is; `part` is [n, of] for a cut longer than one take. `source_cuts` are the cuts whose audio it plays. */
  cut: number; cut_id: string; part: [number, number] | null
  source_cuts: number[]; cut_ids: string[]; starts_mid_cut: boolean; ends_mid_cut: boolean
  frame: { cut: number; cut_id: string; time: number; strategy: 'cut_first_frame' | 'frame_at_time'; continues_previous_shot: boolean }
  opens: ShotEdge; closes: ShotEdge
  scene: string | null; intent: string; lines: ShotLine[]; fixed_voices: FixedVoice[]
  /** The speakers heard in the shot, named as its voices name them. */
  speakers: string[]
  result: ShotResult | null; stale: StaleReason[]; job: 'queued' | 'running' | null
}
export type SequenceResult<T> = { id: string; content: T; origin: string; model_id: string | null; job_id: string | null; parent_id: string | null; created_at: string; completed_by_app: string | null; generation_seconds?: number | null; stale_reason?: string | null }
export type Bible = {
  logline: string; synopsis: string; spoken_language: string; arc: string; pacing: string
  scenes: { id: string; from_cut: number; to_cut: number; location: string; time_of_day: string; lighting: string; mood: string; scenario: string | null }[]
  characters: { name: string; character_context_id: string; role: string; anchors: string; outfit: string; arc: string }[]
  /** The named outfits of each character (the full descriptions live in the project's outfit library). Bibles written before 2026-11-21 have none. */
  outfits?: BibleOutfit[]
  voices: { speaker_label: string; character: string | null; kind: 'character' | 'narrator' | 'unknown'; confidence: 'high' | 'medium' | 'low'; source: 'transcript' | 'director' | 'reviewed'; reason: string }[]
  continuity: string[]; open_questions: string[]
}
export type Plan = {
  shots: { start_time: number; end_time: number; frame_cut_id: string | null; scene: string | null; intent: string }[]; notes: string
  /** The choice at each hard cut that falls inside speech. */
  boundaries?: { after_cut_id: string; choice: BoundaryChoice; reason: string; by: Decider }[]
  /** The user's voice ranges, in episode seconds. */
  voices?: { speaker: string; from: number; to: number; delivery: Delivery }[]
}
export type ReviewIssue = { shot: number; shot_key: string; severity: 'fix' | 'check'; problem: string; suggestion: string; dismissed?: boolean }
export type Review = { verdict: string; issues: ReviewIssue[] }
export type SequenceStep = 'compile' | 'bible' | 'plan' | 'shot' | 'review'
export type SequenceJob = { id: string; status: string; stage: string; error: string | null; created_at: string; started_at: string | null; finished_at: string | null; step: SequenceStep; shot_key: string | null }
/** One reviewed line of the dialogue; times are seconds from the start of the sequence. */
export type CompiledLine = {
  id: string; t0: number; t1: number; text: string; original_text: string; subtitle_text: string | null
  speaker_label: string; character: string | null; character_context_id: string | null
  continues_next: boolean; confidence: 'high' | 'medium' | 'low'; note: string
  source_ids: string[]; subtitle_ids: string[]; parts: { t0: number; t1: number; text: string }[]
}
export type CompiledVoice = { speaker_label: string; character: string | null; character_context_id: string | null; kind: 'character' | 'narrator' | 'unknown'; confidence: 'high' | 'medium' | 'low'; reason: string }
export type Compilation = { lines: CompiledLine[]; voices: CompiledVoice[]; open_questions: string[] }
export type DialogueView = {
  state: DialogueState; result: SequenceResult<Compilation> | null; confirmed: boolean; stale: boolean; has_speech: boolean
  source: { id: string; t0: number; t1: number; text: string; speaker_label: string }[]
  subtitles: { id: string; t0: number; t1: number; text: string }[]
  characters?: { id: string; name: string }[]
  shot_starts?: (ShotPoint & { reason: string; by: Decider; speech: boolean })[]
}
export type SequenceDetail = {
  sequence: SequenceInfo; compilation: DialogueView; cuts: SequenceCut[]; missing: number[]
  timeline: { duration: number; pauses: [number, number][]; boundaries: SequenceBoundary[]; lines: SequenceLine[] } | null
  bible: SequenceResult<Bible> | null; plan: SequenceResult<Plan> | null; plan_problems: string[]
  shots: SequenceShot[]; review: SequenceResult<Review> | null; ready: boolean; jobs: SequenceJob[]; next_cut?: NextCut | null
  limits: { sequence_seconds: number; shot_seconds: number; min_generate_seconds: number }
}
export type RunStep = 'compile' | 'direct' | 'bible' | 'plan' | 'shots' | 'review'
type ExportStatus = { job_id: string; status: string; stage: string; error: string | null; download_url: string | null }

export const sequenceApi = {
  list: (episodeId: string) => api<SequenceSummary[]>(`/episodes/${episodeId}/sequences`),
  create: (episodeId: string, body: { title: string; from_cut: number; to_cut: number; instruction: string }) => api<SequenceDetail>(`/episodes/${episodeId}/sequences`, jsonBody(body)),
  get: (id: string) => api<SequenceDetail>(`/sequences/${id}`),
  update: (id: string, body: { title?: string; instruction?: string }) => api<SequenceDetail>(`/sequences/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  remove: (id: string) => api<{ ok: boolean }>(`/sequences/${id}`, { method: 'DELETE' }),
  dismissReviewIssue: (resultId: string, index: number) => api<SequenceDetail>(`/sequence-results/${resultId}/review-issues/${index}/dismiss`, { method: 'POST' }),
  /** `reference_frame` (one shot only): a keyframe for the prompt's staging, or null to clear it; left out, the shot keeps its own. */
  run: (id: string, body: { step: RunStep; shot_keys?: string[]; only_missing?: boolean; instruction?: string; instruction_source?: 'user' | 'review'; reference_frame?: { ordinal: number } | null }) =>
    api<{ queued: number; job_ids: string[] }>(`/sequences/${id}/run`, jsonBody(body)),
  /** The keyframes a shot can take as its staging reference, and the one its current prompt uses. */
  shotKeyframes: (id: string, shotKey: string) => api<ShotKeyframes>(`/sequences/${id}/shots/${encodeURIComponent(shotKey)}/keyframes`),
  cancel: (id: string) => api<{ cancelled: number }>(`/sequences/${id}/cancel`, { method: 'POST' }),
  /** Add the episode's next cut to the end; its dialogue is compiled and appended, then the whole direction follows. */
  extend: (id: string) => api<SequenceDetail>(`/sequences/${id}/extend`, { method: 'POST' }),
  /** Every version of a shot's prompt, newest first, with the rewrite in progress and the Assistant's unapplied proposals. */
  shotHistory: (id: string, shotKey: string) => api<ShotHistory>(`/sequences/${id}/shots/${encodeURIComponent(shotKey)}/history`),
  revise: (resultId: string, content: unknown) => api<SequenceDetail>(`/sequence-results/${resultId}`, { method: 'PUT', body: JSON.stringify({ content }) }),
  saveDialogue: (id: string, body: { result_id: string; content?: Compilation; points?: ShotPoint[]; confirm?: boolean }) =>
    api<SequenceDetail>(`/sequences/${id}/compilation`, { method: 'PUT', body: JSON.stringify(body) }),
  useSourceDialogue: (id: string) => api<SequenceDetail>(`/sequences/${id}/compilation/use-source`, { method: 'POST' }),
  /** Choose where the shot boundary sits at the hard cut after a cut; the two shots beside it must be written again. */
  setBoundary: (id: string, afterCutId: string, choice: BoundaryChoice) =>
    api<SequenceDetail>(`/sequences/${id}/boundaries`, { method: 'PUT', body: JSON.stringify({ after_cut_id: afterCutId, choice }) }),
  /** Save the shot boundaries placed on the Dialogue timeline (all of them; an empty list hands every boundary back to the director). */
  setShotPoints: (id: string, points: ShotPoint[]) => api<SequenceDetail>(`/sequences/${id}/shot-points`, { method: 'PUT', body: JSON.stringify({ points }) }),
  /** Decide how one speaker is heard in one shot (null: the director decides); the shot is rewritten around it. */
  setVoices: (id: string, shotKey: string, speaker: string, segments: { from: number; delivery: Delivery }[] | null) =>
    api<SequenceDetail>(`/sequences/${id}/voices`, { method: 'PUT', body: JSON.stringify({ shot_key: shotKey, speaker, segments }) }),
  /** Keep the written prompts and clear their stale marks (one shot, or every written shot when no key is given). */
  markReady: (id: string, shotKey?: string) => api<SequenceDetail>(`/sequences/${id}/mark-ready`, { method: 'POST', body: JSON.stringify({ shot_key: shotKey ?? null }) }),
  /** Queue the ZIP (package, audio, initial frames, prompts) and wait until it can be downloaded; returns its URL. */
  async exportZip(id: string, episodeId: string, onStage?: (stage: string) => void) {
    const queued = await api<ExportStatus>(`/v1/sequences/${id}/exports`, { method: 'POST' })
    const status = await new Promise<ExportStatus>((resolve, reject) => {
      let refreshing = false
      let changed = false
      async function refresh() {
        changed = true
        if (refreshing) return
        refreshing = true
        try {
          while (changed) {
            changed = false
            const current = await api<ExportStatus>(`/v1/brief-exports/${queued.job_id}`)
            if (current.status !== 'queued' && current.status !== 'running') {
              unsubscribe()
              resolve(current)
              break
            }
            onStage?.(current.stage)
          }
        } catch (error) {
          unsubscribe()
          reject(error)
        } finally {
          refreshing = false
        }
      }
      const unsubscribe = subscribeEpisodeEvents(episodeId, change => { if (change.jobs) void refresh() })
      void refresh()
    })
    if (status.status !== 'done' || !status.download_url) throw new Error(status.error || 'The export failed')
    return status.download_url
  },
}

export const shotFrameUrl = (sequenceId: string, shotKey: string) => `/api/sequences/${sequenceId}/shots/${shotKey}/frame`
