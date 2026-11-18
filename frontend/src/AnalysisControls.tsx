import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, ArrowRight, ChevronDown, ChevronsRight, Film, Image as ImageIcon, Layers, Play, RotateCcw, Save, SlidersHorizontal } from 'lucide-react'
import { api, type AnalysisConfig, type AnalysisResult, type Cut, type Job, type VisualStage } from './api'
import { CharacterDetectionPanel } from './CharacterDetectionPanel'
import { TargetFrames, useTargetDetections, useTargetSelection } from './TargetFrames'
import { isJobActive, targetQueueMessage, unknownSlots, type TargetQueueResult } from './targetView'
import { FrameImage } from './FrameImage'
import { automaticVideoFps, estimatedVideoFrames } from './analysisFrames'
import { ANSWERS_BATCH_REASONS, CHAIN_BATCH_REASONS, SUMMARY_BATCH_REASONS, batchQueueMessage, type BatchQueueResult as TextBatchResult } from './batchQueueMessage'
import { ActionButton, Elapsed, NoticeStrip, Skeleton, StateChip, StatusChip, WarningLine, type Notice } from './RunUi'
import { cachedConfig, cachedProjectId, loadConfig, loadProjectId, prefetchConfigs, storeConfig } from './runCache'
import { STEP_LABELS, VISUAL_STEPS, activeRunFor, batchCounts, cutStepStates, hasVideoErrors, preflight, queueNotice, videoSettingsErrors, videoSettingsSummary, type CharacterSummary, type StageRow, type StepKey, type StepState } from './runState'

type BatchQueueResult = { queued?: number; skipped?: number; blocked?: { unknown_characters: number[]; not_detected: number[] } }
type Optimistic = Partial<Record<StepKey, StepState>>
type RunKind = 'cut' | 'next' | 'batch' | 'from-here' | 'redo-all'

function cutList(numbers: number[]) {
  return numbers.slice(0, 8).join(', ') + (numbers.length > 8 ? ` and ${numbers.length - 8} more` : '')
}

/** Says what was queued and, for the cuts that were held back, which ones need a person first. */
function fromHereMessage(firstCut: number, result: BatchQueueResult) {
  const unknown = result.blocked?.unknown_characters ?? []
  const undetected = result.blocked?.not_detected ?? []
  const parts = [`Queued ${result.queued ?? 0} cuts from cut ${firstCut}; ${result.skipped ?? 0} already complete were skipped.`]
  if (unknown.length) parts.push(`Not queued, unknown characters: cut ${cutList(unknown)}.`)
  if (undetected.length) parts.push(`Not queued, characters not detected yet: cut ${cutList(undetected)}.`)
  return parts.join(' ')
}

const STAGE_BUTTONS: { stage: VisualStage; label: string; icon: ReactNode; hint: string }[] = [
  { stage: 'initial_frame', label: 'Initial frame', icon: <ImageIcon size={13} />, hint: 'Describe only the first frame: the still image, with each character\'s look, pose, expression and emotion' },
  { stage: 'video_analysis', label: 'Video', icon: <Film size={13} />, hint: 'Analyze only the motion: camera, what each character does, who speaks, emotion and how the cut ends. Uses the first-frame result to tell characters apart, so run the initial frame first' },
]

/** One state for a group of steps: anything running or queued wins, then failures, then complete or not run. */
function overall(states: StepState[]): StepState {
  if (states.includes('running')) return 'running'
  if (states.includes('queued')) return 'queued'
  if (states.includes('failed')) return 'failed'
  return states.every(state => state === 'done') ? 'done' : 'none'
}

function NumberField({ label, unit, value, error, min, max, step, onChange }: { label: string; unit: string; value: number; error?: string; min: number; max: number; step: number | 'any'; onChange: (value: number) => void }) {
  return <label className="block min-w-0"><span className="muted text-[11px]">{label} · {unit}</span>
    <input className={`field !px-2.5 !py-1.5 text-sm ${error ? '!border-[#8a3b46]' : ''}`} type="number" min={min} max={max} step={step} value={Number.isFinite(value) ? value : ''} aria-invalid={!!error} onChange={event => onChange(event.target.valueAsNumber)} />
    <span className={`mt-0.5 block min-h-4 text-[11px] ${error ? 'text-[#ff9fa7]' : 'text-transparent'}`} role={error ? 'alert' : undefined}>{error || '.'}</span></label>
}

type Props = {
  jobs: Job[]; cut: Cut; cuts: Cut[]; episodeId: string; busy: boolean; enabled: boolean
  results: AnalysisResult[]; stages: StageRow[]; prefetchIds: (string | undefined)[]
  charactersTarget: HTMLElement | null
  onStarted: () => Promise<void>; onConfigChange: (config: AnalysisConfig | null) => void
  onPrevious?: () => void; onNext?: () => void; children?: ReactNode
}

export default memo(function AnalysisControls({ cut, cuts, episodeId, busy, enabled, results, stages, prefetchIds, charactersTarget, onStarted, onConfigChange, onNext, jobs, children }: Props) {
  const cutId = cut.id
  const targetFrames = useTargetSelection(cutId, `${cut.start_time}:${cut.end_time}`)
  const targetPeople = useTargetDetections(cutId, `${cut.start_time}:${cut.end_time}`, targetFrames.selection?.revision || '', () => void onStarted())
  const [config, setConfig] = useState<AnalysisConfig | null>(() => cachedConfig(cutId) ?? null)
  const [saved, setSaved] = useState(() => { const known = cachedConfig(cutId); return known ? JSON.stringify(known) : '' })
  const [loadError, setLoadError] = useState('')
  const [projectId, setProjectId] = useState(() => cachedProjectId(episodeId) ?? '')
  const [pending, setPending] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [optimistic, setOptimistic] = useState<Optimistic>({})
  const [characters, setCharacters] = useState<CharacterSummary | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [characterTab, setCharacterTab] = useState<'characters' | 'targets'>('characters')
  const [boundaryDraft, setBoundaryDraft] = useState<Partial<Record<'first' | 'last', { times: number[]; index: number; original: number }>>>({})
  const [boundaryLoading, setBoundaryLoading] = useState<'first' | 'last' | null>(null)
  const boundaryBase = useRef<{ start: number; end: number } | null>(null)
  const [boundaryError, setBoundaryError] = useState('')
  const [movingBoundary, setMovingBoundary] = useState(false)
  const configRef = useRef(config); configRef.current = config
  const savedRef = useRef(saved); savedRef.current = saved
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const previewRequests = useRef(new Map<string, Promise<{ times: number[]; anchor: number }>>())
  const getBoundaryPreview = useCallback((position: 'first' | 'last') => {
    const key = `${cutId}:${cut.start_time}:${cut.end_time}:${cut.first_frame_time}:${cut.last_frame_time}:${position}`
    let request = previewRequests.current.get(key)
    if (!request) {
      request = api<{ times: number[]; anchor: number }>(`/cuts/${cutId}/boundary-preview?position=${position}`)
      previewRequests.current.set(key, request)
      void request.catch(() => previewRequests.current.delete(key))
    }
    return request
  }, [cutId, cut.start_time, cut.end_time, cut.first_frame_time, cut.last_frame_time])
  useEffect(() => {
    // Probe nearby timestamps in the background, so the first click normally paints immediately.
    if (cuts.some(item => item.cut_index === cut.cut_index - 1)) void getBoundaryPreview('first').catch(() => {})
    if (cuts.some(item => item.cut_index === cut.cut_index + 1)) void getBoundaryPreview('last').catch(() => {})
  }, [getBoundaryPreview, cuts, cut.cut_index])

  // The cached settings paint at once; the request only replaces them if the user has not started editing.
  const prefetchRef = useRef(prefetchIds); prefetchRef.current = prefetchIds
  const loadSettings = useCallback(() => {
    setLoadError('')
    loadConfig(cutId).then(value => {
      if (!configRef.current || JSON.stringify(configRef.current) === savedRef.current) { setConfig(value); setSaved(JSON.stringify(value)) }
      prefetchConfigs(prefetchRef.current)
    }).catch(err => { if (!configRef.current) setLoadError((err as Error).message) })
  }, [cutId])
  useEffect(() => { loadSettings() }, [loadSettings])
  useEffect(() => { if (!charactersTarget || projectId) return; let active = true; void loadProjectId(episodeId, cutId).then(id => { if (active) setProjectId(id) }).catch(() => {}); return () => { active = false } }, [charactersTarget, projectId, episodeId, cutId])
  useEffect(() => { onConfigChange(config) }, [config, onConfigChange])
  useEffect(() => () => clearTimeout(noticeTimer.current), [])
  useEffect(() => {
    // Warm the adjacent previews after a pause, without delaying rapid stepping.
    const timer = setTimeout(() => {
      for (const draft of Object.values(boundaryDraft)) {
        if (!draft) continue
        for (const index of [draft.index - 1, draft.index + 1]) {
          const time = draft.times[index]
          if (time === undefined) continue
          const image = new Image()
          image.src = `/api/cuts/${cutId}/boundary-preview-frame?time=${time}`
        }
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [boundaryDraft, cutId])

  const errors = useMemo(() => config ? videoSettingsErrors(config) : {}, [config])
  const invalid = hasVideoErrors(errors)
  useEffect(() => { if (invalid) setSettingsOpen(true) }, [invalid])
  const dirty = !!config && JSON.stringify(config) !== saved
  const active = activeRunFor(jobs, cutId)
  const steps = useMemo(() => {
    const base = cutStepStates(results, stages, jobs, cutId)
    return Object.fromEntries((Object.keys(base) as StepKey[]).map(key => [key, optimistic[key] && base[key] !== 'running' ? optimistic[key] : base[key]])) as Record<StepKey, StepState>
  }, [results, stages, jobs, cutId, optimistic])
  const counts = useMemo(() => batchCounts(cuts, cut), [cuts, cut])
  const warnings = useMemo(() => preflight({ mediaBusy: busy, enabled, errors, characters }), [busy, enabled, errors, characters])

  const boundaryDirty = Object.values(boundaryDraft).some(draft => draft && draft.index !== draft.original)
  const settingsReason = boundaryDirty ? 'Save or discard the boundary preview first' : characters?.saving ? 'Character edits are still saving' : pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : !enabled ? 'Analysis is disabled for this cut' : !config ? 'Settings are still loading' : invalid ? 'Fix the video analysis settings first' : ''
  const batchReason = boundaryDirty ? 'Save or discard the boundary preview first' : characters?.saving ? 'Character edits are still saving' : pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : !config ? 'Settings are still loading' : invalid ? 'Fix the video analysis settings first' : ''
  const textReason = pending ? 'Another action is being queued' : !cut.active ? 'Only active cuts can be queued' : ''
  const consistencyReason = pending ? 'Another action is being queued' : !enabled ? 'Analysis is disabled for this cut' : ''
  const targetBatchReason = pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : ''
  const targetCutReason = targetBatchReason || (!cut.active ? 'Only active cuts can be queued' : !targetFrames.selection?.result ? 'Evaluate this cut first' : '')
  const targetJob = targetPeople.data?.job
  const targetRun = targetPeople.data?.run && !targetPeople.data.run.stale ? targetPeople.data.run : null
  const targetState: StepState = isJobActive(targetJob) ? (targetJob!.status as StepState) : targetJob?.status === 'failed' ? 'failed' : targetRun ? 'done' : 'none'
  const targetUnknown = unknownSlots(targetRun).length
  const allTextReason = pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : ''
  const nextReason = settingsReason || (!onNext ? 'This is the last cut' : '')
  const precedingCut = cuts.find(item => item.cut_index === cut.cut_index - 1)
  const followingCut = cuts.find(item => item.cut_index === cut.cut_index + 1)
  const boundaryBusy = movingBoundary || !!characters?.saving || jobs.some(job => job.status === 'queued' || job.status === 'running')
  async function moveBoundary(position: 'first' | 'last', direction: -1 | 1) {
    if (boundaryBusy || boundaryLoading) return
    setBoundaryError('')
    if (!boundaryDirty) boundaryBase.current = { start: cut.start_time, end: cut.end_time }
    let draft = boundaryDraft[position]
    if (!draft) {
      setBoundaryLoading(position)
      try {
        const result = await getBoundaryPreview(position)
        if (!result.times.length) throw new Error('No frames found near this boundary.')
        const original = result.times.reduce((best, time, index) => Math.abs(time - result.anchor) < Math.abs(result.times[best] - result.anchor) ? index : best, 0)
        draft = { times: result.times, index: original, original }
      } catch (error) { setBoundaryError((error as Error).message); return } finally { setBoundaryLoading(null) }
    }
    let index = draft.index + direction
    // Extend the timestamp window on demand; the first probe is only a small, fast slice.
    if (index < 0 || index >= draft.times.length - (position === 'last' ? 1 : 0)) {
      setBoundaryLoading(position)
      try {
        const edge = draft.times[draft.index]
        const around = Math.max(0, Math.min(cuts[cuts.length - 1]?.end_time ?? cut.end_time, edge + direction * 2))
        const result = await api<{ times: number[] }>(`/cuts/${cutId}/boundary-preview?position=${position}&around=${around}`)
        const times = [...new Set([...draft.times, ...result.times])].sort((a, b) => a - b)
        const originalTime = draft.times[draft.original]
        const currentTime = draft.times[draft.index]
        draft = { times, original: times.indexOf(originalTime), index: times.indexOf(currentTime) }
        index = draft.index + direction
      } catch (error) { setBoundaryError((error as Error).message); return } finally { setBoundaryLoading(null) }
    }
    // The frame after the selected last frame is the new shared boundary.
    if (index < 0 || index >= draft.times.length - (position === 'last' ? 1 : 0)) { setBoundaryError('No more frames are available in that direction.'); return }
    const boundary = draft.times[index + (position === 'last' ? 1 : 0)]
    const start = position === 'first' ? boundary : boundaryDraft.first && boundaryDraft.first.index !== boundaryDraft.first.original ? boundaryDraft.first.times[boundaryDraft.first.index] : cut.start_time
    const end = position === 'last' ? boundary : boundaryDraft.last && boundaryDraft.last.index !== boundaryDraft.last.original ? boundaryDraft.last.times[boundaryDraft.last.index + 1] : cut.end_time
    if (start >= end || (precedingCut && start <= precedingCut.start_time) || (followingCut && end >= followingCut.end_time)) { setBoundaryError('That frame would leave a cut without a positive duration.'); return }
    setBoundaryDraft(value => ({ ...value, [position]: { ...draft!, index } }))
  }
  async function saveBoundaries() {
    if (!boundaryDirty || boundaryBusy) return
    const first = boundaryDraft.first
    const last = boundaryDraft.last
    const start = first && first.index !== first.original ? first.times[first.index] : cut.start_time
    const end = last && last.index !== last.original ? last.times[last.index + 1] : cut.end_time
    setMovingBoundary(true); setBoundaryError('')
    try {
      await api(`/cuts/${cutId}/timing`, { method: 'PUT', body: JSON.stringify({ start_time: start, end_time: end, expected_start_time: boundaryBase.current?.start ?? cut.start_time, expected_end_time: boundaryBase.current?.end ?? cut.end_time }) })
      setBoundaryDraft({}); boundaryBase.current = null
      await onStarted()
    } catch (error) { setBoundaryError((error as Error).message) } finally { setMovingBoundary(false) }
  }

  function show(next: Notice) {
    clearTimeout(noticeTimer.current)
    setNotice(next)
    if (next.tone === 'success') noticeTimer.current = setTimeout(() => setNotice(null), 12000)
  }

  /** Saves the settings only when they changed, so a clean queue is a single request. */
  async function persist(required: boolean) {
    const snapshot = configRef.current
    if (!snapshot || JSON.stringify(snapshot) === savedRef.current) return
    if (hasVideoErrors(videoSettingsErrors(snapshot))) { if (required) throw new Error('Enter even dimensions within the listed limits and a frame rate greater than 0 and at most 30.'); return }
    await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(snapshot) })
    storeConfig(cutId, snapshot)
    setSaved(JSON.stringify(snapshot)); savedRef.current = JSON.stringify(snapshot)
  }

  /**
   * Runs one queue action. Only the clicked button spins. The success message and the queued chips appear as soon as
   * the server accepted the job; the episode refresh that follows happens in the background.
   */
  async function perform(key: string, task: () => Promise<string | Notice>, options: { settings?: boolean; steps?: StepKey[]; afterQueued?: () => void } = {}) {
    if (pending) return
    setPending(key); setNotice(null)
    try {
      await persist(!!options.settings)
      const result = await task()
      show(typeof result === 'string' ? { ...queueNotice(result) } : result)
      if (options.steps?.length) setOptimistic(previous => ({ ...previous, ...Object.fromEntries(options.steps!.map(step => [step, 'queued' as const])) }))
      options.afterQueued?.()
      void onStarted().catch(() => show({ tone: 'warning', lines: ['Queued, but the episode could not refresh.', 'Reload the page to see its status.'] })).finally(() => setOptimistic({}))
    } catch (err) { setNotice({ tone: 'error', lines: [(err as Error).message] }) } finally { setPending(null) }
  }

  const queueVisual = (kind: RunKind) => perform(`visual-${kind}`, async () => {
    const url = kind === 'batch' ? `/episodes/${episodeId}/visual-analysis`
      : kind === 'redo-all' ? `/episodes/${episodeId}/visual-analysis?only_missing=false`
      : kind === 'from-here' ? `/episodes/${episodeId}/visual-analysis?from_cut=${cut.cut_index + 1}&characters_resolved=true`
      : `/cuts/${cutId}/analyze`
    const result = await api<BatchQueueResult>(url, { method: 'POST' })
    return kind === 'redo-all' ? `Visual analysis re-run queued for ${result.queued ?? 0} cuts.`
      : kind === 'batch' ? `Visual analysis queued for ${result.queued ?? 0} cuts; ${result.skipped ?? 0} already complete were skipped.`
      : kind === 'from-here' ? fromHereMessage(cut.cut_index + 1, result) : 'Visual analysis queued for this cut.'
  }, { settings: true, steps: kind === 'cut' || kind === 'next' ? [...VISUAL_STEPS] : [], afterQueued: kind === 'next' ? onNext : undefined })

  const queueStage = (stage: VisualStage) => perform(`stage-${stage}`, async () => {
    await api(`/cuts/${cutId}/analysis-stages/run`, { method: 'POST', body: JSON.stringify({ stage }) })
    return `${STEP_LABELS[stage]} analysis queued for this cut.`
  }, { settings: true, steps: [stage] })

  const queueConsistency = () => perform('consistency-cut', async () => {
    await api(`/cuts/${cutId}/consistency-summary`, { method: 'POST' })
    return 'Consistency summary queued for this cut.'
  }, { steps: ['consistency'] })

  const queueConsistencyFromHere = () => perform('consistency-from', async () => {
    const result = await api<{ queued: number }>(`/episodes/${episodeId}/consistency-summaries?from_cut=${cut.cut_index + 1}`, { method: 'POST' })
    return `Consistency summary queued for ${result.queued} cuts from cut ${cut.cut_index + 1}.`
  })

  const queueTextFromHere = (kind: 'answers' | 'summaries') => perform(`${kind}-from`, async () => {
    const first = cut.cut_index + 1
    const result = await api<TextBatchResult>(`/episodes/${episodeId}/${kind === 'answers' ? 'review-answers' : 'cut-summaries'}?from_cut=${first}`, { method: 'POST' })
    return kind === 'answers'
      ? batchQueueMessage('Production decisions draft', first, result, ANSWERS_BATCH_REASONS)
      : batchQueueMessage('Cut summary', first, result, SUMMARY_BATCH_REASONS, 'they already have a Cut summary')
  })

  const queueTextChain = () => perform('chain-from', async () => {
    const first = cut.cut_index + 1
    const result = await api<TextBatchResult>(`/episodes/${episodeId}/text-chain?from_cut=${first}`, { method: 'POST' })
    return batchQueueMessage('Consistency summary, decisions and Cut summary', first, result, CHAIN_BATCH_REASONS, 'they keep their existing Cut summary')
  })

  const queueAllText = () => perform('text-all', async () => {
    const settings = await api<Pick<AnalysisConfig, 'text_provider' | 'local_thinking' | 'openrouter_model'>>('/text-generation-settings')
    const result = await api<{ cuts_queued: number; skipped: number }>(`/episodes/${episodeId}/text-processing`, { method: 'POST', body: JSON.stringify(settings) })
    return `Text processing queued for ${result.cuts_queued} cuts; ${result.skipped} not ready or already queued.`
  })

  const queueTargets = (kind: 'cut' | 'from-here' | 'all') => perform(`targets-${kind}`, async () => {
    if (kind === 'cut') {
      await api(`/cuts/${cutId}/target-detection`, { method: 'POST' })
      return 'Target-frame characters queued for this cut.'
    }
    const first = kind === 'from-here' ? cut.cut_index + 1 : 1
    return targetQueueMessage(await api<TargetQueueResult>(`/episodes/${episodeId}/target-detection${kind === 'from-here' ? `?from_cut=${first}` : ''}`, { method: 'POST' }), first)
  }, { afterQueued: () => void targetPeople.reload() })

  async function saveSettings() {
    if (!config || pending) return
    if (invalid) { setNotice({ tone: 'error', lines: ['Enter even dimensions within the listed limits and a frame rate greater than 0 and at most 30.'] }); return }
    setPending('save'); setNotice(null)
    try { await persist(true); show({ tone: 'success', lines: ['Settings saved for this cut.'] }) }
    catch (err) { setNotice({ tone: 'error', lines: [(err as Error).message] }) } finally { setPending(null) }
  }

  const cell = (reason: string, node: (disabled: boolean, title: string) => ReactNode) => node(!!reason, reason)
  const none = <span className="muted" aria-label="Not available here" title="Not available here. This action is on the Results tab.">—</span>
  const row = (label: string, hint: string, state: StepState, thisCut: ReactNode, fromHere: ReactNode, all: ReactNode) => <tr className="border-t hairline">
    <th scope="row" className="py-2 pr-2 text-left align-middle font-normal"><span className="block font-semibold text-[#e8ebf1]">{label}</span><span className="muted block text-[11px] leading-4">{hint}</span><span className="mt-1 block"><StateChip state={state} /></span></th>
    <td className="px-1 py-2 text-center align-middle">{thisCut}</td><td className="px-1 py-2 text-center align-middle">{fromHere}</td><td className="px-1 py-2 text-center align-middle">{all}</td></tr>
  const isPending = (key: string) => pending === key
  const batchLabel = (count: number) => `${count} ${count === 1 ? 'cut' : 'cuts'}`

  return <div className="space-y-4">
    {/* Run card: the actions people reach for first stay visible while the rest scrolls. */}
    <section aria-label="Run this cut" className="z-10 space-y-3 lg:[@media(min-height:860px)]:sticky lg:[@media(min-height:860px)]:top-0 rounded-xl border border-[#3a3358] bg-[#1b202a]/95 p-3 shadow-lg shadow-black/30 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">Run analysis <span className="muted font-normal">· cut {cut.cut_index + 1}</span></h3>
        <div className="flex items-center gap-1.5">{dirty && <StatusChip tone="amber" title="Changed settings are saved when you queue a run">Unsaved settings</StatusChip>}</div>
      </div>
      <ul className="flex flex-wrap gap-1" aria-label="Analysis steps for this cut">{(Object.keys(STEP_LABELS) as StepKey[]).map(key => <li key={key}><StateChip state={steps[key]} label={STEP_LABELS[key]} /></li>)}</ul>
      {active && <p role="status" className="flex items-center gap-2 rounded-lg border border-[#4a4363] bg-[#221c33] px-3 py-1.5 text-xs text-[#c3aeff]"><span className="min-w-0 flex-1 truncate">{active.status === 'queued' ? 'Waiting in the queue' : active.stage || 'Running'}</span>{active.status === 'running' && <Elapsed since={active.started_at} />}</p>}
      {warnings.length > 0 && <div className="space-y-1">{warnings.map(item => <WarningLine key={item.text} level={item.level}>{item.text}</WarningLine>)}</div>}
      <div className="flex flex-wrap items-center gap-2">
        <ActionButton variant="primary" icon={<ArrowRight size={14} />} pending={isPending('visual-next')} disabled={!!nextReason} title={nextReason || 'Queue visual analysis for this cut, then open the next cut'} onClick={() => void queueVisual('next')} >Analyze &amp; next</ActionButton>
        <ActionButton icon={<Play size={14} />} pending={isPending('visual-cut')} disabled={!!settingsReason} title={settingsReason || 'Queue visual analysis for this cut (Ctrl+Enter in the context box)'} onClick={() => void queueVisual('cut')} >Analyze cut</ActionButton>
      </div>
      <ActionButton compact icon={<ImageIcon size={13} />} pending={isPending('keyframes')} disabled={busy || !!pending} onClick={() => void perform('keyframes', async () => { await api(`/cuts/${cutId}/keyframe-selection`, { method: 'POST', body: JSON.stringify({}) }); window.dispatchEvent(new Event('framewise:keyframes-changed')); return 'Cut evaluation queued.' })}>Evaluate cut</ActionButton>
      <div className="flex flex-wrap items-center gap-1.5"><span className="muted text-[11px]">Only:</span>{STAGE_BUTTONS.map(item => <ActionButton key={item.stage} compact icon={item.icon} pending={isPending(`stage-${item.stage}`)} disabled={!!settingsReason} title={settingsReason || item.hint} onClick={() => void queueStage(item.stage)}>{item.label}</ActionButton>)}</div>
      {notice && <NoticeStrip notice={notice} onDismiss={() => setNotice(null)} />}
    </section>

    <section aria-label="Adjust cut boundaries" className="space-y-3 rounded-xl border hairline bg-[#171c25] p-3">
      <h3 className="text-sm font-semibold">Adjust cut boundaries</h3>
      <p className="muted text-[11px] leading-4">Preview frames with Earlier and Later, then save the boundaries when ready. The adjacent cuts move with them. Existing results and boxes stay saved; detect characters again if needed.</p>
      <div className="grid grid-cols-2 gap-2">
        {(['first', 'last'] as const).map(position => <figure key={position} className="min-w-0">
          <FrameImage className="aspect-video w-full rounded bg-black object-contain" src={boundaryDraft[position] && boundaryDraft[position]!.index !== boundaryDraft[position]!.original ? `/api/cuts/${cut.id}/boundary-preview-frame?time=${boundaryDraft[position]!.times[boundaryDraft[position]!.index]}` : `/api/cuts/${cut.id}/frame/${position}?v=${position === 'first' ? `${cut.start_time}-${cut.first_frame_time}` : `${cut.end_time}-${cut.last_frame_time}`}`} alt={position === 'first' ? 'Current first frame' : 'Current last frame'} />
          <figcaption className="muted mt-1 text-[11px]">{position === 'first' ? 'First frame' : 'Last frame'}</figcaption>
        </figure>)}
      </div>
      {([['Start', precedingCut], ['End', followingCut ? cut : undefined]] as const).map(([label, boundaryCut]) => <div key={label} className="flex items-center gap-2">
        <span className="w-10 text-xs font-semibold">{label}</span>
        <ActionButton compact icon={<ArrowLeft size={13} />} disabled={boundaryBusy || !!boundaryLoading || !boundaryCut} title={`Preview the ${label.toLowerCase()} boundary one frame earlier`} onClick={() => void moveBoundary(label === 'Start' ? 'first' : 'last', -1)}>Earlier</ActionButton>
        <ActionButton compact icon={<ArrowRight size={13} />} disabled={boundaryBusy || !!boundaryLoading || !boundaryCut} title={`Preview the ${label.toLowerCase()} boundary one frame later`} onClick={() => void moveBoundary(label === 'Start' ? 'first' : 'last', 1)}>Later</ActionButton>
      </div>)}
      <div className="flex gap-2"><ActionButton compact icon={<Save size={13} />} disabled={!boundaryDirty || boundaryBusy || !!boundaryLoading} pending={movingBoundary} onClick={() => void saveBoundaries()}>Save boundaries</ActionButton><ActionButton compact disabled={!boundaryDirty || boundaryBusy || !!boundaryLoading} onClick={() => { setBoundaryDraft({}); boundaryBase.current = null; setBoundaryError('') }}>Discard</ActionButton></div>
      {boundaryLoading && <p role="status" className="muted text-xs">Loading nearby frames…</p>}
      {boundaryError && <p role="alert" className="text-xs text-[#ff9fa7]">{boundaryError}</p>}
    </section>

    {/* Batch actions: what each button does is stated by its column and row, not by a long label. */}
    <section aria-label="Queue more" className="rounded-xl border hairline bg-[#171c25] p-3">
      <h3 className="mb-1 flex items-center gap-2 text-sm font-semibold"><Layers size={15} className="text-[#b496ff]" aria-hidden /> Queue more</h3>
      <table className="w-full text-xs">
        <thead><tr className="muted text-[10px] uppercase tracking-wider"><th className="w-[38%] pb-1.5 text-left font-semibold">Step</th><th className="pb-1.5 font-semibold">This cut</th><th className="pb-1.5 font-semibold" title="This cut and every later cut">From cut {cut.cut_index + 1}</th><th className="pb-1.5 font-semibold">Whole episode</th></tr></thead>
        <tbody>
          {row('Visual analysis', 'First frame + video', overall(VISUAL_STEPS.map(key => steps[key])),
            cell(settingsReason, (disabled, title) => <ActionButton compact icon={<Play size={13} />} pending={isPending('visual-cut')} disabled={disabled} title={title || 'Queue visual analysis for this cut'} onClick={() => void queueVisual('cut')}>Run</ActionButton>),
            cell(batchReason || (!cut.active ? 'Only active cuts can be queued' : ''), (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('visual-from-here')} disabled={disabled} title={title || `Queue this cut and every later cut whose characters are settled (marked as no characters, or every box named). Cuts with unknown or undetected characters are skipped and listed. Up to ${batchLabel(counts.from)}.`} onClick={() => void queueVisual('from-here')}>{batchLabel(counts.from)}</ActionButton>),
            cell(batchReason, (disabled, title) => <ActionButton compact icon={<Layers size={13} />} pending={isPending('visual-batch')} disabled={disabled} title={title || `Queue visual analysis for all ${batchLabel(counts.all)}; completed ones are skipped`} onClick={() => void queueVisual('batch')}>All {counts.all}</ActionButton>))}
          {row('Consistency summary', 'Needs visual results', steps.consistency,
            cell(consistencyReason, (disabled, title) => <ActionButton compact icon={<Play size={13} />} pending={isPending('consistency-cut')} disabled={disabled} title={title || 'Queue a consistency summary for this cut'} onClick={() => void queueConsistency()}>Run</ActionButton>),
            cell(textReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('consistency-from')} disabled={disabled} title={title || 'Queue for this cut and every later enabled cut. Nothing is queued unless each already has its initial frame, video and last frame results.'} onClick={() => void queueConsistencyFromHere()}>{batchLabel(counts.from)}</ActionButton>),
            none)}
          {row('Production decisions', 'Drafts review answers', 'none',
            none,
            cell(textReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('answers-from')} disabled={disabled} title={title || "Draft and save answers to each later cut's human review and suggested checks. Cuts without a consistency summary or anything to decide are skipped and listed; answers you edit while a draft waits are not overwritten."} onClick={() => void queueTextFromHere('answers')}>{batchLabel(counts.from)}</ActionButton>),
            none)}
          {row('Cut summary', 'Needs consistency + decisions', steps.summary,
            none,
            cell(textReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('summaries-from')} disabled={disabled} title={title || 'Queue a Cut summary for this cut and every later cut that has a consistency summary but no summary yet. Cuts whose decisions are still being drafted are skipped and listed.'} onClick={() => void queueTextFromHere('summaries')}>{batchLabel(counts.from)}</ActionButton>),
            cell(allTextReason, (disabled, title) => <ActionButton compact icon={<Layers size={13} />} pending={isPending('text-all')} disabled={disabled} title={title || 'All text steps: queue a consistency summary and then a Cut summary for every ready cut'} onClick={() => void queueAllText()}>All text</ActionButton>))}
          {row('Target-frame characters', 'Needs evaluated frames', targetState,
            cell(targetCutReason, (disabled, title) => <ActionButton compact icon={<Play size={13} />} pending={isPending('targets-cut')} disabled={disabled} title={title || 'Find and order the people on this cut’s evaluated frames. Run it again to keep your names and boxes and redo the rest.'} onClick={() => void queueTargets('cut')}>Run</ActionButton>),
            cell(targetBatchReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('targets-from-here')} disabled={disabled} title={title || 'Queue this cut and every later cut that has evaluated frames and no detection yet. Cuts not evaluated, marked as having no characters or with analysis disabled are skipped and listed.'} onClick={() => void queueTargets('from-here')}>{batchLabel(counts.from)}</ActionButton>),
            cell(targetBatchReason, (disabled, title) => <ActionButton compact icon={<Layers size={13} />} pending={isPending('targets-all')} disabled={disabled} title={title || 'Queue every cut with evaluated frames and no detection yet'} onClick={() => void queueTargets('all')}>All {counts.all}</ActionButton>))}
          {row('All three text steps', 'Runs in order, each waits for the last', 'none',
            none,
            cell(textReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('chain-from')} disabled={disabled} title={title || 'For this cut and every later ready cut, queue a consistency summary, then production-decision answers from it, then a Cut summary from both. Each job starts only after the one before it finishes. Cuts that already have a Cut summary keep it. Cuts without initial frame, video and last frame results, or with jobs already pending, are skipped and listed.'} onClick={() => void queueTextChain()}>{batchLabel(counts.from)}</ActionButton>),
            none)}
        </tbody>
      </table>
      <p className="muted mt-1 text-[11px]">Queue visual analysis first. Text steps work from its results.</p>
      <ActionButton compact className="mt-2" icon={<RotateCcw size={13} />} pending={isPending('visual-redo-all')} disabled={!!batchReason} title={batchReason || `Queue first-frame and video analysis again for all ${batchLabel(counts.all)}, even those already complete. Existing results stay in history as earlier revisions.`} onClick={() => { if (window.confirm(`Re-run visual analysis for all ${batchLabel(counts.all)}, including cuts already analysed?`)) void queueVisual('redo-all') }}>Re-run all visual analysis</ActionButton>
    </section>

    {/* Context for the models. */}
    {config ? <section className="rounded-xl border hairline bg-[#171c25] p-3">
      <label className="block"><span className="mb-1 flex items-center justify-between text-sm font-semibold">Context for this cut<span className={`text-[11px] font-normal tabular-nums ${config.additional_context.length > 2000 ? 'text-[#e6c58b]' : 'muted'}`} title="Local visual models read only the first 2,000 characters">{config.additional_context.length.toLocaleString()} / 2,000</span></span>
        <textarea className="field min-h-24 resize-y !py-2 text-sm" rows={4} maxLength={20000} placeholder="Anything the models should know about this shot…" value={config.additional_context} onChange={event => setConfig({ ...config, additional_context: event.target.value })} onKeyDown={event => {
          if (event.key !== 'Enter' || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.nativeEvent.isComposing) return
          event.preventDefault()
          if (!event.repeat && !settingsReason) void queueVisual('cut')
        }} />
        <span className="muted mt-1 block text-[11px]">Local models use the first 2,000 characters here and 500 per selected context item. Ctrl+Enter queues this cut.</span></label>
    </section> : loadError ? <section className="rounded-xl border border-[#63363e] bg-[#2b191f] p-3 text-xs text-[#ffb6bc]" role="alert"><p>Could not load this cut's settings: {loadError}</p><ActionButton compact className="mt-2" icon={<RotateCcw size={13} />} onClick={loadSettings}>Try again</ActionButton></section>
      : <div className="space-y-2" aria-busy="true" aria-label="Loading settings"><Skeleton className="h-28" /><Skeleton className="h-12" /></div>}

    {/* Defaults are usually right, so the canvas settings stay collapsed with their current values in view. */}
    {config && <details open={settingsOpen} onToggle={event => setSettingsOpen(event.currentTarget.open)} className="group rounded-xl border hairline bg-[#171c25]">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-sm font-semibold"><SlidersHorizontal size={15} className="text-[#6aa8ff]" aria-hidden />Video analysis<span className={`min-w-0 flex-1 truncate text-[11px] font-normal ${invalid ? 'text-[#ff9fa7]' : 'muted'}`}>{videoSettingsSummary(config, cut)}</span>{dirty && <StatusChip tone="amber">Unsaved</StatusChip>}<ChevronDown size={15} className="shrink-0 text-[#8b93a3] transition-transform group-open:rotate-180" aria-hidden /></summary>
      <div className="space-y-2 border-t hairline p-3">
        <div className="grid grid-cols-3 gap-2">
          <NumberField label="Width" unit="px" min={128} max={1920} step={2} value={config.video_width} error={errors.width} onChange={value => setConfig({ ...config, video_width: value })} />
          <NumberField label="Height" unit="px" min={128} max={1080} step={2} value={config.video_height} error={errors.height} onChange={value => setConfig({ ...config, video_height: value })} />
          <NumberField label="Frames per second" unit="fps" min={0.000001} max={30} step="any" value={config.video_fps} error={errors.fps} onChange={value => setConfig({ ...config, video_fps: value, video_frame_budget: null })} />
        </div>
        <p className="text-xs text-[#d5dbe5]" aria-live="polite"><span className="muted">Estimate: </span>{invalid ? 'enter valid settings' : `≈${estimatedVideoFrames(config, cut)} video frames${config.video_frame_budget ? ` (maximum ${config.video_frame_budget}, including padding)` : ' (including padding; actual count may vary)'}`}</p>
        <div className="flex flex-wrap items-center gap-2">
          <ActionButton compact icon={<RotateCcw size={13} />} title="Set the frame rate so the cut yields at most 60 frames" onClick={() => setConfig({ ...config, video_fps: automaticVideoFps(cut.end_time - cut.start_time), video_frame_budget: 60 })}>Limit to 60 frames</ActionButton>
          <ActionButton compact variant={dirty ? 'primary' : 'quiet'} icon={<Save size={13} />} pending={isPending('save')} disabled={!dirty || invalid || (!!pending && !isPending('save'))} title={dirty ? 'Save these settings for this cut' : 'Nothing to save'} onClick={() => void saveSettings()}>{dirty ? 'Save settings' : 'Saved'}</ActionButton>
        </div>
        <p className="muted text-[11px]">The cut is scaled to fit this canvas. Applies when queuing one cut; changes are also saved when you queue.</p>
      </div>
    </details>}

    {charactersTarget && createPortal(<div className="space-y-3">
      <div role="tablist" aria-label="Cut frames" className="flex gap-1 rounded-lg border hairline p-1">
        {([['characters', 'Characters in this cut'], ['targets', 'Target frames']] as const).map(([value, label]) => <button key={value} role="tab" type="button" aria-selected={characterTab === value} onClick={() => setCharacterTab(value)} className={`rounded-md px-4 py-1.5 text-xs font-semibold ${characterTab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}{value === 'targets' && targetUnknown > 0 && <span className="ml-1.5 rounded-full bg-[#4a2f3b] px-1.5 py-0.5 text-[10px] tabular-nums text-[#f4a8c4]" title="People on the evaluated frames with no character assigned">{targetUnknown} unknown</span>}{value === 'targets' && targetFrames.selection?.result && <span className="ml-1.5 rounded-full bg-[#3a4352] px-1.5 py-0.5 text-[10px] tabular-nums" aria-label={`${targetFrames.selection.result.selected_frames.length} target frames`}>{targetFrames.selection.result.selected_frames.length}</span>}</button>)}
      </div>
      <div hidden={characterTab !== 'characters'}>{projectId
        ? <CharacterDetectionPanel cutId={cutId} cutStart={cut.start_time} cutEnd={cut.end_time} projectId={projectId} jobs={jobs} onStarted={onStarted} onSummary={setCharacters} noCharacters={!!cut.no_characters} boundaryKey={`${cut.start_time}:${cut.end_time}`} />
        : <div className="space-y-2 rounded-xl border hairline bg-[#171c25] p-3" aria-busy="true" aria-label="Loading characters"><Skeleton className="h-5 w-48" /><Skeleton className="aspect-video" /></div>}</div>
      {characterTab === 'targets' && (projectId
        ? <TargetFrames cutId={cutId} selection={targetFrames.selection} error={targetFrames.error} projectId={projectId} detections={targetPeople} episodeBusy={busy} onChanged={() => void onStarted()} />
        : <div className="space-y-2 rounded-xl border hairline bg-[#171c25] p-3" aria-busy="true" aria-label="Loading target frames"><Skeleton className="h-5 w-48" /><Skeleton className="aspect-video" /></div>)}
    </div>, charactersTarget)}
    {children}
  </div>
})
