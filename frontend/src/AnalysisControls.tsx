import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowRight, ChevronDown, ChevronsRight, Film, Image as ImageIcon, Layers, Play, RotateCcw, Save, SlidersHorizontal } from 'lucide-react'
import { api, type AnalysisConfig, type AnalysisResult, type Cut, type Job, type VisualStage } from './api'
import { CharacterDetectionPanel } from './CharacterDetectionPanel'
import { automaticVideoFps, estimatedVideoFrames } from './analysisFrames'
import { ANSWERS_BATCH_REASONS, CHAIN_BATCH_REASONS, SUMMARY_BATCH_REASONS, batchQueueMessage, type BatchQueueResult as TextBatchResult } from './batchQueueMessage'
import { ActionButton, Elapsed, NoticeStrip, Skeleton, StateChip, StatusChip, WarningLine, type Notice } from './RunUi'
import { cachedConfig, cachedProjectId, loadConfig, loadProjectId, prefetchConfigs, storeConfig } from './runCache'
import { STEP_LABELS, VISUAL_STEPS, activeRunFor, batchCounts, cutStepStates, hasVideoErrors, preflight, queueNotice, videoSettingsErrors, videoSettingsSummary, type CharacterSummary, type StageRow, type StepKey, type StepState } from './runState'

type BatchQueueResult = { queued?: number; skipped?: number; blocked?: { unknown_characters: number[]; not_detected: number[] } }
type Optimistic = Partial<Record<StepKey, StepState>>
type RunKind = 'cut' | 'next' | 'batch' | 'from-here'

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
  const [config, setConfig] = useState<AnalysisConfig | null>(() => cachedConfig(cutId) ?? null)
  const [saved, setSaved] = useState(() => { const known = cachedConfig(cutId); return known ? JSON.stringify(known) : '' })
  const [loadError, setLoadError] = useState('')
  const [projectId, setProjectId] = useState(() => cachedProjectId(episodeId) ?? '')
  const [pending, setPending] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [optimistic, setOptimistic] = useState<Optimistic>({})
  const [characters, setCharacters] = useState<CharacterSummary | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const configRef = useRef(config); configRef.current = config
  const savedRef = useRef(saved); savedRef.current = saved
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

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

  const settingsReason = pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : !enabled ? 'Analysis is disabled for this cut' : !config ? 'Settings are still loading' : invalid ? 'Fix the video analysis settings first' : ''
  const batchReason = pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : !config ? 'Settings are still loading' : invalid ? 'Fix the video analysis settings first' : ''
  const textReason = pending ? 'Another action is being queued' : !cut.active ? 'Only active cuts can be queued' : ''
  const consistencyReason = pending ? 'Another action is being queued' : !enabled ? 'Analysis is disabled for this cut' : ''
  const allTextReason = pending ? 'Another action is being queued' : busy ? 'Media preparation or detection is running' : ''
  const nextReason = settingsReason || (!onNext ? 'This is the last cut' : '')

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
      : kind === 'from-here' ? `/episodes/${episodeId}/visual-analysis?from_cut=${cut.cut_index + 1}&characters_resolved=true`
      : `/cuts/${cutId}/analyze`
    const result = await api<BatchQueueResult>(url, { method: 'POST' })
    return kind === 'batch' ? `Visual analysis queued for ${result.queued ?? 0} cuts; ${result.skipped ?? 0} already complete were skipped.`
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
      <div className="flex flex-wrap items-center gap-1.5"><span className="muted text-[11px]">Only:</span>{STAGE_BUTTONS.map(item => <ActionButton key={item.stage} compact icon={item.icon} pending={isPending(`stage-${item.stage}`)} disabled={!!settingsReason} title={settingsReason || item.hint} onClick={() => void queueStage(item.stage)}>{item.label}</ActionButton>)}</div>
      {notice && <NoticeStrip notice={notice} onDismiss={() => setNotice(null)} />}
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
          {row('All three text steps', 'Runs in order, each waits for the last', 'none',
            none,
            cell(textReason, (disabled, title) => <ActionButton compact icon={<ChevronsRight size={13} />} pending={isPending('chain-from')} disabled={disabled} title={title || 'For this cut and every later ready cut, queue a consistency summary, then production-decision answers from it, then a Cut summary from both. Each job starts only after the one before it finishes. Cuts that already have a Cut summary keep it. Cuts without initial frame, video and last frame results, or with jobs already pending, are skipped and listed.'} onClick={() => void queueTextChain()}>{batchLabel(counts.from)}</ActionButton>),
            none)}
        </tbody>
      </table>
      <p className="muted mt-1 text-[11px]">Queue visual analysis first. Text steps work from its results.</p>
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

    {charactersTarget && createPortal(projectId
      ? <CharacterDetectionPanel cutId={cutId} projectId={projectId} jobs={jobs} onStarted={onStarted} onSummary={setCharacters} noCharacters={!!cut.no_characters} />
      : <div className="space-y-2 rounded-xl border hairline bg-[#171c25] p-3" aria-busy="true" aria-label="Loading characters"><Skeleton className="h-5 w-48" /><div className="grid gap-3 xl:grid-cols-2"><Skeleton className="aspect-video" /><Skeleton className="aspect-video" /></div></div>, charactersTarget)}
    {children}
  </div>
})
