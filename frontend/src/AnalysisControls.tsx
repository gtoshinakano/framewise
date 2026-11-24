import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, ArrowRight, GalleryHorizontal, Image as ImageIcon, Play, RotateCcw, Save } from 'lucide-react'
import { api, type AnalysisConfig, type AnalysisResult, type Cut, type Job, type VisualStage } from './api'
import { CharacterDetectionPanel } from './CharacterDetectionPanel'
import { TargetFrames, useTargetDetections, useTargetSelection } from './TargetFrames'
import type { RunSection } from './cutRoute'
import { unknownSlots } from './targetView'
import { FrameImage } from './FrameImage'
import { ActionButton, Elapsed, NoticeStrip, Skeleton, StateChip, StatusChip, WarningLine, type Notice } from './RunUi'
import { cachedConfig, cachedProjectId, loadConfig, loadProjectId, prefetchConfigs, storeConfig } from './runCache'
import { STEP_LABELS, VISUAL_STEPS, activeRunFor, cutStepStates, preflight, queueNotice, type CharacterSummary, type StageRow, type StepKey, type StepState } from './runState'

type Optimistic = Partial<Record<StepKey, StepState>>
type RunKind = 'cut' | 'next'

const STAGE_BUTTONS: { stage: VisualStage; label: string; icon: ReactNode; hint: string }[] = [
  { stage: 'initial_frame', label: 'Initial frame', icon: <ImageIcon size={13} />, hint: 'Describe only the first frame: the still image, with each character\'s look, pose, expression and emotion' },
  { stage: 'keyframe_analysis', label: 'Keyframes', icon: <GalleryHorizontal size={13} />, hint: 'Analyze only what changes between the keyframes: each character\'s state per frame, body changes measured from keypoints, the camera measured from the background. Selects the keyframes first if the cut has none; uses the first-frame names, or the Target frames names when they are current. Its performance beats are written afterwards' },
]

type Props = {
  jobs: Job[]; cut: Cut; cuts: Cut[]; episodeId: string; busy: boolean; enabled: boolean
  results: AnalysisResult[]; stages: StageRow[]; prefetchIds: (string | undefined)[]
  charactersTarget: HTMLElement | null
  section: RunSection; onSection: (section: RunSection) => void
  onStarted: () => Promise<void>; onConfigChange: (config: AnalysisConfig | null) => void
  onPrevious?: () => void; onNext?: () => void; children?: ReactNode
}

export default memo(function AnalysisControls({ cut, cuts, episodeId, busy, enabled, results, stages, prefetchIds, charactersTarget, section: characterTab, onSection: setCharacterTab, onStarted, onConfigChange, onNext, jobs, children }: Props) {
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

  const dirty = !!config && JSON.stringify(config) !== saved
  const active = activeRunFor(jobs, cutId)
  const steps = useMemo(() => {
    const base = cutStepStates(results, stages, jobs, cutId)
    return Object.fromEntries((Object.keys(base) as StepKey[]).map(key => [key, optimistic[key] && base[key] !== 'running' ? optimistic[key] : base[key]])) as Record<StepKey, StepState>
  }, [results, stages, jobs, cutId, optimistic])
  const warnings = useMemo(() => preflight({ hasPrerequisites: busy, enabled, characters }), [busy, enabled, characters])

  const boundaryDirty = Object.values(boundaryDraft).some(draft => draft && draft.index !== draft.original)
  const settingsReason = boundaryDirty ? 'Save or discard the boundary preview first' : characters?.saving ? 'Character edits are still saving' : pending ? 'Another action is being queued' : !enabled ? 'Analysis is disabled for this cut' : !config ? 'Settings are still loading' : ''
  const targetRun = targetPeople.data?.run && !targetPeople.data.run.stale ? targetPeople.data.run : null
  const targetUnknown = unknownSlots(targetRun).length
  const nextReason = settingsReason || (!onNext ? 'This is the last cut' : '')
  const precedingCut = cuts.find(item => item.cut_index === cut.cut_index - 1)
  const followingCut = cuts.find(item => item.cut_index === cut.cut_index + 1)
  const boundaryBusy = movingBoundary || !!characters?.saving
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
  async function persist(_required: boolean) {
    const snapshot = configRef.current
    if (!snapshot || JSON.stringify(snapshot) === savedRef.current) return
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

  // Many cuts at once are Run Multiple's (the editor header): it shows what will run and which cuts wait for you first.
  const queueVisual = (kind: RunKind) => perform(`visual-${kind}`, async () => {
    await api(`/cuts/${cutId}/analyze`, { method: 'POST' })
    return 'Visual analysis queued for this cut.'
  }, { settings: true, steps: [...VISUAL_STEPS], afterQueued: kind === 'next' ? onNext : undefined })

  const queueStage = (stage: VisualStage) => perform(`stage-${stage}`, async () => {
    await api(`/cuts/${cutId}/analysis-stages/run`, { method: 'POST', body: JSON.stringify({ stage }) })
    return `${STEP_LABELS[stage]} analysis queued for this cut.`
  }, { settings: true, steps: [stage] })

  async function saveSettings() {
    if (!config || pending) return
    setPending('save'); setNotice(null)
    try { await persist(true); show({ tone: 'success', lines: ['Settings saved for this cut.'] }) }
    catch (err) { setNotice({ tone: 'error', lines: [(err as Error).message] }) } finally { setPending(null) }
  }

  const isPending = (key: string) => pending === key

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
      <ActionButton compact icon={<ImageIcon size={13} />} pending={isPending('keyframes')} disabled={!cut.active || !!pending} onClick={() => void perform('keyframes', async () => { await api(`/cuts/${cutId}/keyframe-selection`, { method: 'POST', body: JSON.stringify({}) }); return 'Cut evaluation queued.' })}>Evaluate cut</ActionButton>
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

    {/* Context for the models. */}
    {config ? <section className="rounded-xl border hairline bg-[#171c25] p-3">
      <label className="block"><span className="mb-1 flex items-center justify-between text-sm font-semibold">Context for this cut<span className={`text-[11px] font-normal tabular-nums ${config.additional_context.length > 2000 ? 'text-[#e6c58b]' : 'muted'}`} title="Local visual models read only the first 2,000 characters">{config.additional_context.length.toLocaleString()} / 2,000</span></span>
        <textarea className="field min-h-24 resize-y !py-2 text-sm" rows={4} maxLength={20000} placeholder="Anything the models should know about this shot…" value={config.additional_context} onChange={event => setConfig({ ...config, additional_context: event.target.value })} onKeyDown={event => {
          if (event.key !== 'Enter' || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.nativeEvent.isComposing) return
          event.preventDefault()
          if (!event.repeat && !settingsReason) void queueVisual('cut')
        }} />
        <span className="muted mt-1 block text-[11px]">Local models use the first 2,000 characters here and 500 per selected context item. Ctrl+Enter queues this cut.</span></label>
      {dirty && <ActionButton compact className="mt-2" variant="primary" icon={<Save size={13} />} pending={isPending('save')} disabled={!!pending && !isPending('save')} title="Save this context for the cut" onClick={() => void saveSettings()}>Save context</ActionButton>}
    </section> : loadError ? <section className="rounded-xl border border-[#63363e] bg-[#2b191f] p-3 text-xs text-[#ffb6bc]" role="alert"><p>Could not load this cut's settings: {loadError}</p><ActionButton compact className="mt-2" icon={<RotateCcw size={13} />} onClick={loadSettings}>Try again</ActionButton></section>
      : <div className="space-y-2" aria-busy="true" aria-label="Loading settings"><Skeleton className="h-28" /><Skeleton className="h-12" /></div>}

    {charactersTarget && createPortal(<div className="space-y-3">
      <div role="tablist" aria-label="Cut frames" className="flex gap-1 rounded-lg border hairline p-1">
        {([['characters', 'Characters in this cut'], ['targets', 'Target frames']] as const).map(([value, label]) => <button key={value} role="tab" type="button" aria-selected={characterTab === value} onClick={() => setCharacterTab(value)} className={`rounded-md px-4 py-1.5 text-xs font-semibold ${characterTab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}{value === 'targets' && targetUnknown > 0 && <span className="ml-1.5 rounded-full bg-[#4a2f3b] px-1.5 py-0.5 text-[10px] tabular-nums text-[#f4a8c4]" title="People on the evaluated frames with no character assigned">{targetUnknown} unknown</span>}{value === 'targets' && targetFrames.selection?.result && <span className="ml-1.5 rounded-full bg-[#3a4352] px-1.5 py-0.5 text-[10px] tabular-nums" aria-label={`${targetFrames.selection.result.selected_frames.length} target frames`}>{targetFrames.selection.result.selected_frames.length}</span>}</button>)}
      </div>
      <div hidden={characterTab !== 'characters'}>{projectId
        ? <CharacterDetectionPanel episodeId={episodeId} cutId={cutId} cutStart={cut.start_time} cutEnd={cut.end_time} projectId={projectId} jobs={jobs} onStarted={onStarted} onSummary={setCharacters} noCharacters={!!cut.no_characters} boundaryKey={`${cut.start_time}:${cut.end_time}`} />
        : <div className="space-y-2 rounded-xl border hairline bg-[#171c25] p-3" aria-busy="true" aria-label="Loading characters"><Skeleton className="h-5 w-48" /><Skeleton className="aspect-video" /></div>}</div>
      {characterTab === 'targets' && (projectId
        ? <TargetFrames cutId={cutId} selection={targetFrames.selection} error={targetFrames.error} projectId={projectId} detections={targetPeople} onChanged={() => void onStarted()} />
        : <div className="space-y-2 rounded-xl border hairline bg-[#171c25] p-3" aria-busy="true" aria-label="Loading target frames"><Skeleton className="h-5 w-48" /><Skeleton className="aspect-video" /></div>)}
    </div>, charactersTarget)}
    {children}
  </div>
})
