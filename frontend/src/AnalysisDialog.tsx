import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, X } from 'lucide-react'
import { PlayerProvider, PlayerSlot, SourcePlayer, usePlayer, usePlayerValue } from './SourcePlayer'
import { prefetchCutJobs, useCutJobs } from './cutJobs'
import { mediaUrl, type AnalysisConfig, type AnalysisResult, type Cut, type Job, type TranscriptSegment, type Waveform } from './api'
import AnalysisControls from './AnalysisControls'
import { useCutStages } from './useCutStages'
import ResultEditor from './ResultEditor'
import { CutSummaryPanel, SUMMARY_TITLE } from './AnimationPromptPanel'
import { FrameTools, FramesDialog } from './DetectionPreview'
import { CutAudioPreview } from './CutAudioPreview'
import { CutRunHistory } from './CutRunHistory'
import AnalysisTimeline from './AnalysisTimeline'
import { ConsistencySummary } from './ConsistencySummary'
import { PerformanceBeats } from './PerformanceBeats'
import CutAssistant from './CutAssistant'
import { CutNeedsBar } from './AgentNotes'
import { CutEscalations } from './CutEscalations'
import { useCutAgentWork } from './useCutAgentWork'
import { firstNeed, hasNeed, placeForNeed, type CutNeeds, type NeedAnchor, type NeedKey } from './cutNeedsView'

import type { DialogTab, RunSection } from './cutRoute'

type DialogProps = { initialTab: DialogTab; initialSection: RunSection; onPlace: (tab: DialogTab, section: RunSection) => void; needs?: CutNeeds; focusKeys: ReadonlySet<NeedKey>; cut: Cut; cuts: Cut[]; previousCut?: Cut; nextCut?: Cut; onNavigate: (cut: Cut) => void; episodeId: string; projectId: string; proxyReady: boolean; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparingAudio: boolean; busy: boolean; jobs: Job[]; jobsKey: string; frameVersion: string; onStarted: () => Promise<void>; onClose: () => void }

export default memo(function AnalysisDialog({ cut, cuts, previousCut, nextCut, onNavigate, onClose, initialTab, initialSection, onPlace, needs, focusKeys, ...contentProps }: DialogProps) {
  const [tab, setTab] = useState<DialogTab>(initialTab)
  const [section, setSection] = useState<RunSection>(initialSection)
  // The editor keeps the address (a link to this cut and place) in step with these.
  useEffect(() => { onPlace(tab, section) }, [tab, section, onPlace])
  // One video for the whole dialog. Stepping to another cut only moves its playhead and the interval it may play, so what the
  // browser has already loaded is kept (the content below is rebuilt for every cut, the video is not).
  const player = useMemo(() => new SourcePlayer({ label: 'Play this cut', preload: 'metadata' }), [])
  useEffect(() => () => player.dispose(), [player])
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); opener?.focus() }
  }, [onClose])
  // Where the person must look: scroll the element marked data-need into view and pulse it. Tabs and lazy panels render a
  // moment later, so it looks again for a couple of seconds; "voices" falls back to the line list when no voice is unknown here.
  const pointing = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pointAt = useCallback((anchor: NeedAnchor) => {
    if (pointing.current) clearTimeout(pointing.current)
    let tries = 0
    const look = () => {
      const root = panel.current
      const target = root?.querySelector<HTMLElement>(`[data-need="${anchor}"]`) ?? (anchor === 'voices' ? root?.querySelector<HTMLElement>('[data-need="lines"]') : null)
      if (target && target.offsetParent !== null) {
        target.scrollIntoView({ block: 'center', behavior: 'smooth' })
        target.classList.remove('need-pulse'); void target.offsetWidth; target.classList.add('need-pulse')
        pointing.current = setTimeout(() => target.classList.remove('need-pulse'), 2600)
        return
      }
      if (++tries < 15) pointing.current = setTimeout(look, 150)
    }
    pointing.current = setTimeout(look, 60)
  }, [])
  useEffect(() => () => { if (pointing.current) clearTimeout(pointing.current) }, [])
  const jumpTo = useCallback((key: NeedKey) => {
    const place = placeForNeed(key)
    setTab(place.tab)
    if (place.section) setSection(place.section)
    pointAt(place.anchor)
  }, [pointAt])
  // Arriving at a cut shows its next human action. The first cut keeps the place the dialog opened on when that is another
  // place (a link names it); every cut reached from here moves to the action.
  const arrived = useRef<string | null>(null)
  const opened = useRef(false)
  useEffect(() => {
    if (!needs || arrived.current === cut.id) return
    arrived.current = cut.id
    const first = !opened.current
    opened.current = true
    const need = firstNeed(needs, focusKeys)
    if (!need) return
    const place = placeForNeed(need.key)
    if (first && (place.tab !== tab || (place.section && place.section !== section))) return
    jumpTo(need.key)
  }, [cut.id, needs])
  const [jump, setJump] = useState(String(cut.cut_index + 1))
  // The timeline is history, so it loads only when asked for; the choice holds while moving between cuts in this dialog.
  const [timelineOpen, setTimelineOpen] = useState(false)
  const [toast, setToast] = useState('')
  useEffect(() => setJump(String(cut.cut_index + 1)), [cut.id, cut.cut_index])
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToast(''), 4000); return () => clearTimeout(timer) }, [toast])
  function goToCut() {
    const number = Number(jump)
    const target = jump.trim() !== '' && Number.isInteger(number) ? cuts[number - 1] : undefined
    if (!target) { setJump(String(cut.cut_index + 1)); setToast(`Cut ${jump.trim() || '(empty)'} does not exist. Enter a number from 1 to ${cuts.length}.`); return }
    if (target.id !== cut.id) onNavigate(target)
  }
  // Stable handlers: the Run tab's memoized panels should not re-render just because the dialog did.
  const navigation = useRef({ previousCut, nextCut, onNavigate }); navigation.current = { previousCut, nextCut, onNavigate }
  const goPrevious = useCallback(() => { const { previousCut: target, onNavigate: go } = navigation.current; if (target) go(target) }, [])
  const goNext = useCallback(() => { const { nextCut: target, onNavigate: go } = navigation.current; if (target) go(target) }, [])
  // Keyboard: "[" goes to the previous cut and "]" to the next. Ignored while typing in a field, with a modifier held
  // (browser shortcuts), on key repeat, or while a nested dialog (run details) is open.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return
      if (event.key !== '[' && event.key !== ']') return
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return
      if (document.querySelectorAll('[aria-modal="true"]').length > 1) return
      event.preventDefault()
      if (event.key === '[') goPrevious(); else goNext()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goPrevious, goNext])
  const title = cut.title || (cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut')
  return <PlayerProvider player={player}><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Analysis for ${title}`} className="flex h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="grid grid-cols-[1fr_auto] items-center gap-3 border-b hairline px-5 py-3 lg:grid-cols-[1fr_auto_1fr]"><div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Cut analysis</p><h2 className="truncate text-base font-semibold">{title}</h2></div><div className="order-3 col-span-2 flex justify-center lg:order-none lg:col-span-1"><div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">{([['results', 'Results'], ['run', 'Run settings'], ['assistant', 'Assistant']] as const).map(([value, label]) => <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`rounded-md px-5 py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}</button>)}</div></div><div className="flex items-center justify-end gap-2"><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!previousCut} onClick={() => previousCut && onNavigate(previousCut)} aria-label="Previous cut analysis" aria-keyshortcuts="[" title="Previous cut (press [ )"><ArrowLeft size={14} /> Previous <kbd className="hidden rounded border border-[#3b4352] px-1 font-mono text-[10px] leading-4 text-[#929aa9] xl:inline">[</kbd></button><input type="number" min={1} max={cuts.length} step={1} value={cut.active ? jump : ''} disabled={!cut.active} onChange={event => setJump(event.target.value)} onBlur={goToCut} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} className="field !w-24 px-2.5 py-2 text-center" aria-label="Go to cut number" /><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!nextCut} onClick={() => nextCut && onNavigate(nextCut)} aria-label="Next cut analysis" aria-keyshortcuts="]" title="Next cut (press ] )"><kbd className="hidden rounded border border-[#3b4352] px-1 font-mono text-[10px] leading-4 text-[#929aa9] xl:inline">]</kbd> Next <ArrowRight size={14} /></button><button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close analysis"><X size={18} /></button></div></header>
      <AnalysisDialogContent key={cut.id} cut={cut} cuts={cuts} needs={needs} onJump={jumpTo} tab={tab} section={section} onSection={setSection} previousId={previousCut?.id} nextId={nextCut?.id} onPrevious={previousCut ? goPrevious : undefined} onNext={nextCut ? goNext : undefined} {...contentProps} />
      {timelineOpen
        ? <AnalysisTimeline cutId={cut.id} onHide={() => setTimelineOpen(false)} />
        : <footer className="shrink-0 border-t hairline px-5 py-1.5 text-center"><button type="button" onClick={() => setTimelineOpen(true)} className="text-xs text-[#a58ae8] hover:text-[#c3aeff] hover:underline">Load Analysis timeline</button></footer>}
    </div>
    {toast && <div role="alert" className="fixed bottom-6 left-1/2 z-[60] flex -translate-x-1/2 items-start gap-3 rounded-xl border border-[#63363e] bg-[#2b191f] px-4 py-3 text-sm text-[#ffb6bc] shadow-lg">{toast}<button onClick={() => setToast('')} aria-label="Dismiss"><X size={16} /></button></div>}
  </div></PlayerProvider>
})

/** Playhead in seconds from the cut's first frame. It follows the video itself, so the dialog does not re-render 60 times a second. */
function PlayheadReadout({ start, end }: { start: number; end: number }) {
  const seconds = usePlayerValue(time => Math.max(0, Math.min(end - start, time - start)))
  return <span className="pointer-events-none absolute left-2 top-2 rounded-md bg-black/70 px-2 py-1 font-mono text-xs tabular-nums text-white" title="Playhead position in seconds, counted from the start of this cut" aria-label={`Playhead ${seconds.toFixed(3)} seconds into the cut`}>{seconds.toFixed(3)}<span className="ml-0.5 text-[#aeb5c1]">s</span></span>
}

type ContentProps = Pick<DialogProps, 'cut' | 'cuts' | 'needs' | 'episodeId' | 'projectId' | 'proxyReady' | 'waveform' | 'transcript' | 'hasAudio' | 'preparingAudio' | 'busy' | 'jobs' | 'jobsKey' | 'frameVersion' | 'onStarted'> & { onJump: (key: NeedKey) => void; tab: DialogTab; section: RunSection; onSection: (section: RunSection) => void; previousId?: string; nextId?: string; onPrevious?: () => void; onNext?: () => void }

const AnalysisDialogContent = memo(function AnalysisDialogContent({ cut, cuts, needs, onJump, previousId, nextId, episodeId, projectId, proxyReady, waveform, transcript, hasAudio, preparingAudio, busy, jobs, jobsKey, frameVersion, onStarted, tab, section, onSection, onPrevious, onNext }: ContentProps) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [resultsLoaded, setResultsLoaded] = useState(false)
  const [resultsRevision, setResultsRevision] = useState(0)
  const [, setConfig] = useState<AnalysisConfig | null>(null)
  const agentWork = useCutAgentWork(episodeId, cut.id)
  const agentMarks = useMemo(() => ({
    decisions: agentWork.decisions,
    // The questions an open Claude review_question escalation of this cut carries.
    claudeQuestions: new Set<string>(agentWork.escalations.filter(item => item.reviewer === 'claude' && item.kind === 'review_question')
      .flatMap(item => (item.evidence?.questions ?? []).map((question: { id: string }) => question.id))),
    onChanged: () => { agentWork.reload(); void onStarted() },
  }), [agentWork.decisions, agentWork.escalations, agentWork.reload, onStarted])
  const player = usePlayer()
  // The shared video as the ref the audio editor reads it through.
  const video = useMemo(() => ({ current: player.element }), [player])
  const [editorTarget, setEditorTarget] = useState<HTMLElement | null>(null)
  const [framesOpen, setFramesOpen] = useState(false)
  const [runVideoOpen, setRunVideoOpen] = useState(false)
  // The video plays this cut only: it pauses on the last frame (frame by frame) and Play from there starts the cut again.
  // Arriving at a cut stops what was playing and puts the playhead at the cut's start; a timing edit only moves it when it falls outside.
  const arriving = useRef(true)
  useEffect(() => {
    if (arriving.current) player.pause()
    player.setRange({ start: cut.start_time, end: cut.end_time }, arriving.current)
    arriving.current = false
  }, [player, cut.start_time, cut.end_time])
  useEffect(() => {
    const element = player.element
    const toStart = () => player.setRange({ start: cut.start_time, end: cut.end_time }, true)
    element.addEventListener('loadedmetadata', toStart)
    return () => element.removeEventListener('loadedmetadata', toStart)
  }, [player, cut.start_time, cut.end_time])
  // The proxy is requested the first time a video is shown (the Run tab waits for the "Load video preview" button).
  const showsVideo = tab === 'results' || runVideoOpen
  useEffect(() => { if (proxyReady && showsVideo) player.load(mediaUrl(episodeId, 'proxy')) }, [player, proxyReady, showsVideo, episodeId])
  const [charactersTarget, setCharactersTarget] = useState<HTMLElement | null>(null)
  const prefetchIds = useMemo(() => [previousId, nextId], [previousId, nextId])
  // The episode's job list is bounded; the cut's own history comes from its endpoint and is read ahead for the neighbours.
  const { jobs: cutJobs, loaded: jobsLoaded } = useCutJobs(cut.id, jobs)
  useEffect(() => { prefetchCutJobs(prefetchIds) }, [prefetchIds])
  const stages = useCutStages(cut.id, tab === 'run', jobsKey)
  const current = results.find(result => result.title === SUMMARY_TITLE && !result.superseded_by)
  const consistency = results.find(result => result.settings.structured_type === 'consistency_summary' && !result.superseded_by)
  // The chat mounts on first visit and then stays mounted so a streaming reply survives tab switches.
  const [assistantMounted, setAssistantMounted] = useState(tab === 'assistant')
  useEffect(() => { if (tab === 'assistant') { setAssistantMounted(true); player.pause() } }, [tab, player])
  const onSummarySaved = useCallback((result: AnalysisResult) => { setResults(previous => [result, ...previous.filter(item => item.id !== result.parent_id)]); setResultsRevision(value => value + 1) }, [])
  function resultsChanged() {
    setResultsRevision(value => value + 1)
    window.dispatchEvent(new Event('framewise:cut-timeline-changed'))
    window.dispatchEvent(new Event('framewise:brief-instructions-changed'))
  }
  // On Results the layout fills the dialog and has no scrollbar, but a hidden-overflow box can still be scrolled by
  // scrollIntoView, focus or a scroll position carried over from the Run tab, which pushes the whole content up with no
  // way back. Keep those boxes at the top there (the wide layout only; narrow screens scroll the page normally).
  const grid = useRef<HTMLDivElement>(null)
  const resultsColumn = useRef<HTMLDivElement>(null)
  const pinTop = useCallback(() => {
    if (tab !== 'results' || !window.matchMedia('(min-width: 1024px)').matches) return
    for (const box of [grid.current, resultsColumn.current]) if (box && box.scrollTop !== 0) box.scrollTop = 0
  }, [tab])
  useLayoutEffect(pinTop, [pinTop])
  return <>
    {assistantMounted && <CutAssistant cut={cut} active={tab === 'assistant'} onResultsChanged={resultsChanged} />}
    <div ref={grid} onScroll={pinTop} className={`${tab === 'assistant' ? 'hidden' : 'grid'} min-h-0 flex-1 gap-5 overflow-y-auto p-5 lg:overflow-hidden ${tab === 'run' ? 'lg:grid-cols-[minmax(0,1fr)_minmax(400px,460px)]' : 'lg:grid-cols-[16rem_minmax(0,1fr)] xl:grid-cols-[18rem_minmax(0,1fr)] 2xl:grid-cols-[19rem_minmax(0,1fr)]'}`}>
      <div ref={resultsColumn} onScroll={pinTop} className={`scrollbar flex flex-col gap-3 ${tab === 'run' ? 'lg:order-2 lg:min-h-0 lg:overflow-y-auto lg:pr-1' : 'order-2 min-h-0 lg:overflow-hidden'}`}>
        {/* Both stay mounted so unsaved settings and drafts survive a tab switch. */}
        <div className={tab === 'results' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <CutNeedsBar needs={needs} escalations={agentWork.escalations} decisions={agentWork.decisions} onJump={onJump} onChanged={agentMarks.onChanged} />
          <CutSummaryPanel otherDecisions={<CutEscalations escalations={agentWork.escalations} onChanged={agentMarks.onChanged} />} editorSlot={<div ref={setEditorTarget} className="editor-overlay" />} review={answers => <ConsistencySummary cutId={cut.id} result={consistency} loaded={resultsLoaded} jobs={cutJobs} disabled={!cut.active} onStarted={onStarted} answers={answers} agent={agentMarks} lead={<PerformanceBeats cutId={cut.id} results={results} jobs={cutJobs} disabled={!cut.active} onStarted={onStarted} />} extra={<ResultEditor cutId={cut.id} refreshKey={`${jobsKey}:${resultsRevision}`} onLoaded={value => { setResults(value); setResultsLoaded(true) }} onChanged={onStarted} editorTarget={editorTarget} />} />} cutId={cut.id} current={current} loaded={resultsLoaded} jobs={cutJobs} disabled={!cut.active || !cut.analysis_enabled} onStarted={onStarted} onSaved={onSummarySaved} />
        </div>
        <div hidden={tab !== 'run'} data-need={section === 'targets' ? 'targets' : 'characters'}>{cut.active
          ? <AnalysisControls jobs={cutJobs} cut={cut} cuts={cuts} episodeId={episodeId} busy={busy} enabled={cut.analysis_enabled} results={results} stages={stages} prefetchIds={prefetchIds} charactersTarget={charactersTarget} section={section} onSection={onSection} onStarted={onStarted} onConfigChange={setConfig} onPrevious={onPrevious} onNext={onNext}><CutRunHistory cutId={cut.id} jobs={cutJobs} loaded={jobsLoaded} stages={stages} onChanged={onStarted} /></AnalysisControls>
          : <div className="space-y-4"><p className="muted text-xs">Archived cuts can be reviewed but not analyzed.</p><CutRunHistory cutId={cut.id} jobs={cutJobs} loaded={jobsLoaded} stages={stages} onChanged={onStarted} /></div>}</div>
      </div>
      <div className={`scrollbar flex flex-col gap-3 lg:overflow-y-auto lg:pr-1 ${tab === 'run' ? 'lg:order-1 lg:min-h-0' : 'order-1 min-h-0'}`}>
        <div ref={setCharactersTarget} className={tab === 'run' ? 'block' : 'hidden'} />
        {tab === 'run' && <button type="button" className="btn btn-quiet self-start" onClick={() => setRunVideoOpen(value => !value)}>{runVideoOpen ? 'Hide video preview' : 'Load video preview'}</button>}
        {showsVideo && <div className={`shrink-0 ${tab === 'run' ? 'mx-auto w-full lg:max-w-[82vh]' : 'mx-auto w-full max-w-sm lg:max-w-none'}`}>{proxyReady
          ? <PlayerSlot className="aspect-video"><PlayheadReadout start={cut.start_time} end={cut.end_time} />{tab === 'results' && <FrameTools cut={cut} results={results} frameVersion={frameVersion} onPreview={() => setFramesOpen(true)} />}</PlayerSlot>
          : <div className="relative aspect-video overflow-hidden rounded-xl border hairline bg-black"><div className="muted flex h-full items-center justify-center text-xs">Run Prepare to create the video preview.</div></div>}</div>}
        {tab === 'results' && <CutAudioPreview cut={cut} waveform={waveform} transcript={transcript} hasAudio={hasAudio} preparing={preparingAudio} video={video} canPlay={proxyReady} projectId={projectId} episodeId={episodeId} onChanged={onStarted} marks={{ voice: hasNeed(needs, 'voice'), lines: hasNeed(needs, 'lines_to_check') || hasNeed(needs, 'speaker_unsure') }} />}
      </div>
    </div>
    {framesOpen && <FramesDialog cut={cut} results={results} frameVersion={frameVersion} onClose={() => setFramesOpen(false)} />}
  </>
})
