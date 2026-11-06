import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, X } from 'lucide-react'
import { mediaUrl, type AnalysisConfig, type AnalysisResult, type Cut, type Job, type TranscriptSegment, type Waveform } from './api'
import AnalysisControls from './AnalysisControls'
import ResultEditor from './ResultEditor'
import { CutSummaryPanel, SUMMARY_TITLE } from './AnimationPromptPanel'
import { DetectionPreview } from './DetectionPreview'
import { RunFramePreview } from './RunFramePreview'
import { CutAudioPreview } from './CutAudioPreview'
import { CutRunHistory } from './CutRunHistory'
import { ConsistencySummary } from './ConsistencySummary'

type DialogProps = { cut: Cut; previousCut?: Cut; nextCut?: Cut; onNavigate: (cut: Cut) => void; episodeId: string; proxyReady: boolean; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparingAudio: boolean; busy: boolean; jobs: Job[]; jobsKey: string; frameVersion: string; onStarted: () => Promise<void>; onClose: () => void }

export default function AnalysisDialog({ cut, previousCut, nextCut, onNavigate, onClose, ...contentProps }: DialogProps) {
  const [tab, setTab] = useState<'results' | 'run'>(() => cut.active && cut.has_analysis_results === false ? 'run' : 'results')
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); opener?.focus() }
  }, [onClose])
  const title = cut.title || (cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut')
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Analysis for ${title}`} className="flex h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="grid grid-cols-[1fr_auto] items-center gap-3 border-b hairline px-5 py-3 lg:grid-cols-[1fr_auto_1fr]"><div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Cut analysis</p><h2 className="truncate text-base font-semibold">{title}</h2></div><div className="order-3 col-span-2 flex justify-center lg:order-none lg:col-span-1"><div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">{([['results', 'Results'], ['run', 'Run settings']] as const).map(([value, label]) => <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`rounded-md px-5 py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}</button>)}</div></div><div className="flex items-center justify-end gap-2"><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!previousCut} onClick={() => previousCut && onNavigate(previousCut)} aria-label="Previous cut analysis"><ArrowLeft size={14} /> Previous</button><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!nextCut} onClick={() => nextCut && onNavigate(nextCut)} aria-label="Next cut analysis">Next <ArrowRight size={14} /></button><button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close analysis"><X size={18} /></button></div></header>
      <AnalysisDialogContent key={cut.id} cut={cut} tab={tab} onPrevious={previousCut ? () => onNavigate(previousCut) : undefined} onNext={nextCut ? () => onNavigate(nextCut) : undefined} {...contentProps} />
    </div>
  </div>
}

type ContentProps = Pick<DialogProps, 'cut' | 'episodeId' | 'proxyReady' | 'waveform' | 'transcript' | 'hasAudio' | 'preparingAudio' | 'busy' | 'jobs' | 'jobsKey' | 'frameVersion' | 'onStarted'> & { tab: 'results' | 'run'; onPrevious?: () => void; onNext?: () => void }

function AnalysisDialogContent({ cut, episodeId, proxyReady, waveform, transcript, hasAudio, preparingAudio, busy, jobs, jobsKey, frameVersion, onStarted, tab, onPrevious, onNext }: ContentProps) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [resultsLoaded, setResultsLoaded] = useState(false)
  const [resultsRevision, setResultsRevision] = useState(0)
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [playTime, setPlayTime] = useState(0)
  function playSegment(time: number) {
    const player = video.current
    if (!player) return
    player.currentTime = Math.min(Math.max(time, cut.start_time), Math.max(cut.start_time, cut.end_time - 0.001))
    void player.play().catch(() => {})
  }
  const [editorTarget, setEditorTarget] = useState<HTMLElement | null>(null)
  const current = results.find(result => result.title === SUMMARY_TITLE && !result.superseded_by)
  const consistency = results.find(result => result.settings.structured_type === 'consistency_summary' && !result.superseded_by)
  return <>
    <div className={`grid min-h-0 flex-1 gap-5 overflow-y-auto p-5 lg:overflow-hidden ${tab === 'run' ? 'lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)]' : 'lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]'}`}>
      <div className={`scrollbar flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-1 ${tab === 'run' ? 'lg:order-2' : ''}`}>
        {/* Both stay mounted so unsaved settings and drafts survive a tab switch. */}
        <div className={tab === 'results' ? 'flex min-h-[30rem] flex-1 flex-col lg:min-h-[26rem]' : 'hidden'}>
          {consistency && <div className="mb-3"><ConsistencySummary cutId={cut.id} result={consistency} disabled={busy || !cut.active} onStarted={onStarted} /></div>}
          <CutSummaryPanel textProvider={config?.text_provider} cutId={cut.id} current={current} loaded={resultsLoaded} jobs={jobs} disabled={busy || !cut.active || !cut.analysis_enabled} onStarted={onStarted} onSaved={result => { setResults(previous => [result, ...previous.filter(item => item.id !== result.parent_id)]); setResultsRevision(value => value + 1) }} />
        </div>
        <div hidden={tab !== 'run'} className="space-y-4">{cut.active ? <AnalysisControls jobs={jobs} cut={cut} episodeId={episodeId} busy={busy} enabled={cut.analysis_enabled} onStarted={onStarted} onConfigChange={setConfig} onPrevious={onPrevious} onNext={onNext} /> : <p className="muted text-xs">Archived cuts can be reviewed but not analyzed.</p>}<CutRunHistory cutId={cut.id} jobs={jobs} onChanged={onStarted} /></div>
      </div>
      <div className={`scrollbar flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-1 ${tab === 'run' ? 'lg:order-1' : ''}`}>
        <div className="shrink-0">{tab === 'results' || !cut.active ? <div className="aspect-video overflow-hidden rounded-xl border hairline bg-black">{proxyReady ? <video ref={video} className="h-full w-full" controls preload="metadata" src={mediaUrl(episodeId, 'proxy')} aria-label="Play this cut" onLoadedMetadata={event => { event.currentTarget.currentTime = cut.start_time }} onSeeking={event => { const player = event.currentTarget; if (player.currentTime < cut.start_time) player.currentTime = cut.start_time; else if (player.currentTime > cut.end_time) player.currentTime = cut.end_time }} onPlay={event => { if (event.currentTarget.currentTime >= cut.end_time) event.currentTarget.currentTime = cut.start_time }} onTimeUpdate={event => { const player = event.currentTarget; setPlayTime(Math.max(0, Math.min(cut.end_time - cut.start_time, player.currentTime - cut.start_time))); if (player.currentTime >= cut.end_time && !player.paused) { player.pause(); if (player.currentTime > cut.end_time) player.currentTime = cut.end_time } }} /> : <div className="muted flex h-full items-center justify-center text-xs">Run Prepare to create the video preview.</div>}</div> : <RunFramePreview cut={cut} episodeId={episodeId} config={config} frameVersion={frameVersion} />}</div>
        {tab === 'results' && <details className="rounded-lg border hairline p-3"><summary className="cursor-pointer text-xs font-semibold text-[#c3aeff]">View analyzed frames and detections</summary><div className="mt-3"><DetectionPreview cut={cut} results={results} frameVersion={frameVersion} /></div></details>}
        {tab === 'results' && <CutAudioPreview cut={cut} waveform={waveform} transcript={transcript} hasAudio={hasAudio} preparing={preparingAudio} time={playTime} onPlaySegment={playSegment} canPlay={proxyReady} />}
        <div hidden={tab !== 'results'} className="scrollbar max-h-56 shrink-0 overflow-y-auto"><ResultEditor cutId={cut.id} refreshKey={`${jobsKey}:${resultsRevision}`} onLoaded={value => { setResults(value); setResultsLoaded(true) }} onChanged={onStarted} editorTarget={editorTarget} /></div>
        <div ref={setEditorTarget} className={tab === 'results' ? 'flex flex-1 flex-col' : 'hidden'} />
      </div>
    </div>
  </>
}
