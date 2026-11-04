import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, X } from 'lucide-react'
import { type AnalysisConfig, type AnalysisResult, type Cut, type Job, type TranscriptSegment, type Waveform } from './api'
import AnalysisControls from './AnalysisControls'
import ResultEditor from './ResultEditor'
import { CutSummaryPanel, SUMMARY_TITLE } from './AnimationPromptPanel'
import { DetectionPreview } from './DetectionPreview'
import { RunFramePreview } from './RunFramePreview'
import { CutAudioPreview } from './CutAudioPreview'

type DialogProps = { cut: Cut; previousCut?: Cut; nextCut?: Cut; onNavigate: (cut: Cut) => void; episodeId: string; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparingAudio: boolean; audioVersion: string; busy: boolean; jobs: Job[]; jobsKey: string; frameVersion: string; onStarted: () => Promise<void>; onClose: () => void }

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
      <header className="flex flex-wrap items-center justify-between gap-3 border-b hairline px-5 py-3"><div className="min-w-0 flex-1"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Cut analysis</p><h2 className="truncate text-base font-semibold">{title}</h2></div><div className="flex items-center gap-2"><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!previousCut} onClick={() => previousCut && onNavigate(previousCut)} aria-label="Previous cut analysis"><ArrowLeft size={14} /> Previous</button><button type="button" className="btn btn-quiet px-2.5 py-2" disabled={!nextCut} onClick={() => nextCut && onNavigate(nextCut)} aria-label="Next cut analysis">Next <ArrowRight size={14} /></button><button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close analysis"><X size={18} /></button></div></header>
      <AnalysisDialogContent key={cut.id} cut={cut} tab={tab} setTab={setTab} onPrevious={previousCut ? () => onNavigate(previousCut) : undefined} onNext={nextCut ? () => onNavigate(nextCut) : undefined} {...contentProps} />
    </div>
  </div>
}

type ContentProps = Pick<DialogProps, 'cut' | 'episodeId' | 'waveform' | 'transcript' | 'hasAudio' | 'preparingAudio' | 'audioVersion' | 'busy' | 'jobs' | 'jobsKey' | 'frameVersion' | 'onStarted'> & { tab: 'results' | 'run'; setTab: (tab: 'results' | 'run') => void; onPrevious?: () => void; onNext?: () => void }

function AnalysisDialogContent({ cut, episodeId, waveform, transcript, hasAudio, preparingAudio, audioVersion, busy, jobs, jobsKey, frameVersion, onStarted, tab, setTab, onPrevious, onNext }: ContentProps) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [resultsLoaded, setResultsLoaded] = useState(false)
  const [resultsRevision, setResultsRevision] = useState(0)
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const [editorTarget, setEditorTarget] = useState<HTMLElement | null>(null)
  const current = results.find(result => result.title === SUMMARY_TITLE && !result.superseded_by)
  return <>
    <div className={`grid min-h-0 flex-1 gap-5 overflow-y-auto p-5 lg:overflow-hidden ${tab === 'run' ? 'lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)]' : 'lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]'}`}>
      <div className={`scrollbar flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-1 ${tab === 'run' ? 'lg:order-2' : ''}`}>
        <div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">{([['results', 'Results'], ['run', 'Run settings']] as const).map(([value, label]) => <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`flex-1 rounded-md py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}</button>)}</div>
        {/* Both stay mounted so unsaved settings and drafts survive a tab switch. */}
        <div className={tab === 'results' ? 'flex min-h-[30rem] flex-1 flex-col lg:min-h-[26rem]' : 'hidden'}>
          <CutSummaryPanel cutId={cut.id} current={current} loaded={resultsLoaded} jobs={jobs} disabled={busy || !cut.active || !cut.analysis_enabled} onStarted={onStarted} onSaved={result => { setResults(previous => [result, ...previous.filter(item => item.id !== result.parent_id)]); setResultsRevision(value => value + 1) }} />
        </div>
        <div hidden={tab !== 'run'}>{cut.active ? <AnalysisControls cutId={cut.id} episodeId={episodeId} busy={busy} enabled={cut.analysis_enabled} onStarted={onStarted} onConfigChange={setConfig} onPrevious={onPrevious} onNext={onNext} /> : <p className="muted text-xs">Archived cuts can be reviewed but not analyzed.</p>}</div>
      </div>
      <div className={`scrollbar flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-1 ${tab === 'run' ? 'lg:order-1' : ''}`}>
        <div className="shrink-0">{tab === 'results' || !cut.active ? <DetectionPreview cut={cut} results={results} frameVersion={frameVersion} /> : <RunFramePreview cut={cut} episodeId={episodeId} config={config} frameVersion={frameVersion} />}</div>
        {tab === 'results' && <CutAudioPreview cut={cut} waveform={waveform} transcript={transcript} hasAudio={hasAudio} preparing={preparingAudio} audioVersion={audioVersion} />}
        <div hidden={tab !== 'results'} className="scrollbar max-h-56 shrink-0 overflow-y-auto"><ResultEditor cutId={cut.id} refreshKey={`${jobsKey}:${resultsRevision}`} onLoaded={value => { setResults(value); setResultsLoaded(true) }} onChanged={onStarted} editorTarget={editorTarget} /></div>
        <div ref={setEditorTarget} className={tab === 'results' ? 'flex flex-1 flex-col' : 'hidden'} />
      </div>
    </div>
  </>
}
