import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { type AnalysisResult, type Cut, type Job } from './api'
import AnalysisControls from './AnalysisControls'
import ResultEditor from './ResultEditor'
import { AnimationPromptPanel, ANIMATION_TITLE } from './AnimationPromptPanel'
import { DetectionPreview } from './DetectionPreview'

export default function AnalysisDialog({ cut, episodeId, busy, job, frameVersion, onStarted, onClose }: { cut: Cut; episodeId: string; busy: boolean; job?: Job; frameVersion: string; onStarted: () => Promise<void>; onClose: () => void }) {
  const [tab, setTab] = useState<'results' | 'run'>('results')
  const [results, setResults] = useState<AnalysisResult[]>([])
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); opener?.focus() }
  }, [onClose])
  const current = [...results].reverse().find(result => result.title === ANIMATION_TITLE && !result.superseded_by)
  const jobKey = `${job?.id || ''}-${job?.status || ''}-${job?.stage || ''}`
  const title = cut.title || `Cut ${cut.cut_index + 1}`
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Analysis for ${title}`} className="flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="flex items-center justify-between gap-3 border-b hairline px-5 py-3"><div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Cut analysis</p><h2 className="truncate text-base font-semibold">{title}</h2></div><button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close analysis"><X size={18} /></button></header>
      <div className="grid min-h-0 flex-1 gap-5 overflow-y-auto p-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:overflow-hidden">
        <div className="scrollbar min-h-0 lg:overflow-y-auto lg:pr-1"><DetectionPreview cut={cut} results={results} frameVersion={frameVersion} /></div>
        <div className="scrollbar flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-1">
          <div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">{([['results', 'Results'], ['run', 'Run settings']] as const).map(([value, label]) => <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`flex-1 rounded-md py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{label}</button>)}</div>
          {/* Both stay mounted so unsaved settings and drafts survive a tab switch. */}
          <div hidden={tab !== 'results'} className="space-y-3">
            <AnimationPromptPanel cutId={cut.id} current={current} job={job} disabled={busy || !cut.active} onStarted={onStarted} />
            <ResultEditor cutId={cut.id} refreshKey={jobKey} onLoaded={setResults} />
          </div>
          <div hidden={tab !== 'run'}>{cut.active ? <AnalysisControls cutId={cut.id} episodeId={episodeId} busy={busy} onStarted={async () => { await onStarted(); setTab('results') }} /> : <p className="muted text-xs">Archived cuts can be reviewed but not analyzed.</p>}</div>
        </div>
      </div>
    </div>
  </div>
}
