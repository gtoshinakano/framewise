import { useEffect, useState } from 'react'
import { ChevronsLeft } from 'lucide-react'
import { FrameImage } from './FrameImage'
import { cutFrameUrl, type Cut } from './api'
import type { AnalysisQueueState } from './analysisQueueStates'
import { CutStatusIcons } from './CutStatus'
import { IconButton } from './EditorUi'
import { formatDuration, formatTimecode } from './timecode'

const PAGE = 30

export function CutGallery({ cuts, selectedId, onSelect, onOpenAnalysis, frameVersion, queueStates, onCollapse }: { cuts: Cut[]; selectedId: string | null; onSelect: (cut: Cut) => void; onOpenAnalysis: (cut: Cut) => void; frameVersion: string; queueStates: Map<string, AnalysisQueueState>; onCollapse: () => void }) {
  const [page, setPage] = useState(0)
  const pages = Math.max(1, Math.ceil(cuts.length / PAGE))
  const visiblePage = Math.min(page, pages - 1)
  useEffect(() => { const index = cuts.findIndex(cut => cut.id === selectedId); if (index >= 0) setPage(Math.floor(index / PAGE)) }, [selectedId])
  useEffect(() => { document.getElementById(`cut-card-${selectedId}`)?.scrollIntoView({ block: 'nearest' }) }, [selectedId, visiblePage])
  return <section className="p-2.5">
    <div className="mb-2 flex items-center justify-between px-1"><h2 className="text-xs font-semibold uppercase tracking-[.12em] text-[#c3c8d2]">Cuts <span className="muted ml-1 font-normal normal-case tracking-normal">{cuts.length}</span></h2>
      <IconButton icon={ChevronsLeft} label="Hide the cut list" size="sm" className="ghost" onClick={onCollapse} /></div>
    {cuts.length ? <><div className="space-y-1">{cuts.slice(visiblePage * PAGE, visiblePage * PAGE + PAGE).map(cut => <button id={`cut-card-${cut.id}`} key={cut.id} onClick={() => onSelect(cut)} onDoubleClick={() => onOpenAnalysis(cut)} title="Click to select, double-click to open the analysis"
      className={`flex w-full items-center gap-2 rounded-lg border p-1.5 text-left ${selectedId === cut.id ? 'border-[#ab8df2] bg-[#282238]' : 'border-transparent bg-[#171c24] hover:border-[#4a4361]'}`}>
      <FrameImage fallbackClassName="flex aspect-video w-20 shrink-0 items-center justify-center rounded bg-[#222938] p-1 text-center text-[9px] text-[#98a1af]" src={cutFrameUrl(cut, 'first', frameVersion)} className="aspect-video w-20 shrink-0 rounded bg-[#222938] object-cover" alt={`First frame of cut ${cut.cut_index + 1}`} loading="lazy" />
      <span className="min-w-0 flex-1 text-[10px] leading-4">
        <span className="flex items-center gap-1.5"><b className="text-[11px] text-[#c3aeff]">{cut.cut_index + 1}</b><strong className="truncate text-xs font-semibold">{cut.title || `Cut ${cut.cut_index + 1}`}</strong></span>
        <span className="muted block truncate font-mono tabular-nums">{formatTimecode(cut.start_time, { precision: 1 })} · {formatDuration(cut.end_time - cut.start_time)}</span>
        <span className="mt-0.5 flex items-center gap-1.5"><CutStatusIcons cut={cut} queueState={queueStates.get(cut.id)} size={11} />{!cut.analysis_enabled && <span className="text-[#e9b971]">Off</span>}</span>
      </span>
    </button>)}</div>{pages > 1 && <div className="mt-2 flex items-center justify-between gap-2 text-xs"><button className="btn btn-quiet !px-2.5 !py-1" disabled={visiblePage === 0} onClick={() => setPage(visiblePage - 1)}>Previous</button><span className="muted">{visiblePage + 1} / {pages}</span><button className="btn btn-quiet !px-2.5 !py-1" disabled={visiblePage >= pages - 1} onClick={() => setPage(visiblePage + 1)}>Next</button></div>}</> : <p className="muted px-1 text-xs leading-5">Run hard-cut detection to list cuts here.</p>}
  </section>
}
