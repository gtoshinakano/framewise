import { useEffect, useState } from 'react'
import { FrameImage } from './FrameImage'
import type { Cut } from './api'

function stamp(seconds: number) {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = (seconds % 60).toFixed(2).padStart(5, '0')
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secs}`
}

export function CutGallery({ cuts, selectedId, onSelect, onOpenAnalysis, frameVersion }: { cuts: Cut[]; selectedId: string | null; onSelect: (cut: Cut) => void; onOpenAnalysis: (cut: Cut) => void; frameVersion: string }) {
  const [page, setPage] = useState(0)
  const pages = Math.max(1, Math.ceil(cuts.length / 24))
  const visiblePage = Math.min(page, pages - 1)
  useEffect(() => { const index = cuts.findIndex(cut => cut.id === selectedId); if (index >= 0) setPage(Math.floor(index / 24)) }, [selectedId])
  return <section className="p-4"><div className="mb-4 flex items-center justify-between"><h2 className="text-sm font-semibold">Detected cuts</h2><span className="muted text-xs">{cuts.length}</span></div>
    {cuts.length ? <><div className="space-y-2">{cuts.slice(visiblePage * 24, visiblePage * 24 + 24).map(cut => <button key={cut.id} onClick={() => onSelect(cut)} onDoubleClick={() => onOpenAnalysis(cut)} className={`flex w-full gap-3 rounded-lg border p-2 text-left ${selectedId === cut.id ? 'border-[#ab8df2] bg-[#282238]' : 'border-[#303641] bg-[#171c24] hover:border-[#6e6388]'}`}><FrameImage fallbackClassName="flex aspect-video w-24 shrink-0 items-center justify-center rounded bg-[#222938] p-1 text-center text-[9px] text-[#98a1af]" src={`/api/cuts/${cut.id}/frame/first?v=${encodeURIComponent(frameVersion)}`} className="aspect-video w-24 shrink-0 rounded bg-[#222938] object-cover" alt={`First frame of cut ${cut.cut_index + 1}`} loading="lazy" /><div className="min-w-0 text-[10px] leading-4"><strong className="block truncate text-xs">{cut.title || `Cut ${cut.cut_index + 1}`}</strong>{!cut.analysis_enabled && <span className="block text-[#e9b971]">Analysis disabled</span>}<span className="muted block">Duration {stamp(cut.end_time - cut.start_time)}</span><span className="muted block">{stamp(cut.start_time)} → {stamp(cut.end_time)}</span><span className="muted block">Frames {cut.first_frame_time.toFixed(2)}s → {cut.last_frame_time.toFixed(2)}s</span></div></button>)}</div>{pages > 1 && <div className="mt-3 flex items-center justify-between gap-2 text-xs"><button className="btn btn-quiet" disabled={visiblePage === 0} onClick={() => setPage(visiblePage - 1)}>Previous</button><span className="muted">{visiblePage + 1} / {pages}</span><button className="btn btn-quiet" disabled={visiblePage >= pages - 1} onClick={() => setPage(visiblePage + 1)}>Next</button></div>}</> : <p className="muted text-xs leading-5">Run hard-cut detection to list cuts here.</p>}
  </section>
}
