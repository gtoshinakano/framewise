import type { Cut } from './api'

function stamp(seconds: number) {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = (seconds % 60).toFixed(2).padStart(5, '0')
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secs}`
}

export function CutGallery({ cuts, selectedId, onSelect, onOpenAnalysis, frameVersion }: { cuts: Cut[]; selectedId: string | null; onSelect: (cut: Cut) => void; onOpenAnalysis: (cut: Cut) => void; frameVersion: string }) {
  return <section className="p-4"><div className="mb-4 flex items-center justify-between"><h2 className="text-sm font-semibold">Detected cuts</h2><span className="muted text-xs">{cuts.length}</span></div>
    {cuts.length ? <div className="space-y-2">{cuts.map(cut => <button key={cut.id} onClick={() => onSelect(cut)} onDoubleClick={() => onOpenAnalysis(cut)} className={`flex w-full gap-3 rounded-lg border p-2 text-left ${selectedId === cut.id ? 'border-[#ab8df2] bg-[#282238]' : 'border-[#303641] bg-[#171c24] hover:border-[#6e6388]'}`}><img src={`/api/cuts/${cut.id}/frame/first?v=${encodeURIComponent(frameVersion)}`} className="aspect-video w-24 shrink-0 rounded bg-[#222938] object-cover" alt={`First frame of cut ${cut.cut_index + 1}`} loading="lazy" /><div className="min-w-0 text-[10px] leading-4"><strong className="block truncate text-xs">{cut.title || `Cut ${cut.cut_index + 1}`}</strong><span className="muted block">Duration {stamp(cut.end_time - cut.start_time)}</span><span className="muted block">{stamp(cut.start_time)} → {stamp(cut.end_time)}</span><span className="muted block">Frames {cut.first_frame_time.toFixed(2)}s → {cut.last_frame_time.toFixed(2)}s</span></div></button>)}</div> : <p className="muted text-xs leading-5">Run hard-cut detection to list cuts here.</p>}
  </section>
}
