import { useState } from 'react'
import type { Cut } from './api'

function stamp(seconds: number) {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export function ArchivedCuts({ cuts, selectedId, onSelect, onOpenAnalysis }: { cuts: Cut[]; selectedId: string | null; onSelect: (cut: Cut) => void; onOpenAnalysis: (cut: Cut) => void }) {
  const [expanded, setExpanded] = useState(false)
  const [limit, setLimit] = useState(24)
  if (!cuts.length) return null
  return <section className="border-t hairline px-6 py-6">
    <button type="button" onClick={() => setExpanded(value => !value)} className="flex w-full items-center justify-between gap-3 text-left">
      <span><strong className="text-sm">Archived cuts</strong><span className="muted ml-2 text-xs">{cuts.length} preserved after redetection</span></span>
      <span className="text-xs text-[#bca7ff]">{expanded ? 'Hide' : 'Show'}</span>
    </button>
    {expanded && <><p className="muted mt-2 text-xs">Earlier boundaries and their notes stay here. Redetection can restore a matching boundary.</p><div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {cuts.slice(0, limit).map(cut => <button key={cut.id} type="button" onClick={() => onSelect(cut)} onDoubleClick={() => onOpenAnalysis(cut)} className={`overflow-hidden rounded-xl border text-left transition ${selectedId === cut.id ? 'border-[#ab8df2] bg-[#211e2d]' : 'border-[#303641] bg-[#171c24] hover:border-[#6e6388]'}`}>
        <img src={`/api/cuts/${cut.id}/frame/first`} className="aspect-video w-full bg-[#222938] object-cover" alt={`${cut.title || 'Archived cut'} preview`} loading="lazy" />
        <span className="block p-3"><strong className="block truncate text-xs">{cut.title || 'Archived cut'}</strong><span className="muted mt-1 block truncate text-[11px]">{stamp(cut.start_time)} → {stamp(cut.end_time)}</span></span>
      </button>)}
    </div>{cuts.length > limit && <button type="button" onClick={() => setLimit(value => value + 24)} className="btn btn-quiet mt-4">Show more</button>}</>}
  </section>
}
