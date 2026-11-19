import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { KINDS } from './AnalysisTimeline'

/** The three steps of the Results tab. Their colours and icons come from the Analysis timeline bar above them. */
export type StepKind = 'consistency_summary' | 'review_answers' | 'cut_summary'

/**
 * One column of the Results workbench: a numbered, colour-coded header, a body that scrolls on its own from `lg` up,
 * and an optional footer that stays in view. The status `tag` floats half outside the top edge so it takes no room
 * from the title (the container above must leave about 12px for it). In the two-column layout (below `2xl`) the
 * panel keeps its natural height and the column it sits in scrolls instead.
 */
export function StepPanel({ step, kind, title, subtitle, updated, tag, actions, footer, className = '', bodyClassName = '', children }: {
  step: number; kind: StepKind; title: string; subtitle: string; updated?: ReactNode; tag?: ReactNode; actions?: ReactNode; footer?: ReactNode; className?: string; bodyClassName?: string; children: ReactNode
}) {
  const { color, icon: Icon } = KINDS[kind]
  return <section aria-label={title} style={{ borderTopColor: color }} className={`relative flex shrink-0 flex-col rounded-xl border border-t-2 hairline bg-[#1b202a] lg:min-h-0 2xl:shrink ${className}`}>
    {tag && <div className="absolute right-3 top-0 z-10 -translate-y-1/2 rounded-full shadow-lg shadow-black/50">{tag}</div>}
    <header className="flex shrink-0 items-center gap-3 border-b hairline px-4 py-3">
      <span style={{ background: color }} className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[13px] font-bold text-[#11111a]" aria-hidden>{step}</span>
      <div className="min-w-0 flex-1"><h3 className="flex items-center gap-1.5 whitespace-nowrap text-sm font-semibold"><Icon size={15} style={{ color }} className="shrink-0" aria-hidden /><span className="truncate">{title}</span>{updated && <span className="shrink-0 font-normal text-[10px] text-[#9aa0a8]">{updated}</span>}</h3><p className="muted truncate text-[11px]">{subtitle}</p></div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </header>
    <div className={`scrollbar flex min-h-0 flex-1 flex-col gap-3 p-4 lg:overflow-y-auto ${bodyClassName}`}>{children}</div>
    {footer && <footer className="shrink-0 border-t hairline p-3">{footer}</footer>}
  </section>
}

/** Secondary details that stay out of the way until asked for. */
export function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  return <details className="group shrink-0 rounded-lg border hairline bg-[#171c25] text-xs">
    <summary className="flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 font-semibold text-[#aeb5c1] hover:text-white"><ChevronDown size={14} className="-rotate-90 transition-transform group-open:rotate-0" aria-hidden />{label}</summary>
    <div className="border-t hairline p-3">{children}</div>
  </details>
}
