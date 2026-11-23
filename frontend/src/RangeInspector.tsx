import { useMemo } from 'react'
import { Bot, Clapperboard, X } from 'lucide-react'
import type { Cut } from './api'
import { PANEL_ACCENT } from './InspectorRail'
import { StatusChip } from './RunUi'
import { formatTimecode } from './timecode'
import { minutesSeconds, NEED_LABEL, rangeSummary, STEP_LABEL, type NeedsMap } from './cutNeedsView'

const SEQUENCE_LIMIT = 180

/** Several cuts selected with Shift+click on the timeline: their times, and what each still needs. */
export function RangeInspector({ cuts, needs, from, to, onSelect, onOpenAnalysis, onClear, onCreateSequence }: {
  cuts: Cut[]; needs: NeedsMap; from: number; to: number
  onSelect: (cut: Cut) => void; onOpenAnalysis: (cut: Cut) => void; onClear: () => void; onCreateSequence: (from: number, to: number) => void
}) {
  const summary = useMemo(() => rangeSummary(cuts, needs, from, to), [cuts, needs, from, to])
  if (!summary) return null
  const first = summary.first.cut_index + 1, last = summary.last.cut_index + 1
  const tooLong = summary.seconds > SEQUENCE_LIMIT
  const people = summary.totals.filter(total => total.kind !== 'step')
  const steps = summary.totals.filter(total => total.kind === 'step')
  return <div className="scrollbar h-full overflow-y-auto p-4">
    <div className="flex items-start justify-between gap-2">
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-[.14em]" style={{ color: PANEL_ACCENT.cut }}>Cuts {first}–{last}<span className="muted ml-2 normal-case tracking-normal">{summary.count} cuts{summary.analysisOff ? ` · ${summary.analysisOff} with analysis off` : ''}</span></p>
        <h2 className="mt-1 text-[15px] font-semibold">Selected range</h2>
      </div>
      <button type="button" onClick={onClear} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Clear the range (Esc)" title="Clear the range (Esc)"><X size={16} /></button>
    </div>
    <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 rounded-lg border hairline bg-[#171c25] p-3 text-xs">
      <div><dt className="muted text-[11px]">Starts</dt><dd className="font-mono tabular-nums">{formatTimecode(summary.start)}</dd></div>
      <div><dt className="muted text-[11px]">Ends</dt><dd className="font-mono tabular-nums">{formatTimecode(summary.end)}</dd></div>
      <div><dt className="muted text-[11px]">Duration</dt><dd className="font-mono tabular-nums">{minutesSeconds(summary.seconds)}</dd></div>
      <div><dt className="muted text-[11px]">Total</dt><dd className="font-mono tabular-nums">{summary.seconds.toFixed(3)} s</dd></div>
    </dl>
    <button type="button" className="btn btn-primary mt-3 w-full" onClick={() => onCreateSequence(first, last)} title={tooLong ? 'A sequence may last at most 3 minutes; the form lets you shorten the range' : 'Open the New sequence form with this range'}>
      <Clapperboard size={15} aria-hidden /> Create sequence
    </button>
    {tooLong && <p className="mt-1 text-[11px] text-[#ffd59a]">This range is longer than the 3 minutes a sequence may last.</p>}

    <h3 className="mb-1.5 mt-4 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Needs you</h3>
    {people.length ? <ul className="space-y-1 text-xs">{people.map(total => <li key={total.key} className="flex items-start gap-2">
      <StatusChip tone={total.kind === 'claude' ? 'purple' : 'amber'} icon={total.kind === 'claude' ? <Bot size={11} aria-hidden /> : undefined} className="shrink-0">{total.label} · {total.cuts.length}</StatusChip>
      <span className="muted min-w-0 break-words pt-0.5 font-mono text-[11px]">{total.cuts.join(', ')}</span>
    </li>)}</ul> : <p className="muted text-xs">Nothing in this range waits for you.</p>}

    <h3 className="mb-1.5 mt-4 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Steps still to run</h3>
    {steps.length ? <ul className="space-y-1 text-xs">{steps.map(total => <li key={total.key} className="flex items-start gap-2">
      <StatusChip tone="grey" className="shrink-0">{total.label} · {total.cuts.length}</StatusChip>
      <span className="muted min-w-0 break-words pt-0.5 font-mono text-[11px]">{total.cuts.join(', ')}</span>
    </li>)}</ul> : <p className="muted text-xs">Every step has run on these cuts.</p>}

    <h3 className="mb-1.5 mt-4 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Per cut</h3>
    <ul className="space-y-1">{summary.rows.map(row => {
      const own = row.needs
      return <li key={row.cut.id}>
        <button type="button" onClick={() => onSelect(row.cut)} onDoubleClick={() => onOpenAnalysis(row.cut)} title="Click to select this cut; double-click to open its analysis"
          className="w-full rounded-lg border border-[#2c323d] bg-[#171c25] px-2.5 py-1.5 text-left hover:border-[#5c497d]">
          <span className="flex items-baseline justify-between gap-2 text-xs"><span className="font-semibold">Cut {row.number}{!row.cut.analysis_enabled && <span className="muted font-normal"> · analysis off</span>}</span><span className="muted font-mono text-[11px] tabular-nums">{row.seconds.toFixed(2)} s</span></span>
          {own && (own.people.length > 0 || own.claude.length > 0 || own.steps.length > 0) ? <span className="mt-1 flex flex-wrap gap-1">
            {own.people.map(item => <StatusChip key={item.key} tone="amber">{NEED_LABEL[item.key]}{item.count > 1 ? ` · ${item.count}` : ''}</StatusChip>)}
            {own.claude.length > 0 && <StatusChip tone="purple" icon={<Bot size={11} aria-hidden />}>Claude · {own.claude.length}</StatusChip>}
            {own.steps.map(step => <StatusChip key={step} tone="grey">{STEP_LABEL[step]}</StatusChip>)}
          </span> : <span className="muted mt-0.5 block text-[11px]">{own ? 'Nothing left to do' : '—'}</span>}
        </button>
      </li>
    })}</ul>
  </div>
}
