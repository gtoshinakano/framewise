import { BoxSelect, ClipboardCheck, Eye, EyeOff, FileText, Filter, ScanEye, ScanFace, Timer, TriangleAlert, UserRound, X, type LucideIcon } from 'lucide-react'
import type { Cut } from './api'
import { CUT_FILTERS, groupFilters, matchesAnyFilter, type CutFilterKey, type FilterDisplay } from './cutFilters'
import { SegmentedControl } from './EditorUi'
import { Popover } from './Popover'

/** Icons per filter. A filter added later without an entry here gets the generic filter icon. */
const FILTER_ICONS: Partial<Record<CutFilterKey, LucideIcon>> = {
  needs_analysis: ScanEye, no_boxes: BoxSelect, unknown_characters: UserRound, unknown_target_characters: ScanFace, needs_consistency: ClipboardCheck,
  needs_summary: FileText, job_failed: TriangleAlert, long_cut: Timer,
}
export const filterIcon = (key: CutFilterKey) => FILTER_ICONS[key] || Filter

/** The Filters button of the timeline toolbar and its popover: every filter by group, with counts and the Dim/Hide switch. */
export function CutFilterPopover({ cuts, active, display, onToggle, onClear, onDisplay, open, onOpenChange }: {
  cuts: Cut[]
  active: ReadonlySet<CutFilterKey>
  display: FilterDisplay
  onToggle: (key: CutFilterKey) => void
  onClear: () => void
  onDisplay: (display: FilterDisplay) => void
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const shown = active.size ? cuts.filter(cut => matchesAnyFilter(cut, active)).length : cuts.length
  return <Popover label="Filter cuts" side="top" align="end" open={open} onOpenChange={onOpenChange} panelClassName="w-[340px] p-0"
    trigger={props => <button type="button" {...props} className="icon-btn icon-btn-sm icon-btn-text" title="Filter cuts (F)" aria-pressed={active.size > 0}>
      <Filter size={14} aria-hidden className={active.size ? 'text-[#bca7ff]' : undefined} /><span>Filters</span>{active.size > 0 && <span className="badge">{active.size}</span>}
    </button>}>
    <div className="flex items-center justify-between gap-2 border-b hairline px-3 py-2.5">
      <div><h3 className="text-sm font-semibold">Filter cuts</h3><p className="muted text-[11px]" aria-live="polite">{active.size ? `Showing ${shown} of ${cuts.length} cuts that need one of these` : `All ${cuts.length} cuts shown`}</p></div>
      <button type="button" className="text-xs text-[#c3aeff] hover:underline disabled:text-[#6f7787] disabled:no-underline" disabled={!active.size} onClick={onClear}>Clear all</button>
    </div>
    <div className="px-2 py-2">
      {groupFilters().map(({ group, filters }) => <fieldset key={group} className="mb-1.5 last:mb-0">
        <legend className="px-1.5 pb-1 pt-1 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">{group}</legend>
        {filters.map(filter => {
          const on = active.has(filter.key)
          const count = cuts.filter(filter.matches).length
          const Icon = filterIcon(filter.key)
          return <label key={filter.key} className={`flex cursor-pointer items-start gap-2.5 rounded-lg px-2 py-1.5 hover:bg-[#222835] ${on ? 'bg-[#1f2330]' : ''}`}>
            <input type="checkbox" className="mt-1 accent-[#a58ae8]" checked={on} onChange={() => onToggle(filter.key)} />
            <span className="mt-[5px] h-2 w-2 shrink-0 rounded-full" style={{ background: filter.color }} aria-hidden />
            <Icon size={14} className="mt-0.5 shrink-0" style={{ color: filter.color }} aria-hidden />
            <span className="min-w-0 flex-1"><span className="flex items-center justify-between gap-2 text-[13px] font-medium"><span>{filter.label}</span><span className={`font-mono text-[11px] tabular-nums ${count ? 'text-[#e4e5ea]' : 'muted'}`}>{count}</span></span>
              <span className="muted block text-[11px] leading-4">{filter.hint}</span></span>
          </label>
        })}
      </fieldset>)}
    </div>
    <div className="flex items-center justify-between gap-3 border-t hairline px-3 py-2.5 text-xs">
      <span className="muted">Other cuts</span>
      <SegmentedControl label="How the timeline shows cuts that match no filter" value={display} onChange={onDisplay}
        options={[{ value: 'dim', label: 'Dim', icon: Eye }, { value: 'hide', label: 'Hide', icon: EyeOff }]} />
    </div>
  </Popover>
}

/** The active filters as removable chips next to the Filters button. More than `limit` collapse into "+N". */
export function ActiveFilterChips({ active, onToggle, limit = 3 }: { active: ReadonlySet<CutFilterKey>; onToggle: (key: CutFilterKey) => void; limit?: number }) {
  const filters = CUT_FILTERS.filter(filter => active.has(filter.key))
  if (!filters.length) return null
  return <span className="flex min-w-0 items-center gap-1">
    {filters.slice(0, limit).map(filter => <span key={filter.key} className="chip" title={filter.hint}>
      <span className="h-2 w-2 rounded-full" style={{ background: filter.color }} aria-hidden />{filter.label}
      <button type="button" onClick={() => onToggle(filter.key)} aria-label={`Remove filter ${filter.label}`}><X size={12} /></button>
    </span>)}
    {filters.length > limit && <span className="chip" title={filters.slice(limit).map(filter => filter.label).join(', ')}>+{filters.length - limit}</span>}
  </span>
}
