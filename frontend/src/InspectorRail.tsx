import { useRef, type KeyboardEvent, type ReactNode } from 'react'
import { AudioLines, Clapperboard, Layers3, LoaderCircle, Scissors, type LucideIcon } from 'lucide-react'

export type InspectorTab = 'cut' | 'chunk' | 'audio' | 'sequences'

/** Each panel keeps one accent colour, used by its rail tab and its heading. */
export const PANEL_ACCENT: Record<InspectorTab, string> = { cut: '#bca7ff', chunk: '#7bc5cb', audio: '#e89aab', sequences: '#e0a96d' }

const TABS: { value: InspectorTab; label: string; title: string; icon: LucideIcon }[] = [
  { value: 'cut', label: 'Cut', title: 'Selected cut: notes, frames and timing', icon: Scissors },
  { value: 'chunk', label: 'Chunk', title: 'Selected chunk: working segment notes', icon: Layers3 },
  { value: 'audio', label: 'Audio', title: 'Transcript, transcription tools and voices', icon: AudioLines },
  { value: 'sequences', label: 'Sequence', title: 'Sequences: short video directions', icon: Clapperboard },
]

/** The vertical tab list on the right edge of the editor. It never runs out of room for another panel. */
export function InspectorRail({ value, onChange, busy, counts }: { value: InspectorTab; onChange: (tab: InspectorTab) => void; busy: Partial<Record<InspectorTab, boolean>>; counts: Partial<Record<InspectorTab, number>> }) {
  const list = useRef<HTMLDivElement>(null)
  function onKey(event: KeyboardEvent<HTMLDivElement>) {
    const keys = ['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft', 'Home', 'End']
    if (!keys.includes(event.key)) return
    event.preventDefault()
    const index = TABS.findIndex(tab => tab.value === value)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? TABS.length - 1 : (index + (event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length
    onChange(TABS[next].value)
    list.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }
  return <div ref={list} role="tablist" aria-label="Inspector panels" aria-orientation="vertical" onKeyDown={onKey} className="flex flex-row lg:flex-col">
    {TABS.map(tab => {
      let badge: ReactNode = null
      if (busy[tab.value]) badge = <LoaderCircle size={11} className="rail-badge animate-spin text-[#f5c678] motion-reduce:animate-none" aria-label="Working" />
      else if (counts[tab.value]) badge = <span className="rail-badge badge !h-[14px] !min-w-[14px] !text-[9px]">{counts[tab.value]}</span>
      return <button key={tab.value} type="button" role="tab" id={`inspector-tab-${tab.value}`} aria-controls="inspector-panel" aria-selected={value === tab.value} tabIndex={value === tab.value ? 0 : -1}
        title={tab.title} onClick={() => onChange(tab.value)} className="rail-tab" style={{ ['--accent' as string]: PANEL_ACCENT[tab.value] }}>
        <tab.icon size={18} aria-hidden /><span>{tab.label}</span>{badge}
      </button>
    })}
  </div>
}
