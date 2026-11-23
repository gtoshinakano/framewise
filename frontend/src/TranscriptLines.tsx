import { useEffect, useMemo, useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { Check, Merge, Plus, Scissors, Trash2, X } from 'lucide-react'
import type { ContextItem, Cut, TranscriptSegment } from './api'
import { IconButton } from './EditorUi'
import { contiguousRange, crossing, speakerColor, speakerConfidence, stamp } from './lineEditView'

/** The line being worked on: an existing one opened for editing, a new one, or the result of a merge that is not saved yet. */
export type Draft = {
  kind: 'edit' | 'new' | 'merge'
  /** The edited line; null for a new one. */
  id: string | null
  /** The lines a merge replaces. */
  ids: string[]
  start: number
  end: number
  text: string
  speaker_label: string
  character_context_id: string | null
  /** What the draft started as, to tell whether it holds changes. */
  base: { start: number; end: number; text: string; speaker_label: string; character_context_id: string | null }
}

export function draftDirty(draft: Draft) {
  const { base } = draft
  return draft.kind === 'merge' || draft.text !== base.text || draft.start !== base.start || draft.end !== base.end || draft.character_context_id !== base.character_context_id
}

type Props = {
  cut: Pick<Cut, 'start_time' | 'end_time'>
  lines: TranscriptSegment[]
  characters: ContextItem[]
  names: Map<string, string>
  selected: string[]
  onSelect: (ids: string[]) => void
  draft: Draft | null
  onDraft: (draft: Draft) => void
  busy: boolean
  /** The playhead, in episode seconds. */
  time: number
  onEdit: (line: TranscriptSegment) => void
  onSave: () => void
  onCancel: () => void
  onDelete: () => void
  /** `caret` is the textarea's cursor offset. */
  onSplit: (caret: number) => void
  canSplit: boolean
  onMerge: () => void
  onAdd: () => void
  onAssign: (line: TranscriptSegment, characterId: string | null) => void
  hint?: ReactNode
}

type Item = { type: 'line'; line: TranscriptSegment } | { type: 'draft' }

const INTERACTIVE = 'select,button,textarea,input,a'

export function TranscriptLines({ cut, lines, characters, names, selected, onSelect, draft, onDraft, busy, time, onEdit, onSave, onCancel, onDelete, onSplit, canSplit, onMerge, onAdd, onAssign, hint }: Props) {
  const area = useRef<HTMLTextAreaElement>(null)
  const anchor = useRef<string | null>(null)
  const dragging = useRef(false)
  useEffect(() => {
    const stop = () => { dragging.current = false }
    window.addEventListener('pointerup', stop); window.addEventListener('pointercancel', stop)
    return () => { window.removeEventListener('pointerup', stop); window.removeEventListener('pointercancel', stop) }
  }, [])

  const items = useMemo<Item[]>(() => {
    const result: Item[] = []
    let placed = false
    for (const line of lines) {
      const replaced = draft && ((draft.kind === 'edit' && draft.id === line.id) || (draft.kind === 'merge' && draft.ids.includes(line.id)))
      if (replaced) { if (!placed) { result.push({ type: 'draft' }); placed = true } continue }
      if (draft?.kind === 'new' && !placed && line.start_time > draft.start) { result.push({ type: 'draft' }); placed = true }
      result.push({ type: 'line', line })
    }
    if (draft && !placed) result.push({ type: 'draft' })
    return result
  }, [lines, draft])
  const order = useMemo(() => lines.filter(line => !(draft?.kind === 'merge' && draft.ids.includes(line.id))).map(line => line.id), [lines, draft])

  // The selected line scrolls into view when it was picked on the graph.
  const list = useRef<HTMLUListElement>(null)
  useEffect(() => {
    if (selected.length !== 1) return
    list.current?.querySelector(`[data-line="${CSS.escape(selected[0])}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  function press(event: PointerEvent, id: string) {
    if (event.button !== 0 || (event.target as HTMLElement).closest(INTERACTIVE)) return
    if (event.shiftKey && anchor.current) onSelect(contiguousRange(order, anchor.current, id))
    else { anchor.current = id; onSelect([id]) }
    dragging.current = true
  }
  function enter(id: string) {
    if (dragging.current && anchor.current) onSelect(contiguousRange(order, anchor.current, id))
  }
  function rowKey(event: KeyboardEvent, line: TranscriptSegment) {
    if (event.target !== event.currentTarget) return
    if (event.key === 'Enter') { event.preventDefault(); onEdit(line) }
    else if (event.key === ' ') { event.preventDefault(); anchor.current = line.id; onSelect([line.id]) }
    else if (event.key === 'Escape') onSelect([])
  }
  const speakerSelect = (value: string | null, label: string, onChange: (id: string | null) => void, unassignedStyle: boolean) =>
    <select aria-label={label} disabled={busy} value={value || ''} onChange={event => onChange(event.target.value || null)}
      className={`field shrink-0 !w-[6.2rem] !rounded-md !py-0.5 !text-[10px] ${unassignedStyle ? '!border-[#6d5526] !text-[#f0c979]' : ''}`}>
      <option value="">Unassigned</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
    </select>

  function draftRow() {
    if (!draft) return null
    const color = speakerColor(draft.character_context_id ? names.get(draft.character_context_id) || draft.speaker_label : draft.speaker_label || 'Unknown speaker')
    const valid = !!draft.text.trim() && draft.end > draft.start
    const title = draft.kind === 'new' ? 'New line' : draft.kind === 'merge' ? `Merged line from ${draft.ids.length} lines` : 'Editing'
    const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Escape') { event.preventDefault(); onCancel() }
      else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); if (valid && !busy) onSave() }
    }
    return <li key="draft" className="rounded-md border border-[#7a4a57] bg-[#1d171c] p-2" style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
      <div className="mb-1 flex items-center gap-1.5">
        <span className="min-w-0 flex-1 truncate text-[10px] font-bold uppercase tracking-wide" style={{ color }}>{title}</span>
        <span className="muted shrink-0 text-[10px] tabular-nums">{stamp(draft.start)}–{stamp(draft.end)}</span>
        {speakerSelect(draft.character_context_id, 'Who speaks this line?', id => onDraft({ ...draft, character_context_id: id }), !draft.character_context_id)}
      </div>
      <textarea ref={area} autoFocus aria-label="Line text" className="field min-h-[3.25rem] w-full resize-y !py-1.5 text-[12px] leading-[1.45]" maxLength={20000} value={draft.text} disabled={busy}
        placeholder={draft.kind === 'new' ? 'What is said here? Drag the handles on the waveform to time it.' : undefined}
        onChange={event => onDraft({ ...draft, text: event.target.value })} onKeyDown={onKey} />
      <div className="mt-1.5 flex items-center gap-1">
        {draft.kind === 'edit' && <IconButton icon={Scissors} label="Split at the cursor" size="sm" disabled={busy || !canSplit} onClick={() => onSplit(area.current?.selectionStart ?? 0)}
          title={canSplit ? 'Split the line at the text cursor' : 'Save the new timing before splitting'} />}
        <IconButton icon={Check} label="Save" shortcut="Ctrl+Enter" size="sm" tone="primary" disabled={busy || !valid || !draftDirty(draft)} onClick={onSave} />
        {draft.kind === 'edit' && <IconButton icon={Trash2} label="Delete this line" size="sm" tone="danger" disabled={busy} onClick={onDelete} />}
        <IconButton icon={X} label="Cancel" shortcut="Esc" size="sm" disabled={busy} onClick={onCancel} />
        <span className="flex-1" />
        {draft.kind === 'merge' && <span className="muted text-[10px]">Not saved yet</span>}
      </div>
    </li>
  }

  return <div className="flex min-h-0 flex-1 flex-col">
    <div className="mb-1.5 flex shrink-0 items-center gap-2">
      <p className="label !mb-0 flex-1">Spoken words</p>
      {selected.length >= 2 && !draft && <>
        <span className="muted text-[10px]">{selected.length} selected</span>
        <IconButton icon={Merge} label="Merge the selected lines" size="sm" text="Merge" disabled={busy} onClick={onMerge} />
        <IconButton icon={X} label="Clear the selection" size="sm" onClick={() => onSelect([])} />
      </>}
      <IconButton icon={Plus} label="Add a line at the playhead" size="sm" text="Add" disabled={busy || !!draft} onClick={onAdd} />
    </div>
    {items.length ? <ul ref={list} role="listbox" aria-multiselectable aria-label="Spoken lines" className="scrollbar min-h-0 max-h-80 flex-1 space-y-1 overflow-y-auto pr-0.5 lg:max-h-none">
      {items.map(item => {
        if (item.type === 'draft') return draftRow()
        const line = item.line
        const active = time >= line.start_time && time < line.end_time
        const isSelected = selected.includes(line.id)
        const assigned = line.character_context_id ? names.get(line.character_context_id) : null
        const speaker = assigned || line.speaker_label || 'Unknown speaker'
        const color = speakerColor(speaker)
        const side = crossing(line, cut)
        const sure = speakerConfidence(line, names)
        return <li key={line.id} data-line={line.id} role="option" aria-selected={isSelected} tabIndex={0} className="flex select-none items-start gap-1.5 outline-none focus-visible:[&>div]:ring-1 focus-visible:[&>div]:ring-[#c3aeff]"
          onPointerDown={event => press(event, line.id)} onPointerEnter={() => enter(line.id)} onKeyDown={event => rowKey(event, line)} onDoubleClick={event => { if (!(event.target as HTMLElement).closest(INTERACTIVE)) onEdit(line) }}
          title="Double-click to edit">
          <div style={{ borderLeftColor: color }} className={`min-w-0 flex-1 cursor-pointer rounded-md border-l-[3px] px-2 py-1 text-left ${isSelected ? 'bg-[#3a3058] ring-1 ring-[#8f78d8]' : active ? 'bg-[#322a48]' : 'bg-[#232a35] hover:bg-[#2a3140]'}`}>
            <span className="flex items-center gap-1.5"><span style={{ color }} className="block min-w-0 truncate text-[10px] font-bold uppercase tracking-wide">{speaker}</span>
              {side && <span className="shrink-0 rounded border border-[#6d5526] px-1 text-[9px] text-[#f0c979]" title={`${stamp(line.start_time)}–${stamp(line.end_time)}`}>{side === 'before' ? 'starts before cut' : side === 'after' ? 'continues after cut' : 'spans the cut'}</span>}
              {sure && <span className={`shrink-0 rounded border px-1 text-[9px] ${sure.blocking ? 'border-[#8a6a2a] bg-[#3a2c12] text-[#f5c678]' : 'border-[#4a4a3a] text-[#c9bd8a]'}`} title={sure.detail}>{sure.label}</span>}
              <span className="muted ml-auto shrink-0 text-[9px] tabular-nums">{stamp(line.start_time)}–{stamp(line.end_time)}</span></span>
            <span className="block break-words text-[12px] leading-[1.45]">{line.text}</span>
          </div>
          {speakerSelect(line.character_context_id, assigned ? `Character speaking "${line.text.slice(0, 48)}"` : 'Who speaks this line?', id => onAssign(line, id), !line.character_context_id)}
        </li>
      })}
    </ul> : draft ? <ul className="space-y-1">{draftRow()}</ul> : <p className="muted text-[11px]">No transcript segments overlap this cut.</p>}
    <p className="muted mt-1.5 shrink-0 text-[10px] leading-4">{hint}</p>
  </div>
}
