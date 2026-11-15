import { useEffect, useId, useState, type KeyboardEvent } from 'react'
import { LocateFixed, Play } from 'lucide-react'
import { clampTime, formatDuration, formatTimecode, nudgeStep, parseTimecode } from './timecode'

const HELP = 'Type 1:23.4, 83.4 or +0.25. ↑/↓ nudge 0.1 s, with Shift 1 s, with Alt 0.01 s.'

/**
 * A time field that shows and accepts timecodes. It commits on Enter or when it loses focus, Escape
 * restores the last value, and the arrow keys nudge it. `playhead` adds a button that copies the playhead.
 */
export function TimecodeInput({ value, onChange, label, min = 0, max, playhead, disabled, compact }: {
  value: number | null
  onChange: (seconds: number) => void
  label: string
  min?: number
  max?: number
  playhead?: number
  disabled?: boolean
  compact?: boolean
}) {
  const id = useId()
  const shown = value === null ? '' : formatTimecode(value)
  const [text, setText] = useState(shown)
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { if (!editing) { setText(shown); setError('') } }, [shown, editing])
  function commit(raw = text) {
    if (!raw.trim()) { setText(shown); setError(''); return }
    const parsed = parseTimecode(raw, value)
    if (parsed === null) { setError('Not a time. Try 1:23.4 or 83.4'); return }
    const next = clampTime(parsed, min, max)
    setError(''); setText(formatTimecode(next))
    if (next !== value) onChange(next)
  }
  function onKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') { event.preventDefault(); commit() }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setText(shown); setError('') }
    else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault()
      const base = parseTimecode(text, value) ?? value ?? min
      const next = clampTime(base + (event.key === 'ArrowUp' ? 1 : -1) * nudgeStep(event), min, max)
      setText(formatTimecode(next)); setError('')
      if (next !== value) onChange(next)
    }
  }
  return <div className="min-w-0">
    <label htmlFor={id} className={compact ? 'label !mb-1 !text-[10px]' : 'label'}>{label}</label>
    <div className={`time-field ${error ? 'time-field-invalid' : ''} ${disabled ? 'opacity-50' : ''}`}>
      <input id={id} type="text" inputMode="decimal" spellCheck={false} autoComplete="off" disabled={disabled} value={text} placeholder="—" title={HELP} aria-invalid={!!error}
        onFocus={event => { setEditing(true); event.currentTarget.select() }}
        onBlur={() => { setEditing(false); commit() }}
        onChange={event => { setText(event.target.value); setError('') }} onKeyDown={onKey} />
      {playhead !== undefined && <button type="button" disabled={disabled} onClick={() => { const next = clampTime(playhead, min, max); setText(formatTimecode(next)); setError(''); onChange(next) }}
        title={`Set to playhead (${formatTimecode(playhead)})`} aria-label={`Set ${label.toLowerCase()} to playhead`}><LocateFixed size={14} aria-hidden /></button>}
    </div>
    {error && <p role="alert" className="mt-1 text-[11px] text-[#ff9fa7]">{error}</p>}
  </div>
}

/** Start and end fields with the resulting length, a warning when the range is empty and an optional play button. */
export function TimeRangeInput({ start, end, onChange, min = 0, max, playhead, onPlay, lockStart, lockEnd, labels = ['Start', 'End'], compact }: {
  start: number | null
  end: number | null
  onChange: (range: { start: number | null; end: number | null }) => void
  min?: number
  max?: number
  playhead?: number
  onPlay?: (start: number, end: number) => void
  lockStart?: boolean
  lockEnd?: boolean
  labels?: [string, string]
  compact?: boolean
}) {
  const valid = start !== null && end !== null && end > start
  return <div>
    <div className="grid grid-cols-2 gap-2">
      <TimecodeInput label={labels[0]} value={start} min={min} max={max} playhead={playhead} disabled={lockStart} compact={compact} onChange={value => onChange({ start: value, end })} />
      <TimecodeInput label={labels[1]} value={end} min={min} max={max} playhead={playhead} disabled={lockEnd} compact={compact} onChange={value => onChange({ start, end: value })} />
    </div>
    <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
      {start !== null && end !== null ? valid ? <span className="muted">Length <span className="font-mono tabular-nums text-[#e4e5ea]">{formatDuration(end - start)}</span></span>
        : <span role="alert" className="text-[#ff9fa7]">The end must be after the start.</span> : <span className="muted">Choose a start and an end.</span>}
      {onPlay && <button type="button" disabled={!valid} onClick={() => valid && onPlay(start!, end!)} className="inline-flex items-center gap-1 text-[#c3aeff] hover:text-white disabled:text-[#6f7787]"><Play size={12} aria-hidden /> Play</button>}
    </div>
  </div>
}
