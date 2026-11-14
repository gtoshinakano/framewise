import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, Clapperboard, LoaderCircle, Plus, Trash2 } from 'lucide-react'
import type { Cut } from './api'
import { sequenceApi, type SequenceSummary } from './sequenceApi'
import { clock, rangeSeconds } from './sequenceView'

const LIMIT_SECONDS = 180

/** The Sequences tab of the episode inspector: the episode's sequences and a form to create one from a range of cuts. */
export function SequencePanel({ episodeId, cuts, selectedCut, sequences, onChanged, onOpen }: {
  episodeId: string; cuts: Cut[]; selectedCut?: Cut; sequences: SequenceSummary[]; onChanged: () => Promise<void> | void; onOpen: (sequenceId: string) => void
}) {
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [title, setTitle] = useState('')
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const selectedNumber = selectedCut?.active ? selectedCut.cut_index + 1 : null
  // An empty form starts at the selected cut, so "select the first cut, then the last" is all it takes.
  useEffect(() => { if (selectedNumber && !from && !to) { setFrom(String(selectedNumber)); setTo(String(selectedNumber)) } }, [selectedNumber])
  const first = Number(from), last = Number(to)
  const valid = Number.isInteger(first) && Number.isInteger(last) && from.trim() !== '' && to.trim() !== ''
  const seconds = valid ? rangeSeconds(cuts, first, last) : null
  const tooLong = seconds !== null && seconds > LIMIT_SECONDS
  const share = seconds === null ? 0 : Math.min(100, seconds / LIMIT_SECONDS * 100)

  async function create() {
    if (seconds === null || tooLong) return
    setWorking(true); setError('')
    try {
      const detail = await sequenceApi.create(episodeId, { title: title.trim(), from_cut: first, to_cut: last, instruction: instruction.trim() })
      setTitle(''); setInstruction('')
      await onChanged()
      onOpen(detail.sequence.id)
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }

  async function remove(sequence: SequenceSummary) {
    if (!window.confirm(`Delete the sequence “${sequence.title}” with its bible, plan, shot prompts and chats? The cuts and their analysis stay.`)) return
    try { await sequenceApi.remove(sequence.id); await onChanged() } catch (err) { setError((err as Error).message) }
  }

  return <div className="scrollbar max-h-[calc(100vh-115px)] overflow-y-auto p-5">
    <p className="text-xs font-semibold uppercase tracking-[.14em] text-[#e0a96d]">Sequences</p>
    <h2 className="mt-1 text-lg font-semibold">Short video directions</h2>
    <p className="muted mt-2 text-xs leading-5">A sequence directs a range of cuts as one short video of at most 3 minutes on the original audio: a bible, a shot plan and one MiniMax H3 prompt per shot of at most 15 seconds.</p>

    <div className="mt-5 space-y-2">
      {!sequences.length && <p className="muted rounded-lg border border-dashed border-[#343b49] p-3 text-xs leading-5">No sequences yet. Choose a range of analysed cuts below.</p>}
      {sequences.map(sequence => <div key={sequence.id} className="flex items-stretch gap-1">
        <button type="button" onClick={() => onOpen(sequence.id)} className="min-w-0 flex-1 rounded-lg border border-[#343b49] bg-[#1b202a] p-3 text-left hover:border-[#e0a96d]">
          <span className="flex items-center gap-2"><Clapperboard size={14} className="shrink-0 text-[#e0a96d]" aria-hidden /><span className="truncate text-sm font-semibold">{sequence.title}</span>{sequence.pending_jobs > 0 && <LoaderCircle size={13} className="shrink-0 animate-spin text-[#d6c8ff]" aria-label="Jobs running" />}</span>
          <span className="muted mt-1 block text-[11px]">{sequence.first_cut ? `Cuts ${sequence.first_cut}–${sequence.last_cut}` : 'Cuts archived'} · {clock(sequence.duration)}</span>
          <span className="mt-2 flex flex-wrap gap-1 text-[10px]">
            <Chip on={sequence.has_bible}>Bible</Chip>
            <Chip on={sequence.has_plan}>{sequence.has_plan ? `${sequence.shots_planned} shots` : 'Plan'}</Chip>
            <Chip on={sequence.shots_planned > 0 && sequence.shots_written >= sequence.shots_planned}>{sequence.shots_written}/{sequence.shots_planned} prompts</Chip>
            <Chip on={sequence.has_review}>Review</Chip>
          </span>
          {(sequence.cuts_changed || sequence.cuts_missing) && <span className="mt-2 flex items-center gap-1 text-[11px] text-[#ffd59a]"><AlertTriangle size={12} aria-hidden /> {sequence.cuts_missing ? 'Its cuts were archived' : 'Cuts changed: direct it again'}</span>}
        </button>
        <button type="button" onClick={() => void remove(sequence)} className="rounded-lg border border-[#343b49] px-2 text-[#929aa9] hover:border-[#5b3439] hover:text-[#ffaaa7]" aria-label={`Delete sequence ${sequence.title}`} title="Delete sequence"><Trash2 size={14} /></button>
      </div>)}
    </div>

    <div className="mt-6 rounded-lg border border-[#343b49] p-3">
      <p className="mb-3 text-xs font-semibold">New sequence</p>
      <div className="grid grid-cols-2 gap-2">
        <label><span className="label">From cut</span><input className="field" type="number" min={1} max={cuts.length} step={1} value={from} onChange={event => setFrom(event.target.value)} /></label>
        <label><span className="label">To cut</span><input className="field" type="number" min={1} max={cuts.length} step={1} value={to} onChange={event => setTo(event.target.value)} /></label>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <button type="button" className="btn btn-quiet !px-2 !py-1.5 !text-[11px]" disabled={!selectedNumber} onClick={() => selectedNumber && setFrom(String(selectedNumber))} title="Start the range at the cut selected in the timeline">From selected cut</button>
        <button type="button" className="btn btn-quiet !px-2 !py-1.5 !text-[11px]" disabled={!selectedNumber} onClick={() => selectedNumber && setTo(String(selectedNumber))} title="End the range at the cut selected in the timeline">To selected cut</button>
      </div>
      <div className="mt-3" aria-live="polite">
        <div className="flex items-baseline justify-between text-[11px]"><span className="muted">Length</span><span className={tooLong ? 'font-semibold text-[#ffaaa7]' : 'text-[#e4e5ea]'}>{seconds === null ? '—' : clock(seconds)} / {clock(LIMIT_SECONDS)}</span></div>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[#262d39]"><div className={`h-full ${tooLong ? 'bg-[#ff898c]' : 'bg-[#e0a96d]'}`} style={{ width: `${share}%` }} /></div>
        {valid && seconds === null && <p className="mt-1 text-[11px] text-[#ffd59a]">Those cut numbers do not form a range of this episode.</p>}
        {tooLong && <p className="mt-1 text-[11px] text-[#ffaaa7]">A sequence may last at most 3 minutes. Choose fewer cuts.</p>}
      </div>
      <label className="mt-3 block"><span className="label">Title</span><input className="field" maxLength={200} value={title} onChange={event => setTitle(event.target.value)} placeholder={valid ? `Cuts ${from}–${to}` : 'Name this sequence'} /></label>
      <label className="mt-3 block"><span className="label">Direction for the short video</span><textarea className="field min-h-24 resize-y" maxLength={8000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="Optional: what this short video should achieve, its tone, what to emphasise…" /></label>
      <button type="button" className="btn btn-primary mt-3 w-full" disabled={working || seconds === null || tooLong} onClick={() => void create()}>{working ? <LoaderCircle size={15} className="animate-spin" /> : <Plus size={15} />} Create sequence</button>
      {error && <p role="alert" className="mt-2 text-[11px] leading-4 text-[#ffaaa7]">{error}</p>}
    </div>
  </div>
}

function Chip({ on, children }: { on: boolean; children: ReactNode }) {
  return <span className={`rounded px-1.5 py-0.5 ${on ? 'bg-[#173a2b] text-[#9ff0c8]' : 'bg-[#262b35] text-[#8b93a3]'}`}>{children}</span>
}
