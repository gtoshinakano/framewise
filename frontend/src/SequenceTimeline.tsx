import type { MouseEvent } from 'react'
import { usePlayerValue } from './SourcePlayer'
import type { SequenceDetail, SequenceShot, ShotDialogue } from './sequenceApi'
import { clock, rulerStep, shotStatus, spokenTimes, timecode, type Tone } from './sequenceView'

const BLOCKS: Record<Tone, string> = {
  good: 'border-[#2f7a57] bg-[#173a2b] text-[#9ff0c8]', busy: 'border-[#5a4a8c] bg-[#2b2442] text-[#d6c8ff]', warn: 'border-[#8a6a2a] bg-[#4a3a17] text-[#ffd59a]',
  bad: 'border-[#8a3a44] bg-[#3d2025] text-[#ffaaa7]', idle: 'border-[#3a4150] bg-[#262b35] text-[#aeb5c1]',
}
const pct = (seconds: number, duration: number) => `${Math.max(0, Math.min(100, seconds / duration * 100))}%`
const minutes = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, '0')}`

/** Where each spoken line of a shot starts (shot seconds): the prompt's `At` times once written, else the planned lines. */
function spokenMarks(shot: SequenceShot) {
  const dialogue: ShotDialogue[] = shot.result?.derived.dialogue || shot.lines
  const said = (t: number) => dialogue.find(line => Math.abs(line.t0 - t) < 0.002)
  if (shot.result?.prompt) return spokenTimes(shot.result.prompt).map(t => ({ t, planned: false, line: said(t) }))
  return shot.lines.map(line => ({ t: line.t0, planned: true, line }))
}

/**
 * The whole sequence on one line: every shot at its length, the hard cuts, and a tick where each spoken line
 * of a shot starts. Clicking a shot seeks there; clicking a tick plays from that line.
 */
export default function SequenceTimeline({ detail, selectedKey, onPick }: {
  detail: SequenceDetail; selectedKey?: string; onPick: (shot: SequenceShot, seconds: number, play: boolean) => void
}) {
  const duration = detail.timeline?.duration || detail.sequence.duration || 1
  const step = rulerStep(duration)
  const ticks = Array.from({ length: Math.floor(duration / step) + 1 }, (_, index) => index * step)
  const placed = (detail.timeline?.boundaries || []).flatMap(boundary => boundary.fixed ? [{ id: boundary.id, t: boundary.fixed.t }] : [])
  function seek(shot: SequenceShot, event: MouseEvent<HTMLButtonElement>) {
    // A keyboard press has no pointer position: start at the beginning of the shot.
    const box = event.currentTarget.getBoundingClientRect()
    const ratio = event.detail === 0 || !box.width ? 0 : (event.clientX - box.left) / box.width
    onPick(shot, Math.max(0, Math.min(shot.duration, ratio * shot.duration)), false)
  }
  return <section aria-label="Sequence timeline" className="shrink-0 rounded-xl border hairline bg-[#1b202a] px-4 pb-3 pt-2.5">
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
      <h3 className="text-xs font-semibold">Sequence timeline</h3>
      <span className="muted">{detail.shots.length} shots · {clock(duration)}</span>
      <span className="muted flex items-center gap-1.5"><i className="inline-block h-2.5 w-0.5 bg-[#5ad1e6]" /> spoken line in the prompt</span>
      <span className="muted flex items-center gap-1.5"><i className="inline-block h-2.5 border-l border-dashed border-[#5ad1e6]" /> planned, not written yet</span>
      <span className="muted flex items-center gap-1.5"><i className="inline-block h-2.5 border-l border-dashed border-[#c9cfda]" /> hard cut</span>
      {placed.length > 0 && <span className="muted flex items-center gap-1.5"><i className="inline-block h-2 w-2 rotate-45 bg-[#e0a96d]" /> boundary you placed</span>}
      <Readout detail={detail} />
    </div>
    <div className="relative mx-2 mt-1.5 select-none">
      <div className="relative h-4 text-[10px] text-[#8b93a3]" aria-hidden="true">
        {ticks.map(t => <span key={t} className={`absolute top-0 font-mono ${t === 0 ? '' : '-translate-x-1/2'}`} style={{ left: pct(t, duration) }}>{minutes(t)}</span>)}
      </div>
      <div className="relative h-9">
        {detail.shots.map(shot => {
          const status = shotStatus(shot)
          const selected = shot.key === selectedKey
          return <button key={shot.key} type="button" onClick={event => seek(shot, event)} aria-current={selected}
            title={`Shot ${shot.index} · ${clock(shot.t0)}–${clock(shot.t1)} · ${shot.duration.toFixed(3)} s · ${status.label}\nClick to show this shot from that point`}
            className={`absolute inset-y-0 flex flex-col justify-center overflow-hidden rounded border px-1 text-left leading-tight ${BLOCKS[status.tone]} ${selected ? 'z-10 outline outline-2 outline-[#e0a96d]' : 'hover:brightness-125'}`}
            style={{ left: pct(shot.t0, duration), width: `calc(${pct(shot.t1 - shot.t0, duration)} - 2px)` }}>
            <span className="truncate text-[11px] font-semibold">{shot.index}</span>
            <span className="truncate font-mono text-[9px] opacity-80">{shot.duration.toFixed(1)}s</span>
          </button>
        })}
        {detail.cuts.filter(cut => cut.t0 > 0.001).map(cut => <span key={cut.id} aria-hidden="true" className="pointer-events-none absolute -bottom-1 -top-1 z-20 border-l border-dashed border-[#c9cfda]/70" style={{ left: pct(cut.t0, duration) }} />)}
        {placed.map(item => <span key={item.id} aria-hidden="true" className="pointer-events-none absolute -top-1 z-20 h-2 w-2 -translate-x-1/2 rotate-45 bg-[#e0a96d]" style={{ left: pct(item.t, duration) }} />)}
      </div>
      <div className="relative mt-1 h-4">
        {detail.shots.flatMap(shot => spokenMarks(shot).map(({ t, planned, line }, index) =>
          <button key={`${shot.key}-${index}`} type="button" onClick={() => onPick(shot, t, true)}
            title={`Shot ${shot.index} · ${timecode(t)}${planned ? ' (planned; the prompt is not written yet)' : ''}${line ? ` · ${line.character_name || line.speaker_label || 'Unknown'}: ${line.text}` : ''}\nClick to play from here`}
            aria-label={`Play shot ${shot.index} from ${timecode(t)}`}
            className="group absolute inset-y-0 w-2.5 -translate-x-1/2" style={{ left: pct(shot.t0 + t, duration) }}>
            <span className={`mx-auto block h-full w-0 border-l-2 ${planned ? 'border-dashed border-[#5ad1e6]/60' : 'border-[#5ad1e6]'} group-hover:border-white`} />
          </button>))}
      </div>
      <Playhead start={detail.sequence.start_time} duration={duration} />
    </div>
  </section>
}

/** The playhead across the timeline; only this line re-renders while the video plays. */
function Playhead({ start, duration }: { start: number; duration: number }) {
  const at = usePlayerValue(time => {
    const offset = time - start
    return offset >= 0 && offset <= duration ? Math.round(offset * 100) / 100 : null
  })
  if (at === null) return null
  return <span aria-hidden="true" className="pointer-events-none absolute -top-0.5 bottom-0 z-30 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_4px_rgba(0,0,0,.8)]" style={{ left: pct(at, duration) }} />
}

/** Which shot the playhead is in and where, on that shot's clock (the prompt's timecodes). */
function Readout({ detail }: { detail: SequenceDetail }) {
  const text = usePlayerValue(time => {
    const shot = detail.shots.find(item => time >= item.start_time - 0.0005 && time < item.end_time) || detail.shots.find(item => Math.abs(time - item.end_time) < 0.0005)
    return shot ? `Shot ${shot.index} · ${timecode(Math.min(shot.duration, Math.max(0, time - shot.start_time)))}` : ''
  })
  return text ? <span className="ml-auto font-mono text-[11px] tabular-nums text-[#e8ebf1]" title="Playhead on the clock of the shot it is in, as the prompt writes its times">{text}</span> : null
}
