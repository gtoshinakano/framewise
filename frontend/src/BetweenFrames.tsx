import { useState } from 'react'
import { ImagePlus, Loader2 } from 'lucide-react'
import { api } from './api'

type Candidate = { ordinal: number; timestamp_seconds: number; relative_timestamp_seconds: number }
type Between = { available: number; frames: Candidate[] }

/** One request per gap and selection: hovering the same strip again does not read the video again. */
const requests = new Map<string, Promise<Between>>()

function between(cutId: string, revision: string, after: number) {
  const key = `${cutId}:${revision}:${after}`
  let request = requests.get(key)
  if (!request) {
    request = api<Between>(`/cuts/${encodeURIComponent(cutId)}/keyframe-selection/between?after=${after}`)
    requests.set(key, request)
    request.catch(() => requests.delete(key))
  }
  return request
}

const candidateUrl = (cutId: string, frame: Candidate) =>
  `/api/cuts/${encodeURIComponent(cutId)}/keyframe-selection/candidates/${frame.ordinal}?time=${frame.timestamp_seconds}`

/**
 * The gap after one selected frame. Hovering or focusing it shows up to four frames evenly spaced through the gap;
 * clicking one adds it to the frames the keyframe analysis reads. Mount it with the selection's revision as part of
 * its key so a changed selection starts from a fresh read.
 */
export function BetweenFrames({ cutId, revision, after, from, to, maxFrames, full, disabled, onAdd }: {
  cutId: string; revision: string; after: number; from: string; to: string | null
  maxFrames: number; full: boolean; disabled: boolean; onAdd: (ordinal: number) => void
}) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<{ data?: Between; error?: string }>({})
  function show() {
    setOpen(true)
    if (state.data || state.error) return
    between(cutId, revision, after).then(data => setState({ data }), reason => setState({ error: (reason as Error).message }))
  }
  const where = to ? `between ${from} and ${to}` : `after ${from}`
  return <div role="group" aria-label={`Frames ${where}`} className="rounded-lg border border-dashed hairline text-[11px]"
    onMouseEnter={show} onMouseLeave={() => setOpen(false)}
    onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false) }}>
    <button type="button" aria-expanded={open} className="muted flex w-full items-center justify-center gap-1 px-2 py-1 hover:text-[#c3aeff] focus-visible:text-[#c3aeff]"
      onFocus={show} onClick={() => open ? setOpen(false) : show()}>
      <ImagePlus size={12} aria-hidden />Add a frame {where}
    </button>
    {open && <div className="space-y-1 px-2 pb-2">
      {!state.data && !state.error && <p className="muted flex items-center gap-1"><Loader2 size={12} className="animate-spin" aria-hidden />Reading the frames…</p>}
      {state.error && <p role="alert" className="text-[#ffb6bc]">{state.error}</p>}
      {state.data && !state.data.frames.length && <p className="muted">There are no frames {where}.</p>}
      {full && <p className="text-[#e6c58b]">This cut already has {maxFrames} frames, the most it can hold. Remove one to add another.</p>}
      {!!state.data?.frames.length && <div className="grid grid-cols-4 gap-1.5">{state.data.frames.map(frame => {
        const time = `${frame.relative_timestamp_seconds.toFixed(3)}s`
        return <button key={frame.ordinal} type="button" disabled={disabled || full} title={`Add the frame at ${time} to the analysis`}
          aria-label={`Add the frame at ${time}`} onClick={() => onAdd(frame.ordinal)}
          className="group relative overflow-hidden rounded border hairline bg-black hover:border-[#bba7ef] focus-visible:border-[#bba7ef] disabled:opacity-40">
          <img loading="lazy" src={candidateUrl(cutId, frame)} alt="" className="aspect-video w-full object-contain" />
          <span className="absolute inset-x-0 bottom-0 bg-black/70 px-1 text-left tabular-nums">{time}</span>
          <span aria-hidden className="absolute right-1 top-1 hidden rounded bg-[#bba7ef] p-0.5 text-black group-hover:block group-focus-visible:block"><ImagePlus size={10} /></span>
        </button>
      })}</div>}
      {state.data && state.data.available > state.data.frames.length && <p className="muted">{state.data.available} frames in this gap; these {state.data.frames.length} are evenly spaced.</p>}
    </div>}
  </div>
}
