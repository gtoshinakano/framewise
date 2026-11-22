import { useEffect, useRef, useState } from 'react'
import { Trash2, X } from 'lucide-react'
import { api, type EpisodeAudioRange } from './api'
import { TimeRangeInput } from './TimecodeInput'
import type { PlayheadStore } from './playhead'

export function AudioRangeDialog({ episodeId, initial, duration, playhead, onClose, onSaved }: {
  episodeId: string
  initial: Pick<EpisodeAudioRange, 'start_time' | 'end_time' | 'kind'> & { id?: string }
  duration: number
  playhead: PlayheadStore
  onClose: () => void
  onSaved: () => Promise<void>
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [draft, setDraft] = useState(initial)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    const panel = dialog.current
    panel?.showModal()
    return () => panel?.close()
  }, [])
  const valid = draft.start_time >= 0 && draft.end_time > draft.start_time && draft.end_time <= duration
  async function submit(remove = false) {
    setBusy(true)
    setError('')
    try {
      const path = `/episodes/${episodeId}/audio-ranges${draft.id ? '/' + draft.id : ''}`
      await api(path, remove ? { method: 'DELETE' } : { method: draft.id ? 'PUT' : 'POST', body: JSON.stringify(draft) })
      await onSaved()
      onClose()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return <dialog ref={dialog} aria-modal="true" aria-labelledby="audio-range-title"
    onCancel={event => { event.preventDefault(); if (!busy) onClose() }}
    className="m-auto w-[min(92vw,28rem)] rounded-xl border hairline bg-[#1b202a] p-5 text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <div className="mb-4 flex items-center justify-between">
      <h2 id="audio-range-title" className="font-semibold">{draft.id ? 'Edit excluded audio range' : 'Exclude an audio range'}</h2>
      <button type="button" aria-label="Close audio range" className="icon-btn" onClick={onClose} disabled={busy}><X size={16} /></button>
    </div>
    <form className="space-y-4" onSubmit={event => { event.preventDefault(); if (valid) void submit() }}>
      <label className="block text-xs">Kind<select className="field mt-1" value={draft.kind}
        onChange={event => setDraft({ ...draft, kind: event.target.value as EpisodeAudioRange['kind'] })}>
        <option value="OP">Opening</option><option value="ED">Ending</option><option value="repeat">Repeated audio</option><option value="manual">Manual exclusion</option>
      </select></label>
      <TimeRangeInput compact start={draft.start_time} end={draft.end_time} min={0} max={duration} playhead={playhead}
        onChange={range => setDraft({ ...draft, start_time: range.start ?? draft.start_time, end_time: range.end ?? draft.end_time })} />
      {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
      <div className="flex justify-between gap-2">
        {draft.id ? <button type="button" className="btn btn-quiet text-[#ff9fa7]" disabled={busy} onClick={() => void submit(true)}><Trash2 size={14} /> Delete range</button> : <span />}
        <button type="submit" className="btn btn-primary" disabled={busy || !valid}>Save range</button>
      </div>
    </form>
  </dialog>
}
