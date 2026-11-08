import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { api, type AudioSelection, type Job, type TranscriptSegment } from './api'

const MIN_SECONDS = 0.5

function clock(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${(seconds - minutes * 60).toFixed(2).padStart(5, '0')}`
}

// A line belongs to the part by its midpoint, the same rule the audio worker uses when it replaces lines.
function inside(segment: TranscriptSegment, selection: AudioSelection) {
  const middle = (segment.start_time + segment.end_time) / 2
  return middle >= selection.start && middle < selection.end
}

export default function TranscribeRangePanel({ episodeId, duration, currentTime, segments, selection, onSelectionChange, playRange, hasAudio, busy, jobs, onQueued }: {
  episodeId: string
  duration: number
  currentTime: number
  segments: TranscriptSegment[]
  selection: AudioSelection | null
  onSelectionChange: (selection: AudioSelection | null) => void
  playRange: (start: number, end: number) => void
  hasAudio: boolean
  busy: boolean
  jobs: Job[]
  onQueued: () => Promise<void>
}) {
  const [replaceManual, setReplaceManual] = useState(false)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const playhead = Math.round(currentTime * 100) / 100
  const valid = !!selection && Number.isFinite(selection.start) && Number.isFinite(selection.end) && selection.start >= 0 && selection.end <= duration + 0.001 && selection.end - selection.start >= MIN_SECONDS
  const lines = selection && valid ? segments.filter(segment => inside(segment, selection)) : []
  const byHand = lines.filter(segment => segment.origin !== 'model')
  const speakers = new Set(lines.map(segment => segment.speaker_label).filter(Boolean)).size
  const latest = jobs.find(job => job.pipeline === 'transcribe')
  const failure = latest?.status === 'failed' && latest.stage !== 'Cancelled' ? latest.error : null
  const status = latest?.status === 'queued' ? 'Queued for the audio worker.' : latest?.status === 'running' ? latest.stage : latest?.status === 'done' && latest.stage !== 'Complete' ? `Last run: ${latest.stage}` : ''
  useEffect(() => { setError('') }, [selection?.start, selection?.end])
  useEffect(() => { if (!byHand.length) setReplaceManual(false) }, [byHand.length])
  function edit(change: Partial<AudioSelection>) {
    const base = selection || { start: playhead, end: Math.min(duration, playhead + 10) }
    onSelectionChange({ ...base, ...change })
  }
  async function run() {
    if (!selection || !valid) return
    setWorking(true); setError('')
    try {
      await api(`/episodes/${episodeId}/transcribe-range`, { method: 'POST', body: JSON.stringify({ start_time: selection.start, end_time: Math.min(selection.end, duration), replace_manual: replaceManual }) })
      await onQueued()
    } catch (err) { setError((err as Error).message) }
    finally { setWorking(false) }
  }
  const reason = !hasAudio ? 'This video has no audio track.' : !duration ? 'Prepare the video first.' : busy ? 'Another job is running for this episode.' : ''
  return <section aria-labelledby="redo-heading" className="mt-5 space-y-3 rounded-lg border hairline p-3">
    <h3 id="redo-heading" className="flex items-center gap-2 text-xs font-semibold"><RefreshCw size={14} /> Redo part of the transcript</h3>
    <p className="muted text-[11px] leading-4">Shift-drag on the waveform to choose the audio, or type the times. Only this part is transcribed again and its speakers identified again; the rest of the transcript stays as it is, and replaced lines remain under Show history.</p>
    <div className="grid grid-cols-2 gap-2">
      <label><span className="label">Start · seconds</span><input type="number" className="field" step="0.01" min="0" max={duration} value={selection ? selection.start : ''} placeholder="—" onChange={event => edit({ start: Number(event.target.value) })} /><button type="button" className="muted mt-1 text-[11px] hover:text-white" onClick={() => edit({ start: playhead })}>Set to playhead</button></label>
      <label><span className="label">End · seconds</span><input type="number" className="field" step="0.01" min="0" max={duration} value={selection ? selection.end : ''} placeholder="—" onChange={event => edit({ end: Number(event.target.value) })} /><button type="button" className="muted mt-1 text-[11px] hover:text-white" onClick={() => edit({ end: playhead })}>Set to playhead</button></label>
    </div>
    {!selection && <button type="button" className="btn btn-quiet w-full" disabled={!duration} onClick={() => edit({})}>Select 10 seconds from the playhead</button>}
    {selection && !valid && <p role="alert" className="text-[11px] text-[#ff9fa7]">Choose at least {MIN_SECONDS} seconds inside the video, with the end after the start.</p>}
    {selection && valid && <p className="text-[11px]" aria-live="polite">{clock(selection.start)} – {clock(selection.end)} · {(selection.end - selection.start).toFixed(1)}s · {lines.length} current {lines.length === 1 ? 'line' : 'lines'}{speakers ? ` · ${speakers} ${speakers === 1 ? 'speaker' : 'speakers'}` : ''}{byHand.length ? ` · ${byHand.length} edited by hand` : ''}</p>}
    {byHand.length > 0 && <label className="flex items-start gap-2 text-[11px]"><input type="checkbox" className="mt-0.5" checked={replaceManual} onChange={event => setReplaceManual(event.target.checked)} /><span>Also replace the {byHand.length} {byHand.length === 1 ? 'line' : 'lines'} I edited by hand. Otherwise they are kept and the new text is not added over them.</span></label>}
    <div className="flex gap-2">
      <button type="button" className="btn btn-quiet flex-1" disabled={!hasAudio || !valid} onClick={() => selection && playRange(selection.start, selection.end)}>Play selection</button>
      <button type="button" className="btn btn-quiet flex-1" disabled={!selection} onClick={() => onSelectionChange(null)}>Clear</button>
    </div>
    <button type="button" className="btn btn-primary w-full" disabled={working || !valid || !!reason} onClick={() => void run()}><RefreshCw size={14} /> Transcribe this part again</button>
    {reason && <p className="muted text-[11px]">{reason}</p>}
    {status && <p role="status" className="text-xs text-[#7bc5cb]">{status}</p>}
    {failure && latest?.status === 'failed' && <p role="alert" className="text-xs text-[#ff9fa7]">The last transcription stopped: {failure}</p>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
  </section>
}
