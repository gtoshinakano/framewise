import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

type TargetFrame = { source_frame_index: number; relative_timestamp_seconds: number; reason: string }
export type Selection = { revision?: string | null; status: string; error?: string | null; result_status?: string; result: null | { shot_change_score: number; change_level: string; coverage_limited: boolean; selected_frames: TargetFrame[] } }

export function useTargetSelection(cutId: string, boundaryKey: string) {
  const [selection, setSelection] = useState<Selection | null>(null)
  const [error, setError] = useState('')
  const current = useRef(0)
  const load = useCallback(() => {
    const request = ++current.current
    void api<Selection>(`/cuts/${cutId}/keyframe-selection`).then(value => {
      if (request === current.current) { setSelection(value); setError('') }
    }).catch(reason => { if (request === current.current) setError((reason as Error).message) })
  }, [cutId])
  useEffect(() => {
    setSelection(null); setError(''); load()
    return () => { current.current++ }
  }, [load, boundaryKey])
  useEffect(() => {
    window.addEventListener('framewise:keyframes-changed', load)
    return () => window.removeEventListener('framewise:keyframes-changed', load)
  }, [load])
  useEffect(() => {
    if (selection?.status !== 'queued' && selection?.status !== 'running') return
    const timer = setInterval(load, 3000)
    return () => clearInterval(timer)
  }, [load, selection?.status])
  return { selection, error }
}

export function TargetFrames({ cutId, selection, error }: { cutId: string; selection: Selection | null; error: string }) {
  return <section aria-label="Target frames" className="space-y-3 rounded-xl border hairline bg-[#171c25] p-3">
    <h3 className="text-sm font-semibold">Target frames</h3>
    {error && <p role="alert" className="text-xs text-[#ffb6bc]">{error}</p>}
    {!selection && !error && <p className="muted text-xs">Loading target frames…</p>}
    {selection?.status === 'missing' && <p className="muted text-xs">Evaluate this cut in Run settings to select target frames.</p>}
    {(selection?.status === 'queued' || selection?.status === 'running') && <p role="status" className="text-xs text-[#c3aeff]">{selection.status === 'queued' ? 'Evaluation queued' : 'Evaluating this cut…'}</p>}
    {selection?.status === 'failed' && <p role="alert" className="text-xs text-[#ffb6bc]">Evaluation failed: {selection.error}</p>}
    {selection?.status === 'cancelled' && <p className="muted text-xs">Evaluation cancelled.</p>}
    {selection && !selection.result && selection.result_status === 'stale' && selection.status === 'done' && <p className="muted text-xs">Selection is stale. Evaluate the cut again.</p>}
    {selection?.result && <>
      <p className="muted text-xs">Change: {selection.result.change_level} ({selection.result.shot_change_score.toFixed(1)} / 100)</p>
      {selection.result.coverage_limited && <p className="text-xs text-[#e6c58b]">Some detected changes may not be represented in the selected frames.</p>}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">{selection.result.selected_frames.map((frame, index) => <figure key={frame.source_frame_index} className="overflow-hidden rounded-lg border hairline bg-black">
        <img loading="lazy" className="aspect-video w-full object-contain" src={`/api/cuts/${encodeURIComponent(cutId)}/keyframe-selection/frames/${frame.source_frame_index}?revision=${encodeURIComponent(selection.revision || "")}`} alt={`Target frame ${index + 1} at ${frame.relative_timestamp_seconds.toFixed(3)} seconds`} />
        <figcaption className="flex justify-between gap-2 p-2 text-xs"><span>Frame {index + 1} · {frame.relative_timestamp_seconds.toFixed(3)}s</span><span className="muted">{frame.reason.replaceAll('_', ' ')}</span></figcaption>
      </figure>)}</div>
    </>}
  </section>
}
