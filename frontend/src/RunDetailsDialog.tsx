import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'

type RunDetails = {
  status: string; error: string | null; input: unknown; results: AnalysisResult[]
  video_windows: { window_index: number; start_time: number; end_time: number; content: unknown }[]
  rejected_responses: { window_index: number; start_time: number; end_time: number; error: string; response: string }[]
}

function readable(value: unknown) {
  if (typeof value === 'string') {
    try { return JSON.stringify(JSON.parse(value), null, 2) } catch { return value }
  }
  return JSON.stringify(value, null, 2) ?? 'Unavailable'
}

function DataBlock({ value }: { value: unknown }) {
  return <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded bg-[#10141b] p-3 text-xs leading-5 text-[#d5dbe5]">{readable(value)}</pre>
}

export function RunDetailsDialog({ job, cutId, onClose }: { job: Job; cutId: string; onClose: () => void }) {
  const [data, setData] = useState<RunDetails | null>(null)
  const [error, setError] = useState('')
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    let active = true
    api<RunDetails>(`/jobs/${job.id}/cuts/${cutId}/run-details`).then(value => { if (active) setData(value) }).catch(reason => { if (active) setError((reason as Error).message) })
    return () => { active = false; opener?.focus() }
  }, [job.id, cutId])
  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="run-details-title" className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border hairline bg-[#1b202a] shadow-2xl outline-none">
      <header className="flex items-center justify-between border-b hairline p-4"><div><h2 id="run-details-title" className="text-base font-semibold">Analysis run details</h2><p className="muted mt-1 text-xs break-all">Run {job.id} · {job.status}</p></div><button type="button" onClick={onClose} aria-label="Close run details" className="rounded p-2 hover:bg-[#303746]"><X size={18} /></button></header>
      <div className="space-y-4 overflow-y-auto p-4">
        {error && <p role="alert" className="text-sm text-[#ff9fa7]">{error}</p>}
        {!data && !error && <p className="muted text-sm">Loading run details…</p>}
        {data && <>
          <p className="muted text-xs leading-5">Inputs show the saved queue snapshot and result-specific prompts and context. Some model requests, temporary video clips, and raw successful responses are not saved; these records may not contain every byte sent to a model.</p>
          <section><h3 className="mb-2 text-sm font-semibold">Queued input and settings</h3><DataBlock value={data.input} /></section>
          {data.error && <section><h3 className="mb-2 text-sm font-semibold text-[#ff9fa7]">Run error</h3><DataBlock value={data.error} /></section>}
          <section><h3 className="mb-2 text-sm font-semibold">Saved outputs</h3>
            {!data.results.length && !data.video_windows.length && !data.rejected_responses.length && <p className="muted text-xs">No outputs saved for this run yet.</p>}
            <div className="space-y-2">{data.results.map(result => <details key={result.id} className="rounded border hairline p-3"><summary className="cursor-pointer text-sm font-semibold">{result.title || result.kind} <span className="muted text-xs font-normal">· {result.model_id || result.origin}{result.job_id !== job.id ? ' · reused from an earlier run' : ''}</span></summary><div className="mt-3 space-y-3"><div><p className="label">Output</p><DataBlock value={result.content} /></div>{result.prompt && <div><p className="label">Saved prompt</p><DataBlock value={result.prompt} /></div>}<div><p className="label">Saved input context</p><DataBlock value={result.context} /></div><div><p className="label">Model settings</p><DataBlock value={result.settings} /></div>{['initial_frame', 'last_frame'].includes(String(result.settings.structured_type)) && <a href={`/api/analysis-results/${result.id}/frame`} target="_blank" rel="noreferrer" className="text-xs text-[#c3aeff] underline">Open analyzed frame</a>}</div></details>)}
            {data.video_windows.map(window => <details key={`window-${window.window_index}`} className="rounded border hairline p-3"><summary className="cursor-pointer text-sm">Video window · {window.start_time.toFixed(3)}–{window.end_time.toFixed(3)}s</summary><div className="mt-3"><DataBlock value={window.content} /></div></details>)}
            {data.rejected_responses.map((item, index) => <details key={`rejected-${index}`} className="rounded border hairline p-3"><summary className="cursor-pointer text-sm text-[#ff9fa7]">Rejected video response · {item.start_time.toFixed(3)}–{item.end_time.toFixed(3)}s</summary><p className="my-2 text-xs text-[#ff9fa7]">{item.error}</p><DataBlock value={item.response} /></details>)}</div>
          </section>
        </>}
      </div>
    </div>
  </div>
}
