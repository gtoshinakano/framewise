import { useState } from 'react'
import { api } from './api'
import { JsonContextEditor, type JsonObject } from './JsonContextEditor'

type RequestPreview = { request: { model: string; system: string; summary_system?: string; input: JsonObject }; base_id: string | null; source_job: { id: string; kind: 'analysis' | 'summary'; stage: string; error: string | null; finished_at: string | null; reconstructed: boolean } | null }
type Source = 'current' | 'failed' | 'failed_summary'

export function SummaryRequestEditor({ cutId, disabled, onStarted }: { cutId: string; disabled: boolean; textProvider?: 'local' | 'openrouter'; onStarted: () => Promise<void> }) {
  const [preview, setPreview] = useState<RequestPreview | null>(null)
  const [source, setSource] = useState<Source>('current')
  const [raw, setRaw] = useState<string | null>(null)
  const [dirty, setDirty] = useState(false)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  async function load(next: Source) {
    if (dirty && !window.confirm('Replace your edited request with a fresh preview?')) return
    setWorking(true); setError(''); setMessage('')
    try {
      const value = await api<RequestPreview>(`/cuts/${cutId}/summary-request-preview?source=${next}`)
      setPreview(value); setSource(next); setRaw(null); setDirty(false)
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  function update(request: RequestPreview['request']) { setPreview(previous => previous && { ...previous, request }); setDirty(true); setMessage('') }
  function parseRaw() {
    if (raw === null) return preview?.request.input
    try {
      const parsed: unknown = JSON.parse(raw)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('The input must be a JSON object.')
      setError(''); return parsed as JsonObject
    } catch (err) { setError(`Invalid request input: ${(err as Error).message}`); return null }
  }
  function applyRaw() {
    const input = parseRaw()
    if (input && preview) { update({ ...preview.request, input }); setRaw(null) }
  }
  async function queue() {
    const input = parseRaw()
    if (!preview || !input) return
    setWorking(true); setError(''); setMessage('')
    try {
      const settings = await api<{ openrouter_model: string }>('/text-generation-settings')
      const request_override = { ...preview.request, model: settings.openrouter_model, input }
      const fullAnalysis = preview.source_job?.kind === 'analysis'
      const path = fullAnalysis ? `/cuts/${cutId}/retry-analysis-request` : `/cuts/${cutId}/regenerate-summary`
      const body = fullAnalysis ? { source_job_id: preview.source_job?.id, request_override } : { mode: 'regenerate', instruction: '', preview_checked: true, preview_base_id: preview.base_id, source_job_id: preview.source_job?.id, request_override }
      await api(path, { method: 'POST', body: JSON.stringify(body) })
      setDirty(false); setRaw(null); setPreview(previous => previous && { ...previous, request: { ...previous.request, input } }); setMessage('Request queued with the edits shown here.')
      try { await onStarted() } catch { setMessage('Request queued, but the episode could not refresh. Refresh the page to see its status.') }
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  return <details className="rounded-lg border hairline bg-[#171c25] p-3 text-xs">
    <summary className="cursor-pointer font-semibold text-[#c3aeff]">Inspect &amp; edit the next summary request</summary>
    <p className="muted mt-2 leading-5">Prepare a summary from current results, or retry the latest failed analysis using its saved frame evidence and the text engine selected in the top navigation. Review and edit what will be sent before queueing. Changes apply only to this run.</p>
    <div className="mt-3 flex flex-wrap gap-2"><button type="button" className="btn btn-quiet" disabled={working} onClick={() => load('current')}>Load current results</button><button type="button" className="btn btn-quiet" disabled={working} onClick={() => load('failed')}>Load last failed analysis</button><button type="button" className="btn btn-quiet" disabled={working} onClick={() => load('failed_summary')}>Load last failed summary</button></div>
    {preview && <div className="mt-3 space-y-3">
      <p className="muted">Source: {source === 'current' ? 'current context and results' : `last failed ${preview.source_job?.kind}${preview.source_job?.finished_at ? ` · ${preview.source_job.finished_at} UTC` : ''}`}.</p>
      {preview.source_job && <div className="rounded border border-[#614c3c] p-2 text-[#e6c58b]"><p>Previous step: {preview.source_job.stage}</p>{preview.source_job.error && <p className="mt-1 whitespace-pre-wrap break-words">{preview.source_job.error}</p>}<p className="mt-1">{preview.source_job.reconstructed ? 'This request was reconstructed from saved data, so it may differ from the earlier request. Intermediate responses and temporary frame IDs were not saved.' : 'This is the saved edited request from the earlier run.'}</p></div>}
      <details className="rounded border hairline p-2"><summary className="cursor-pointer font-semibold">System instruction</summary><textarea aria-label="System instruction" className="field mt-2 min-h-40 resize-y font-mono text-xs" value={preview.request.system} onChange={event => update({ ...preview.request, system: event.target.value })} /></details>
      {preview.source_job?.kind === 'analysis' && <details className="rounded border hairline p-2"><summary className="cursor-pointer font-semibold">Second step: summary instruction</summary><p className="muted mt-2">After the first call succeeds, the same input and its returned assessments are sent with this instruction. The generated assessments replace any input field named assessments.</p><textarea aria-label="Summary system instruction" className="field mt-2 min-h-40 resize-y font-mono text-xs" value={preview.request.summary_system || ''} onChange={event => update({ ...preview.request, summary_system: event.target.value })} /></details>}
      <div><div className="flex flex-wrap items-center justify-between gap-2"><h4 className="font-semibold">Request input</h4>{raw === null ? <button type="button" className="text-[#c3aeff] hover:underline" onClick={() => setRaw(JSON.stringify(preview.request.input, null, 2))}>Edit as JSON</button> : <button type="button" className="text-[#c3aeff] hover:underline" onClick={applyRaw}>Apply JSON &amp; use field editor</button>}</div>
        <p className="muted my-2">Expand fields to edit text, rename or remove fields, and add values or list items.</p>
        {raw === null ? <JsonContextEditor value={preview.request.input} onChange={input => update({ ...preview.request, input })} /> : <textarea aria-label="Request input JSON" spellCheck={false} className="field min-h-80 resize-y font-mono text-xs" value={raw} onChange={event => { setRaw(event.target.value); setDirty(true) }} />}
      </div>
      <p className="muted">The request uses temperature 0.2 and an initial limit of 6,144 output tokens. {preview.source_job?.kind === 'analysis' ? 'Both analysis steps request JSON responses; no downloadable frame IDs are included in this retry.' : 'The summary requests plain text.'} Automatic retries can add a repair instruction and increase the token limit.</p>
      <button type="button" className="btn btn-primary w-full" disabled={working || disabled || !preview.request.model.trim() || !preview.request.system.trim() || (preview.source_job?.kind === 'analysis' && !preview.request.summary_system?.trim())} onClick={queue}>Queue this edited {preview.source_job?.kind === 'analysis' ? 'analysis' : 'summary'}</button>
    </div>}
    {error && <p role="alert" className="mt-3 whitespace-pre-wrap text-[#ff9fa7]">{error}</p>}{message && <p role="status" className="mt-3 text-[#7bdcb2]">{message}</p>}
  </details>
}
