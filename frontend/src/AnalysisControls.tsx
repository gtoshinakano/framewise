import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, Minus, Plus } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisConfig } from './api'

export default function AnalysisControls({ cutId, episodeId, busy, enabled, onStarted, onConfigChange, onPrevious, onNext }: { cutId: string; episodeId: string; busy: boolean; enabled: boolean; onStarted: () => Promise<void>; onConfigChange: (config: AnalysisConfig | null) => void; onPrevious?: () => void; onNext?: () => void }) {
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const [queries, setQueries] = useState('')
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  useEffect(() => {
    let active = true
    setConfig(null); setContext(null); setError(''); setMessage('')
    Promise.all([api<AnalysisConfig>(`/cuts/${cutId}/analysis-config`), api<AnalysisContext>(`/cuts/${cutId}/analysis-context`)])
      .then(([nextConfig, nextContext]) => { if (active) { setConfig(nextConfig); setQueries(nextConfig.detection_queries.join(', ')); setContext(nextContext) } })
      .catch(err => { if (active) setError((err as Error).message) })
    return () => { active = false }
  }, [cutId])
  useEffect(() => { onConfigChange(config) }, [config, onConfigChange])
  function setFrameCount(count: number) {
    setConfig(current => current && { ...current, frame_count: count, frame_mode: count === 2 ? 'endpoints' : 'uniform' })
  }
  async function save(run: 'none' | 'cut' | 'next' | 'batch') {
    if (!config || working) return
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify({ ...config, detection_queries: queries.split(',').map(value => value.trim()).filter(Boolean) }) })
      if (run !== 'none') {
        await api(run === 'batch' ? `/episodes/${episodeId}/analyze` : `/cuts/${cutId}/analyze`, { method: 'POST' })
        setMessage(run === 'batch' ? 'Enabled cuts queued using their saved settings.' : 'Analysis queued.')
        try { await onStarted() } catch { setMessage('Analysis queued, but the episode could not refresh. Refresh the page to see its status.') }
        if (run === 'next') onNext?.()
      } else setMessage('Settings saved for this cut.')
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const count = config ? config.frame_mode === 'endpoints' ? 2 : config.frame_count : 4
  const unavailable = !config || !context || working || busy || !enabled
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4 space-y-4">
    <div><h3 className="text-sm font-semibold">Prepare this cut</h3><p className="muted mt-1 text-[11px]">Review the frames, select the context, then queue and continue.</p></div>
    {config && context && <>
      <div>
        <span className="label" id="frame-count-label">Frames to analyze</span>
        <div className="flex flex-wrap items-center gap-2" aria-labelledby="frame-count-label">
          <button type="button" className="btn btn-quiet px-3 py-2" aria-label="Remove one frame" disabled={count <= 2} onClick={() => setFrameCount(count - 1)}><Minus size={15} /></button>
          <output className="min-w-8 text-center text-sm font-semibold tabular-nums" aria-live="polite">{count}</output>
          <button type="button" className="btn btn-quiet px-3 py-2" aria-label="Add one frame" disabled={count >= 32} onClick={() => setFrameCount(count + 1)}><Plus size={15} /></button>
          <button type="button" className="btn btn-quiet text-xs" disabled={count === 2} onClick={() => setFrameCount(2)}>First &amp; last</button>
          <button type="button" className="btn btn-quiet text-xs" disabled={count === 4} onClick={() => setFrameCount(4)}>4 frames</button>
        </div>
        <p className="muted mt-2 text-[11px]">{count === 2 ? 'First and last frames.' : 'First and last, with evenly spaced frames between them.'}</p>
      </div>
      <fieldset><legend className="label">Characters &amp; other context</legend><div className="grid max-h-64 grid-cols-1 gap-x-3 gap-y-2 overflow-y-auto sm:grid-cols-2">{context.context_items.length ? context.context_items.map(item => <label key={item.id} className="flex min-w-0 items-start gap-2 text-xs"><input type="checkbox" checked={config.context_ids.includes(item.id)} onChange={event => setConfig({ ...config, context_ids: event.target.checked ? [...config.context_ids, item.id] : config.context_ids.filter(id => id !== item.id) })} /><span className="min-w-0 break-words">{item.title} <span className="muted">· {item.kind.replaceAll('_', ' ')}</span></span></label>) : <p className="muted text-xs">No registered context for this episode.</p>}</div></fieldset>
    </>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}{message && <p role="status" className="text-xs text-[#7bdcb2]">{message}</p>}
    {!enabled && <p className="muted text-xs">This cut is disabled for analysis. Enable it in the cut editor to queue it.</p>}
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn btn-primary" disabled={unavailable || !onNext} onClick={() => void save('next')}>Queue &amp; next <ArrowRight size={14} /></button>
      <button className="btn btn-quiet" disabled={unavailable} onClick={() => void save('cut')}>Queue cut</button>
      <button className="btn btn-quiet" disabled={working || !onPrevious} onClick={onPrevious}><ArrowLeft size={14} /> Previous</button>
      <button className="btn btn-quiet" disabled={working || !onNext} onClick={onNext}>Next <ArrowRight size={14} /></button>
    </div>
    {config && <details className="border-t hairline pt-3"><summary className="cursor-pointer text-xs font-semibold text-[#c3aeff]">More settings &amp; batch actions</summary><div className="mt-3 space-y-3">
      <label className="block"><span className="label">Objects to detect</span><textarea className="field min-h-16 resize-y" value={queries} onChange={event => setQueries(event.target.value)} /><span className="muted mt-1 block text-[11px]">Comma-separated visual categories, such as person, coat, or bicycle.</span></label>
      <label className="block"><span className="label">Additional context for this cut</span><textarea className="field min-h-24 resize-y" maxLength={20000} value={config.additional_context} onChange={event => setConfig({ ...config, additional_context: event.target.value })} onKeyDown={event => {
        if (event.key !== 'Enter' || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.nativeEvent.isComposing) return
        event.preventDefault()
        if (!event.repeat && !unavailable) void save('cut')
      }} /><span className="muted mt-1 block text-[11px]">Press Ctrl+Enter to queue this cut.</span></label>
      <label className="block"><span className="label">OpenRouter model</span><input className="field" maxLength={200} value={config.openrouter_model} onChange={event => setConfig({ ...config, openrouter_model: event.target.value })} /><span className="muted mt-1 block text-[11px]">The selected model may charge your OpenRouter account. Project synopsis, selected context, cut and adjacent notes, transcript and current analysis results, chunk notes, frame descriptions, detections, and OCR text are sent to OpenRouter.</span></label>
      <div className="flex flex-wrap gap-2"><button className="btn btn-quiet" disabled={working || busy || !context} onClick={() => void save('batch')} title="Uses each enabled cut's saved preset; cuts without one use four frames with no registered context">Queue enabled cuts</button><button className="btn btn-quiet" disabled={working || !context} onClick={() => void save('none')}>Save preset</button></div>
    </div></details>}
  </div>
}
