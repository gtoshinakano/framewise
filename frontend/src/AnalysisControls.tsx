import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type AnalysisContext, type AnalysisConfig } from './api'

export default function AnalysisControls({ cutId, episodeId, busy, actionsTarget, onStarted }: { cutId: string; episodeId: string; busy: boolean; actionsTarget: HTMLElement | null; onStarted: () => Promise<void> }) {
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
  async function save(run: 'none' | 'cut' | 'batch') {
    if (!config) return
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify({ ...config, detection_queries: queries.split(',').map(value => value.trim()).filter(Boolean) }) })
      if (run !== 'none') {
        await api(run === 'cut' ? `/cuts/${cutId}/analyze` : `/episodes/${episodeId}/analyze`, { method: 'POST' })
        await onStarted()
        setMessage(run === 'cut' ? 'Analysis queued.' : 'All active cuts queued using their saved settings.')
      } else setMessage('Settings saved for this cut.')
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4 space-y-3">
    <div><h3 className="text-xs font-semibold">Analyze this cut</h3><p className="muted mt-2 text-[11px] leading-5">Choose the frames and context for this cut. Each run adds editable draft results.</p></div>
    {config && context && <>
      <label className="block"><span className="label">Frames to analyze</span><select className="field" value={config.frame_mode} onChange={event => setConfig({ ...config, frame_mode: event.target.value as AnalysisConfig['frame_mode'] })}><option value="endpoints">First and last frames</option><option value="uniform">Evenly spaced frames</option></select></label>
      {config.frame_mode === 'uniform' && <label className="block"><span className="label">Number of frames · 2–32</span><input className="field" type="number" min="2" max="32" value={config.frame_count} onChange={event => setConfig({ ...config, frame_count: Number(event.target.value) })} /></label>}
      <label className="block"><span className="label">Objects to detect</span><textarea className="field min-h-16 resize-y" value={queries} onChange={event => setQueries(event.target.value)} /><span className="muted mt-1 block text-[11px]">Comma-separated visual categories, such as person, coat, or bicycle.</span></label>
      <fieldset><legend className="label">Registered context</legend><div className="max-h-40 space-y-2 overflow-y-auto">{context.context_items.length ? context.context_items.map(item => <label key={item.id} className="flex items-start gap-2 text-xs"><input type="checkbox" checked={config.context_ids.includes(item.id)} onChange={event => setConfig({ ...config, context_ids: event.target.checked ? [...config.context_ids, item.id] : config.context_ids.filter(id => id !== item.id) })} /><span>{item.title} <span className="muted">· {item.kind.replaceAll('_', ' ')}</span></span></label>) : <p className="muted text-xs">No registered context for this episode.</p>}</div></fieldset>
      <label className="block"><span className="label">Additional context for this cut</span><textarea className="field min-h-24 resize-y" maxLength={20000} value={config.additional_context} onChange={event => setConfig({ ...config, additional_context: event.target.value })} /></label>
      <label className="block"><span className="label">OpenRouter model</span><input className="field" maxLength={200} value={config.openrouter_model} onChange={event => setConfig({ ...config, openrouter_model: event.target.value })} /><span className="muted mt-1 block text-[11px]">The selected model may charge your OpenRouter account. Project synopsis, selected context, cut and adjacent notes, approved transcript and results, chunk notes, frame descriptions, detections, and OCR text are sent to OpenRouter.</span></label>
    </>}
    {/* The actions live in the dialog footer so they stay visible on every tab. */}
    {actionsTarget && createPortal(<div className="space-y-2">
      {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}{message && <p role="status" className="text-xs text-[#7bdcb2]">{message}</p>}
      <div className="flex flex-wrap items-center gap-2"><button className="btn btn-primary" disabled={!config || working || busy} onClick={() => save('cut')}>Analyze cut</button><button className="btn btn-quiet" disabled={!config || working || busy} onClick={() => save('batch')} title="Uses each cut's saved preset; cuts without one use first and last frames with no registered context">Analyze all cuts</button><button className="btn btn-quiet" disabled={!config || working || busy} onClick={() => save('none')}>Save preset</button></div>
    </div>, actionsTarget)}
  </div>
}
