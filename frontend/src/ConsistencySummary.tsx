import { useState } from 'react'
import { RefreshCw, Sparkles } from 'lucide-react'
import { api, type AnalysisResult } from './api'

const fields = [
  ['summary', 'Summary'], ['clear', 'Clear evidence'], ['confusing', 'Conflicting or unclear'],
  ['human_review', 'Human review'], ['recommended_checks', 'Suggested checks'],
] as const

export function ConsistencySummary({ cutId, result, disabled, onStarted }: { cutId: string; result: AnalysisResult; disabled: boolean; onStarted: () => Promise<void> }) {
  const [working, setWorking] = useState(false)
  const [message, setMessage] = useState('')
  let content: Record<string, unknown> = {}
  try { const parsed: unknown = JSON.parse(result.content); if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) content = parsed as Record<string, unknown> } catch { /* The editor can repair invalid older results. */ }
  async function queue() {
    setWorking(true); setMessage('')
    try {
      await api(`/cuts/${cutId}/consistency-summary`, { method: 'POST' })
      setMessage('Consistency summary queued.'); await onStarted()
    } catch (error) { setMessage((error as Error).message) } finally { setWorking(false) }
  }
  return <section aria-label="Consistency summary" className="space-y-3 rounded-xl border hairline bg-[#1b202a] p-4">
    <h3 className="flex items-center gap-2 text-xs font-semibold"><Sparkles size={14} className="text-[#bca7ff]" /> Consistency summary</h3>
    {fields.map(([key, label]) => <div key={key}><h4 className="text-[11px] font-semibold text-[#c3aeff]">{label}</h4><p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5">{typeof content[key] === 'string' ? content[key] as string : 'Unavailable'}</p></div>)}
    <p className="muted text-[11px]">Review or edit this result and its previous versions in Analysis fields.</p>
    {message && <p role="status" className="text-xs text-[#c3aeff]">{message}</p>}
    <button type="button" className="btn btn-quiet" disabled={working || disabled} onClick={() => void queue()}><RefreshCw size={14} /> Queue consistency summary</button>
  </section>
}
