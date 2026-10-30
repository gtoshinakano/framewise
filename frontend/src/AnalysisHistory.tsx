import { useEffect, useState } from 'react'
import { api, type Job } from './api'

function duration(job: Job, now: number) {
  if (!job.started_at) return 'Not started'
  const parse = (value: string) => Date.parse(value.replace(' ', 'T') + 'Z')
  const seconds = Math.max(0, Math.floor(((job.finished_at ? parse(job.finished_at) : now) - parse(job.started_at)) / 1000))
  if (!Number.isFinite(seconds)) return 'Unavailable'
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}m ${String(seconds % 60).padStart(2, '0')}s`
}

export function AnalysisHistory({ jobs, onChanged }: { jobs: Job[]; onChanged: () => Promise<void> }) {
  const [now, setNow] = useState(Date.now())
  const [error, setError] = useState('')
  const [working, setWorking] = useState<string | null>(null)
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer) }, [])
  const analysis = jobs.filter(job => job.pipeline === 'analyze' || job.pipeline === 'regen')
  async function action(job: Job, name: 'retry' | 'cancel') {
    setWorking(job.id); setError('')
    try { await api(`/jobs/${job.id}/${name}`, { method: 'POST' }); await onChanged() }
    catch (err) { setError((err as Error).message) }
    finally { setWorking(null) }
  }
  return <section className="mx-6 mt-5 rounded-lg border hairline p-3 text-xs">
    <details open={analysis.some(job => job.status === 'queued' || job.status === 'running')}>
      <summary className="cursor-pointer font-semibold">Analysis queue &amp; history ({analysis.length})</summary>
      <p className="muted mt-2">Each run uses the presets, cut boundaries, and context saved when it was queued. Later changes apply to future runs.</p>
      {error && <p role="alert" className="mt-2 text-[#ff9fa7]">{error}</p>}
      {analysis.length ? <div className="mt-3 max-h-64 space-y-2 overflow-y-auto">{analysis.map(job => <div key={job.id} className="rounded border hairline p-2">
        <div className="flex flex-wrap items-center justify-between gap-2"><strong>{job.pipeline === 'regen' ? 'Summary' : 'Video analysis'} · {job.stage === 'Cancelled' ? 'Cancelled' : job.status}</strong><span className="muted">{job.created_at} UTC · {duration(job, now)}</span></div>
        <p className="muted mt-1">{job.stage}{job.retry_of ? ' · retry' : ''}</p>
        {job.error && <p className="mt-1 whitespace-pre-wrap text-[#ffb6bc]">{job.error}</p>}
        {job.status === 'queued' && <button className="btn btn-quiet mt-2" disabled={working !== null} onClick={() => action(job, 'cancel')}>Cancel queued run</button>}
        {job.pipeline === 'analyze' && job.status === 'failed' && job.stage !== 'Cancelled' && <button className="btn btn-quiet mt-2" disabled={working !== null} onClick={() => action(job, 'retry')}>Retry with saved settings</button>}
      </div>)}</div> : <p className="muted mt-2">No analysis runs yet.</p>}
    </details>
  </section>
}
