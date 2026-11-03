import { useEffect, useRef, useState } from 'react'
import { api, type Cut, type Job } from './api'

type Outputs = { voices: { id: string; start_time: number; end_time: number; model_id: string; character_name: string }[]; results: { id: string; cut_id: string; title: string; kind: string; model_id: string | null; superseded_by: string | null }[]; segments: { id: string; start_time: number; end_time: number; text: string; speaker_label: string; superseded_by: string | null; deleted: number }[]; evidence: { cut_id: string; completed: number; frame_count: number | null }[] }
function duration(job: Job, now: number) {
  if (!job.started_at) return 'Not started'
  const parse = (value: string) => Date.parse(value.replace(' ', 'T') + 'Z')
  const seconds = Math.max(0, Math.floor(((job.finished_at ? parse(job.finished_at) : now) - parse(job.started_at)) / 1000))
  if (!Number.isFinite(seconds)) return 'Unavailable'
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}m ${String(seconds % 60).padStart(2, '0')}s`
}

export function AnalysisHistory({ jobs, cuts, onSelectCut, onChanged }: { jobs: Job[]; cuts: Cut[]; onSelectCut: (cut: Cut) => void; onChanged: () => Promise<void> }) {
  const [now, setNow] = useState(Date.now())
  const [open, setOpen] = useState(jobs.some(job => ['queued', 'running', 'failed'].includes(job.status)))
  const [error, setError] = useState('')
  const [working, setWorking] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [outputs, setOutputs] = useState<Outputs | null>(null)
  const outputRequest = useRef(0)
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer) }, [])
  async function action(job: Job, name: 'retry' | 'cancel' | 'delete') {
    if (name === 'delete' && !window.confirm('Delete this run history? Its saved run evidence and retry links are removed, and results and transcripts will no longer identify this run. Their content remains. This cannot be undone.')) return
    setWorking(job.id); setError('')
    try { await api(name === 'delete' ? `/jobs/${job.id}` : `/jobs/${job.id}/${name}`, { method: name === 'delete' ? 'DELETE' : 'POST' }); if (name === 'delete') { outputRequest.current++; setExpanded(null); setOutputs(null) }; await onChanged() }
    catch (err) { setError((err as Error).message) }
    finally { setWorking(null) }
  }
  async function showOutputs(job: Job) {
    const request = ++outputRequest.current
    if (expanded === job.id) { setExpanded(null); return }
    setExpanded(job.id); setOutputs(null); setError('')
    try { const value = await api<Outputs>(`/jobs/${job.id}/outputs`); if (request === outputRequest.current) setOutputs(value) } catch (err) { if (request === outputRequest.current) setError((err as Error).message) }
  }
  async function removeEvidence(job: Job, cutId: string) {
    if (!window.confirm('Delete this run’s saved evidence for this cut? A retry may need to analyze its frames again. Generated results remain. Unreferenced files can then be removed in Storage & cleanup.')) return
    setWorking(job.id); setError('')
    const request = ++outputRequest.current
    try { await api(`/jobs/${job.id}/evidence/${cutId}`, { method: 'DELETE' }); const value = await api<Outputs>(`/jobs/${job.id}/outputs`); if (request === outputRequest.current) setOutputs(value); await onChanged() } catch (err) { setError((err as Error).message) } finally { setWorking(null) }
  }
  const label = (job: Job) => ({ analyze: 'Video analysis', regen: 'Summary', prepare: 'Prepare', hardcuts: 'Hard-cut detection', cutframes: 'Cut previews', transcribe: 'Transcription', voice_sample: 'Voice sample' } as Record<string, string>)[job.pipeline] || job.pipeline.replaceAll('_', ' ')
  return <section className="mx-6 mb-5 rounded-lg border hairline p-3 text-xs">
    <div>
      <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)} className="cursor-pointer font-semibold">Runs, queue &amp; history ({jobs.length}) {jobs.some(job => job.status === 'failed') && <span className="ml-2 text-[#ff9fa7]">· {jobs.filter(job => job.status === 'failed').length} failed</span>}</button>
      {open && <><p className="muted mt-2">Runs retain their settings, errors and links to generated data. Later edits create revisions.</p>
      {error && <p role="alert" className="mt-2 text-[#ff9fa7]">{error}</p>}
      {jobs.length ? <div className="mt-3 max-h-96 space-y-2 overflow-y-auto">{jobs.map(job => <div key={job.id} className="rounded border hairline p-2">
        <div className="flex flex-wrap items-center justify-between gap-2"><strong>{label(job)} · {job.stage === 'Cancelled' ? 'Cancelled' : job.status}</strong><span className="muted">{job.created_at} UTC · {duration(job, now)}</span></div>
        <p className="muted mt-1 break-all">Run {job.id}{job.retry_of ? ` · retry of ${job.retry_of}` : ''}</p>
        {job.cuts?.length > 0 && <ul className="mt-2 space-y-1">{job.cuts.map(item => {
          const cut = cuts.find(candidate => candidate.id === item.id)
          const text = `${item.cut_index == null ? 'Cut' : `Cut ${item.cut_index + 1}`}${item.title ? ` · ${item.title}` : ''}`
          const timing = item.start_time != null && item.end_time != null ? ` · ${item.start_time.toFixed(2)}–${item.end_time.toFixed(2)}s` : ''
          return <li key={item.id} className="text-[#c8cfda]">{cut ? <button className="text-[#c3aeff] hover:underline" onClick={() => onSelectCut(cut)}>{text}</button> : <span>{text}</span>}{timing} <span className={item.completed ? 'text-[#7bdcb2]' : 'muted'}>· {item.completed ? 'Completed' : 'Not completed'}</span></li>
        })}</ul>}
        <p className="muted mt-1">{job.stage}</p>{job.error && <p className="mt-1 whitespace-pre-wrap text-[#ffb6bc]">{job.error}</p>}
        <div className="flex flex-wrap gap-2"><button className="btn btn-quiet mt-2" onClick={() => showOutputs(job)}>{expanded === job.id ? 'Hide generated data' : 'Show generated data'}</button>
        {((job.status === 'queued' && ['analyze', 'regen', 'transcribe', 'voice_sample'].includes(job.pipeline)) || (job.pipeline === 'regen' && job.status === 'running')) && <button className="btn btn-quiet mt-2" disabled={working !== null} onClick={() => action(job, 'cancel')}>{job.status === 'queued' ? 'Cancel queued run' : 'Cancel summary'}</button>}
        {job.pipeline === 'analyze' && job.status === 'failed' && job.stage !== 'Cancelled' && <button className="btn btn-quiet mt-2" disabled={working !== null} onClick={() => action(job, 'retry')}>Retry with saved settings</button>}
        {job.status !== 'queued' && job.status !== 'running' && <button className="btn btn-quiet mt-2 text-[#ff9fa7]" disabled={working !== null} onClick={() => action(job, 'delete')}>Delete run history</button>}</div>
        {expanded === job.id && <div className="mt-2 rounded bg-[#151a22] p-2">{outputs ? <><p className="muted">{outputs.results.length} result revisions · {outputs.segments.length} transcript revisions · {outputs.evidence.length} saved cut evidence records · {outputs.voices.length} voice samples</p>{outputs.voices.map(voice => <p key={voice.id} className="mt-1">Voice sample · {voice.character_name} · {voice.start_time.toFixed(2)}–{voice.end_time.toFixed(2)}s · {voice.model_id}</p>)}{outputs.results.map(result => { const cut = cuts.find(item => item.id === result.cut_id); return <p key={result.id} className="mt-1 break-words">{cut ? <button className="text-[#c3aeff] hover:underline" onClick={() => onSelectCut(cut)}>{cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut'}</button> : 'Missing cut'} · {result.title || result.kind}{result.superseded_by ? ' · historical revision' : ''} {result.model_id && <span className="muted">· {result.model_id}</span>}</p> })}{outputs.evidence.map(item => { const cut = cuts.find(candidate => candidate.id === item.cut_id); return <div className="mt-2 flex flex-wrap items-center gap-2" key={item.cut_id}><span>Saved evidence · {cut ? cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut' : item.cut_id} · {item.completed ? 'Completed' : 'Incomplete'} · {item.frame_count === null ? 'Unreadable evidence' : `${item.frame_count} frames`}</span>{item.frame_count !== 0 && <button className="text-[#ff9fa7] hover:underline" disabled={working !== null} onClick={() => removeEvidence(job, item.cut_id)}>Delete saved evidence</button>}</div> })}{outputs.segments.map(segment => <p key={segment.id} className="mt-1">{segment.start_time.toFixed(2)}–{segment.end_time.toFixed(2)}s · {segment.speaker_label || 'Unknown speaker'}: {segment.text}{segment.superseded_by ? ' · historical revision' : ''}{segment.deleted ? ' · deleted' : ''}</p>)}</> : <span className="muted">Loading generated data…</span>}</div>}
      </div>)}</div> : <p className="muted mt-2">No runs yet.</p>}
      </>}
    </div>
  </section>
}
