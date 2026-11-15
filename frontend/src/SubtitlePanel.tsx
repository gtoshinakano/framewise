import { useEffect, useState } from 'react'
import { Captions } from 'lucide-react'
import { api, type EpisodeSubtitles, type Job, type SubtitleLine, type SubtitleRun } from './api'

const LANGUAGES = ['Portuguese', 'Brazilian Portuguese', 'English', 'Spanish', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Chinese', 'Russian', 'Arabic', 'Hindi', 'Turkish', 'Indonesian']
const PAGE = 40

function clock(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, '0')}`
}

export default function SubtitlePanel({ episodeId, ready, busy, savedLanguage, run, jobs, onChanged }: {
  episodeId: string
  ready: boolean
  busy: boolean
  savedLanguage: string | null
  run: SubtitleRun | null
  jobs: Job[]
  onChanged: () => Promise<void>
}) {
  const [language, setLanguage] = useState(savedLanguage || run?.language || '')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)
  const [lines, setLines] = useState<SubtitleLine[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(0)
  const active = jobs.find(job => job.pipeline === 'subtitles' && ['queued', 'running'].includes(job.status))
  const transcribing = jobs.some(job => ['transcribe', 'voice_sample'].includes(job.pipeline) && ['queued', 'running'].includes(job.status))
  const latest = jobs.find(job => job.pipeline === 'subtitles')
  const failure = latest && latest.status === 'failed' && latest.stage !== 'Cancelled' ? latest.error : null
  useEffect(() => { if (savedLanguage) setLanguage(current => current || savedLanguage) }, [savedLanguage])
  useEffect(() => {
    if (!open) return
    let current = true
    setLoading(true)
    api<EpisodeSubtitles>(`/episodes/${episodeId}/subtitles`)
      .then(value => { if (current) { setLines(value.lines); setPage(0) } })
      .catch(err => { if (current) setError((err as Error).message) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [open, episodeId, run?.id])
  async function extract() {
    if (!language.trim()) return
    setWorking(true); setError('')
    try { await api(`/episodes/${episodeId}/subtitles`, { method: 'POST', body: JSON.stringify({ language: language.trim() }) }); await onChanged() }
    catch (err) { setError((err as Error).message) }
    finally { setWorking(false) }
  }
  const pageCount = Math.max(1, Math.ceil(lines.length / PAGE))
  const currentPage = Math.min(page, pageCount - 1)
  const reason = !ready ? 'Prepare the video first.' : transcribing ? 'Wait for audio transcription to finish.' : busy && !active ? 'Another job is running for this episode.' : ''
  return <div className="space-y-3">
    <details className="text-[11px] leading-4"><summary className="muted cursor-pointer select-none hover:text-white">How it works</summary>
      <p className="muted mt-1">Run this after transcription finishes. The whole video is read once, not cut by cut; each cut's analysis then sees the subtitle lines that overlap it. Every subtitle in the video must be in the one language you give, and the model is told so; separate-language tracks need separate videos.</p></details>
    <div className="flex items-end gap-2">
      <label className="min-w-0 flex-1"><span className="label !mb-1 !text-[10px]">Subtitle language · required</span>
        <input className="field !py-2" list="subtitle-languages" value={language} maxLength={60} placeholder="e.g. Portuguese" disabled={working || !!active} onChange={event => setLanguage(event.target.value)} />
        <datalist id="subtitle-languages">{LANGUAGES.map(name => <option key={name} value={name} />)}</datalist>
      </label>
      <button className="btn btn-primary !py-2" disabled={working || !!reason || !!active || !language.trim()} onClick={() => void extract()} title={run ? 'Extract subtitles again for the entire video' : 'Extract subtitles for the entire video'}>
        <Captions size={14} /> {run ? 'Extract again' : 'Extract'}
      </button>
    </div>
    {reason && <p className="muted text-[11px]">{reason}</p>}
    {active && <p role="status" className="text-xs text-[#e89aab]">{active.status === 'queued' ? 'Queued for the inference worker.' : active.stage}</p>}
    {failure && !active && <p role="alert" className="text-xs text-[#ff9fa7]">The last extraction stopped: {failure}</p>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
    {run && <div className="space-y-1 text-[11px]">
      <p><strong>{run.line_count ?? lines.length} lines</strong> in {run.language}{run.finished_at ? ` · ${new Date(run.finished_at.replace(' ', 'T') + 'Z').toLocaleString()}` : ''}</p>
      {typeof run.band?.top === 'number' && typeof run.band?.bottom === 'number' && <p className="muted">Subtitle strip at {(run.band.top * 100).toFixed(0)}–{(run.band.bottom * 100).toFixed(0)}% of the frame height{run.band.source === 'default' ? ' (no strip was detected, so the lower third was used)' : ''}.</p>}
      {run.diagnostics?.warning && <p role="alert" className="text-[#f2c879]">{run.diagnostics.warning}</p>}
      <button type="button" className="text-[#e89aab] hover:underline" aria-expanded={open} onClick={() => setOpen(value => !value)}>{open ? 'Hide lines' : 'Review lines'}</button>
    </div>}
    {open && run && <div className="space-y-2">
      {loading ? <p className="muted text-xs">Loading…</p> : lines.length ? <ol className="space-y-1.5">{lines.slice(currentPage * PAGE, (currentPage + 1) * PAGE).map(line => <li key={line.id} className="rounded border hairline p-2 text-xs">
        <span className="muted block font-mono text-[10px] tabular-nums">{clock(line.start_time)} – {clock(line.end_time)}{line.confidence === 'low' ? ' · hard to read' : ''}</span>
        <span className="block whitespace-pre-line">{line.text}</span></li>)}</ol> : <p className="muted text-xs">No subtitle lines were found.</p>}
      {pageCount > 1 && <div className="flex items-center justify-between text-xs"><button className="btn btn-quiet !px-2.5 !py-1" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button><span className="muted">Page {currentPage + 1} of {pageCount}</span><button className="btn btn-quiet !px-2.5 !py-1" disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)}>Next</button></div>}
    </div>}
  </div>
}
