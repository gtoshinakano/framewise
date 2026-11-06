import { useCallback, useEffect, useState } from 'react'
import { api, type ContextItem, type Job } from './api'

type Frame = { position: 'first' | 'last'; time: number; image_size: [number, number] }
type Detection = { id: string; position: 'first' | 'last'; box: [number, number, number, number]; character_context_id: string | null; character_name: string | null; label_source: string | null; match_reason: string | null; is_reference: boolean }
type DetectionRun = { id: string; job_id: string | null; stale: boolean; frames: Frame[]; detections: Detection[] }
type Reference = { id: string; character_name: string; run_id: string; position: string; box: [number, number, number, number]; image_size: [number, number] }

export function CharacterDetectionPanel({ cutId, projectId, jobs, onStarted }: { cutId: string; projectId: string; jobs: Job[]; onStarted: () => Promise<void> }) {
  const [run, setRun] = useState<DetectionRun | null>(null)
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [references, setReferences] = useState<Reference[]>([])
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const [drawing, setDrawing] = useState<'first' | 'last' | null>(null)
  const [start, setStart] = useState<[number, number] | null>(null)
  const [current, setCurrent] = useState<[number, number] | null>(null)
  const job = jobs.find(item => item.pipeline === 'character_detect' && item.cuts?.some(cut => cut.id === cutId))
  const active = job?.status === 'queued' || job?.status === 'running'
  const episodeBusy = jobs.some(item => item.status === 'queued' || item.status === 'running')
  const refresh = useCallback(async () => {
    const [next, items, refs] = await Promise.all([
      api<DetectionRun | null>(`/cuts/${cutId}/character-detections`),
      api<ContextItem[]>(`/projects/${projectId}/context`),
      api<Reference[]>(`/projects/${projectId}/character-references`),
    ])
    setRun(next); setCharacters(items.filter(item => item.kind === 'character')); setReferences(refs)
  }, [cutId, projectId])
  useEffect(() => { void refresh().catch(reason => setError((reason as Error).message)) }, [refresh, job?.status, job?.id])
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => { void api<{ id: string; status: string }>(`/jobs/${job.id}/character-detection-status`).then(value => { if (value.status !== 'queued' && value.status !== 'running') void onStarted() }).catch(() => {}) }, 3000)
    return () => clearInterval(timer)
  }, [active, job?.id, onStarted])
  async function action(fn: () => Promise<unknown>) {
    setWorking(true); setError('')
    try { await fn(); await refresh() } catch (reason) { setError((reason as Error).message) } finally { setWorking(false) }
  }
  function coords(event: React.PointerEvent<SVGSVGElement>, frame: Frame): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect()
    return [Math.max(0, Math.min(frame.image_size[0], (event.clientX - rect.left) * frame.image_size[0] / rect.width)), Math.max(0, Math.min(frame.image_size[1], (event.clientY - rect.top) * frame.image_size[1] / rect.height))]
  }
  async function finish(frame: Frame, end: [number, number]) {
    if (!run || !start) return
    const box = [Math.round(Math.min(start[0], end[0])), Math.round(Math.min(start[1], end[1])), Math.round(Math.max(start[0], end[0])), Math.round(Math.max(start[1], end[1]))]
    setStart(null); setCurrent(null); setDrawing(null)
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return
    await action(() => api(`/character-detection-runs/${run.id}/boxes`, { method: 'POST', body: JSON.stringify({ position: frame.position, box }) }))
  }
  return <section className="space-y-3 rounded-lg border hairline p-3">
    <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-xs font-semibold">Characters in this cut</h3><p className="muted mt-1 text-[11px]">Detect people in the first and last frames, then label them before analysis.</p></div><button type="button" className="btn btn-quiet" disabled={working || episodeBusy} onClick={() => void action(async () => { await api(`/cuts/${cutId}/character-detection`, { method: 'POST' }); await onStarted() })}>Detect characters</button></div>
    {active && <p role="status" className="text-xs text-[#c3aeff]">{job?.stage || 'Detection queued'}…</p>}
    {job?.status === 'failed' && <p role="alert" className="text-xs text-[#ff9fa7]">{job.error || 'Character detection failed.'}</p>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
    {run?.stale && <p className="text-xs text-[#e6c58b]">Cut timing changed. Detect again before using these labels in analysis.</p>}
    {run && <div className="space-y-3">{run.frames.map(frame => {
      const people = run.detections.filter(item => item.position === frame.position)
      const img = `/api/character-detection-runs/${run.id}/frame/${frame.position}`
      return <div key={frame.position} className="space-y-2"><div className="flex justify-between"><strong className="text-xs">{frame.position === 'first' ? 'First' : 'Last'} frame · {frame.time.toFixed(2)}s</strong><button type="button" className="text-xs text-[#c3aeff] hover:underline" disabled={working || !!active || run.stale} onClick={() => { setDrawing(drawing === frame.position ? null : frame.position); setStart(null); setCurrent(null) }}>{drawing === frame.position ? 'Cancel drawing' : 'Draw missing person'}</button></div>
        <div className="relative overflow-hidden bg-black" style={{ aspectRatio: `${frame.image_size[0]} / ${frame.image_size[1]}` }}><img src={img} alt={`${frame.position} detected frame`} className="absolute inset-0 h-full w-full" /><svg viewBox={`0 0 ${frame.image_size[0]} ${frame.image_size[1]}`} className={`absolute inset-0 h-full w-full ${drawing === frame.position ? 'touch-none cursor-crosshair' : 'pointer-events-none'}`} aria-label={`${frame.position} frame character boxes`} onPointerDown={event => { if (drawing !== frame.position) return; event.currentTarget.setPointerCapture(event.pointerId); const p = coords(event, frame); setStart(p); setCurrent(p) }} onPointerMove={event => { if (drawing === frame.position && start) setCurrent(coords(event, frame)) }} onPointerUp={event => { if (drawing === frame.position && start) void finish(frame, coords(event, frame)) }}>
          {people.map((person, index) => <g key={person.id}><rect x={person.box[0]} y={person.box[1]} width={person.box[2] - person.box[0]} height={person.box[3] - person.box[1]} fill="#bca7ff" fillOpacity=".12" stroke="#c3aeff" strokeWidth="3" /><text x={person.box[0] + 4} y={Math.max(20, person.box[1] - 4)} fill="white" stroke="black" strokeWidth="3" paintOrder="stroke" fontSize="20">{index + 1}. {person.character_name || 'Unknown'}</text></g>)}
          {drawing === frame.position && start && current && <rect x={Math.min(start[0], current[0])} y={Math.min(start[1], current[1])} width={Math.abs(start[0] - current[0])} height={Math.abs(start[1] - current[1])} fill="none" stroke="white" strokeWidth="3" />}</svg></div>
        {!people.length && <p className="muted text-xs">No people detected. You can draw a missing person.</p>}
        {people.map((person, index) => <div key={person.id} className="rounded border hairline p-2 text-xs"><div className="flex flex-wrap items-center gap-2"><span>Person {index + 1}</span><select className="field min-w-32 flex-1" value={person.character_context_id || ''} disabled={working || !!active || run.stale || person.is_reference} onChange={event => void action(() => api(`/character-detections/${person.id}`, { method: 'PUT', body: JSON.stringify({ character_context_id: event.target.value || null }) }))}><option value="">Unknown</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>{person.character_context_id && <button type="button" className="btn btn-quiet" disabled={working || !!active || run.stale} onClick={() => void action(() => api(`/character-detections/${person.id}/reference`, { method: person.is_reference ? 'DELETE' : 'POST' }))}>{person.is_reference ? 'Remove reference' : 'Save as reference'}</button>}<button type="button" className="text-[#ff9fa7] hover:underline" disabled={working || !!active || person.is_reference} onClick={() => void action(() => api(`/character-detections/${person.id}`, { method: 'DELETE' }))}>Remove box</button></div>{person.label_source === 'auto' && <p className="mt-1 text-[#e6c58b]">Automatic match · unverified{person.match_reason ? `: ${person.match_reason}` : ''}</p>}</div>)}
      </div>
    })}</div>}
    {!!references.length && <details><summary className="cursor-pointer text-xs text-[#c3aeff]">Project character references ({references.length})</summary><div className="mt-2 grid grid-cols-2 gap-2">{references.map(ref => <div key={ref.id} className="rounded border hairline p-2"><svg viewBox={`${ref.box[0]} ${ref.box[1]} ${ref.box[2]-ref.box[0]} ${ref.box[3]-ref.box[1]}`} className="h-24 w-full bg-black"><image href={`/api/character-detection-runs/${ref.run_id}/frame/${ref.position}`} width={ref.image_size[0]} height={ref.image_size[1]} /></svg><div className="mt-1 flex justify-between gap-1 text-xs"><span>{ref.character_name}</span><button type="button" disabled={working} className="text-[#ff9fa7] hover:underline" onClick={() => void action(() => api(`/character-references/${ref.id}`, { method: 'DELETE' }))}>Remove</button></div></div>)}</div></details>}
    <p className="muted text-[11px]">Create character names in project context before labeling. Saved references are reused across the project. Automatic matches are unverified until you queue this cut. Review and correct or clear all named boxes first; Queue cut treats their names as verified. Up to four reference images accompany single-cut frame analysis; video analysis receives text and box hints.</p>
  </section>
}
