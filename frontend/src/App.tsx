import Storage from './Storage'
import AnalysisDialog from './AnalysisDialog'
import { AnalysisHistory } from './AnalysisHistory'
import { analysisQueueProgress } from './queueProgress'
import { analysisQueueStates } from './analysisQueueStates'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, AudioLines, ChevronRight, Clapperboard, Film, FolderOpen, Layers3, Pause, Play, Plus, RefreshCw, Save, Scissors, Settings2, Trash2, UploadCloud, Video, WandSparkles, X } from 'lucide-react'
import { api, jsonBody, mediaUrl, upload, type Chunk, type ContextItem, type Cut, type Episode, type EpisodeStatus, type EpisodeSummary, type Project, type TranscriptSegment, type Waveform } from './api'
import { CutFramePair } from './CutFramePair'
import { ArchivedCuts } from './ArchivedCuts'
import { Timeline } from './Timeline'
import { CutGallery } from './CutGallery'
import TranscriptEditor from './TranscriptEditor'

function formatTime(seconds?: number | null) {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const min = Math.floor(seconds / 60)
  return `${String(Math.floor(min / 60)).padStart(2, '0')}:${String(min % 60).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

function useRoute() {
  const current = () => window.location.hash.slice(1) || (/^\/(project|episode)\//.test(window.location.pathname) ? window.location.pathname : '/')
  const [route, setRoute] = useState(current)
  useEffect(() => { const onHash = () => setRoute(current()); window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash) }, [])
  return route
}

function go(path: string) { window.location.hash = path }

function Brand() {
  return <button onClick={() => go('/')} className="flex items-center gap-3 text-left" aria-label="Go to projects">
    <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#bca7ff] text-[#171224]"><Clapperboard size={20} strokeWidth={2.4} /></span>
    <span className="text-[17px] font-bold tracking-tight text-white">framewise<span className="text-[#bca7ff]">.</span></span>
  </button>
}

function Status({ status, stage }: { status: string; stage?: string }) {
  return <span className="inline-flex items-center gap-2 text-xs text-[#b8bfcb]"><i className={`status-dot status-${status}`} />{stage || status}</span>
}

function ErrorBanner({ message, clear }: { message: string; clear: () => void }) {
  return <div className="mb-5 flex items-start justify-between gap-4 rounded-xl border border-[#63363e] bg-[#2b191f] px-4 py-3 text-sm text-[#ffb6bc]">{message}<button onClick={clear} aria-label="Dismiss"><X size={16} /></button></div>
}

function UploadDialog({ projectId, onClose, onDone }: { projectId?: string; onClose: () => void; onDone: (projectId: string, episodeId: string) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [projectTitle, setProjectTitle] = useState('')
  const [episodeTitle, setEpisodeTitle] = useState(projectId ? '' : 'Episode 1')
  const [progress, setProgress] = useState<number | null>(null)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  async function createBlank() {
    if (!projectTitle.trim()) return
    setCreating(true); setError('')
    try { const project = await api<Project>('/projects', jsonBody({ title: projectTitle })); onClose(); go(`/project/${project.id}`) }
    catch (err) { setError((err as Error).message) } finally { setCreating(false) }
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!file) return
    const body = new FormData()
    body.append('file', file)
    if (projectId) body.append('title', episodeTitle)
    else { body.append('project_title', projectTitle); body.append('episode_title', episodeTitle) }
    setError(''); setProgress(0)
    try {
      if (projectId) {
        const result = await upload<Episode>(`/projects/${projectId}/episodes`, body, setProgress)
        onDone(projectId, result.id)
      } else {
        const result = await upload<{ project: Project; episode: Episode }>('/import', body, setProgress)
        onDone(result.project.id, result.episode.id)
      }
    } catch (err) { setError((err as Error).message); setProgress(null) }
  }
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4" onMouseDown={event => { if (event.target === event.currentTarget && progress === null) onClose() }}>
    <form onSubmit={submit} className="panel w-full max-w-lg p-7 shadow-2xl">
      <div className="flex items-start justify-between"><div><div className="mb-2 text-xs font-semibold uppercase tracking-[.18em] text-[#ac94ed]">New material</div><h2 className="text-2xl font-semibold">{projectId ? 'Add episode' : 'Import a video'}</h2><p className="muted mt-2 text-sm">Your video is stored as the original. A smaller preview and working assets are created automatically.</p></div><button type="button" onClick={onClose} aria-label="Close" className="muted"><X size={20} /></button></div>
      <div className="mt-6 space-y-5">
        {!projectId && <label><span className="label">Project name</span><input className="field" value={projectTitle} onChange={event => setProjectTitle(event.target.value)} maxLength={200} required placeholder="e.g. Pilot reference" /></label>}
        <label><span className="label">Episode name</span><input className="field" value={episodeTitle} onChange={event => setEpisodeTitle(event.target.value)} maxLength={200} required placeholder="e.g. Episode 01" /></label>
        <label className="block cursor-pointer rounded-xl border border-dashed border-[#565072] bg-[#1c1c2a] p-7 text-center hover:border-[#bca7ff]"><UploadCloud className="mx-auto mb-3 text-[#bca7ff]" size={30} /><span className="block text-sm font-medium">{file ? file.name : 'Choose a video file'}</span><span className="muted mt-1 block text-xs">MP4, MOV, MKV, WebM, AVI, M4V · up to 40 GB by default</span><input type="file" accept="video/*,.mkv,.m4v" className="hidden" onChange={event => setFile(event.target.files?.[0] || null)} required /></label>
      </div>
      {progress !== null && <div className="mt-5"><div className="mb-2 flex justify-between text-xs text-[#c2c8d2]"><span>Uploading</span><span>{Math.round(progress * 100)}%</span></div><div className="h-1.5 rounded-full bg-[#353a48]"><div className="h-full rounded-full bg-[#bca7ff]" style={{ width: `${progress * 100}%` }} /></div></div>}
      {error && <p className="mt-4 text-sm text-[#ff9fa7]">{error}</p>}
      <div className="mt-7 flex flex-wrap justify-end gap-2"><button type="button" className="btn btn-quiet" onClick={onClose}>Cancel</button>{!projectId && <button type="button" disabled={!projectTitle.trim() || progress !== null || creating} className="btn btn-quiet" onClick={createBlank}>Create empty project</button>}<button disabled={!file || progress !== null || creating} className="btn btn-primary" type="submit"><UploadCloud size={16} /> Upload video</button></div>
    </form>
  </div>
}

function Home() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const refresh = useCallback(async () => { try { setProjects(await api<Project[]>('/projects')); setError('') } catch (err) { setError((err as Error).message) } finally { setLoading(false) } }, [])
  useEffect(() => { refresh(); const timer = setInterval(refresh, 5000); return () => clearInterval(timer) }, [refresh])
  return <div className="min-h-screen"><header className="border-b hairline bg-[#10131a]"><div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4"><Brand /><button onClick={() => go('/storage')} className="btn btn-quiet">Storage &amp; cleanup</button></div></header>
    <main className="mx-auto max-w-7xl px-6 py-12"><div className="mb-10 flex flex-wrap items-end justify-between gap-5"><div><p className="mb-3 text-xs font-semibold uppercase tracking-[.18em] text-[#b49cf1]">Workspace</p><h1 className="text-4xl font-semibold tracking-tight text-white">Your projects</h1><p className="muted mt-3 text-sm">Organize reference videos, explore cuts, and build a reusable analysis record.</p></div><button onClick={() => setUploading(true)} className="btn btn-primary"><Plus size={17} /> New project</button></div>
      {error && <ErrorBanner message={error} clear={() => setError('')} />}
      {loading ? <div className="muted py-20 text-center">Loading projects…</div> : projects.length === 0 ? <div className="panel flex flex-col items-center px-6 py-20 text-center"><div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-[#302b45] text-[#c3aeff]"><Film size={30} /></div><h2 className="text-xl font-semibold">Start with a video</h2><p className="muted mt-2 max-w-md text-sm">Import a reference video to create a project and its first episode. Preparation begins as soon as the upload finishes.</p><button onClick={() => setUploading(true)} className="btn btn-primary mt-7"><UploadCloud size={17} /> Import video</button></div> : <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{projects.map(project => <button key={project.id} onClick={() => go(`/project/${project.id}`)} className="panel group p-6 text-left transition hover:border-[#6d5b9b] hover:bg-[#191d27]"><div className="mb-8 flex items-center justify-between"><div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#27233a] text-[#bca7ff]"><FolderOpen size={21} /></div><ArrowRight size={18} className="muted transition group-hover:translate-x-1 group-hover:text-[#c3aeff]" /></div><h2 className="truncate text-lg font-semibold text-white">{project.title}</h2><p className="muted mt-1 line-clamp-2 h-10 text-sm">{project.synopsis || 'No synopsis yet'}</p><div className="mt-7 flex items-center justify-between border-t hairline pt-4 text-xs text-[#a0a8b5]"><span>{project.episodes.length} {project.episodes.length === 1 ? 'episode' : 'episodes'}</span><span>{new Date(`${project.created_at}Z`).toLocaleDateString()}</span></div></button>)}</div>}
    </main>{uploading && <UploadDialog onClose={() => setUploading(false)} onDone={(_, episodeId) => go(`/episode/${episodeId}`)} />}</div>
}

function ProjectView({ id }: { id: string }) {
  const [project, setProject] = useState<Project | null>(null)
  const [title, setTitle] = useState('')
  const [synopsis, setSynopsis] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const refresh = useCallback(async () => { try { const value = await api<Project>(`/projects/${id}`); setProject(value); setError('') } catch (err) { setError((err as Error).message) } }, [id])
  useEffect(() => { refresh(); const timer = setInterval(refresh, 5000); return () => clearInterval(timer) }, [refresh])
  useEffect(() => { if (project) { setTitle(project.title); setSynopsis(project.synopsis) } }, [project?.id])
  async function save() { setSaving(true); try { const result = await api<Project>(`/projects/${id}`, { method: 'PATCH', body: JSON.stringify({ title, synopsis }) }); setProject({ ...result, episodes: project?.episodes || [] }); setError('') } catch (err) { setError((err as Error).message) } finally { setSaving(false) } }
  async function remove() { if (!project || !window.confirm(`Delete “${project.title}” and all its episodes and media? This cannot be undone.`)) return; try { await api(`/projects/${id}`, { method: 'DELETE' }); go('/') } catch (err) { setError((err as Error).message) } }
  return <div className="min-h-screen"><header className="border-b hairline bg-[#10131a]"><div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4"><Brand /><button onClick={() => go('/')} className="btn btn-quiet"><ArrowLeft size={16} /> All projects</button></div></header><main className="mx-auto max-w-7xl px-6 py-10">
    {error && <ErrorBanner message={error} clear={() => setError('')} />}{!project ? <p className="muted">Loading project…</p> : <><div className="mb-10 flex flex-wrap items-end justify-between gap-4"><div><p className="mb-3 text-xs font-semibold uppercase tracking-[.18em] text-[#b49cf1]">Project</p><h1 className="text-3xl font-semibold">{project.title}</h1><p className="muted mt-2 text-sm">{project.episodes.length} episodes · Created {new Date(`${project.created_at}Z`).toLocaleDateString()}</p></div><button onClick={() => setUploading(true)} className="btn btn-primary"><Plus size={17} /> Add episode</button></div>
      <div className="grid gap-7 lg:grid-cols-[1fr_320px]"><section><div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-semibold">Episodes</h2><span className="muted text-xs">Original videos and analysis</span></div><div className="panel overflow-hidden">{project.episodes.length === 0 ? <div className="muted p-12 text-center text-sm">No episodes yet. Add a video to begin.</div> : project.episodes.map((episode, index) => <button key={episode.id} onClick={() => go(`/episode/${episode.id}`)} className="flex w-full items-center gap-4 border-b hairline p-5 text-left transition last:border-b-0 hover:bg-[#1b202b]"><div className="flex h-14 w-20 shrink-0 items-center justify-center rounded-lg bg-[#252b38] text-[#a991e8]"><Video size={25} /></div><div className="min-w-0 flex-1"><div className="mb-1 flex items-center gap-2"><span className="text-xs font-semibold text-[#8d82ad]">{String(index + 1).padStart(2, '0')}</span><strong className="truncate text-sm text-white">{episode.title}</strong></div><div className="flex flex-wrap items-center gap-4"><Status status={episode.status} stage={episode.stage} /><span className="muted text-xs">{formatTime(episode.duration)}</span></div>{episode.error && <p className="mt-2 truncate text-xs text-[#ff9fa7]">{episode.error}</p>}</div><ChevronRight size={18} className="muted" /></button>)}</div></section>
      <aside className="panel h-fit p-6"><div className="mb-5 flex items-center gap-2 text-sm font-semibold"><Layers3 size={17} className="text-[#bca7ff]" /> Project context</div><label className="mb-4 block"><span className="label">Name</span><input className="field" value={title} onChange={event => setTitle(event.target.value)} maxLength={200} /></label><label className="block"><span className="label">Synopsis / context</span><textarea className="field min-h-36 resize-y" value={synopsis} onChange={event => setSynopsis(event.target.value)} placeholder="Describe the story, visual style, or notes that future analysis should know…" /></label><button onClick={save} disabled={saving || !title.trim()} className="btn btn-quiet mt-4 w-full"><Save size={15} /> Save context</button><div className="mt-7 border-t hairline pt-5"><button onClick={remove} className="text-xs text-[#e9919a] hover:underline"><Trash2 size={14} className="mr-1 inline" /> Delete project</button></div></aside></div><ContextLibrary projectId={id} episodes={project.episodes} /></>}
  </main>{uploading && <UploadDialog projectId={id} onClose={() => setUploading(false)} onDone={(_, episodeId) => go(`/episode/${episodeId}`)} />}</div>
}

type ContextDraft = Pick<ContextItem, 'kind' | 'title' | 'content' | 'episode_id'> & { id?: string }
const contextKinds: { value: ContextItem['kind']; label: string }[] = [
  { value: 'chapter_synopsis', label: 'Chapter synopsis' }, { value: 'chapter_happenings', label: 'Chapter happenings' },
  { value: 'character', label: 'Character' }, { value: 'world', label: 'World' },
  { value: 'scenario', label: 'Scenario' }, { value: 'object', label: 'Object' }, { value: 'general', label: 'General' },
]

function ContextLibrary({ projectId, episodes }: { projectId: string; episodes: EpisodeSummary[] }) {
  const [items, setItems] = useState<ContextItem[]>([])
  const [draft, setDraft] = useState<ContextDraft | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const load = useCallback(async () => { try { setItems(await api<ContextItem[]>(`/projects/${projectId}/context`)) } catch (err) { setError((err as Error).message) } }, [projectId])
  useEffect(() => { load() }, [load])
  async function save() {
    if (!draft) return
    setSaving(true)
    try {
      const path = draft.id ? `/context/${draft.id}` : `/projects/${projectId}/context`
      const result = await api<ContextItem>(path, { method: draft.id ? 'PUT' : 'POST', body: JSON.stringify(draft) })
      setDraft(result); setError(''); await load()
    } catch (err) { setError((err as Error).message) } finally { setSaving(false) }
  }
  async function remove() {
    if (!draft?.id || !window.confirm(`Delete context item “${draft.title}”?`)) return
    try { await api(`/context/${draft.id}`, { method: 'DELETE' }); setDraft(null); await load() } catch (err) { setError((err as Error).message) }
  }
  return <section className="mt-8"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">Context library</h2><p className="muted mt-1 text-xs">Reusable story, character, and world details for later analysis.</p></div><button onClick={() => setDraft({ kind: 'general', title: '', content: '', episode_id: null })} className="btn btn-quiet"><Plus size={15} /> Add context</button></div>
    {error && <ErrorBanner message={error} clear={() => setError('')} />}
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_420px]"><div className="panel min-h-52 overflow-hidden">{items.length ? items.map(item => <button key={item.id} onClick={() => setDraft(item)} className={`flex w-full items-center justify-between gap-4 border-b hairline px-5 py-4 text-left last:border-b-0 hover:bg-[#202531] ${draft?.id === item.id ? 'bg-[#27223a]' : ''}`}><div className="min-w-0"><div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[#a993e1]">{contextKinds.find(kind => kind.value === item.kind)?.label}{item.episode_id ? ` · ${episodes.find(episode => episode.id === item.episode_id)?.title || 'Episode'}` : ' · Project'}</div><strong className="block truncate text-sm">{item.title}</strong><p className="muted mt-1 truncate text-xs">{item.content || 'No details yet'}</p></div><ChevronRight size={16} className="muted shrink-0" /></button>) : <div className="muted flex h-full min-h-52 items-center justify-center px-6 text-center text-sm">Add a character, world, scenario, or chapter note.</div>}</div>
      <div className="panel p-5">{draft ? <div className="space-y-4"><div className="flex items-center justify-between"><h3 className="text-sm font-semibold">{draft.id ? 'Edit context' : 'New context'}</h3><button onClick={() => setDraft(null)} className="muted" aria-label="Close context editor"><X size={17} /></button></div><div className="grid gap-3 sm:grid-cols-2"><label><span className="label">Type</span><select className="field" value={draft.kind} onChange={event => setDraft({ ...draft, kind: event.target.value as ContextItem['kind'] })}>{contextKinds.map(kind => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label><label><span className="label">Scope</span><select className="field" value={draft.episode_id || ''} onChange={event => setDraft({ ...draft, episode_id: event.target.value || null })}><option value="">Entire project</option>{episodes.map(episode => <option key={episode.id} value={episode.id}>{episode.title}</option>)}</select></label></div><label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="e.g. Main character, Chapter 2" /></label><label className="block"><span className="label">Details</span><textarea className="field min-h-36 resize-y" value={draft.content} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Write details that future analysis should use…" /></label><div className="flex items-center justify-between gap-3"><button onClick={remove} disabled={!draft.id} className="text-xs text-[#e9919a] disabled:opacity-30"><Trash2 size={14} className="mr-1 inline" /> Delete</button><button onClick={save} disabled={saving || !draft.title.trim()} className="btn btn-primary"><Save size={15} /> Save context</button></div></div> : <div className="muted flex min-h-52 items-center justify-center text-center text-sm">Select an item to edit, or add new context.</div>}</div></div></section>
}

function Editor({ id }: { id: string }) {
  const [episode, setEpisode] = useState<Episode | null>(null)
  const [project, setProject] = useState<Project | null>(null)
  const [waveform, setWaveform] = useState<Waveform | null>(null)
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Cut | null>(null)
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)
  const [selectedTranscriptId, setSelectedTranscriptId] = useState<string | null>(null)
  const [chunkDraft, setChunkDraft] = useState<Chunk | null>(null)
  const [time, setTime] = useState(0)
  const [seekRevision, setSeekRevision] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [cutStart, setCutStart] = useState('')
  const [cutEnd, setCutEnd] = useState('')
  const [error, setError] = useState('')
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const closeAnalysis = useCallback(() => setAnalysisOpen(false), [])
  const [activeTab, setActiveTab] = useState<'cut' | 'chunk' | 'audio' | 'settings'>('cut')
  const [settings, setSettings] = useState<Episode['settings'] | null>(null)
  const video = useRef<HTMLVideoElement>(null)
  const playbackEnd = useRef<number | null>(null)
  const latestEpisode = useRef<Episode | null>(null)
  const lastFullRefresh = useRef(0)
  const waveformVersion = useRef<string | null>(null)
  const queueSession = useRef<{ episodeId: string; jobs: Set<string> }>({ episodeId: id, jobs: new Set() })
  const [queueProgress, setQueueProgress] = useState({ completed: 0, total: 0 })
  const refresh = useCallback(async () => {
    try {
      const value = await api<Episode>(`/episodes/${id}`)
      latestEpisode.current = value
      if (queueSession.current.episodeId !== id) queueSession.current = { episodeId: id, jobs: new Set() }
      setQueueProgress(analysisQueueProgress(value.jobs, queueSession.current.jobs))
      setEpisode(value); setSettings(current => current || value.settings)
      const preparedBy = value.jobs.find(job => job.pipeline === 'prepare' && job.status === 'done')?.id || 'initial'
      const preparing = value.jobs.some(job => job.pipeline === 'prepare' && (job.status === 'queued' || job.status === 'running'))
      const waveformKey = `${id}:${preparedBy}`
      if (!preparing && waveformVersion.current !== waveformKey) {
        const nextWaveform = await api<Waveform>(`/episodes/${id}/waveform`)
        const latest = latestEpisode.current
        const latestPrepare = latest?.jobs.find(job => job.pipeline === 'prepare' && job.status === 'done')?.id || 'initial'
        const prepareRunning = latest?.jobs.some(job => job.pipeline === 'prepare' && (job.status === 'queued' || job.status === 'running'))
        if (latest?.id === id && !prepareRunning && latestPrepare === preparedBy) {
          setWaveform(nextWaveform)
          waveformVersion.current = waveformKey
        }
      }
      setTranscript(await api<TranscriptSegment[]>(`/episodes/${id}/transcript`))
      lastFullRefresh.current = Date.now()
      setError('')
    } catch (err) { setError((err as Error).message) }
  }, [id])
  useEffect(() => {
    let active = true
    void refresh()
    const timer = setInterval(async () => {
      try {
        const status = await api<EpisodeStatus>(`/episodes/${id}/status`)
        if (!active) return
        const current = latestEpisode.current
        const changed = !current || status.status !== current.status || status.stage !== current.stage || status.job_id !== (current.jobs[0]?.id || null)
        if (changed || status.status === 'queued' || status.status === 'processing' || Date.now() - lastFullRefresh.current > 30000) await refresh()
      } catch (err) { if (active) setError((err as Error).message) }
    }, 2500)
    return () => { active = false; clearInterval(timer) }
  }, [id, refresh])
  useEffect(() => { if (episode) api<Project>(`/projects/${episode.project_id}`).then(setProject).catch(() => {}) }, [episode?.project_id])
  useEffect(() => { setSelectedId(null); setDraft(null); setSelectedChunkId(null); setSelectedTranscriptId(null); setChunkDraft(null); setWaveform(null); setTranscript([]); setSettings(null); setTime(0); playbackEnd.current = null }, [id])
  useEffect(() => { const cut = [...(episode?.cuts || []), ...(episode?.archived_cuts || [])].find(item => item.id === selectedId); if (cut && (!draft || draft.id !== cut.id)) setDraft(cut) }, [episode?.cuts, episode?.archived_cuts, selectedId])
  useEffect(() => { const chunk = episode?.chunks.find(item => item.id === selectedChunkId); if (chunk && (!chunkDraft || chunkDraft.id !== chunk.id)) setChunkDraft(chunk) }, [episode?.chunks, selectedChunkId])
  const selected = [...(episode?.cuts || []), ...(episode?.archived_cuts || [])].find(cut => cut.id === selectedId)
  const selectedChunk = episode?.chunks.find(chunk => chunk.id === selectedChunkId)
  const duration = episode?.duration || 0
  function seek(value: number) { const target = Math.max(0, Math.min(duration, value)); playbackEnd.current = null; if (video.current) video.current.currentTime = target; setTime(target); setSeekRevision(revision => revision + 1) }
  function openCutAnalysis(cut: Cut) { selectCut(cut, true); setAnalysisOpen(true) }
  function selectCut(cut: Cut, keepAnalysisOpen = false) { if (!keepAnalysisOpen) setAnalysisOpen(false); video.current?.pause(); setSelectedId(cut.id); setDraft(cut); setCutStart(String(cut.start_time)); setCutEnd(String(cut.end_time)); setActiveTab('cut'); seek(cut.first_frame_time) }
  function selectChunk(chunk: Chunk) { setSelectedChunkId(chunk.id); setChunkDraft(chunk); setActiveTab('chunk'); seek(chunk.start_time) }
  function selectTranscript(segment: TranscriptSegment) { setSelectedTranscriptId(segment.id); setActiveTab('audio'); seek(segment.start_time) }
  function playRange(start: number, end: number) { const player = video.current; if (!player) return; player.currentTime = start; playbackEnd.current = end; void player.play().catch(err => setError((err as Error).message)) }
  function onVideoTimeUpdate(player: HTMLVideoElement) { const end = playbackEnd.current; if (!player.seeking && end !== null && player.currentTime >= end) { player.pause(); player.currentTime = end; playbackEnd.current = null }; setTime(player.currentTime) }
  async function run(pipeline: 'prepare' | 'hardcuts') { try { await api(`/episodes/${id}/run`, jsonBody({ pipeline })); await refresh() } catch (err) { setError((err as Error).message) } }
  async function cancelJob(jobId: string) { try { await api(`/jobs/${jobId}/cancel`, { method: 'POST' }); await refresh() } catch (err) { setError((err as Error).message) } }
  async function toggleAnalysis() { if (!selected) return; try { const value = await api<Cut>(`/cuts/${selected.id}/analysis-enabled`, { method: 'PUT', body: JSON.stringify({ analysis_enabled: !selected.analysis_enabled }) }); setDraft(current => current?.id === value.id ? { ...current, analysis_enabled: value.analysis_enabled } : current); await refresh() } catch (err) { setError((err as Error).message) } }
  async function saveCut() { if (!draft) return; try { const value = await api<Cut>(`/cuts/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: draft.title, description: draft.description, notes: draft.notes, results: draft.results }) }); setDraft(value); await refresh() } catch (err) { setError((err as Error).message) } }
  async function editTimeline(action: 'timing' | 'merge-next' | 'delete') {
    if (!selected || !selected.active) return
    if (action !== 'timing' && !window.confirm(action === 'delete' ? 'Remove this cut? Its interval will be added to a neighboring cut and its notes will be archived.' : 'Merge this cut with the next cut? The next cut and its notes will be archived.')) return
    try {
      const path = action === 'delete' ? `/cuts/${selected.id}` : `/cuts/${selected.id}/${action}`
      const result = await api<{ selected_id: string }>(path, action === 'timing' ? { method: 'PUT', body: JSON.stringify({ start_time: Number(cutStart), end_time: Number(cutEnd) }) } : { method: action === 'delete' ? 'DELETE' : 'POST' })
      const value = await api<Episode>(`/episodes/${id}`)
      latestEpisode.current = value; setEpisode(value)
      const next = value.cuts.find(cut => cut.id === result.selected_id)
      if (next) selectCut(next)
    } catch (err) { setError((err as Error).message) }
  }
  async function saveChunk() { if (!chunkDraft) return; try { const value = await api<Chunk>(`/chunks/${chunkDraft.id}`, { method: 'PUT', body: JSON.stringify({ title: chunkDraft.title, notes: chunkDraft.notes }) }); setChunkDraft(value); await refresh() } catch (err) { setError((err as Error).message) } }
  async function saveSettings() { if (!settings) return; try { await api(`/episodes/${id}/settings`, { method: 'PUT', body: JSON.stringify(settings) }); await refresh() } catch (err) { setError((err as Error).message) } }
  async function remove() { if (!episode || !window.confirm(`Delete “${episode.title}” and all its media? This cannot be undone.`)) return; try { await api(`/episodes/${id}`, { method: 'DELETE' }); go(`/project/${episode.project_id}`) } catch (err) { setError((err as Error).message) } }
  const busy = episode?.jobs.some(job => job.status === 'running' || job.status === 'queued') || false
  const mediaBusy = episode?.jobs.some(job => ['prepare', 'hardcuts', 'cutframes'].includes(job.pipeline) && (job.status === 'running' || job.status === 'queued')) || false
  const cutQueueStates = analysisQueueStates(episode?.jobs || [])
  const analysisBusy = episode?.jobs.some(job => ['prepare', 'hardcuts'].includes(job.pipeline) && (job.status === 'running' || job.status === 'queued')) || false
  return <div className="editor-shell bg-[#0d1016]"><header className="flex h-16 items-center justify-between gap-4 border-b hairline bg-[#10131a] px-5"><div className="flex min-w-0 items-center gap-4"><Brand /><span className="hidden h-5 w-px bg-[#333945] sm:block" /><button onClick={() => go(`/project/${episode?.project_id || ''}`)} className="muted hidden truncate text-sm hover:text-white sm:block">{project?.title || 'Project'}</button><ChevronRight size={15} className="muted hidden sm:block" /><span className="truncate text-sm font-semibold">{episode?.title || 'Loading…'}</span></div><div className="flex items-center gap-3"><Status status={episode?.status || 'queued'} stage={episode?.stage} /><span className="whitespace-nowrap text-xs text-[#b8bfcb]" title="Finished items, including failures, out of this active analysis queue. Progress starts when the queue is first seen and resets when it is empty." aria-label={`Analysis queue: ${queueProgress.completed} of ${queueProgress.total} items finished`}>Queue <span className="font-mono tabular-nums">{queueProgress.completed}/{queueProgress.total}</span></span><button onClick={() => go('/')} className="btn btn-quiet hidden sm:flex"><FolderOpen size={15} /> Projects</button></div></header>
    {error && <div className="px-5 pt-4"><ErrorBanner message={error} clear={() => setError('')} /></div>}
    {!episode ? <div className="muted p-12">Loading episode…</div> : <div className="editor-workspace"><aside className="editor-cuts scrollbar"><CutGallery cuts={episode.cuts} selectedId={selectedId} onSelect={selectCut} onOpenAnalysis={openCutAnalysis} frameVersion={`${episode.jobs[0]?.id || ""}-${episode.jobs[0]?.status || ""}`} /><ArchivedCuts cuts={episode.archived_cuts} selectedId={selectedId} onSelect={selectCut} onOpenAnalysis={openCutAnalysis} /></aside><main className="editor-stage min-w-0 border-r hairline"><div className="flex flex-wrap items-center justify-between gap-4 border-b hairline px-6 py-4"><div><p className="text-xs font-semibold uppercase tracking-[.16em] text-[#a58ae8]">Episode editor</p><h1 className="mt-1 text-xl font-semibold">{episode.title}</h1></div><div className="flex flex-wrap items-center gap-2"><button disabled={busy} onClick={() => run('prepare')} className="btn btn-quiet" title="Rebuild preview, frames, audio, and waveform"><RefreshCw size={15} /> Prepare</button><button disabled={busy} onClick={() => run('hardcuts')} className="btn btn-primary"><Scissors size={15} /> {episode.cuts.length ? 'Redetect cuts' : 'Detect hard cuts'}</button></div></div>
      {episode.jobs.filter(job => ['transcribe', 'voice_sample'].includes(job.pipeline) && job.status === 'queued').map(job => <div key={job.id} className="mx-6 mt-3 flex items-center justify-between gap-3 rounded-lg border hairline p-3 text-xs"><span>Waiting for the audio worker.</span><button className="btn btn-quiet" onClick={() => cancelJob(job.id)}>Cancel queued job</button></div>)}
      {episode.error && <div className="mx-6 mt-5 rounded-lg border border-[#63363e] bg-[#2b191f] p-3 text-xs text-[#ffb6bc]">{episode.error}</div>}
      <div className="px-6 py-6"><div className="preview-stage relative mx-auto aspect-video max-w-5xl overflow-hidden rounded-xl border border-[#303540] bg-black shadow-2xl">{episode.proxy_ready ? <video ref={video} src={mediaUrl(id, 'proxy')} className="h-full w-full" controls onSeeking={() => setSeekRevision(revision => revision + 1)} onTimeUpdate={event => onVideoTimeUpdate(event.currentTarget)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} /> : <div className="flex h-full flex-col items-center justify-center text-center"><Film size={36} className="mb-4 text-[#a58ae8]" /><strong className="text-sm">{busy ? episode.stage : 'Preview unavailable'}</strong><p className="muted mt-2 text-xs">{busy ? 'Video preparation runs in the background.' : 'Run Prepare to create the preview.'}</p></div>}</div>
      <div className="mx-auto mt-4 flex max-w-5xl items-center justify-between text-xs text-[#9ca5b3]"><div className="flex items-center gap-3"><button onClick={() => { if (!video.current) return; video.current.paused ? video.current.play() : video.current.pause() }} className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#272d39] text-white">{playing ? <Pause size={14} /> : <Play size={14} />}</button><span className="font-mono">{formatTime(time)} <span className="text-[#5d6675]">/</span> {formatTime(duration)}</span></div><span>{episode.width && episode.height ? `${episode.width} × ${episode.height}` : '—'} · {episode.original_filename}</span></div></div>
      <AnalysisHistory jobs={episode.jobs} cuts={[...episode.cuts, ...episode.archived_cuts]} onSelectCut={openCutAnalysis} onChanged={refresh} />
    </main><aside className="editor-inspector scrollbar bg-[#131720]"><div className="flex border-b hairline"><button onClick={() => setActiveTab('cut')} className={`flex min-w-0 flex-1 items-center justify-center gap-1 whitespace-nowrap py-4 text-[11px] font-semibold ${activeTab === 'cut' ? 'border-b-2 border-[#bca7ff] text-white' : 'muted'}`}><Scissors size={15} /> Cut</button><button onClick={() => setActiveTab('chunk')} className={`flex min-w-0 flex-1 items-center justify-center gap-1 whitespace-nowrap py-4 text-[11px] font-semibold ${activeTab === 'chunk' ? 'border-b-2 border-[#bca7ff] text-white' : 'muted'}`}><Layers3 size={15} /> Chunk</button><button onClick={() => setActiveTab('audio')} className={`flex min-w-0 flex-1 items-center justify-center gap-1 whitespace-nowrap py-4 text-[11px] font-semibold ${activeTab === 'audio' ? 'border-b-2 border-[#bca7ff] text-white' : 'muted'}`}><AudioLines size={15} /> Audio</button><button onClick={() => setActiveTab('settings')} className={`flex min-w-0 flex-1 items-center justify-center gap-1 whitespace-nowrap py-4 text-[11px] font-semibold ${activeTab === 'settings' ? 'border-b-2 border-[#bca7ff] text-white' : 'muted'}`}><Settings2 size={15} /> Setup</button></div>
      {activeTab === 'cut' ? draft && selected ? <div className="scrollbar max-h-[calc(100vh-115px)] overflow-y-auto p-5"><div className="mb-6 flex items-start justify-between"><div><p className="text-xs font-semibold uppercase tracking-[.14em] text-[#a58ae8]">{selected.active ? `Cut ${selected.cut_index + 1}` : 'Archived cut'}</p><h2 className="mt-1 text-lg font-semibold">{formatTime(selected.start_time)} → {formatTime(selected.end_time)}</h2></div><span className="rounded-md bg-[#282438] px-2 py-1 text-[10px] text-[#c3aeff]">{(selected.end_time - selected.start_time).toFixed(1)}s</span></div><div className="space-y-5">{selected.active && <div className="rounded-lg border border-[#343b49] p-3"><p className="mb-3 text-xs font-semibold">Cut timing</p><div className="grid grid-cols-2 gap-2"><label><span className="label">Start · seconds</span><input className="field" type="number" step="0.001" value={cutStart} disabled={mediaBusy || selected.cut_index === 0} onChange={event => setCutStart(event.target.value)} /></label><label><span className="label">End · seconds</span><input className="field" type="number" step="0.001" value={cutEnd} disabled={mediaBusy || selected.cut_index === episode.cuts.length - 1} onChange={event => setCutEnd(event.target.value)} /></label></div><p className="muted mt-2 text-[11px]">Adjacent cuts adjust to keep the entire video covered.</p><button onClick={() => editTimeline('timing')} disabled={mediaBusy || cutStart.trim() === '' || cutEnd.trim() === ''} className="btn btn-quiet mt-3 w-full">Apply timing</button><div className="mt-2 flex gap-2"><button onClick={() => editTimeline('merge-next')} disabled={mediaBusy || selected.cut_index === episode.cuts.length - 1} className="btn btn-quiet flex-1">Merge next</button><button onClick={() => editTimeline('delete')} disabled={mediaBusy || episode.cuts.length < 2} className="btn btn-danger flex-1">Delete cut</button></div></div>}{selected.active && <button onClick={toggleAnalysis} className="btn btn-quiet w-full">{selected.analysis_enabled ? 'Disable analysis for this cut' : 'Enable analysis for this cut'}</button>}<CutFramePair key={`${selected.id}-${episode.jobs[0]?.id || ""}-${episode.jobs[0]?.status || ""}`} cut={selected} onSeek={seek} frameVersion={`${episode.jobs[0]?.id || ""}-${episode.jobs[0]?.status || ""}`} /><label className="block"><span className="label">Title</span><input className="field" value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="Name this cut" /></label><label className="block"><span className="label">Cut description</span><textarea className="field min-h-28 resize-y" value={draft.description} onChange={event => setDraft({ ...draft, description: event.target.value })} placeholder="What happens in this shot?" /></label><label className="block"><span className="label">Working notes</span><textarea className="field min-h-28 resize-y" value={draft.notes} onChange={event => setDraft({ ...draft, notes: event.target.value })} placeholder="Visual details, scene context, follow-up ideas…" /></label><button onClick={() => setAnalysisOpen(true)} className="btn btn-primary w-full"><WandSparkles size={15} /> Analysis &amp; results</button>{analysisOpen && <AnalysisDialog cut={selected} previousCut={selected.active ? episode.cuts[selected.cut_index - 1] : undefined} nextCut={selected.active ? episode.cuts[selected.cut_index + 1] : undefined} onNavigate={cut => selectCut(cut, true)} episodeId={id} waveform={waveform} transcript={transcript} hasAudio={episode.has_audio} preparingAudio={episode.jobs.some(job => job.pipeline === 'prepare' && (job.status === 'queued' || job.status === 'running'))} audioVersion={episode.jobs.find(job => job.pipeline === 'prepare' && job.status === 'done')?.id || 'initial'} busy={analysisBusy} jobs={episode.jobs} jobsKey={episode.jobs.map(job => `${job.id}-${job.status}-${job.stage}`).join('|')} frameVersion={`${episode.jobs[0]?.id || ""}-${episode.jobs[0]?.status || ""}`} onStarted={refresh} onClose={closeAnalysis} />}<div className="flex gap-2"><button onClick={() => playRange(selected.start_time, selected.end_time)} className="btn btn-quiet flex-1"><Play size={15} /> Play cut</button><button onClick={saveCut} className="btn btn-primary flex-1"><Save size={15} /> Save cut</button></div></div><div className="mt-6 flex gap-2"><button onClick={() => { const previous = episode.cuts[selected.cut_index - 1]; if (previous) selectCut(previous) }} disabled={!selected.active || selected.cut_index === 0} className="btn btn-quiet flex-1"><ArrowLeft size={14} /> Previous</button><button onClick={() => { const next = episode.cuts[selected.cut_index + 1]; if (next) selectCut(next) }} disabled={!selected.active || selected.cut_index === episode.cuts.length - 1} className="btn btn-quiet flex-1">Next <ArrowRight size={14} /></button></div></div> : <div className="p-8 text-center"><div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[#282438] text-[#bca7ff]"><Scissors size={21} /></div><h2 className="text-sm font-semibold">Select a cut</h2><p className="muted mt-2 text-xs leading-5">Choose a segment in the timeline or cut list to inspect and edit its notes.</p></div> : activeTab === 'chunk' ? selectedChunk && chunkDraft ? <div className="p-5"><p className="text-xs font-semibold uppercase tracking-[.14em] text-[#7bc5cb]">Chunk {selectedChunk.chunk_index + 1}</p><h2 className="mt-2 text-lg font-semibold">{formatTime(selectedChunk.start_time)} → {formatTime(selectedChunk.end_time)}</h2><p className="muted mt-2 text-xs">A working segment for long-video analysis.</p><label className="mt-6 block"><span className="label">Title</span><input className="field" value={chunkDraft.title} onChange={event => setChunkDraft({ ...chunkDraft, title: event.target.value })} placeholder="Name this chunk" /></label><label className="mt-5 block"><span className="label">Working notes</span><textarea className="field min-h-36 resize-y" value={chunkDraft.notes} onChange={event => setChunkDraft({ ...chunkDraft, notes: event.target.value })} placeholder="Chapter happenings, context, or observations…" /></label><div className="mt-5 flex gap-2"><button onClick={() => playRange(selectedChunk.start_time, selectedChunk.end_time)} className="btn btn-quiet flex-1"><Play size={15} /> Play chunk</button><button onClick={saveChunk} className="btn btn-primary flex-1"><Save size={15} /> Save</button></div></div> : <div className="muted p-8 text-center text-sm">Select a chunk in the timeline to inspect it.</div> : activeTab === 'audio' ? <TranscriptEditor processing={!!busy} hasAudio={episode.has_audio} episodeId={id} projectId={episode.project_id} duration={duration} currentTime={time} segments={transcript} selectedId={selectedTranscriptId} onSelect={setSelectedTranscriptId} onSaved={refresh} playRange={playRange} /> : <div className="p-5"><div className="mb-6"><h2 className="text-base font-semibold">Processing settings</h2><p className="muted mt-2 text-xs leading-5">Choose a balance of speed and detail. Run Prepare or Detect hard cuts again to apply changes.</p></div>{settings && <div className="space-y-5"><label className="block"><span className="label">Preview max width · pixels</span><input className="field" type="number" min="320" max="1920" value={settings.proxy_width} onChange={event => setSettings({ ...settings, proxy_width: Number(event.target.value) })} /><p className="muted mt-1 text-[11px]">Lower values make preparation and playback lighter.</p></label><label className="block"><span className="label">Preview quality · CRF</span><input className="field" type="number" min="18" max="35" value={settings.proxy_crf} onChange={event => setSettings({ ...settings, proxy_crf: Number(event.target.value) })} /><p className="muted mt-1 text-[11px]">Lower values keep more detail and use more disk. Range 18–35.</p></label><label className="block"><span className="label">Encoding speed</span><select className="field" value={settings.proxy_preset} onChange={event => setSettings({ ...settings, proxy_preset: event.target.value as typeof settings.proxy_preset })}>{['ultrafast','superfast','veryfast','faster','fast','medium'].map(value => <option key={value} value={value}>{value}</option>)}</select><p className="muted mt-1 text-[11px]">Faster presets prepare quickly; slower presets make smaller files.</p></label><label className="block"><span className="label">Frame extraction interval · seconds</span><input className="field" type="number" min="0.5" max="60" step="0.5" value={settings.frame_interval} onChange={event => setSettings({ ...settings, frame_interval: Number(event.target.value) })} /></label><label className="block"><span className="label">Scene change threshold</span><input className="field" type="number" min="0.05" max="0.9" step="0.01" value={settings.scene_threshold} onChange={event => setSettings({ ...settings, scene_threshold: Number(event.target.value) })} /><p className="muted mt-1 text-[11px]">Lower detects more candidate hard cuts. Range 0.05–0.9.</p></label><label className="block"><span className="label">Chunk duration · seconds</span><input className="field" type="number" min="10" max="3600" step="10" value={settings.chunk_duration} onChange={event => setSettings({ ...settings, chunk_duration: Number(event.target.value) })} /><p className="muted mt-1 text-[11px]">Timeline working segments. Range 10–3600 seconds.</p></label><button onClick={saveSettings} className="btn btn-primary w-full"><Save size={15} /> Save settings</button></div>}<div className="mt-8 border-t hairline pt-5"><h3 className="mb-3 text-xs font-semibold">Episode details</h3><div className="space-y-2 text-xs"><div className="flex justify-between"><span className="muted">Duration</span><span>{formatTime(duration)}</span></div><div className="flex justify-between"><span className="muted">Audio track</span><span>{episode.has_audio ? 'Available' : 'None'}</span></div><div className="flex justify-between"><span className="muted">Frames</span><span>Every {episode.settings.frame_interval}s</span></div></div><button onClick={remove} className="mt-7 text-xs text-[#e9919a] hover:underline"><Trash2 size={14} className="mr-1 inline" /> Delete episode</button></div></div>}
    </aside><div className="editor-timeline scrollbar"><Timeline key={id} duration={duration} waveform={waveform} hasAudio={episode.has_audio} preparing={episode.jobs.some(job => job.pipeline === 'prepare' && (job.status === 'queued' || job.status === 'running'))} seekRevision={seekRevision} chunks={episode.chunks} cuts={episode.cuts} cutQueueStates={cutQueueStates} transcript={transcript} currentTime={time} selectedCutId={selectedId} selectedChunkId={selectedChunkId} selectedTranscriptId={selectedTranscriptId} onSeek={seek} onSelectCut={selectCut} onOpenCutAnalysis={openCutAnalysis} onSelectChunk={selectChunk} onSelectTranscript={selectTranscript} /></div></div>}</div>
}

export default function App() {
  const route = useRoute()
  if (route === '/storage') return <Storage />
  if (route.startsWith('/project/')) return <ProjectView key={route} id={route.slice(9)} />
  if (route.startsWith('/episode/')) return <Editor key={route} id={route.slice(9)} />
  return <Home />
}
