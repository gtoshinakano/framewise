import { LocalModelControl } from './LocalModelControl'
import { VisualEngineControl } from './VisualEngineControl'
import Storage from './Storage'
import AnalysisDialog from './AnalysisDialog'
import { adjacentCut, nextNeedingAction, useCutFilters } from './cutFilters'
import { analysisQueueStates } from './analysisQueueStates'
import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { ArrowLeft, ArrowRight, ChevronRight, Clapperboard, Film, FolderOpen, Layers3, Merge, PanelLeftOpen, Pause, Play, Plus, Redo2, Save, Scissors, Trash2, Undo2, UploadCloud, Video, X } from 'lucide-react'
import { api, jsonBody, mediaUrl, upload, type AudioSelection, type Chunk, type ContextItem, type Cut, type Episode, type EpisodeStatus, type EpisodeSummary, type Job, type Project, type TranscriptSegment, type Waveform } from './api'
import { ArchivedCuts } from './ArchivedCuts'
import { Timeline } from './Timeline'
import { CutGallery } from './CutGallery'
import TranscriptEditor, { type AudioView } from './TranscriptEditor'
import { SequencePanel } from './SequencePanel'
import SequenceDialog from './SequenceDialog'
import { sequenceApi, type SequenceSummary } from './sequenceApi'
import { ToastProvider, useToast } from './Toasts'
import { usePref } from './layoutPrefs'
import { EditorHeader, type CharacterStep } from './EditorHeader'
import { InspectorRail, type InspectorTab } from './InspectorRail'
import { CutInspector } from './CutInspector'
import { ChunkInspector } from './ChunkInspector'
import { QueueDrawer } from './QueueDrawer'
import type { QueueSummary } from './queueView'
import { EpisodeSettingsDialog } from './EpisodeSettingsDialog'
import { IconButton } from './EditorUi'
import { useCutEdits } from './useCutEdits'
import { useEditorShortcuts } from './useEditorShortcuts'
import { formatTimecode } from './timecode'

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
  return <div className="min-h-screen"><header className="border-b hairline bg-[#10131a]"><div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4"><Brand /><div className="flex items-center gap-3"><VisualEngineControl /><LocalModelControl /><button onClick={() => go('/storage')} className="btn btn-quiet">Storage &amp; cleanup</button></div></div></header>
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
  return <div className="min-h-screen"><header className="border-b hairline bg-[#10131a]"><div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4"><Brand /><div className="flex items-center gap-3"><VisualEngineControl /><LocalModelControl /><button onClick={() => go('/')} className="btn btn-quiet"><ArrowLeft size={16} /> All projects</button></div></div></header><main className="mx-auto max-w-7xl px-6 py-10">
    {error && <ErrorBanner message={error} clear={() => setError('')} />}{!project ? <p className="muted">Loading project…</p> : <><div className="mb-10 flex flex-wrap items-end justify-between gap-4"><div><p className="mb-3 text-xs font-semibold uppercase tracking-[.18em] text-[#b49cf1]">Project</p><h1 className="text-3xl font-semibold">{project.title}</h1><p className="muted mt-2 text-sm">{project.episodes.length} episodes · Created {new Date(`${project.created_at}Z`).toLocaleDateString()}</p></div><button onClick={() => setUploading(true)} className="btn btn-primary"><Plus size={17} /> Add episode</button></div>
      <div className="grid gap-7 lg:grid-cols-[1fr_320px]"><section><div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-semibold">Episodes</h2><span className="muted text-xs">Original videos and analysis</span></div><div className="panel overflow-hidden">{project.episodes.length === 0 ? <div className="muted p-12 text-center text-sm">No episodes yet. Add a video to begin.</div> : project.episodes.map((episode, index) => <button key={episode.id} onClick={() => go(`/episode/${episode.id}`)} className="flex w-full items-center gap-4 border-b hairline p-5 text-left transition last:border-b-0 hover:bg-[#1b202b]"><div className="flex h-14 w-20 shrink-0 items-center justify-center rounded-lg bg-[#252b38] text-[#a991e8]"><Video size={25} /></div><div className="min-w-0 flex-1"><div className="mb-1 flex items-center gap-2"><span className="text-xs font-semibold text-[#8d82ad]">{String(index + 1).padStart(2, '0')}</span><strong className="truncate text-sm text-white">{episode.title}</strong></div><div className="flex flex-wrap items-center gap-4"><Status status={episode.status} stage={episode.stage} /><span className="muted text-xs">{formatTime(episode.duration)}</span></div>{episode.error && <p className="mt-2 truncate text-xs text-[#ff9fa7]">{episode.error}</p>}</div><ChevronRight size={18} className="muted" /></button>)}</div></section>
      <aside className="panel h-fit p-6"><div className="mb-5 flex items-center gap-2 text-sm font-semibold"><Layers3 size={17} className="text-[#bca7ff]" /> Project context</div><label className="mb-4 block"><span className="label">Name</span><input className="field" value={title} onChange={event => setTitle(event.target.value)} maxLength={200} /></label><label className="block"><span className="label">Synopsis / context</span><textarea className="field min-h-36 resize-y" value={synopsis} onChange={event => setSynopsis(event.target.value)} placeholder="Describe the story, visual style, or notes that future analysis should know…" /></label><button onClick={save} disabled={saving || !title.trim()} className="btn btn-quiet mt-4 w-full"><Save size={15} /> Save context</button><div className="mt-7 border-t hairline pt-5"><button onClick={remove} className="text-xs text-[#e9919a] hover:underline"><Trash2 size={14} className="mr-1 inline" /> Delete project</button></div></aside></div><ContextLibrary projectId={id} episodes={project.episodes} /></>}
  </main>{uploading && <UploadDialog projectId={id} onClose={() => setUploading(false)} onDone={(_, episodeId) => go(`/episode/${episodeId}`)} />}</div>
}

type ContextDraft = Pick<ContextItem, 'kind' | 'title' | 'content' | 'episode_id' | 'is_main'> & { id?: string }
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
      const result = await api<ContextItem>(path, { method: draft.id ? 'PUT' : 'POST', body: JSON.stringify({ ...draft, is_main: draft.kind === 'character' && !!draft.is_main }) })
      setDraft(result); setError(''); await load()
    } catch (err) { setError((err as Error).message) } finally { setSaving(false) }
  }
  async function remove() {
    if (!draft?.id || !window.confirm(`Delete context item “${draft.title}”?`)) return
    try { await api(`/context/${draft.id}`, { method: 'DELETE' }); setDraft(null); await load() } catch (err) { setError((err as Error).message) }
  }
  return <section className="mt-8"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">Context library</h2><p className="muted mt-1 text-xs">Reusable story, character, and world details for later analysis.</p></div><button onClick={() => setDraft({ kind: 'general', title: '', content: '', episode_id: null })} className="btn btn-quiet"><Plus size={15} /> Add context</button></div>
    {error && <ErrorBanner message={error} clear={() => setError('')} />}
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_420px]"><div className="panel min-h-52 overflow-hidden">{items.length ? items.map(item => <button key={item.id} onClick={() => setDraft(item)} className={`flex w-full items-center justify-between gap-4 border-b hairline px-5 py-4 text-left last:border-b-0 hover:bg-[#202531] ${draft?.id === item.id ? 'bg-[#27223a]' : ''}`}><div className="min-w-0"><div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[#a993e1]">{contextKinds.find(kind => kind.value === item.kind)?.label}{item.kind === 'character' && item.is_main ? ' · Main' : ''}{item.episode_id ? ` · ${episodes.find(episode => episode.id === item.episode_id)?.title || 'Episode'}` : ' · Project'}</div><strong className="block truncate text-sm">{item.title}</strong><p className="muted mt-1 truncate text-xs">{item.content || 'No details yet'}</p></div><ChevronRight size={16} className="muted shrink-0" /></button>) : <div className="muted flex h-full min-h-52 items-center justify-center px-6 text-center text-sm">Add a character, world, scenario, or chapter note.</div>}</div>
      <div className="panel p-5">{draft ? <div className="space-y-4"><div className="flex items-center justify-between"><h3 className="text-sm font-semibold">{draft.id ? 'Edit context' : 'New context'}</h3><button onClick={() => setDraft(null)} className="muted" aria-label="Close context editor"><X size={17} /></button></div><div className="grid gap-3 sm:grid-cols-2"><label><span className="label">Type</span><select className="field" value={draft.kind} onChange={event => setDraft({ ...draft, kind: event.target.value as ContextItem['kind'] })}>{contextKinds.map(kind => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label><label><span className="label">Scope</span><select className="field" value={draft.episode_id || ''} onChange={event => setDraft({ ...draft, episode_id: event.target.value || null })}><option value="">Entire project</option>{episodes.map(episode => <option key={episode.id} value={episode.id}>{episode.title}</option>)}</select></label></div><label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="e.g. Main character, Chapter 2" /></label>{draft.kind === 'character' && <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={!!draft.is_main} onChange={event => setDraft({ ...draft, is_main: event.target.checked ? 1 : 0 })} /> Main character</label>}<label className="block"><span className="label">Details</span><textarea className="field min-h-36 resize-y" value={draft.content} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Write details that future analysis should use…" /></label><div className="flex items-center justify-between gap-3"><button onClick={remove} disabled={!draft.id} className="text-xs text-[#e9919a] disabled:opacity-30"><Trash2 size={14} className="mr-1 inline" /> Delete</button><button onClick={save} disabled={saving || !draft.title.trim()} className="btn btn-primary"><Save size={15} /> Save context</button></div></div> : <div className="muted flex min-h-52 items-center justify-center text-center text-sm">Select an item to edit, or add new context.</div>}</div></div></section>
}

const DEFAULT_CUTS_WIDTH = 260
const MIN_TIMELINE = 160
function Editor({ id }: { id: string }) {
  const toast = useToast()
  const [episode, setEpisode] = useState<Episode | null>(null)
  const [project, setProject] = useState<Project | null>(null)
  const [waveform, setWaveform] = useState<Waveform | null>(null)
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)
  const [selectedTranscriptId, setSelectedTranscriptId] = useState<string | null>(null)
  const [audioSelection, setAudioSelection] = useState<AudioSelection | null>(null)
  const [time, setTime] = useState(0)
  const [seekRevision, setSeekRevision] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [cutDetection, setCutDetection] = useState<{ cutId: string; jobId: string } | null>(null)
  const [loadError, setLoadError] = useState('')
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const closeAnalysis = useCallback(() => setAnalysisOpen(false), [])
  const [activeTab, setActiveTab] = useState<InspectorTab>('cut')
  const [audioView, setAudioView] = useState<AudioView>('lines')
  const [sequences, setSequences] = useState<SequenceSummary[]>([])
  const [sequenceOpen, setSequenceOpen] = useState<string | null>(null)
  const closeSequence = useCallback(() => setSequenceOpen(null), [])
  const refreshSequences = useCallback(() => { void sequenceApi.list(id).then(setSequences).catch(() => {}) }, [id])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const closeSettings = useCallback(() => setSettingsOpen(false), [])
  // The queue drawer, opened at the top or at Needs attention.
  const [queueOpen, setQueueOpen] = useState<false | 'top' | 'attention'>(false)
  const closeQueue = useCallback(() => setQueueOpen(false), [])
  const [queueSummary, setQueueSummary] = useState<QueueSummary | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [cutsCollapsed, setCutsCollapsed] = usePref('editor.cutsCollapsed', false)
  const [timelineHeight, setTimelineHeight] = usePref<number | null>('editor.timelineHeight', null)
  const video = useRef<HTMLVideoElement>(null)
  const playbackEnd = useRef<number | null>(null)
  const latestEpisode = useRef<Episode | null>(null)
  const lastFullRefresh = useRef(0)
  const waveformVersion = useRef<string | null>(null)
  const showEpisode = useCallback((value: Episode) => {
    latestEpisode.current = value
    setEpisode(value)
  }, [])
  const refresh = useCallback(async () => {
    try {
      const value = await api<Episode>(`/episodes/${id}`)
      showEpisode(value)
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
      setSequences(await sequenceApi.list(id))
      lastFullRefresh.current = Date.now()
      setLoadError('')
    } catch (err) { setLoadError((err as Error).message) }
  }, [id, showEpisode])
  useEffect(() => {
    let active = true
    setQueueSummary(null)
    const tick = async () => {
      try {
        const status = await api<EpisodeStatus>(`/episodes/${id}/status`)
        if (!active) return
        setQueueSummary(status.queue)
        const current = latestEpisode.current
        const changed = !current || current.id !== id || status.status !== current.status || status.stage !== current.stage || status.job_id !== (current.jobs[0]?.id || null)
        if (changed || status.status === 'queued' || status.status === 'processing' || Date.now() - lastFullRefresh.current > 30000) await refresh()
      } catch (err) { if (active) setLoadError((err as Error).message) }
    }
    void tick()
    const timer = setInterval(tick, 2500)
    return () => { active = false; clearInterval(timer) }
  }, [id, refresh])
  // Cut Assistance edits the transcript outside this editor; reload it as soon as a proposal is applied.
  useEffect(() => { const reload = () => { void refresh() }; window.addEventListener('framewise:transcript-changed', reload); return () => window.removeEventListener('framewise:transcript-changed', reload) }, [refresh])
  useEffect(() => { if (episode) api<Project>(`/projects/${episode.project_id}`).then(setProject).catch(() => {}) }, [episode?.project_id])
  useEffect(() => { setSelectedId(null); setSelectedChunkId(null); setSelectedTranscriptId(null); setAudioSelection(null); setWaveform(null); setTranscript([]); setSequences([]); setSequenceOpen(null); setTime(0); playbackEnd.current = null }, [id])
  const selected = [...(episode?.cuts || []), ...(episode?.archived_cuts || [])].find(cut => cut.id === selectedId)
  const selectedChunk = episode?.chunks.find(chunk => chunk.id === selectedChunkId)
  const cutDetectionJob = cutDetection?.cutId === selectedId ? episode?.jobs.find(job => job.id === cutDetection.jobId) : undefined
  const duration = episode?.duration || 0
  const cutFilters = useCutFilters()
  function seek(value: number) { const target = Math.max(0, Math.min(duration, value)); playbackEnd.current = null; if (video.current) video.current.currentTime = target; setTime(target); setSeekRevision(revision => revision + 1) }
  function openCutAnalysis(cut: Cut) { selectCut(cut, true); setActiveTab('cut'); setAnalysisOpen(true) }
  function selectCut(cut: Cut, keepAnalysisOpen = false) { if (!keepAnalysisOpen) setAnalysisOpen(false); video.current?.pause(); setSelectedId(cut.id); setActiveTab(current => current === 'sequences' ? current : 'cut'); seek(cut.first_frame_time) }
  function selectChunk(chunk: Chunk) { setSelectedChunkId(chunk.id); setActiveTab('chunk'); seek(chunk.start_time) }
  function selectTranscript(segment: TranscriptSegment) { setSelectedTranscriptId(segment.id); setActiveTab('audio'); setAudioView('lines'); seek(segment.start_time) }
  function selectAudio(range: AudioSelection | null) { setAudioSelection(range); if (range) { setActiveTab('audio'); setAudioView('tools') } }
  function playRange(start: number, end: number) { const player = video.current; if (!player) return; player.currentTime = start; playbackEnd.current = end; void player.play().catch(err => toast({ message: (err as Error).message, tone: 'error' })) }
  function togglePlay() { const player = video.current; if (!player) return; if (player.paused) void player.play().catch(() => {}); else player.pause() }
  function onVideoTimeUpdate(player: HTMLVideoElement) { const end = playbackEnd.current; if (!player.seeking && end !== null && player.currentTime >= end) { player.pause(); player.currentTime = end; playbackEnd.current = null }; setTime(player.currentTime) }
  const fail = (err: unknown) => toast({ message: (err as Error).message, tone: 'error' })
  async function run(pipeline: 'prepare' | 'hardcuts') {
    if (pipeline === 'hardcuts' && episode?.cuts.length && !window.confirm('Detect the hard cuts of the whole video again? Notes and results stay on cuts whose boundary is found again; the others are archived. Earlier cut edits can no longer be undone.')) return
    try { await api(`/episodes/${id}/run`, jsonBody({ pipeline })); await refresh(); toast({ message: pipeline === 'prepare' ? 'Rebuilding the preview and waveform.' : 'Detecting hard cuts.' }) } catch (err) { fail(err) }
  }
  const [bulkDetectionWorking, setBulkDetectionWorking] = useState(false)
  async function queueCharacterStep(path: CharacterStep) {
    if (bulkDetectionWorking) return
    const rerun = path === 'character-detection-rerun'
    if (rerun && !window.confirm('Redo the automatic character labels in every cut? Your manual labels and the boxes you drew stay exactly as they are.')) return
    setBulkDetectionWorking(true)
    try {
      const result = await api<{ queued: number }>(`/episodes/${id}/${path}`, { method: 'POST' })
      await refresh()
      const boxes = path === 'character-boxes'
      const label = boxes ? 'box detection' : path === 'character-detection-all' ? 'character detection' : 'character recognition'
      const cuts = `${result.queued} ${result.queued === 1 ? 'cut' : 'cuts'}`
      toast(result.queued ? { tone: 'success', message: rerun ? `Queued recognition again for ${cuts}; your manual labels are kept.` : `Queued ${label} for ${cuts}.` }
        : { message: boxes ? 'No cuts need box detection.' : rerun ? 'Every box already has your label.' : 'No cuts with boxes need recognition. Run “Find character boxes” first.' })
    } catch (err) { fail(err) } finally { setBulkDetectionWorking(false) }
  }
  async function detectWithin(threshold: number) { if (!selected) return; try { const result = await api<{ job_id: string }>(`/cuts/${selected.id}/detect-hardcuts`, jsonBody({ scene_threshold: threshold })); setCutDetection({ cutId: selected.id, jobId: result.job_id }); await refresh() } catch (err) { fail(err) } }
  async function toggleAnalysis() { if (!selected) return; try { await api<Cut>(`/cuts/${selected.id}/analysis-enabled`, { method: 'PUT', body: JSON.stringify({ analysis_enabled: !selected.analysis_enabled }) }); await refresh() } catch (err) { fail(err) } }
  async function saveNotes(notes: Pick<Cut, 'title' | 'description' | 'notes'>) { if (!selected) return; try { await api<Cut>(`/cuts/${selected.id}`, { method: 'PUT', body: JSON.stringify({ ...notes, results: selected.results }) }); await refresh(); toast({ tone: 'success', message: `Saved the notes of cut ${selected.cut_index + 1}.` }) } catch (err) { fail(err) } }
  async function saveChunk(chunk: Chunk) { try { await api<Chunk>(`/chunks/${chunk.id}`, { method: 'PUT', body: JSON.stringify({ title: chunk.title, notes: chunk.notes }) }); await refresh(); toast({ tone: 'success', message: `Saved chunk ${chunk.chunk_index + 1}.` }) } catch (err) { fail(err) } }
  async function saveSettings(settings: Episode['settings']) { try { await api(`/episodes/${id}/settings`, { method: 'PUT', body: JSON.stringify(settings) }); await refresh(); toast({ tone: 'success', message: 'Settings saved. Run Prepare or redetect cuts to apply them.' }); return true } catch (err) { fail(err); return false } }
  async function remove() { if (!episode || !window.confirm(`Delete “${episode.title}” and all its media? This cannot be undone.`)) return; try { await api(`/episodes/${id}`, { method: 'DELETE' }); go(`/project/${episode.project_id}`) } catch (err) { fail(err) } }
  // After a cut edit the playhead stays where it is, so several splits in a row need no scrubbing back.
  const cutEdits = useCutEdits({ episodeId: id, toast, onEpisode: showEpisode, onSelect: cutId => { setSelectedId(cutId); setActiveTab(current => current === 'sequences' ? current : 'cut') } })
  const jobs = episode?.jobs || []
  const active = (job: Job) => job.status === 'running' || job.status === 'queued'
  const busy = jobs.some(active)
  const mediaBusy = jobs.some(job => ['prepare', 'hardcuts', 'cutframes'].includes(job.pipeline) && active(job))
  // Cut edits wait for preview jobs by themselves; only these two replace or rebuild cuts underneath them.
  const cutsBusy = jobs.some(job => ['prepare', 'hardcuts'].includes(job.pipeline) && active(job))
  const cutQueueStates = analysisQueueStates(jobs)
  const analysisBusy = jobs.some(job => (['prepare', 'hardcuts', 'character_detect'].includes(job.pipeline) || job.workflow === 'boundary_frame') && active(job))
  const characterBlock = bulkDetectionWorking ? 'Queuing…' : !episode?.cuts.length ? 'Detect hard cuts first.' : jobs.some(job => job.pipeline !== 'character_detect' && active(job)) ? 'Wait for the running jobs to finish.' : ''
  const frameVersion = `${jobs[0]?.id || ''}-${jobs[0]?.status || ''}`
  const history = episode?.cut_history
  const activeSelected = selected?.active ? selected : undefined
  const historyReason = cutsBusy ? 'Wait for the cut detection or Prepare to finish' : cutEdits.working ? 'Working…' : ''
  const editReason = !activeSelected ? 'Select a cut first' : historyReason
  // Split works on the cut under the playhead, like a video editor, so several splits in a row need no reselecting.
  const splitTarget = episode?.cuts.find(cut => time >= cut.start_time + 0.1 && time <= cut.end_time - 0.1)
  const splitReason = historyReason || (!episode?.cuts.length ? 'Detect hard cuts first' : !splitTarget ? 'Move the playhead inside a cut, at least 0.1 s from its edges' : '')
  const mergeReason = editReason || (activeSelected && activeSelected.cut_index === (episode?.cuts.length || 0) - 1 ? 'This is the last cut' : '')
  const deleteReason = editReason || ((episode?.cuts.length || 0) < 2 ? 'The only cut cannot be deleted' : '')
  const split = () => { if (splitTarget && !splitReason) void cutEdits.edit('split', splitTarget.id, { time }) }
  const undo = () => { if (history?.undo && !historyReason) void cutEdits.undo() }
  const redo = () => { if (history?.redo && !historyReason) void cutEdits.redo() }
  const step = (direction: 1 | -1) => { if (!activeSelected || !episode) return; const target = episode.cuts[activeSelected.cut_index + direction]; if (target) selectCut(target) }
  useEditorShortcuts(!!episode && !analysisOpen && !sequenceOpen && !settingsOpen && !queueOpen, {
    playPause: togglePlay, split, undo, redo, previous: () => step(-1), next: () => step(1),
    nextNeeding: () => { if (!episode) return; const next = nextNeedingAction(episode.cuts, selectedId, cutFilters.active); if (next) selectCut(next) },
    filters: () => setFiltersOpen(true),
  })
  const editTools = <>
    <IconButton icon={Undo2} size="sm" label={history?.undo ? `Undo: ${history.undo.label}` : 'Nothing to undo'} shortcut="Ctrl+Z" title={historyReason || (history?.undo ? `Undo: ${history.undo.label} (Ctrl+Z)` : 'Nothing to undo')} disabled={!history?.undo || !!historyReason} onClick={undo} />
    <IconButton icon={Redo2} size="sm" label={history?.redo ? `Redo: ${history.redo.label}` : 'Nothing to redo'} shortcut="Ctrl+Shift+Z" title={historyReason || (history?.redo ? `Redo: ${history.redo.label} (Ctrl+Shift+Z)` : 'Nothing to redo')} disabled={!history?.redo || !!historyReason} onClick={redo} />
    <span className="mx-1 h-5 w-px bg-[#2c323d]" aria-hidden />
    <IconButton icon={Scissors} size="sm" text="Split" label="Split the cut under the playhead" title={splitReason || `Split cut ${(splitTarget?.cut_index ?? 0) + 1} at the playhead (S). Undo with Ctrl+Z.`} disabled={!!splitReason} onClick={split} />
    <IconButton icon={Merge} size="sm" text="Merge" label="Merge the selected cut with the next one" title={mergeReason || 'Merge the selected cut with the next one. Undo with Ctrl+Z.'} disabled={!!mergeReason} onClick={() => activeSelected && void cutEdits.edit('merge-next', activeSelected.id)} />
    <IconButton icon={Trash2} size="sm" tone="danger" label="Delete the selected cut" title={deleteReason || 'Delete the selected cut; its time joins a neighbour and its notes are archived. Undo with Ctrl+Z.'} disabled={!!deleteReason} onClick={() => activeSelected && void cutEdits.edit('delete', activeSelected.id)} />
  </>
  const resize = useRef<{ pointerId: number; y: number; height: number } | null>(null)
  const timelineBox = useRef<HTMLDivElement>(null)
  const clampHeight = (value: number) => Math.round(Math.max(MIN_TIMELINE, Math.min(window.innerHeight * 0.6, value)))
  const workspaceStyle = { '--cuts-w': cutsCollapsed ? '0px' : `${DEFAULT_CUTS_WIDTH}px`, ...(timelineHeight ? { '--timeline-h': `${timelineHeight}px` } : {}) } as CSSProperties
  const audioBusy = jobs.some(job => ['transcribe', 'voice_sample', 'subtitles'].includes(job.pipeline) && active(job))
  return <div className="editor-shell bg-[#0d1016]">
    <EditorHeader projectId={episode?.project_id} projectTitle={project?.title} episodeTitle={episode?.title} status={episode?.status || 'queued'} stage={episode?.stage}
      hasCuts={!!episode?.cuts.length} mediaBlocked={busy ? 'Wait for the running jobs to finish.' : ''} characterBlock={characterBlock} onCharacterStep={step => void queueCharacterStep(step)}
      onDetectCuts={() => void run('hardcuts')} onPrepare={() => void run('prepare')}
      queue={queueSummary} running={jobs.some(job => job.status === 'running')} onOpenQueue={focus => setQueueOpen(focus || 'top')}
      onOpenSettings={() => setSettingsOpen(true)} onDeleteEpisode={() => void remove()} />
    {!episode ? <div className="muted p-12">{loadError || 'Loading episode…'}</div> : <div className="editor-workspace" style={workspaceStyle}>
      <aside className="editor-cuts scrollbar" hidden={cutsCollapsed} aria-label="Cut list">
        <CutGallery cuts={episode.cuts} selectedId={selectedId} onSelect={selectCut} onOpenAnalysis={openCutAnalysis} frameVersion={frameVersion} queueStates={cutQueueStates} onCollapse={() => setCutsCollapsed(true)} />
        <ArchivedCuts cuts={episode.archived_cuts} selectedId={selectedId} onSelect={selectCut} onOpenAnalysis={openCutAnalysis} />
      </aside>
      <main className="editor-stage min-w-0">
        {cutsCollapsed && <button type="button" onClick={() => setCutsCollapsed(false)} className="absolute left-0 top-3 z-10 flex items-center gap-1 rounded-r-lg border border-l-0 border-[#353c48] bg-[#1b202a] py-2 pl-1.5 pr-2 text-[11px] font-semibold text-[#c3c8d2] hover:bg-[#272e3a]" title="Show the cut list"><PanelLeftOpen size={14} aria-hidden /> Cuts</button>}
        {loadError && <p role="alert" className="mx-4 mt-3 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-xs text-[#ffb6bc]">Could not refresh the episode: {loadError}</p>}
        {episode.error && <p className="mx-4 mt-3 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-xs text-[#ffb6bc]">{episode.error}</p>}
        <div className={`flex min-h-0 flex-1 flex-col pb-2 pr-4 pt-3 ${cutsCollapsed ? 'pl-16' : 'pl-4'}`}>
          <div className="preview-stage relative overflow-hidden rounded-xl border border-[#303540] bg-black shadow-2xl">{episode.proxy_ready ? <video ref={video} src={mediaUrl(id, 'proxy')} className="h-full w-full" controls onSeeking={() => setSeekRevision(revision => revision + 1)} onTimeUpdate={event => onVideoTimeUpdate(event.currentTarget)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} /> : <div className="flex h-full flex-col items-center justify-center text-center"><Film size={36} className="mb-4 text-[#a58ae8]" /><strong className="text-sm">{busy ? episode.stage : 'Preview unavailable'}</strong><p className="muted mt-2 text-xs">{busy ? 'Video preparation runs in the background.' : 'Use Cuts › Rebuild preview & waveform to create the preview.'}</p></div>}</div>
          <div className="mt-2 flex items-center justify-between gap-3 text-xs text-[#9ca5b3]">
            <div className="flex items-center gap-3"><IconButton icon={playing ? Pause : Play} label={playing ? 'Pause' : 'Play'} shortcut="Space" size="sm" onClick={togglePlay} /><span className="font-mono tabular-nums text-[#e8dfff]">{formatTimecode(time)} <span className="text-[#5d6675]">/</span> <span className="text-[#9ca5b3]">{formatTimecode(duration)}</span></span></div>
            <span className="min-w-0 truncate" title={episode.original_filename}>{episode.width && episode.height ? `${episode.width} × ${episode.height}` : '—'} · {episode.original_filename}</span>
          </div>
        </div>
      </main>
      <aside id="inspector-panel" className="editor-inspector bg-[#131720]" role="tabpanel" aria-labelledby={`inspector-tab-${activeTab}`}>
        {activeTab === 'cut' ? selected ? <CutInspector cut={selected} cuts={episode.cuts} queueState={cutQueueStates.get(selected.id)} time={time} duration={duration} mediaBusy={mediaBusy} busy={busy} frameVersion={frameVersion}
          sceneThreshold={episode.settings.scene_threshold ?? 0.32} detectionJob={cutDetectionJob} onSelect={selectCut} onPlay={playRange} onSeek={seek} onOpenAnalysis={() => setAnalysisOpen(true)}
          onToggleAnalysis={() => void toggleAnalysis()} onSaveNotes={saveNotes} onApplyTiming={(start, end) => void cutEdits.edit('timing', selected.id, { start, end })} onDetectWithin={threshold => void detectWithin(threshold)}
          onCopy={text => { void navigator.clipboard?.writeText(text).then(() => toast({ message: `Copied ${text}`, duration: 2000 })).catch(() => {}) }} />
          : <div className="p-8 text-center"><div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[#282438] text-[#bca7ff]"><Scissors size={21} /></div><h2 className="text-sm font-semibold">Select a cut</h2><p className="muted mt-2 text-xs leading-5">Choose a segment in the timeline or the cut list to see its notes, frames and timing.</p></div>
        : activeTab === 'chunk' ? <ChunkInspector chunk={selectedChunk} onPlay={playRange} onSave={saveChunk} />
        : activeTab === 'audio' ? <TranscriptEditor processing={busy} hasAudio={episode.has_audio} subtitleLanguage={episode.subtitle_language} subtitleRun={episode.subtitles.run} jobs={jobs} episodeId={id} projectId={episode.project_id} duration={duration} currentTime={time} segments={transcript} selectedId={selectedTranscriptId} onSelect={setSelectedTranscriptId} onSaved={refresh} playRange={playRange} audioSelection={audioSelection} onAudioSelection={selectAudio} view={audioView} onViewChange={setAudioView} />
        : <SequencePanel episodeId={id} cuts={episode.cuts} selectedCut={selected} sequences={sequences} onChanged={refresh} onOpen={setSequenceOpen} />}
      </aside>
      <nav className="editor-rail" aria-label="Inspector">
        <InspectorRail value={activeTab} onChange={setActiveTab} busy={{ audio: audioBusy, sequences: sequences.some(sequence => sequence.pending_jobs > 0) }} counts={{ sequences: sequences.length }} />
      </nav>
      <div ref={timelineBox} className="editor-timeline border-t hairline">
        <div role="separator" aria-orientation="horizontal" aria-label="Resize the timeline" aria-valuenow={timelineHeight || undefined} tabIndex={0} className="resize-handle" title="Drag to resize the timeline; double-click to reset"
          onPointerDown={event => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); resize.current = { pointerId: event.pointerId, y: event.clientY, height: timelineBox.current?.getBoundingClientRect().height || 300 } }}
          onPointerMove={event => { const drag = resize.current; if (drag?.pointerId === event.pointerId) setTimelineHeight(clampHeight(drag.height - (event.clientY - drag.y))) }}
          onPointerUp={() => { resize.current = null }} onPointerCancel={() => { resize.current = null }} onDoubleClick={() => setTimelineHeight(null)}
          onKeyDown={event => { if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return; event.preventDefault(); const current = timelineHeight || timelineBox.current?.getBoundingClientRect().height || 300; setTimelineHeight(clampHeight(current + (event.key === 'ArrowUp' ? 24 : -24))) }} />
        <div className="scrollbar min-h-0 flex-1 overflow-auto">
          <Timeline key={id} activeFilters={cutFilters.active} filterDisplay={cutFilters.display} onToggleFilter={cutFilters.toggle} onClearFilters={cutFilters.clear} onFilterDisplay={cutFilters.setDisplay} filtersOpen={filtersOpen} onFiltersOpenChange={setFiltersOpen} editTools={editTools}
            duration={duration} waveform={waveform} hasAudio={episode.has_audio} preparing={jobs.some(job => job.pipeline === 'prepare' && active(job))} seekRevision={seekRevision} chunks={episode.chunks} cuts={episode.cuts} cutQueueStates={cutQueueStates} transcript={transcript} currentTime={time} selectedCutId={selectedId} selectedChunkId={selectedChunkId} selectedTranscriptId={selectedTranscriptId} onSeek={seek} onSelectCut={selectCut} onOpenCutAnalysis={openCutAnalysis} onSelectChunk={selectChunk} onSelectTranscript={selectTranscript} audioSelection={audioSelection} onSelectAudio={selectAudio} sequences={sequences} onOpenSequence={setSequenceOpen} />
        </div>
      </div>
    </div>}
    {episode && analysisOpen && selected && <AnalysisDialog cut={selected} cuts={episode.cuts} previousCut={selected.active ? adjacentCut(episode.cuts, selected, -1, cutFilters.active) : undefined} nextCut={selected.active ? adjacentCut(episode.cuts, selected, 1, cutFilters.active) : undefined} onNavigate={cut => selectCut(cut, true)} episodeId={id} projectId={episode.project_id} proxyReady={episode.proxy_ready} waveform={waveform} transcript={transcript} hasAudio={episode.has_audio} preparingAudio={jobs.some(job => job.pipeline === 'prepare' && active(job))} busy={analysisBusy} jobs={jobs} jobsKey={jobs.map(job => `${job.id}-${job.status}-${job.stage}`).join('|')} frameVersion={frameVersion} onStarted={refresh} onClose={closeAnalysis} />}
    {episode && sequenceOpen && <SequenceDialog key={sequenceOpen} sequenceId={sequenceOpen} episodeId={id} proxyReady={episode.proxy_ready} onClose={closeSequence} onChanged={refreshSequences} />}
    {episode && queueOpen && <QueueDrawer episodeId={id} focus={queueOpen === 'attention' ? 'attention' : null} onClose={closeQueue} onChanged={refresh} onSummary={setQueueSummary}
      onOpenCut={cutId => { const cut = [...episode.cuts, ...episode.archived_cuts].find(item => item.id === cutId); if (cut) openCutAnalysis(cut); else toast({ tone: 'error', message: 'That cut no longer exists.' }) }}
      onOpenSequence={setSequenceOpen} />}
    {episode && settingsOpen && <EpisodeSettingsDialog episode={episode} initial={episode.settings} onSave={saveSettings} onDelete={() => void remove()} onClose={closeSettings} />}
  </div>
}

export default function App() {
  const route = useRoute()
  return <ToastProvider>{route === '/storage' ? <Storage /> : route.startsWith('/project/') ? <ProjectView key={route} id={route.slice(9)} /> : route.startsWith('/episode/') ? <Editor key={route} id={route.slice(9)} /> : <Home />}</ToastProvider>
}
