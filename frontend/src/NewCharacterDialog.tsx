import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { CheckCircle2, Loader2, Save, UserPlus, X, XCircle } from 'lucide-react'
import { api, type ContextItem } from './api'
import { Switch } from './EditorUi'
import { ActionButton, Elapsed } from './RunUi'

export type NewCharacter = { title: string; content: string; is_main: boolean; episode_only: boolean }
export type NewCharacterResult = { character: ContextItem; job_id: string }
type ProfileStatus = { id: string; status: string; stage: string; error: string | null }
type Box = [number, number, number, number]

/**
 * Creates a project character from one character box. The box is labelled with it, and a queued job
 * describes the box (character sheet and outfit sheet) and appends the text to the character's Details.
 * Closing the dialog never stops the job.
 */
export function NewCharacterDialog({ projectId, box, frameUrl, imageSize, boxNumber, onCreate, onClose }: {
  projectId: string
  box: Box
  frameUrl: string
  imageSize: [number, number]
  boxNumber: number
  onCreate: (input: NewCharacter) => Promise<NewCharacterResult>
  onClose: () => void
}) {
  const [input, setInput] = useState<NewCharacter>({ title: '', content: '', is_main: false, episode_only: false })
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [created, setCreated] = useState<{ result: NewCharacterResult; since: string } | null>(null)
  const [status, setStatus] = useState<ProfileStatus | null>(null)
  const [details, setDetails] = useState<{ item: ContextItem; draft: string } | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.querySelector<HTMLInputElement>('input')?.focus()
    return () => opener?.focus()
  }, [])
  const jobId = created?.result.job_id
  const finished = status?.status === 'done' || status?.status === 'failed' || status?.status === 'cancelled'
  useEffect(() => {
    if (!jobId || finished) return
    let alive = true
    const poll = () => void api<ProfileStatus>(`/jobs/${jobId}/character-profile-status`).then(value => { if (alive) setStatus(value) }).catch(() => {})
    poll()
    const timer = setInterval(poll, 3000)
    return () => { alive = false; clearInterval(timer) }
  }, [jobId, finished])
  // The finished job wrote the sheets into Details; read the character again to show them.
  useEffect(() => {
    if (status?.status !== 'done' || !created) return
    void api<ContextItem[]>(`/projects/${projectId}/context`).then(items => {
      const item = items.find(entry => entry.id === created.result.character.id)
      if (item) setDetails({ item, draft: item.content })
    }).catch(reason => setError((reason as Error).message))
  }, [status?.status, created, projectId])
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!input.title.trim() || creating) return
    setCreating(true); setError('')
    try {
      const result = await onCreate({ ...input, title: input.title.trim() })
      setCreated({ result, since: new Date().toISOString() })
    } catch (reason) { setError((reason as Error).message) } finally { setCreating(false) }
  }
  async function saveDetails() {
    if (!details) return
    setSaving(true); setError('')
    try {
      const { item, draft } = details
      const result = await api<ContextItem>(`/context/${item.id}`, { method: 'PUT', body: JSON.stringify({ kind: item.kind, title: item.title, content: draft, episode_id: item.episode_id, is_main: !!item.is_main }) })
      setDetails({ item: result, draft: result.content }); setSaved(true)
    } catch (reason) { setError((reason as Error).message) } finally { setSaving(false) }
  }
  const name = created?.result.character.title
  return createPortal(<div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="new-character-title" className="max-h-[92vh] w-full max-w-2xl overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl outline-none">
      <div className="flex items-start justify-between gap-4"><div><h2 id="new-character-title" className="flex items-center gap-2 text-base font-semibold"><UserPlus size={17} className="text-[#e2709c]" aria-hidden /> {name ? name : 'Add new character'}</h2><p className="muted mt-1 text-xs leading-5">Creates a character for the whole project from box {boxNumber}, labels the box with it, and has Qwen describe the character and the outfit from this box into the character's Details.</p></div><button type="button" onClick={onClose} aria-label="Close new character" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button></div>
      <div className="mt-5 grid gap-5 sm:grid-cols-[180px_minmax(0,1fr)]">
        <svg viewBox={`${box[0]} ${box[1]} ${box[2] - box[0]} ${box[3] - box[1]}`} className="h-56 w-full rounded-lg bg-black sm:h-64" role="img" aria-label={`Box ${boxNumber}`}><image href={frameUrl} width={imageSize[0]} height={imageSize[1]} /></svg>
        {!created ? <form className="space-y-4" onSubmit={event => void submit(event)}>
          <label className="block"><span className="label">Name</span><input className="field" maxLength={200} value={input.title} onChange={event => setInput({ ...input, title: event.target.value })} placeholder="The name used everywhere in the project" required /></label>
          <label className="block"><span className="label">Character info</span><textarea className="field min-h-28 resize-y" maxLength={50000} value={input.content} onChange={event => setInput({ ...input, content: event.target.value })} placeholder="Who they are: role, relationships, personality, how they speak…" /></label>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
            <Switch checked={input.is_main} onChange={value => setInput({ ...input, is_main: value })} label="Main character" hint="Mark as one of the series' main characters" />
            <Switch checked={input.episode_only} onChange={value => setInput({ ...input, episode_only: value })} label="Only in this episode" hint="Scope the character to this episode instead of the whole project" />
          </div>
          {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
          <div className="flex justify-end gap-2"><button type="button" className="btn btn-quiet" onClick={onClose}>Cancel</button><button type="submit" className="btn btn-primary !px-3 !py-2 !text-xs" disabled={creating || !input.title.trim()} aria-busy={creating || undefined}>{creating ? <Loader2 size={14} className="animate-spin" aria-hidden /> : <UserPlus size={14} />}Create and describe</button></div>
        </form> : <div className="space-y-3 text-xs">
          <p className="flex items-center gap-2 text-[#7bdcb2]"><CheckCircle2 size={14} aria-hidden />Created {name} and labelled box {boxNumber}.</p>
          {!finished && <p role="status" className="flex items-center gap-2 rounded-lg border border-[#4a4363] bg-[#221c33] px-3 py-2 text-[#c3aeff]"><Loader2 size={14} className="animate-spin" aria-hidden />{status?.status === 'running' ? `${status.stage}…` : 'Queued · starts when the episode’s current job finishes'}<span className="ml-auto"><Elapsed since={created.since} /></span></p>}
          {!finished && <p className="muted">You can close this dialog: the description continues in the queue and is added to the character's Details.</p>}
          {(status?.status === 'failed' || status?.status === 'cancelled') && <p role="alert" className="flex items-start gap-2 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-[#ffb6bc]"><XCircle size={14} className="mt-0.5 shrink-0" aria-hidden />{status.status === 'cancelled' ? 'The description was cancelled.' : status.error || 'The description failed.'} The character and label are kept; retry it from the queue.</p>}
          {status?.status === 'done' && !details && !error && <p className="muted flex items-center gap-2"><Loader2 size={14} className="animate-spin" aria-hidden />Loading Details…</p>}
          {details && <label className="block"><span className="label">Details {status?.stage && <span className="muted normal-case">· {status.stage}</span>}</span><textarea aria-label="Character details" className="field min-h-72 resize-y font-mono text-xs" value={details.draft} onChange={event => { setDetails({ ...details, draft: event.target.value }); setSaved(false) }} /></label>}
          {error && <p role="alert" className="text-[#ff9fa7]">{error}</p>}
          <div className="flex items-center justify-end gap-2">{saved && <span className="text-[#7bdcb2]">Saved</span>}{details && <ActionButton variant="primary" pending={saving} disabled={details.draft === details.item.content} icon={<Save size={14} />} onClick={() => void saveDetails()}>Save details</ActionButton>}<button type="button" className="btn btn-quiet" onClick={onClose}>Close</button></div>
        </div>}
      </div>
    </div>
  </div>, document.body)
}
