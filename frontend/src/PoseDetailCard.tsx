import { useEffect, useMemo, useState } from 'react'
import { ChevronDown, Eye, EyeOff, TriangleAlert } from 'lucide-react'
import { api } from './api'
import { StatusChip } from './RunUi'
import { BODY_POSES, EYE_GAZES, TARGET_KINDS, applyEdit, entryIndex, label, snapAngle, summarize, viewOf, viewWords, type FrameContent, type Pose, type PoseEdit } from './poseView'

export type PoseBox = { id: string; box: [number, number, number, number]; character_context_id: string | null; character_name: string | null }
type Source = { id: string; content: FrameContent | null }

const KEY = 'framewise.poseCard'
function storedOpen(): boolean {
  try { return localStorage.getItem(KEY) !== 'closed' } catch { return true }
}
function rememberOpen(open: boolean) {
  try { localStorage.setItem(KEY, open ? 'open' : 'closed') } catch { /* The choice then lasts for this visit only. */ }
}

// Every box card of a frame asks for the same result; one request serves them all until the cut's jobs change.
const requests = new Map<string, Promise<Source | null>>()
function loadSource(cutId: string, key: string): Promise<Source | null> {
  const id = `${cutId}:${key}`
  let found = requests.get(id)
  if (!found) {
    found = api<{ id: string; type: string; content: FrameContent | null }[]>(`/cuts/${cutId}/structured-analysis`)
      .then(rows => { const row = rows.find(item => item.type === 'initial_frame'); return row ? { id: row.id, content: row.content } : null })
    requests.set(id, found)
    found.catch(() => requests.delete(id))
    if (requests.size > 40) requests.delete(requests.keys().next().value as string)
  }
  return found
}

const field = 'field !w-auto !px-2 !py-1 !text-[12px]'

/**
 * What the first-frame analysis measured for the character in one box: state, which way the torso and head face, the
 * eyes and what they look at. It can be hidden, and a person can correct a value; that saves a new revision of the
 * first-frame result by hand, like any other edit of that result.
 */
export function PoseDetailCard({ cutId, person, frameBoxes, refreshKey, disabled }: { cutId: string; person: PoseBox; frameBoxes: PoseBox[]; refreshKey: string; disabled: boolean }) {
  const [open, setOpen] = useState(storedOpen)
  const [source, setSource] = useState<Source | null | undefined>(undefined)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [version, setVersion] = useState(0)
  useEffect(() => {
    if (!open || !person.character_context_id) return
    let active = true
    loadSource(cutId, `${refreshKey}:${version}`).then(value => { if (active) { setSource(value); setError('') } }).catch(reason => { if (active) setError((reason as Error).message) })
    return () => { active = false }
  }, [open, cutId, refreshKey, version, person.character_context_id])
  const content = source?.content ?? null
  const index = useMemo(() => entryIndex(content, person, frameBoxes), [content, person, frameBoxes])
  const entry = index >= 0 ? content?.visible_characters?.[index] : undefined
  const pose: Pose | undefined = entry?.pose_detail
  const summary = summarize(pose)
  const peers = (content?.visible_characters ?? []).filter((item, position) => position !== index && item.character).map(item => String(item.character))
  async function save(edit: PoseEdit) {
    if (!source || !content || index < 0) return
    setSaving(true); setError('')
    try {
      await api(`/analysis-results/${source.id}`, { method: 'PUT', body: JSON.stringify({ title: 'Initial frame description', content: JSON.stringify(applyEdit(content, index, edit)) }) })
      requests.clear(); setSource(undefined); setVersion(number => number + 1)
    } catch (reason) {
      const message = (reason as Error).message
      setError(/newer revision/i.test(message) ? 'The first-frame result changed meanwhile. Reloaded; try again.' : message)
      requests.clear(); setSource(undefined); setVersion(number => number + 1)
    } finally { setSaving(false) }
  }
  const locked = disabled || saving
  const toggle = <button type="button" onClick={() => { setOpen(!open); rememberOpen(!open) }} aria-expanded={open} className="inline-flex items-center gap-1 text-[11px] font-semibold text-[#c3aeff] hover:underline">
    {open ? <EyeOff size={12} aria-hidden /> : <Eye size={12} aria-hidden />}{open ? 'Hide pose' : 'Show pose'}<ChevronDown size={12} className={`transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden /></button>
  if (!person.character_context_id) return null  // the analysis only measures characters the user named
  return <div className="mt-2 rounded-lg border border-[#2c3342] bg-[#171c25] px-2.5 py-2 text-[11px]" aria-label={`Pose of box ${person.character_name ?? ''}`}>
    <div className="flex items-center justify-between gap-2"><span className="font-semibold uppercase tracking-[.08em] text-[#929aa9]">Pose from the first frame</span>{toggle}</div>
    {open && <div className="mt-2 space-y-2">
      {source === undefined && !error && <p className="muted">Loading…</p>}
      {source === null && <p className="muted">The first frame has not been analysed yet.</p>}
      {source && !summary && <p className="muted">{index < 0 ? 'This character is not in the saved first-frame analysis. Run the first-frame analysis again after naming the box.' : 'No pose was measured for this character. Run the first-frame analysis again to get one.'}</p>}
      {summary && pose && <>
        <dl className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-center gap-x-2 gap-y-1.5">
          <dt className="muted">State</dt>
          <dd><select aria-label="Body state" className={field} value={pose.body_pose ?? 'unclear'} disabled={locked} onChange={event => void save({ body_pose: event.target.value })}>{BODY_POSES.map(value => <option key={value} value={value}>{label(value)}</option>)}</select></dd>
          <dt className="muted">Torso</dt>
          <dd className="flex flex-wrap items-center gap-1.5"><Angle name="Torso direction" value={pose.body_yaw} limit={180} disabled={locked} onSave={value => void save({ body_yaw: value })} /><span>{viewWords(viewOf(pose.body_yaw), pose.body_yaw)}</span></dd>
          <dt className="muted">Head</dt>
          <dd className="flex flex-wrap items-center gap-1.5"><Angle name="Head direction" value={pose.head_yaw} limit={180} disabled={locked} onSave={value => void save({ head_yaw: value })} /><span>{viewWords(viewOf(pose.head_yaw), pose.head_yaw)}</span></dd>
          <dt className="muted">Head tilt</dt>
          <dd className="flex items-center gap-1.5"><Angle name="Head tilt" value={pose.head_pitch} limit={90} disabled={locked} onSave={value => void save({ head_pitch: value })} /><span className="muted">negative looks down</span></dd>
          <dt className="muted">Eyes</dt>
          <dd><select aria-label="Eye direction" className={field} value={pose.eye_gaze ?? 'unclear'} disabled={locked} onChange={event => void save({ eye_gaze: event.target.value })}>{EYE_GAZES.map(value => <option key={value} value={value}>{label(value)}</option>)}</select></dd>
          <dt className="muted">Looking at</dt>
          <dd className="flex flex-wrap items-center gap-1.5"><select aria-label="Looking at" className={field} value={pose.gaze_target ?? 'unclear'} disabled={locked} onChange={event => void save({ gaze_target: event.target.value })}>
            {peers.map(name => <option key={name} value={name}>{name}</option>)}{TARGET_KINDS.map(value => <option key={value} value={value}>{label(value)}</option>)}</select>
            {pose.gaze_target === 'object' && <ObjectName value={pose.gaze_object ?? ''} disabled={locked} onSave={value => void save({ gaze_object: value })} />}</dd>
        </dl>
        {pose.edited && <StatusChip tone="purple" title="You corrected this block; it is saved as your revision of the first-frame result.">Edited by you</StatusChip>}
        {summary.flags.map(text => <p key={text} className="flex items-start gap-1.5 rounded-md border border-[#6b5a2f] bg-[#2a2314] px-2 py-1 text-[#f0d9a5]"><TriangleAlert size={12} className="mt-0.5 shrink-0" aria-hidden />{text}</p>)}
      </>}
      {error && <p role="alert" className="text-[#ff9fa7]">{error}</p>}
    </div>}
  </div>
}

function Angle({ name, value, limit, disabled, onSave }: { name: string; value: number | null | undefined; limit: number; disabled: boolean; onSave: (value: number | null) => void }) {
  const [text, setText] = useState(value === null || value === undefined ? '' : String(value))
  useEffect(() => setText(value === null || value === undefined ? '' : String(value)), [value])
  const commit = () => {
    const next = text.trim() === '' ? null : snapAngle(text, limit)
    if (next !== (value ?? null)) onSave(next)
    else setText(value === null || value === undefined ? '' : String(value))
  }
  return <input aria-label={name} className={`${field} w-16 text-right`} inputMode="numeric" value={text} disabled={disabled} placeholder="—" title="Degrees, in steps of 15. Empty when it cannot be seen." onChange={event => setText(event.target.value)} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} />
}

function ObjectName({ value, disabled, onSave }: { value: string; disabled: boolean; onSave: (value: string) => void }) {
  const [text, setText] = useState(value)
  useEffect(() => setText(value), [value])
  return <input aria-label="Object looked at" className={`${field} w-28`} value={text} maxLength={60} disabled={disabled} placeholder="which object" onChange={event => setText(event.target.value)} onBlur={() => { if (text !== value) onSave(text) }} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} />
}
