import { CharacterSelect } from './CharacterSelect'
import { useEffect, useRef, useState } from 'react'
import { Plus, Trash2, X } from 'lucide-react'
import { api, type ContextItem } from './api'
import { IconButton, SegmentedControl, Switch } from './EditorUi'
import {
  addPair, MODE_OPTIONS, orderCharacters, RELEVANCE_OPTIONS, relevanceOf, removePair, sameProfile, statText, TOPIC_HELP, UNSURE_OPTIONS,
  withRelevance, type AutopilotProfile, type ProfileResponse, type WhenUnsure,
} from './autopilotView'

const heading = 'mb-1 mt-5 text-sm font-semibold'
const intro = 'muted mb-2 text-xs leading-5'

/** What the user tells the autopilot once per project: what it may apply, how to settle review questions, which characters matter. */
export function AutopilotProfileDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [data, setData] = useState<ProfileResponse | null>(null)
  const [draft, setDraft] = useState<AutopilotProfile | null>(null)
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [pair, setPair] = useState<[string, string]>(['', ''])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const url = `/projects/${projectId}/autopilot-profile`
  useEffect(() => {
    dialog.current?.showModal()
    const controller = new AbortController()
    Promise.all([api<ProfileResponse>(url, { signal: controller.signal }), api<ContextItem[]>(`/projects/${projectId}/context`, { signal: controller.signal })])
      .then(([response, context]) => {
        setData(response)
        setDraft(response.profile)
        setCharacters(orderCharacters(context.filter(item => item.kind === 'character' && !item.unregistered)))
      })
      .catch(err => { if (!controller.signal.aborted) setError((err as Error).message) })
    return () => { controller.abort(); dialog.current?.close() }
  }, [projectId, url])
  const dirty = !!draft && !sameProfile(draft, data?.profile ?? null)
  function change(next: AutopilotProfile) { setDraft(next); setSaved(false) }
  async function save() {
    if (!draft) return
    setBusy(true)
    setError('')
    try {
      const response = await api<ProfileResponse>(url, { method: 'PUT', body: JSON.stringify({ profile: draft }) })
      setData(response)
      setDraft(response.profile)
      setSaved(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  function close() {
    if (busy) return
    if (dirty && !window.confirm('Close without saving your changes to the autopilot profile?')) return
    onClose()
  }
  const name = (id: string) => characters.find(item => item.id === id)?.title ?? 'Removed character'
  return <dialog ref={dialog} aria-modal="true" aria-labelledby="autopilot-title"
    onCancel={event => { event.preventDefault(); close() }}
    className="m-auto max-h-[90vh] w-[min(96vw,52rem)] overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <div className="mb-1 flex items-center justify-between gap-2">
      <h2 id="autopilot-title" className="font-semibold">Autopilot profile</h2>
      <IconButton icon={X} label="Close the autopilot profile" size="sm" disabled={busy} onClick={close} />
    </div>
    <p className="muted text-xs leading-5">Shared by every episode of this project. Agents read it before they decide anything for you; everything they apply is logged and can be undone. Leave anything you have no opinion on as it is.</p>
    {error && <p role="alert" className="mt-3 text-xs text-[#ff9fa7]">{error}</p>}
    {!draft || !data ? <p className="muted mt-4 text-sm">{error ? '' : 'Loading the profile…'}</p> : <form onSubmit={event => { event.preventDefault(); void save() }}>

      <h3 className={heading}>What agents may do</h3>
      <p className={intro}><strong>Apply</strong>: the agent decides and you can undo it. <strong>Ask me</strong>: it puts its proposal in your inbox. <strong>Off</strong>: it leaves that work to you.</p>
      <ul className="divide-y divide-[#333945] rounded-lg border hairline">{Object.entries(data.decisions).map(([key, label]) =>
        <li key={key} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-xs">
          <span>{label}</span>
          <SegmentedControl label={label} value={draft.modes[key] ?? 'apply'} options={MODE_OPTIONS}
            onChange={mode => change({ ...draft, modes: { ...draft.modes, [key]: mode } })} />
        </li>)}
      </ul>

      <h3 className={heading}>Claude as second reviewer</h3>
      <p className={intro}>Local models decide first. Only what they cannot settle goes to Claude, in your Claude Code session; this caps how much one run may send.</p>
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <Switch label="Ask Claude when local models are unsure" checked={draft.claude.enabled} onChange={enabled => change({ ...draft, claude: { ...draft.claude, enabled } })} />
        <label className="flex items-center gap-2">At most
          <input type="number" min={0} max={500} className="field w-20" value={draft.claude.max_items_per_run} disabled={!draft.claude.enabled}
            onChange={event => change({ ...draft, claude: { ...draft.claude, max_items_per_run: Math.max(0, Math.min(500, Number(event.target.value) || 0)) } })} />
          items per run</label>
      </div>

      <h3 className={heading}>Review questions</h3>
      <p className={intro}>The consistency review asks about what the analysis could not settle. Write a standing rule where you always answer the same way, and choose what the agent does when the evidence does not decide. The counts are this project's current reviews and your saved answers.</p>
      <div className="space-y-2">{data.topics.map(topic => {
        const help = TOPIC_HELP[topic] ?? { label: topic, covers: '', example: '' }
        const entry = draft.topics[topic] ?? { rule: '', when_unsure: 'evidence' as WhenUnsure }
        const setTopic = (next: Partial<typeof entry>) => change({ ...draft, topics: { ...draft.topics, [topic]: { ...entry, ...next } } })
        return <fieldset key={topic} className="rounded-lg border hairline px-3 py-2">
          <legend className="px-1 text-xs font-semibold">{help.label}</legend>
          <p className="muted text-[11px] leading-4">{help.covers}{help.covers ? ' · ' : ''}{statText(data.topic_stats[topic])}</p>
          <div className="mt-1.5 grid gap-2 md:grid-cols-[1fr_auto] md:items-start">
            <textarea className="field min-h-[2.5rem] text-xs" maxLength={600} rows={1} aria-label={`${help.label}: standing rule`}
              placeholder={help.example ? `e.g. ${help.example}` : 'A standing rule, if you have one'} value={entry.rule}
              onChange={event => setTopic({ rule: event.target.value })} />
            <label className="text-[11px]">When unsure
              <select className="field mt-0.5 text-xs" value={entry.when_unsure} onChange={event => setTopic({ when_unsure: event.target.value as WhenUnsure })}>
                {UNSURE_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select></label>
          </div>
        </fieldset>
      })}</div>

      <h3 className={heading}>Characters</h3>
      <p className={intro}><strong>Always matters</strong>: box them wherever they appear, even when only a small part is visible. <strong>Background</strong>: never box them. <strong>Auto</strong>: the agent decides from whether they speak, are looked at or touched, and keep appearing.</p>
      {!characters.length ? <p className="muted text-xs">This project has no characters yet.</p> :
        <ul className="max-h-72 divide-y divide-[#333945] overflow-y-auto rounded-lg border hairline">{characters.map(item =>
          <li key={item.id} className="flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 text-xs">
            <span>{item.title}{item.is_main ? <span className="muted"> · main</span> : null}</span>
            <SegmentedControl label={`${item.title}: relevance`} value={relevanceOf(draft, item.id)} options={RELEVANCE_OPTIONS}
              onChange={relevance => change(withRelevance(draft, item.id, relevance))} />
          </li>)}
        </ul>}

      <h3 className={heading}>Look-alikes</h3>
      <p className={intro}>Characters the agent must never tell apart on its own: a box that could be either one always comes to you.</p>
      {!!draft.look_alikes.length && <ul className="mb-2 space-y-1">{draft.look_alikes.map(([a, b], index) =>
        <li key={`${a}-${b}`} className="flex items-center gap-2 text-xs">
          <span>{name(a)} ↔ {name(b)}</span>
          <IconButton icon={Trash2} label={`Remove ${name(a)} and ${name(b)}`} size="sm" onClick={() => change({ ...draft, look_alikes: removePair(draft.look_alikes, index) })} />
        </li>)}
      </ul>}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {[0, 1].map(side => <CharacterSelect projectId={characters[0]?.project_id} key={side} className="field w-48 text-xs" aria-label={side ? 'Second character' : 'First character'} value={pair[side]}
          onChange={event => setPair(side ? [pair[0], event.target.value] : [event.target.value, pair[1]])}>
          <option value="">Choose a character</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
        </CharacterSelect>)}
        <IconButton icon={Plus} label="Add this pair" text="Add pair" size="sm" disabled={!pair[0] || !pair[1] || pair[0] === pair[1]}
          onClick={() => { change({ ...draft, look_alikes: addPair(draft.look_alikes, pair[0], pair[1]) }); setPair(['', '']) }} />
      </div>

      <h3 className={heading}>Notes for the agents</h3>
      <p className={intro}>Anything else they should know when they decide for this project: how the show marks flashbacks, recurring situations, what you always correct.</p>
      <textarea className="field min-h-[6rem] text-xs" maxLength={4000} value={draft.notes} placeholder="e.g. Flashbacks have soft, blurred edges. Phone calls: the voice on the phone is off-screen."
        onChange={event => change({ ...draft, notes: event.target.value })} />

      <div className="sticky bottom-0 -mx-5 -mb-5 mt-4 flex items-center justify-end gap-2 border-t hairline bg-[#1b202a] px-5 py-3">
        <span className="muted mr-auto text-xs" aria-live="polite">{saved && !dirty ? 'Saved.' : dirty ? 'Unsaved changes' : data.updated_at ? `Last saved ${data.updated_at} UTC` : 'Defaults: nothing saved yet'}</span>
        <button type="button" className="btn btn-quiet" disabled={busy} onClick={close}>Close</button>
        <button type="submit" className="btn btn-primary" disabled={busy || !dirty}>Save profile</button>
      </div>
    </form>}
  </dialog>
}
