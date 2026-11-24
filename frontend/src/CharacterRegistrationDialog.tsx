import { useEffect, useRef, useState } from 'react'
import { Loader2, Plus, Save, Search, Trash2, X } from 'lucide-react'
import { api, jsonBody, type ContextItem, type EpisodeSummary, type Job, type Project } from './api'
import { IconButton } from './EditorUi'
import { CharacterRegistrationFrame, cropFrame, useFrameUrl, type CapturedFrame, type RegistrationFrame } from './CharacterRegistrationFrame'
import { RegistrationVoice, type VoiceDraft, type VoiceSample } from './RegistrationVoice'
import { useCharacters, loadCharacters, rememberCharacter } from './runCache'
import type { Box } from './targetView'
import { WebSearchSources } from './WebSearchSources'

type Glossary = { id?: string; part: 'family' | 'given'; spelling: string; reading: string }
type Outfit = { id?: string; name: string; description: string; summary: string }
type Entry = {
  character: ContextItem; version: string; glossary: Glossary[]; outfits: Outfit[]; voice: VoiceSample[]; thumbnail: boolean
}
export type RegistrationResult = Pick<Entry, 'character' | 'version'> & { voice_job_ids: string[]; profile_job_id: string | null }
type Draft = { title: string; content: string; episode_id: string | null; is_main: boolean }
type Proof = { source: number; quote: string }
type Sourced<T> = T & { evidence: Proof[] }
type TextSuggestion = Sourced<{ value: string }>
type Hit = { id: number; title: string; url: string; excerpt: string }
type Research = {
  hits: Hit[]; warnings: { source: string; message: string }[]; configured: boolean
  suggestions: { japanese_name: TextSuggestion | null; appearance: TextSuggestion | null; names: Sourced<Glossary>[]
    outfits: Sourced<Omit<Outfit, 'summary'>>[]; candidates: { label: string; source: number }[] }
}

const base = (projectId: string) => `/projects/${projectId}/character-registration`
const blank = { title: '', content: '', episode_id: null, is_main: false }

function uuid() {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 15) | 64
  bytes[8] = (bytes[8] & 63) | 128
  const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
  return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-')
}

type Props = {
  projectId: string; episodeId?: string; projectTitle?: string; episodes?: EpisodeSummary[]
  characterId?: string; chooseCharacter?: boolean; frame?: CapturedFrame; original?: RegistrationFrame; time?: number
  saveFromBox?: (body: FormData) => Promise<RegistrationResult>
  onSaved?: (result: RegistrationResult) => void; onDeleted?: () => void; onClose: () => void
}

export function CharacterRegistrationDialog(props: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const characters = useCharacters(props.projectId)
  const [selected, setSelected] = useState(props.characterId || '')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    const panel = dialog.current
    panel?.showModal()
    return () => panel?.close()
  }, [])

  return <dialog ref={dialog} aria-modal="true" aria-labelledby="registration-heading" onKeyDown={event => { if (event.key === 'Escape') event.stopPropagation() }} onCancel={event => { event.preventDefault(); if (!busy) props.onClose() }}
    className="m-auto max-h-[92vh] w-[min(95vw,65rem)] overflow-auto rounded-xl border hairline bg-[#1b202a] text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <header className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b hairline bg-[#1b202a] px-5 py-3">
      <h2 className="text-base font-semibold" id="registration-heading">{props.chooseCharacter ? 'Edit character' : props.characterId ? 'Character details' : 'Register character'}</h2>
      <IconButton icon={X} size="sm" label="Close character registration" onClick={props.onClose} disabled={busy} />
    </header>
    {props.chooseCharacter && !selected ? <div className="p-5">
      <label className="block text-sm">Character<select aria-label="Character" className="field mt-1" value={selected} onChange={event => setSelected(event.target.value)}>
        <option value="">Choose a character…</option>
        {characters.filter(item => !item.unregistered).map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
      </select></label>
    </div> : <RegistrationForm key={selected} {...props} characterId={selected || undefined} setBusy={setBusy} />}
  </dialog>
}

function RegistrationForm({ projectId, projectTitle, episodes, episodeId, characterId, frame, original, time, saveFromBox, onSaved, onDeleted, onClose, setBusy }: Props & { setBusy: (busy: boolean) => void }) {
  const frameUrl = useFrameUrl(frame)
  const researchRequest = useRef<AbortController | null>(null)
  const [ids] = useState(() => ({ character: characterId || uuid(), save: uuid() }))
  const [project, setProject] = useState<{ title: string; episodes: EpisodeSummary[] } | null>(projectTitle && episodes ? { title: projectTitle, episodes } : null)
  const [series, setSeries] = useState(projectTitle || '')
  const [entry, setEntry] = useState<Entry | null>(null)
  const [draft, setDraft] = useState<Draft>(blank)
  const [glossary, setGlossary] = useState<Glossary[]>([])
  const [outfits, setOutfits] = useState<Outfit[]>([])
  const [voice, setVoice] = useState<VoiceDraft>({})
  const [removed, setRemoved] = useState<string[]>([])
  const [box, setBox] = useState<Box | null>(original?.box || null)
  const [tab, setTab] = useState<'details' | 'voice'>('details')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState<RegistrationResult | null>(null)
  const [researching, setResearching] = useState(false)
  const [researchError, setResearchError] = useState('')
  const [research, setResearch] = useState<Research | null>(null)
  const [applied, setApplied] = useState<string[]>([])
  const [sourceLinks, setSourceLinks] = useState<Record<string, string>>({})
  const ready = (!characterId || !!entry) && (!frame || !!box)
  const voiceTooShort = Object.values(voice).some(ranges => ranges.length && ranges.reduce((total, range) => total + range.end_time - range.start_time, 0) < 2)
  const saveAllowed = ready && !!draft.title.trim() && !voiceTooShort && !saving
  const image: RegistrationFrame | undefined = frame && frameUrl ? { url: frameUrl, size: [frame.canvas.width, frame.canvas.height] } : original
  const thumbnail = entry?.thumbnail ? `/api${base(projectId)}/${entry.character.id}/image` : null

  useEffect(() => {
    const request = new AbortController()
    if (!project) api<Project>(`/projects/${projectId}`, { signal: request.signal }).then(value => {
      setProject(value); setSeries(current => current || value.title)
    }).catch(reason => { if (!request.signal.aborted) setError(reason.message) })
    if (characterId) api<Entry>(`${base(projectId)}/${characterId}`, { signal: request.signal }).then(value => {
      setEntry(value)
      setDraft({ title: value.character.title, content: value.character.content, episode_id: value.character.episode_id, is_main: !!value.character.is_main })
      setGlossary(value.glossary); setOutfits(value.outfits)
    }).catch(reason => { if (!request.signal.aborted) setError(reason.message) })
    return () => { request.abort(); researchRequest.current?.abort() }
    // A different character remounts this form.
  }, [projectId, characterId])

  function resetResearch() {
    researchRequest.current?.abort()
    setResearching(false); setResearch(null); setApplied([]); setResearchError('')
  }

  async function autofill(hit?: Hit, candidate?: string) {
    resetResearch()
    const request = new AbortController()
    researchRequest.current = request
    setResearching(true)
    try {
      const result = await api<Research>(`/projects/${projectId}/character-research`, {
        ...jsonBody({ name: draft.title, series, source_url: hit?.url, candidate }), signal: request.signal,
      })
      if (!request.signal.aborted) setResearch(result)
    } catch (reason) { if (!request.signal.aborted) setResearchError((reason as Error).message) }
    finally { if (!request.signal.aborted) setResearching(false) }
  }

  function apply(key: string, evidence: Proof[], update: () => void) {
    update()
    setApplied(list => [...list, key])
    setSourceLinks(existing => {
      const next = { ...existing }
      for (const citation of evidence) {
        const hit = research?.hits.find(item => item.id === citation.source)
        if (hit) next[hit.url] = hit.title
      }
      return next
    })
  }

  function appendDetails(title: string, value: string) {
    setDraft(current => ({ ...current, content: [current.content.trimEnd(), `## ${title}\n${value}`].filter(Boolean).join('\n\n') }))
  }

  async function save(describe = false) {
    if (!saveAllowed) return
    setSaving(true); setBusy(true); setError('')
    try {
      const links = Object.keys(sourceLinks).filter(url => !draft.content.includes(url))
      const content = links.length ? [draft.content.trimEnd(), '## Sources\n' + links.map(url => `- <${url}>`).join('\n')].filter(Boolean).join('\n\n') : draft.content
      const body = new FormData()
      body.set('draft', JSON.stringify({ ...draft, content, character_id: ids.character, save_id: ids.save, version: entry?.version,
        describe, glossary, outfits, voice: Object.entries(voice).filter(([, ranges]) => ranges.length).map(([episode_id, ranges]) => ({ episode_id, ranges })), remove_voice: removed }))
      if (frame && box) body.set('thumbnail', await cropFrame(frame, box), 'character.jpg')
      const result = saveFromBox ? await saveFromBox(body) : await api<RegistrationResult>(base(projectId), { method: 'POST', body })
      rememberCharacter(result.character)
      void loadCharacters(projectId, true).catch(() => {})
      if (result.voice_job_ids.length || result.profile_job_id) setSaved(result)
      onSaved?.(result)
      if (!result.voice_job_ids.length && !result.profile_job_id) onClose()
    } catch (reason) { setError((reason as Error).message) }
    finally { setSaving(false); setBusy(false) }
  }

  async function deleteCharacter() {
    if (!characterId || !onDeleted || !window.confirm(`Delete “${draft.title}” and its references?`)) return
    setSaving(true); setBusy(true); setError('')
    try {
      await api(`/context/${characterId}`, { method: 'DELETE' })
      void loadCharacters(projectId, true).catch(() => {})
      onDeleted()
    } catch (reason) { setError((reason as Error).message) }
    finally { setSaving(false); setBusy(false) }
  }

  if (saved) return <SavedRegistration saved={saved} projectId={projectId} onClose={onClose} />

  const citationLinks = (evidence: Proof[]) => <span className="flex flex-wrap gap-x-2 text-[11px]">
    {evidence.map((proof, i) => {
      const hit = research?.hits.find(item => item.id === proof.source)
      return hit && <a key={i} href={hit.url} title={proof.quote} target="_blank" rel="noreferrer" className="text-[#cbb9ff] underline">{hit.title}</a>
    })}
  </span>

  const suggestions = research?.suggestions
  const cards: { key: string; title: string; text: string; evidence: Proof[]; action: string; conflicts: boolean; present?: boolean; use: () => void }[] = []
  for (const key of ['japanese_name', 'appearance'] as const) {
    const item = suggestions?.[key]
    if (!item) continue
    const title = key === 'japanese_name' ? 'Japanese name' : 'Appearance'
    cards.push({ key, title, text: item.value, evidence: item.evidence, action: 'Use suggestion',
      conflicts: draft.content.includes(`## ${title}\n`),
      present: draft.content.includes(item.value), use: () => appendDetails(title, item.value) })
  }
  suggestions?.names.forEach((item, i) => {
    cards.push({ key: `name-${i}`, title: item.part === 'given' ? 'Given name' : 'Family name',
      text: `${item.spelling} · ${item.reading}`, evidence: item.evidence, action: 'Use suggestion', conflicts: false,
      present: glossary.some(row => row.part === item.part && row.spelling === item.spelling && row.reading === item.reading),
      use: () => setGlossary(rows => [...rows, { part: item.part, spelling: item.spelling, reading: item.reading }]) })
  })
  suggestions?.outfits.forEach((item, i) => {
    const matches = (row: Outfit) => row.name.trim().toLowerCase() === item.name.trim().toLowerCase()
    const existing = outfits.find(matches)
    cards.push({ key: `outfit-${i}`, title: item.name, text: item.description, evidence: item.evidence,
      action: existing ? 'Replace description' : 'Use suggestion', conflicts: !!existing, present: existing?.description === item.description,
      use: () => setOutfits(rows => rows.some(matches) ? rows.map(row => matches(row) ? { ...row, description: item.description, summary: '' } : row) :
        [...rows, { name: item.name, description: item.description, summary: '' }]) })
  })
  const newCards = cards.filter(item => !item.conflicts && !item.present && !applied.includes(item.key))
  const anySuggestion = !!(cards.length || suggestions?.candidates.length)
  return <>
    <div className={`grid gap-5 p-5 ${image || thumbnail ? 'md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]' : ''}`}>
      {(image || thumbnail) && <aside className="space-y-2">
        {image ? <CharacterRegistrationFrame frame={frame} episodeId={episodeId} image={image} box={box} onBox={setBox} />
          : <img className="max-h-80 rounded-lg bg-black object-contain" src={thumbnail!} alt={`Saved crop of ${entry?.character.title}`} />}
      </aside>}
      <div className="min-w-0 space-y-4">
        {error && <p role="alert" className="text-sm text-[#ff9fa7]">{error}</p>}
        {characterId && !entry && !error && <p role="status" className="muted text-xs">Loading character details…</p>}
        {frame && !box && <p role="status" className="muted text-sm">Mark the character in the frame to begin.</p>}
        {ready && <>
          <div className="flex gap-2 border-b hairline pb-2" aria-label="Registration sections">
            <button type="button" aria-pressed={tab === 'details'} className={`btn ${tab === 'details' ? 'btn-primary' : 'btn-quiet'} !py-1.5`} onClick={() => setTab('details')}>Details</button>
            <button type="button" aria-pressed={tab === 'voice'} className={`btn ${tab === 'voice' ? 'btn-primary' : 'btn-quiet'} !py-1.5`} onClick={() => setTab('voice')}>Voice samples</button>
          </div>
          {tab === 'voice' ? <RegistrationVoice episodes={project?.episodes || []} initialEpisode={episodeId || draft.episode_id || undefined}
            time={frame?.time ?? time} draft={voice} onDraft={setVoice} samples={entry?.voice || []} removed={removed} onRemove={setRemoved} /> : <>
            <label className="block"><span className="label">Character name</span><input className="field" maxLength={200} value={draft.title}
              onChange={event => { resetResearch(); setDraft({ ...draft, title: event.target.value }) }} /></label>
            <div className="rounded-lg border hairline p-3">
              <label className="block"><span className="label">Series title for search</span><input className="field" maxLength={200} value={series}
                onChange={event => { resetResearch(); setSeries(event.target.value) }} /></label>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button type="button" className="btn btn-quiet !py-2" disabled={!draft.title.trim() || researching} onClick={() => void autofill()}>
                  {researching ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />} {researching ? 'Searching and checking sources…' : 'Autofill from web'}
                </button>
                <WebSearchSources />
              </div>
              {researchError && <p role="alert" className="mt-2 text-xs text-[#ff9fa7]">{researchError}</p>}
              {research && <div aria-live="polite" className="mt-2 space-y-2 text-xs">
                {!anySuggestion && <p className="muted">{!research.configured ? 'Add a web search source to use autofill.' : 'No supported details found. You can enter them below.'}</p>}
                {research.warnings.length > 0 && <details className="muted"><summary>Some sources were unavailable</summary>
                  <ul>{research.warnings.map(item => <li key={item.source}>{item.source} · {item.message}</li>)}</ul></details>}
                {suggestions && suggestions.candidates.length > 0 && <div className="space-y-1">
                  <p>Choose the matching character.</p>
                  {suggestions.candidates.map((candidate, i) => {
                    const hit = research.hits.find(item => item.id === candidate.source)
                    return hit && <div className="flex items-center gap-2" key={i}>
                      <button className="btn btn-quiet !py-1" type="button" onClick={() => void autofill(hit, candidate.label)}>{candidate.label}</button>
                      <a className="text-[#cbb9ff] underline" href={hit.url} target="_blank" rel="noreferrer">Source</a>
                    </div>
                  })}
                </div>}
                {newCards.length > 1 && <button className="btn btn-primary !py-1.5" type="button"
                  onClick={() => newCards.forEach(item => apply(item.key, item.evidence, item.use))}>Use all new details</button>}
                {cards.map(item => <SuggestionCard key={item.key} title={item.title} text={item.text} sources={citationLinks(item.evidence)}
                  applied={item.present || applied.includes(item.key)} action={item.action}
                  onUse={() => apply(item.key, item.evidence, item.use)} />)}

              </div>}
            </div>
            <label className="block"><span className="label">Details</span><textarea aria-label="Details" className="field min-h-28 resize-y" maxLength={50000} value={draft.content}
              onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Japanese name, appearance, and character notes" /></label>
            <div className="flex flex-wrap items-end gap-3">
              <label className="block min-w-40 flex-1"><span className="label">Scope</span><select className="field" value={draft.episode_id || ''} onChange={event => setDraft({ ...draft, episode_id: event.target.value || null })}>
                <option value="">Entire project</option>
                {(project?.episodes || []).filter(item => !saveFromBox || item.id === episodeId).map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
              </select></label>
              <label className="flex items-center gap-2 py-2 text-sm"><input type="checkbox" checked={draft.is_main}
                onChange={event => setDraft({ ...draft, is_main: event.target.checked })} /> Main character</label>
            </div>
            <section className="space-y-2">
              <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold">Name glossary</h3>
                <IconButton icon={Plus} label="Add name entry" size="sm" text="Add entry" onClick={() => setGlossary([...glossary, { part: 'family', spelling: '', reading: '' }])} /></div>
              {glossary.map((item, i) => <div key={i} className="flex flex-wrap items-end gap-2">
                <label className="text-xs">Part<select className="field mt-1 !w-auto" value={item.part} onChange={event => setGlossary(glossary.map((row, j) => i === j ? { ...row, part: event.target.value as Glossary['part'] } : row))}>
                  <option value="family">Family</option><option value="given">Given</option></select></label>
                {(['spelling', 'reading'] as const).map(key => <label className="min-w-20 flex-1 text-xs" key={key}>{key === 'spelling' ? 'Spelling' : 'Kana reading'}
                  <input className="field mt-1" maxLength={120} value={item[key]} onChange={event => setGlossary(glossary.map((row, j) => i === j ? { ...row, [key]: event.target.value } : row))} /></label>)}
                <IconButton icon={Trash2} label={`Remove name entry ${i + 1}`} size="sm" onClick={() => setGlossary(glossary.filter((_, j) => i !== j))} />
              </div>)}
            </section>
            <section className="space-y-2">
              <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold">Outfits</h3>
                <IconButton icon={Plus} label="Add outfit" size="sm" text="Add outfit" onClick={() => setOutfits([...outfits, { name: '', description: '', summary: '' }])} /></div>
              {outfits.map((item, i) => <div key={i} className="space-y-2 rounded-lg border hairline p-2">
                <div className="flex items-end gap-2">
                  <label className="flex-1 text-xs">Outfit name<input className="field mt-1" maxLength={120} value={item.name}
                    onChange={event => setOutfits(outfits.map((row, j) => i === j ? { ...row, name: event.target.value } : row))} /></label>
                  <IconButton icon={Trash2} label={`Remove outfit ${i + 1}`} size="sm" onClick={() => setOutfits(outfits.filter((_, j) => i !== j))} />
                </div>
                <label className="block text-xs">Description<textarea className="field mt-1 resize-y" maxLength={6000} value={item.description}
                  onChange={event => setOutfits(outfits.map((row, j) => i === j ? { ...row, description: event.target.value } : row))} /></label>
                <label className="block text-xs">Summary<input className="field mt-1" maxLength={400} value={item.summary}
                  onChange={event => setOutfits(outfits.map((row, j) => i === j ? { ...row, summary: event.target.value } : row))} /></label>
              </div>)}
            </section>
          </>}
        </>}
      </div>
    </div>
    <footer className="sticky bottom-0 flex flex-wrap items-center justify-end gap-2 border-t hairline bg-[#1b202a] px-5 py-3">
      {entry && onDeleted && <IconButton icon={Trash2} tone="danger" label="Delete character" text="Delete character" className="mr-auto" disabled={saving} onClick={() => void deleteCharacter()} />}
      {voiceTooShort && <p className="mr-auto text-xs text-[#f5c678]">Each episode sample needs at least 2 seconds of audio.</p>}
      <button type="button" className="btn btn-quiet" disabled={saving} onClick={onClose}>Cancel</button>
      {saveFromBox && <button type="button" className="btn btn-quiet" disabled={!saveAllowed} onClick={() => void save(true)}>Save and describe from box</button>}
      <button type="button" className="btn btn-primary" disabled={!saveAllowed} onClick={() => void save()}><Save size={15} /> {saving ? 'Saving…' : 'Save character'}</button>
    </footer>
  </>
}

function SuggestionCard({ title, text, sources, applied, onUse, action = 'Use suggestion' }: {
  title: string; text: string; sources: React.ReactNode; applied: boolean; action?: string; onUse: () => void
}) {
  return <div className="rounded border hairline p-2">
    <div className="flex items-start justify-between gap-2"><strong>{title}</strong>
      <button type="button" className="text-[#cbb9ff] disabled:text-[#a5acb9]" disabled={applied} onClick={onUse}>{applied ? 'In draft' : action}</button></div>
    <p className="mb-1 whitespace-pre-wrap">{text}</p>{sources}
  </div>
}

function SavedRegistration({ saved, projectId, onClose }: { saved: RegistrationResult; projectId: string; onClose: () => void }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [samples, setSamples] = useState<VoiceSample[]>([])
  useEffect(() => {
    const controller = new AbortController()
    const ids = [...saved.voice_job_ids, ...(saved.profile_job_id ? [saved.profile_job_id] : [])]
    let timer: number
    async function poll() {
      try {
        const values = await Promise.all(ids.map(id => api<Job>(`/v1/jobs/${id}`, { signal: controller.signal })))
        setJobs(values)
        if (values.some(job => job.status === 'queued' || job.status === 'running')) timer = window.setTimeout(poll, 1400)
        else if (values.some(job => job.pipeline === 'voice_sample' && job.status === 'done')) {
          const result = await api<Entry>(`${base(projectId)}/${saved.character.id}`, { signal: controller.signal })
          setSamples(result.voice.filter(item => saved.voice_job_ids.includes(item.id) && item.preview_available))
        }
      } catch {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, 3000)
      }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [saved, projectId])
  return <div className="space-y-4 p-5">
    <p role="status" className="text-sm text-[#7bdcb2]">Saved {saved.character.title}.</p>
    {!jobs.length && <p className="muted text-sm">Checking queued work…</p>}
    {jobs.map(job => <div key={job.id} className="rounded border hairline p-2 text-sm">
      <strong>{job.pipeline === 'voice_sample' ? 'Voice sample' : 'Frame description'}</strong> · {job.status === 'done' ? 'Ready' : job.stage || job.status}
      {job.error && <p className="text-xs text-[#ff9fa7]">{job.error}</p>}
      {samples.some(sample => sample.id === job.id) && <audio className="mt-2 w-full" controls preload="none" aria-label="New voice sample" src={`/api/voice-samples/${job.id}/audio`} />}
    </div>)}
    <div className="flex justify-end"><button type="button" className="btn btn-primary" onClick={onClose}>Done</button></div>
  </div>
}
