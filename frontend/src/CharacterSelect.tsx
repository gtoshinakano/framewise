import { Children, isValidElement, useEffect, useId, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, UserRound } from 'lucide-react'
import { api } from './api'
import { CharacterOptions } from './CharacterOptions'
import type { Cut, Episode, Project } from './api'
import type { TargetDetections } from './targetView'

type Option = { value: string; label: string; disabled?: boolean; group?: string }
type Props = { children: ReactNode; value: string; onChange: (event: ChangeEvent<HTMLSelectElement>) => void; projectId?: string; imageIds?: Record<string, string>; disabled?: boolean; required?: boolean; className?: string; 'aria-label'?: string }
type Matcher = { id: string; character_context_id: string; state: string; cut_id: string; position: string }
type Evidence = { pictures: Record<string, string>; counts: Record<string, number> }
type DetectionRun = { detections: { position: string; character_context_id: string | null }[] } | null
const CACHE_DURATION_MS = 6 * 60 * 60 * 1000
const imageCache = new Map<string, { at: number; promise: Promise<Evidence> }>()
let activePicker: string | null = null
const emptyEvidence: Evidence = { pictures: {}, counts: {} }
const pictureCache = new Map<string, Record<string, string>>()
const pictureListeners = new Map<string, Set<(pictures: Record<string, string>) => void>>()
async function loadEvidence(projectId: string): Promise<Evidence> {
  const [matcher, references, project] = await Promise.allSettled([
    api<Matcher[]>(`/projects/${projectId}/character-matcher`),
    api<{ id: string; character_context_id: string }[]>(`/projects/${projectId}/character-references`),
    api<Project>(`/projects/${projectId}`),
  ])
  const pictures: Record<string, string> = {}
  if (references.status === 'fulfilled') for (const row of references.value) if (!pictures[row.character_context_id]) pictures[row.character_context_id] = `/api/character-references/${row.id}/image`
  const seen = new Map<string, Set<string>>()
  const add = (character: string | null, frame: string) => {
    if (!character) return
    if (!seen.has(character)) seen.set(character, new Set())
    seen.get(character)!.add(frame)
  }
  if (matcher.status === 'fulfilled') for (const row of [...matcher.value].sort((a, b) => Number(b.state === 'active') - Number(a.state === 'active'))) {
    if (!pictures[row.character_context_id]) pictures[row.character_context_id] = `/api/character-matcher/${row.id}/image`
    add(row.character_context_id, `${row.cut_id}:${row.position}`)
  }
  pictureCache.set(projectId, pictures)
  pictureListeners.get(projectId)?.forEach(listener => listener(pictures))
  if (project.status === 'rejected') throw project.reason
  const episodes = await Promise.all(project.value.episodes.map(episode => api<Episode>(`/episodes/${episode.id}`)))
  const cuts: Cut[] = episodes.flatMap(episode => episode.cuts)
  const tasks: (() => Promise<void>)[] = []
  for (const cut of cuts) {
    if (cut.has_boxes) tasks.push(async () => {
      const run = await api<DetectionRun>(`/cuts/${cut.id}/character-detections`)
      for (const box of run?.detections ?? []) add(box.character_context_id, `${cut.id}:${box.position}`)
    })
    if (cut.has_target_boxes) tasks.push(async () => {
      const run = (await api<TargetDetections>(`/cuts/${cut.id}/target-detections`)).run
      if (!run) return
      const slots = new Map(run.slots.map(slot => [slot.slot, slot.character_context_id]))
      const frames = new Map(run.frames.map(frame => [frame.ordinal, frame.time]))
      for (const box of run.detections) {
        const time = frames.get(box.frame_ordinal)
        // The first target frame can be the same picture as the opening box.
        const key = time !== undefined && Math.abs(time - cut.first_frame_time) < 0.001 ? 'first' : `target:${time ?? box.frame_ordinal}`
        add(slots.get(box.slot) ?? null, `${cut.id}:${key}`)
      }
    })
  }
  let next = 0
  await Promise.all(Array.from({ length: Math.min(6, tasks.length) }, async () => {
    while (next < tasks.length) await tasks[next++]()
  }))
  return { pictures, counts: Object.fromEntries([...seen].map(([character, frames]) => [character, frames.size])) }
}
function images(projectId: string) {
  const cached = imageCache.get(projectId)
  if (cached && Date.now() - cached.at < CACHE_DURATION_MS) return cached.promise
  const promise = loadEvidence(projectId).then(data => { const entry = imageCache.get(projectId); if (entry?.promise === promise) entry.at = Date.now(); return data }).catch(error => { if (imageCache.get(projectId)?.promise === promise) imageCache.delete(projectId); throw error })
  imageCache.set(projectId, { at: Date.now(), promise })
  return promise
}
function optionsFrom(children: ReactNode, group?: string): Option[] {
  const result: Option[] = []
  Children.forEach(children, child => {
    if (!isValidElement<{ children?: ReactNode; value?: string; disabled?: boolean; label?: string; characters?: Parameters<typeof CharacterOptions>[0]['characters']; allowNew?: boolean }>(child)) return
    if (child.type === CharacterOptions && child.props.characters) result.push(...optionsFrom(CharacterOptions({ characters: child.props.characters, allowNew: child.props.allowNew }), group))
    else if (child.type === 'option') result.push({ value: String(child.props.value ?? ''), label: String(child.props.children ?? ''), disabled: child.props.disabled, group })
    else result.push(...optionsFrom(child.props.children, child.type === 'optgroup' ? child.props.label : group))
  })
  return result
}
function Avatar({ src, name }: { src?: string; name: string }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [src])
  return <span className="grid h-12 w-12 shrink-0 place-items-center overflow-hidden rounded-md bg-[#303746] text-[#aeb5c1]">{src && !failed ? <img src={src} alt="" loading="lazy" className="h-full w-full object-cover" onError={() => setFailed(true)} /> : <UserRound size={22} aria-label={name ? undefined : 'No image'} />}</span>
}
export function CharacterSelect({ children, value, onChange, projectId, imageIds, disabled, required, className = '', 'aria-label': ariaLabel }: Props) {
  const [open, setOpen] = useState(false)
  const [activeValue, setActiveValue] = useState(value)
  const [evidence, setEvidence] = useState<Evidence>(emptyEvidence)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const rawOptions = useMemo(() => optionsFrom(children), [children])
  const list = useMemo(() => {
    const isCharacter = (item: Option) => !!item.value && !item.value.startsWith('__') && (!imageIds || Object.hasOwn(imageIds, item.value))
    const characters = rawOptions.filter(isCharacter).map(item => ({ ...item, group: undefined }))
    characters.sort((a, b) => (evidence.counts[imageIds?.[b.value] ?? b.value] ?? 0) - (evidence.counts[imageIds?.[a.value] ?? a.value] ?? 0) || a.label.localeCompare(b.label, undefined, { sensitivity: 'base', numeric: true }))
    const before = rawOptions.filter(item => !isCharacter(item) && !item.value)
    const after = rawOptions.filter(item => !isCharacter(item) && !!item.value)
    return [...before, ...characters, ...after]
  }, [rawOptions, evidence, imageIds])
  const characterIndices = useMemo(() => list.flatMap((item, index) => item.value && !item.disabled && !item.value.startsWith('__') && (!imageIds || Object.hasOwn(imageIds, item.value)) ? [index] : []), [list, imageIds])
  const selected = list.find(item => item.value === value)
  const active = Math.max(0, list.findIndex(item => item.value === activeValue))
  const [position, setPosition] = useState({ top: 0, bottom: undefined as number | undefined, left: 0, width: 200, maxHeight: 240 })
  const trigger = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const id = useId()
  const enabled = list.map((item, index) => item.disabled ? -1 : index).filter(index => index >= 0)
  const show = () => { if (disabled) return; activePicker = id; clearTimeout(timer.current); if (!open) setActiveValue(value); setOpen(true) }
  const closeLater = () => { timer.current = setTimeout(() => { if (!trigger.current?.matches(':focus-visible')) setOpen(false) }, 180) }
  useEffect(() => () => clearTimeout(timer.current), [])
  useEffect(() => { if (disabled) setOpen(false) }, [disabled])
  useEffect(() => {
    if (!open) return
    let alive = true
    const receivePictures = (pictures: Record<string, string>) => { if (alive) setEvidence(current => ({ ...current, pictures })) }
    if (projectId) {
      const listeners = pictureListeners.get(projectId) ?? new Set()
      listeners.add(receivePictures); pictureListeners.set(projectId, listeners)
      receivePictures(pictureCache.get(projectId) ?? {})
      setFailed(false); setLoading(true)
      void images(projectId).then(data => { if (alive) setEvidence(data) }).catch(() => { if (alive) { setFailed(true); setEvidence(current => ({ ...current, counts: {} })) } }).finally(() => { if (alive) setLoading(false) })
    }
    else { setEvidence(emptyEvidence); setLoading(false); setFailed(false) }
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect(); if (!rect) return
      const below = window.innerHeight - rect.bottom, above = rect.top
      const height = Math.min(280, Math.max(90, (below >= 200 || below >= above ? below : above) - 8))
      const opensBelow = below >= 200 || below >= above
      setPosition({ top: opensBelow ? rect.bottom : 0, bottom: opensBelow ? undefined : window.innerHeight - rect.top, left: Math.max(4, Math.min(rect.left, window.innerWidth - Math.max(220, rect.width) - 4)), width: Math.max(220, rect.width), maxHeight: height })
    }
    place(); window.addEventListener('resize', place); window.addEventListener('scroll', place, true)
    const outside = (event: PointerEvent) => { if (!trigger.current?.contains(event.target as Node) && !panel.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => { alive = false; if (projectId) { const listeners = pictureListeners.get(projectId); listeners?.delete(receivePictures); if (!listeners?.size) pictureListeners.delete(projectId) }; window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); document.removeEventListener('pointerdown', outside) }
  }, [open, projectId])
  const choose = (index: number) => { const item = list[index]; if (!item || item.disabled) return; onChange({ target: { value: item.value } } as ChangeEvent<HTMLSelectElement>); setOpen(false); trigger.current?.focus() }
  useEffect(() => {
    if (!open || disabled) return
    const shortcut = (event: KeyboardEvent) => {
      if (activePicker !== id || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.repeat || !/^[1-9]$/.test(event.key)) return
      const target = event.target as HTMLElement | null
      if (target !== document.body && target !== trigger.current) return
      const index = characterIndices[Number(event.key) - 1]
      if (index === undefined) return
      event.preventDefault(); event.stopPropagation(); event.stopImmediatePropagation(); choose(index)
    }
    document.addEventListener('keydown', shortcut, true)
    return () => document.removeEventListener('keydown', shortcut, true)
  }, [open, disabled, characterIndices, list, onChange, id])
  const move = (direction: number) => { const pos = enabled.indexOf(active); const next = enabled[(pos + direction + enabled.length) % enabled.length]; if (next !== undefined) setActiveValue(list[next].value) }
  useEffect(() => { if (open) panel.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }) }, [open, active])
  const root = trigger.current?.closest('dialog') || document.body
  const layout = className.split(' ').filter(part => /^(?:!?w-|min-w-|max-w-|flex-1|shrink-0|mt-)/.test(part)).join(' ' )
  return <span className={`relative inline-block min-w-0 ${layout || 'w-full'}`} onMouseEnter={show} onMouseLeave={closeLater}>
    {required && <input tabIndex={-1} aria-hidden="true" required value={value} onChange={() => {}} onInvalid={event => { event.preventDefault(); trigger.current?.focus(); show() }} className="absolute h-px w-px opacity-0" />}
    <button ref={trigger} type="button" disabled={disabled} aria-label={ariaLabel} role="combobox" aria-expanded={open} aria-controls={id} aria-activedescendant={open ? `${id}-${active}` : undefined} className={`field flex w-full min-w-0 items-center gap-1.5 text-left disabled:opacity-50 ${className.replace(/(?:!?w-|min-w-|max-w-|flex-1|shrink-0|mt-)\S*/g, '')}`} onClick={() => { if (open) setOpen(false); else { setActiveValue(value); show() } }} onKeyDown={event => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); if (!open) { setActiveValue(value); show() } else move(event.key === 'ArrowDown' ? 1 : -1) }
      else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (open) choose(active); else show() }
      else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false) }
      else if (event.key === 'Tab') setOpen(false)
    }}><span className="min-w-0 flex-1 truncate">{selected?.label ?? value}</span><ChevronDown size={14} className="shrink-0" /></button>
    {open && createPortal(<div ref={panel} id={id} role="listbox" aria-label={ariaLabel} onMouseEnter={() => clearTimeout(timer.current)} onMouseLeave={closeLater} style={{ position: 'fixed', top: position.bottom === undefined ? position.top : undefined, bottom: position.bottom, left: position.left, width: position.width, maxHeight: position.maxHeight, zIndex: 1000 }} className="overflow-y-auto rounded-lg border border-[#454b5b] bg-[#1b202a] p-1 shadow-xl">
      {loading && <div className="px-2 py-1 text-[10px] text-[#aeb5c1]">Counting character frames…</div>}
      {failed && <div role="status" className="px-2 py-1 text-[10px] text-[#f0c979]">Frame counts unavailable; sorted by name.</div>}
      {list.map((item, index) => <div key={`${item.value}-${index}`}>{item.group && (index === 0 || list[index - 1].group !== item.group) && <div className="px-2 pt-2 text-[10px] text-[#aeb5c1]">{item.group}</div>}<div id={`${id}-${index}`} role="option" aria-selected={index === active} aria-disabled={item.disabled || undefined} onMouseEnter={() => setActiveValue(item.value)} onMouseDown={event => event.preventDefault()} onClick={() => choose(index)} className={`flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-xs ${item.disabled ? 'opacity-40' : index === active ? 'bg-[#514576] text-white' : 'text-[#e0e2e8] hover:bg-[#303540]'}`}><Avatar src={evidence.pictures[imageIds?.[item.value] ?? item.value]} name={item.label} /><span className="min-w-0 flex-1 break-words">{item.label}</span>{characterIndices.indexOf(index) >= 0 && characterIndices.indexOf(index) < 9 && <kbd className="rounded border border-[#454b5b] px-1.5 py-0.5 text-[10px] text-[#aeb5c1]" aria-label={`Shortcut ${characterIndices.indexOf(index) + 1}`}>{characterIndices.indexOf(index) + 1}</kbd>}{item.value === value && <span aria-hidden>✓</span>}</div></div>)}
    </div>, root)}
  </span>
}
