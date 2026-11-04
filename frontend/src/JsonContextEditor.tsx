import { useState } from 'react'
import { ChevronDown, ChevronRight, Plus, Trash2 } from 'lucide-react'

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }
const defaultValue = (kind: string): Json => kind === 'object' ? {} : kind === 'array' ? [] : kind === 'number' ? 0 : kind === 'boolean' ? false : kind === 'null' ? null : ''

export function JsonContextEditor({ value, onChange }: { value: JsonObject; onChange: (value: JsonObject) => void }) {
  return <JsonNode value={value} onChange={next => onChange(next as JsonObject)} depth={0} />
}

function JsonNode({ value, onChange, depth }: { value: Json; onChange: (value: Json) => void; depth: number }) {
  const [open, setOpen] = useState(depth === 0)
  const [newKey, setNewKey] = useState('')
  const [newType, setNewType] = useState('string')
  const [error, setError] = useState('')
  const isArray = Array.isArray(value)
  const isObject = value !== null && typeof value === 'object' && !isArray
  if (!isArray && !isObject) {
    if (typeof value === 'string') return <textarea aria-label="Text value" className="field min-h-16 resize-y text-xs" value={value} onChange={event => onChange(event.target.value)} />
    if (typeof value === 'number') return <input aria-label="Number value" className="field text-xs" type="number" value={value} onChange={event => { if (event.target.value !== '' && Number.isFinite(Number(event.target.value))) onChange(Number(event.target.value)) }} />
    if (typeof value === 'boolean') return <label className="flex items-center gap-2"><input type="checkbox" checked={value} onChange={event => onChange(event.target.checked)} /> {String(value)}</label>
    return <span className="muted">null</span>
  }
  const entries: [string, Json][] = isArray ? (value as Json[]).map((item, index) => [String(index), item]) : Object.entries(value as JsonObject)
  function change(key: string, next: Json) {
    if (isArray) { const copy = [...value as Json[]]; copy[Number(key)] = next; onChange(copy) }
    else onChange({ ...value as JsonObject, [key]: next })
  }
  function remove(key: string) {
    if (isArray) onChange((value as Json[]).filter((_, index) => index !== Number(key)))
    else { const copy = { ...value as JsonObject }; delete copy[key]; onChange(copy) }
  }
  function rename(oldKey: string, newName: string) {
    if (newName === oldKey) return
    if (!newName.trim() || Object.hasOwn(value as JsonObject, newName)) { setError('Field names must be nonempty and unique.'); return }
    setError('')
    onChange(Object.fromEntries(entries.map(([key, item]) => [key === oldKey ? newName : key, item])))
  }
  function add() {
    if (isArray) onChange([...(value as Json[]), defaultValue(newType)])
    else {
      if (!newKey.trim() || Object.hasOwn(value as JsonObject, newKey)) { setError('Enter a unique field name.'); return }
      onChange({ ...value as JsonObject, [newKey]: defaultValue(newType) }); setNewKey('')
    }
    setError(''); setOpen(true)
  }
  return <div className={depth ? 'border-l border-[#414958] pl-3' : ''}>
    {depth > 0 && <button type="button" className="flex items-center gap-1 text-[#c3aeff] hover:underline" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{isArray ? 'List' : 'Fields'} ({entries.length})</button>}
    {open && <div className="space-y-2">{entries.map(([key, item]) => <div key={key} className="rounded border border-[#343b49] p-2">
      <div className="mb-1 flex items-center gap-2">{isArray ? <span className="min-w-0 flex-1 text-[#c8cfda]">Item {Number(key) + 1}</span> : <FieldName key={key} name={key} onRename={name => rename(key, name)} />}
        <select aria-label={`Value type for ${key}`} className="rounded border border-[#414958] bg-[#151a22] p-1 text-[11px]" value={item === null ? 'null' : Array.isArray(item) ? 'array' : typeof item} onChange={event => change(key, defaultValue(event.target.value))}>
          {['string', 'number', 'boolean', 'object', 'array', 'null'].map(type => <option key={type} value={type}>{type}</option>)}
        </select><button type="button" aria-label={`Remove ${key}`} className="text-[#ff9fa7]" onClick={() => remove(key)}><Trash2 size={14} /></button></div>
      <JsonNode value={item} onChange={next => change(key, next)} depth={depth + 1} />
    </div>)}<div className="flex flex-wrap gap-2">{!isArray && <input aria-label="New field name" className="field min-w-24 flex-1 text-xs" placeholder="New field name" value={newKey} onChange={event => setNewKey(event.target.value)} />}
      <select aria-label="New value type" className="rounded border border-[#414958] bg-[#151a22] p-1 text-xs" value={newType} onChange={event => setNewType(event.target.value)}>{['string', 'number', 'boolean', 'object', 'array', 'null'].map(type => <option key={type} value={type}>{type}</option>)}</select>
      <button type="button" className="btn btn-quiet" onClick={add}><Plus size={13} /> Add {isArray ? 'item' : 'field'}</button></div>{error && <p role="alert" className="text-[#ff9fa7]">{error}</p>}</div>}
  </div>
}

function FieldName({ name, onRename }: { name: string; onRename: (name: string) => void }) {
  const [draft, setDraft] = useState(name)
  return <input aria-label={`Field name ${name}`} className="min-w-0 flex-1 rounded border border-transparent bg-transparent p-1 text-xs font-semibold text-[#e8ebf1] hover:border-[#414958] focus:border-[#8e75ca]" value={draft} onChange={event => setDraft(event.target.value)} onBlur={() => { onRename(draft); setDraft(name) }} />
}
