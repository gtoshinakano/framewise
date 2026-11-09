import { createContext, memo, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, ChevronsDownUp, ChevronsUpDown, ListTree, Search, WrapText } from 'lucide-react'
import { CopyButton, Markdown } from './Markdown'
import { MAX_RENDER_CHARS, isLongString, looksLikeMarkdown, nodeOf, parseValue, promptOutline, startsCollapsed, stringify } from './dataView'

type Command = { id: number; open: boolean } | null
const TreeCommand = createContext<{ command: Command; query: string }>({ command: null, query: '' })

const C = { key: 'text-[#c3aeff]', string: 'text-[#7bdcb2]', number: 'text-[#e6c58b]', bool: 'text-[#7fb4ff]', nul: 'text-[#8b93a3]', punct: 'text-[#8b93a3]' }

function Highlight({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>
  const lower = text.toLowerCase(); const needle = query.toLowerCase()
  const parts: ReactNode[] = []
  let at = 0
  for (let hit = lower.indexOf(needle); hit !== -1; hit = lower.indexOf(needle, at)) {
    if (hit > at) parts.push(text.slice(at, hit))
    parts.push(<mark key={hit} className="rounded-sm bg-[#e6c58b] px-0.5 text-[#1b1f27]">{text.slice(hit, hit + needle.length)}</mark>)
    at = hit + needle.length
  }
  parts.push(text.slice(at))
  return <>{parts}</>
}

function countMatches(text: string, query: string) {
  if (!query) return 0
  const lower = text.toLowerCase(); const needle = query.toLowerCase()
  let count = 0
  for (let hit = lower.indexOf(needle); hit !== -1; hit = lower.indexOf(needle, hit + needle.length)) count++
  return count
}

/** Makes "Label:" openings of sentences stand out in long run-on model text. Display only; Raw shows the exact text. */
const LABEL_AT_SENTENCE_START = /(^|[.!?]\s+|\n)([A-Z][A-Za-z/-]*(?: [A-Za-z/-]+){0,2}):(?=\s)/g
function Labelled({ text }: { text: string }) {
  const parts: ReactNode[] = []
  let at = 0
  for (const match of text.matchAll(LABEL_AT_SENTENCE_START)) {
    const start = (match.index ?? 0) + match[1].length
    if (start > at) parts.push(text.slice(at, start))
    parts.push(<strong key={start} className="font-semibold text-[#c3aeff]">{match[2]}:</strong>)
    at = start + match[2].length + 1
  }
  parts.push(text.slice(at))
  return <>{parts}</>
}

const CLIP = 6000

/** A long string inside JSON gets its own wrapped block; very long ones are clipped until asked for. */
function StringValue({ value, query }: { value: string; query: string }) {
  const [all, setAll] = useState(false)
  const clipped = !all && value.length > CLIP
  if (!isLongString(value)) return <span className={`${C.string} break-words`}>"<Highlight text={value} query={query} />"</span>
  return <div className="mt-1 rounded-md border-l-2 border-[#2f6b55] bg-[#0f141b] px-3 py-2">
    <div className={`whitespace-pre-wrap break-words ${C.string}`}><Highlight text={clipped ? value.slice(0, CLIP) : value} query={query} /></div>
    {value.length > CLIP && <button type="button" className="mt-1 text-[11px] text-[#c3aeff] hover:underline" onClick={() => setAll(!all)}>{all ? 'Show less' : `Show all ${value.length.toLocaleString()} characters`}</button>}
  </div>
}

function Scalar({ value, query }: { value: unknown; query: string }) {
  if (value === null || value === undefined) return <span className={C.nul}>null</span>
  if (typeof value === 'number') return <span className={C.number}>{String(value)}</span>
  if (typeof value === 'boolean') return <span className={C.bool}>{String(value)}</span>
  return <StringValue value={String(value)} query={query} />
}

const JsonNode = memo(function JsonNode({ name, value, depth, last }: { name?: string; value: unknown; depth: number; last: boolean }) {
  const { command, query } = useContext(TreeCommand)
  const info = nodeOf(value)
  const container = info.kind === 'object' || info.kind === 'array'
  const [open, setOpen] = useState(() => depth === 0 || !startsCollapsed(value, depth))
  useEffect(() => { if (command && container) setOpen(command.open) }, [command, container])
  const comma = last ? '' : ','
  const label = name !== undefined && <><span className={C.key}><Highlight text={name} query={query} /></span><span className={C.punct}>: </span></>
  if (!container) {
    return <div className="py-px pl-4">{label}<Scalar value={value} query={query} /><span className={C.punct}>{comma}</span></div>
  }
  const array = info.kind === 'array'
  const entries: [string | undefined, unknown][] = array ? (value as unknown[]).map(item => [undefined, item]) : Object.entries(value as object)
  const [openBrace, closeBrace] = array ? ['[', ']'] : ['{', '}']
  return <div className="py-px">
    <button type="button" className="group flex w-full items-start gap-0.5 text-left hover:bg-[#1c2230]" onClick={() => setOpen(!open)} aria-expanded={open}>
      <span className="mt-0.5 shrink-0 text-[#8b93a3]">{open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}</span>
      <span>{label}<span className={C.punct}>{openBrace}</span>{!open && <span className="text-[#8b93a3]"> {entries.length} {array ? (entries.length === 1 ? 'item' : 'items') : (entries.length === 1 ? 'key' : 'keys')} <span className={C.punct}>{closeBrace}{comma}</span></span>}</span>
    </button>
    {open && <>
      <div className="ml-[6px] border-l border-[#2a3140] pl-3">{entries.length === 0 ? <span className="muted pl-4 text-[11px]">empty</span> : entries.map(([key, item], index) => <JsonNode key={index} name={key} value={item} depth={depth + 1} last={index === entries.length - 1} />)}</div>
      <div className={`pl-4 ${C.punct}`}>{closeBrace}{comma}</div>
    </>}
  </div>
})

const TAG = /^\s*<(Picture|Audio|Video|d|shot|scene)(?:\s+\d+)?[^>]*>/i

function PromptLines({ text, query, wrap, lineRefs }: { text: string; query: string; wrap: boolean; lineRefs: React.MutableRefObject<Map<number, HTMLElement>> }) {
  const outline = useMemo(() => new Map(promptOutline(text).map(item => [item.line, item.level])), [text])
  return <div className={`font-mono text-xs leading-5 ${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'}`}>{text.split('\n').map((line, index) => {
    const level = outline.get(index)
    const tone = level === 1 ? 'font-semibold text-[#c3aeff]' : level === 2 ? 'font-semibold text-[#9ec5ff]' : TAG.test(line) ? 'text-[#e6c58b]' : 'text-[#d5dbe5]'
    return <div key={index} ref={element => { if (element) lineRefs.current.set(index, element); else lineRefs.current.delete(index) }} className="flex gap-3 hover:bg-[#161c27]"><span className="w-8 shrink-0 select-none text-right text-[#566074]">{index + 1}</span><span className={`min-w-0 flex-1 ${tone}`}><Highlight text={line || ' '} query={query} /></span></div>
  })}</div>
}

type Variant = 'auto' | 'prompt' | 'error'

/**
 * Shows a saved value the way it reads best: JSON as a collapsible coloured tree, Markdown as formatted text,
 * prompts with line numbers and a section outline. "Raw" always shows the exact saved text.
 */
export function DataView({ value, variant = 'auto', maxHeight = 'max-h-[28rem]', empty = 'Nothing saved' }: { value: unknown; variant?: Variant; maxHeight?: string; empty?: string }) {
  const parsed = useMemo(() => parseValue(value), [value])
  const raw = useMemo(() => parsed.kind === 'json' ? stringify(parsed.value) : parsed.text, [parsed])
  const markdown = parsed.kind === 'text' && variant === 'auto' && looksLikeMarkdown(raw)
  const prompt = parsed.kind === 'text' && variant === 'prompt'
  const [mode, setMode] = useState<'formatted' | 'raw'>('formatted')
  const [wrap, setWrap] = useState(true)
  const [query, setQuery] = useState('')
  const [command, setCommand] = useState<Command>(null)
  const [showAll, setShowAll] = useState(false)
  const lineRefs = useRef(new Map<number, HTMLElement>())
  const body = useRef<HTMLDivElement>(null)
  const outline = useMemo(() => prompt ? promptOutline(raw) : [], [prompt, raw])
  const matches = useMemo(() => countMatches(raw, query.trim()), [raw, query])
  const search = query.trim()
  useEffect(() => { if (search && parsed.kind === 'json') setCommand(previous => ({ id: (previous?.id ?? 0) + 1, open: true })) }, [search, parsed.kind])
  if (!raw.trim()) return <p className="muted rounded-md border border-dashed hairline px-3 py-2 text-xs">{empty}</p>
  const big = raw.length > MAX_RENDER_CHARS && !showAll
  const shown = big ? raw.slice(0, MAX_RENDER_CHARS) : raw
  const lines = raw.split('\n').length
  const formatted = mode === 'formatted'
  const tone = variant === 'error' ? 'border-[#63363e] bg-[#1d1215] text-[#ffb6bc]' : 'hairline bg-[#10141b] text-[#d5dbe5]'
  const jump = (line: number) => lineRefs.current.get(line)?.scrollIntoView({ block: 'start', behavior: 'smooth' })
  return <div className={`overflow-hidden rounded-lg border ${variant === 'error' ? 'border-[#63363e]' : 'hairline'}`}>
    <div className="flex flex-wrap items-center gap-1.5 border-b hairline bg-[#171c25] px-2 py-1.5 text-[11px]">
      <div className="flex rounded-md border hairline p-0.5" role="group" aria-label="View as">
        {(['formatted', 'raw'] as const).map(item => <button key={item} type="button" aria-pressed={mode === item} onClick={() => setMode(item)} className={`rounded px-2 py-0.5 font-semibold ${mode === item ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{item === 'formatted' ? (parsed.kind === 'json' ? 'Tree' : markdown ? 'Formatted' : 'Text') : 'Raw'}</button>)}
      </div>
      {parsed.kind === 'json' && formatted && <>
        <button type="button" className="inline-flex items-center gap-1 rounded px-1.5 py-1 text-[#aeb5c1] hover:bg-[#303746] hover:text-white" onClick={() => setCommand({ id: (command?.id ?? 0) + 1, open: true })} title="Expand all"><ChevronsUpDown size={13} />Expand</button>
        <button type="button" className="inline-flex items-center gap-1 rounded px-1.5 py-1 text-[#aeb5c1] hover:bg-[#303746] hover:text-white" onClick={() => setCommand({ id: (command?.id ?? 0) + 1, open: false })} title="Collapse all"><ChevronsDownUp size={13} />Collapse</button>
      </>}
      {(!formatted || prompt || parsed.kind === 'text') && <button type="button" aria-pressed={wrap} onClick={() => setWrap(!wrap)} className={`inline-flex items-center gap-1 rounded px-1.5 py-1 ${wrap ? 'bg-[#272e3a] text-white' : 'text-[#aeb5c1] hover:text-white'}`} title="Wrap long lines"><WrapText size={13} />Wrap</button>}
      <label className="relative ml-auto flex items-center"><Search size={12} className="pointer-events-none absolute left-2 text-[#8b93a3]" /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find" aria-label="Find in this block" className="w-32 rounded-md border hairline bg-[#10141b] py-1 pl-6 pr-2 text-[11px] text-white focus:border-[#9c80f8]" /></label>
      {search && <span className={matches ? 'text-[#e6c58b]' : 'text-[#ff9fa7]'} aria-live="polite">{matches ? `${matches} match${matches === 1 ? '' : 'es'}` : 'No match'}</span>}
      <span className="muted tabular-nums">{lines.toLocaleString()} lines · {raw.length.toLocaleString()} chars</span>
      <CopyButton text={raw} />
    </div>
    {outline.length > 2 && formatted && <div className="flex items-center gap-1.5 overflow-x-auto border-b hairline bg-[#141922] px-2 py-1.5 text-[11px]"><ListTree size={13} className="shrink-0 text-[#8b93a3]" aria-hidden />{outline.filter(item => item.level < 3).slice(0, 30).map(item => <button key={item.line} type="button" onClick={() => jump(item.line)} className={`shrink-0 rounded border px-1.5 py-0.5 hover:text-white ${item.level === 1 ? 'border-[#4a4363] text-[#c3aeff]' : 'border-[#34435f] text-[#9ec5ff]'}`}>{item.label}</button>)}</div>}
    <div ref={body} className={`scrollbar overflow-auto px-3 py-2 ${maxHeight} ${tone}`}>
      {!formatted
        ? <pre className={`font-mono text-xs leading-5 ${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'}`}><Highlight text={shown} query={search} /></pre>
        : parsed.kind === 'text' && !markdown && !prompt
        ? <pre className={`font-sans text-[13px] leading-6 ${variant === 'error' ? '' : 'text-[#dfe4ee]'} ${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'}`}>{search ? <Highlight text={shown} query={search} /> : <Labelled text={shown} />}</pre>
        : parsed.kind === 'json'
          ? <div className="font-mono text-xs leading-5"><TreeCommand.Provider value={{ command, query: search }}><JsonNode value={parsed.value} depth={0} last /></TreeCommand.Provider></div>
          : prompt ? <PromptLines text={shown} query={search} wrap={wrap} lineRefs={lineRefs} />
            : search ? <pre className={`font-sans text-[13px] leading-6 ${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'}`}><Highlight text={shown} query={search} /></pre>
              : <Markdown text={shown} />}
      {big && <button type="button" className="mt-2 text-[11px] text-[#c3aeff] hover:underline" onClick={() => setShowAll(true)}>Showing the first {MAX_RENDER_CHARS.toLocaleString()} of {raw.length.toLocaleString()} characters. Show all</button>}
    </div>
  </div>
}
