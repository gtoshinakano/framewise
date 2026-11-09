import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronRight, FileText, Info, ListChecks, Settings2, X } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { DataView } from './DataView'
import { durationText, runFacts } from './dataView'
import { CopyButton } from './Markdown'
import { JobChip, Skeleton, StatusChip, formatElapsed, resultAccent, type Tone } from './RunUi'
import type { StageRow } from './runState'

type RunDetails = {
  pipeline?: string; status: string; error: string | null; input: unknown; results: AnalysisResult[]
  video_windows: { window_index: number; start_time: number; end_time: number; content: unknown }[]
  rejected_responses: { window_index: number; start_time: number; end_time: number; error: string; response: string }[]
}

type Tab = 'overview' | 'input' | 'outputs' | 'prompts' | 'windows' | 'errors'
type ResultView = 'output' | 'prompt' | 'context' | 'settings'

const pipelineName = (job: Job) => job.pipeline === 'regen' ? 'Cut summary' : job.pipeline === 'consistency_summary' ? 'Consistency summary' : 'Visual analysis'
const stageTone = (status: string): Tone => status === 'done' || status === 'complete' ? 'green' : status === 'running' ? 'purple' : status === 'failed' || status === 'interrupted' ? 'red' : 'grey'
const statusWord = (status: string) => status === 'done' ? 'Succeeded' : status === 'failed' ? 'Failed' : status === 'running' ? 'Running' : status === 'queued' ? 'Queued' : status
const hasData = (value: unknown) => value !== null && value !== undefined && !(typeof value === 'object' && Object.keys(value as object).length === 0) && value !== ''

function ResultCard({ result, job }: { result: AnalysisResult; job: Job }) {
  const accent = resultAccent(result.settings, result.title)
  const views = ([['output', 'Output', true], ['prompt', 'Saved prompt', !!result.prompt], ['context', 'Input context', hasData(result.context)], ['settings', 'Model settings', hasData(result.settings)]] as const).filter(item => item[2])
  const [view, setView] = useState<ResultView>('output')
  const [open, setOpen] = useState(true)
  const frame = ['initial_frame', 'last_frame'].includes(String(result.settings.structured_type))
  return <article className="overflow-hidden rounded-xl border hairline bg-[#171c25]" style={{ borderLeft: `4px solid ${accent.color}` }}>
    <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full flex-wrap items-center gap-2 px-3 py-2.5 text-left hover:bg-[#1c2230]">
      {open ? <ChevronDown size={15} className="text-[#8b93a3]" /> : <ChevronRight size={15} className="text-[#8b93a3]" />}
      <span className="text-sm font-semibold" style={{ color: accent.color }}>{result.title || accent.label}</span>
      {result.model_id && <StatusChip tone="blue">{result.model_id}</StatusChip>}
      <StatusChip tone="grey">{result.origin}</StatusChip>
      {result.job_id !== job.id && <StatusChip tone="amber" title="This result was produced by an earlier run and reused as a checkpoint">Reused from an earlier run</StatusChip>}
      <span className="muted ml-auto text-[11px]">{result.created_at}</span>
    </button>
    {open && <div className="space-y-3 border-t hairline p-3">
      <div className="flex flex-wrap gap-1" role="tablist" aria-label={`${result.title} sections`}>{views.map(([key, label]) => <button key={key} type="button" role="tab" aria-selected={view === key} onClick={() => setView(key)} className={`rounded-md px-2.5 py-1 text-[11px] font-semibold ${view === key ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:bg-[#232a36] hover:text-white'}`}>{label}</button>)}</div>
      {view === 'output' && <div className={frame ? 'grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,16rem)]' : ''}>
        <DataView value={result.content} maxHeight="max-h-[calc(90vh-22rem)]" />
        {frame && <a href={`/api/analysis-results/${result.id}/frame`} target="_blank" rel="noreferrer" className="block self-start overflow-hidden rounded-lg border hairline bg-black" title="Open the analyzed frame in a new tab"><img src={`/api/analysis-results/${result.id}/frame`} alt={`${result.title} analyzed frame`} loading="lazy" decoding="async" className="w-full" /></a>}
      </div>}
      {view === 'prompt' && <DataView value={result.prompt} variant="prompt" maxHeight="max-h-[calc(90vh-23rem)]" />}
      {view === 'context' && <DataView value={result.context} maxHeight="max-h-[calc(90vh-23rem)]" />}
      {view === 'settings' && <DataView value={result.settings} maxHeight="max-h-[calc(90vh-23rem)]" />}
    </div>}
  </article>
}

function Windows({ windows }: { windows: RunDetails['video_windows'] }) {
  const [openRows, setOpenRows] = useState<Set<number>>(() => new Set(windows.length === 1 ? [windows[0].window_index] : []))
  const from = Math.min(...windows.map(item => item.start_time)); const to = Math.max(...windows.map(item => item.end_time))
  const span = Math.max(0.001, to - from)
  return <div className="overflow-hidden rounded-xl border hairline"><table className="w-full text-left text-xs">
    <thead className="bg-[#171c25] text-[10px] uppercase tracking-wider text-[#929aa9]"><tr><th className="w-12 px-3 py-2 font-semibold">#</th><th className="px-3 py-2 font-semibold">Time range</th><th className="hidden w-1/3 px-3 py-2 font-semibold sm:table-cell">Position in cut</th></tr></thead>
    <tbody>{windows.map(item => {
      const open = openRows.has(item.window_index)
      const toggle = () => setOpenRows(previous => { const next = new Set(previous); if (next.has(item.window_index)) next.delete(item.window_index); else next.add(item.window_index); return next })
      return [<tr key={item.window_index} onClick={toggle} className="cursor-pointer border-t hairline hover:bg-[#1c2230]">
        <td className="px-3 py-2 font-mono text-[#4cc9c0]"><span className="inline-flex items-center gap-1">{open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{item.window_index + 1}</span></td>
        <td className="px-3 py-2 font-mono tabular-nums"><button type="button" aria-expanded={open} onClick={event => { event.stopPropagation(); toggle() }} className="text-left hover:underline">{item.start_time.toFixed(3)}s → {item.end_time.toFixed(3)}s</button> <span className="muted">({(item.end_time - item.start_time).toFixed(2)}s)</span></td>
        <td className="hidden px-3 py-2 sm:table-cell"><div className="relative h-2 rounded-full bg-[#232a36]"><div className="absolute top-0 h-2 rounded-full bg-[#4cc9c0]" style={{ left: `${(item.start_time - from) / span * 100}%`, width: `${Math.max(1.5, (item.end_time - item.start_time) / span * 100)}%` }} /></div></td>
      </tr>, open && <tr key={`${item.window_index}-content`} className="border-t hairline bg-[#141922]"><td colSpan={3} className="p-3"><DataView value={item.content} maxHeight="max-h-[24rem]" /></td></tr>]
    })}</tbody>
  </table></div>
}

export function RunDetailsDialog({ job, cutId, stages = [], onClose }: { job: Job; cutId: string; stages?: StageRow[]; onClose: () => void }) {
  const [data, setData] = useState<RunDetails | null>(null)
  const [error, setError] = useState('')
  const [tab, setTab] = useState<Tab | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    let active = true
    api<RunDetails>(`/jobs/${job.id}/cuts/${cutId}/run-details`).then(value => { if (active) setData(value) }).catch(reason => { if (active) setError((reason as Error).message) })
    return () => { active = false; opener?.focus() }
  }, [job.id, cutId])
  const facts = useMemo(() => data ? runFacts([data.input, ...data.results.map(result => result.settings)]) : [], [data])
  const prompts = useMemo(() => data?.results.filter(result => result.prompt) ?? [], [data])
  const errorCount = data ? (data.error ? 1 : 0) + data.rejected_responses.length : 0
  const tabs: [Tab, string, number | null, boolean][] = data ? [
    ['overview', 'Overview', null, true], ['input', 'Input', null, hasData(data.input)], ['outputs', 'Outputs', data.results.length, data.results.length > 0],
    ['prompts', 'Prompts', prompts.length, prompts.length > 0], ['windows', 'Windows', data.video_windows.length, data.video_windows.length > 0], ['errors', 'Errors', errorCount, errorCount > 0],
  ] : []
  const current: Tab = tab ?? (errorCount > 0 ? 'errors' : data && data.results.length ? 'outputs' : 'overview')
  const duration = durationText(job.started_at, job.finished_at)
  const stageList = stages.length ? stages : []
  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="run-details-title" className="flex h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="shrink-0 border-b hairline bg-[#1b202a] px-5 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Analysis run</p>
            <h2 id="run-details-title" className="mt-0.5 flex flex-wrap items-center gap-2 text-base font-semibold">{pipelineName(job)} <JobChip status={statusWord(data?.status ?? job.status)} /></h2>
            <p className="muted mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
              <span className="inline-flex items-center gap-1 font-mono" title={job.id}>{job.id.slice(0, 8)}<CopyButton text={job.id} label="Copy id" /></span>
              <span>Queued {job.created_at} UTC</span>{job.started_at && <span>Started {job.started_at}</span>}{job.finished_at && <span>Finished {job.finished_at}</span>}{duration && <span className="font-semibold text-[#d5dbe5]">Took {duration}</span>}
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close run details" className="rounded-md p-2 text-[#aeb5c1] hover:bg-[#303746] hover:text-white"><X size={18} /></button>
        </div>
        {stageList.length > 0 && <ol className="mt-2 flex flex-wrap items-center gap-1.5" aria-label="Stages">{stageList.map((stage, index) => <li key={stage.id} className="flex items-center gap-1.5">{index > 0 && <span className="text-[#566074]" aria-hidden>→</span>}<StatusChip tone={stageTone(stage.status)} title={stage.error || undefined}>{stage.label} · {stage.status} · {formatElapsed(stage.elapsed_seconds)}</StatusChip></li>)}</ol>}
        {data && <div className="mt-3 flex gap-1 overflow-x-auto" role="tablist" aria-label="Run details sections">{tabs.filter(item => item[3]).map(([key, label, count]) => <button key={key} role="tab" aria-selected={current === key} onClick={() => setTab(key)} className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold ${current === key ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:bg-[#232a36] hover:text-white'}`}>{label}{count !== null && <span className={`rounded-full px-1.5 text-[10px] ${key === 'errors' ? 'bg-[#63363e] text-[#ffb6bc]' : 'bg-[#2a3140] text-[#c8cfda]'}`}>{count}</span>}</button>)}</div>}
      </header>
      <div className="scrollbar min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {error && <p role="alert" className="rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-sm text-[#ffb6bc]">{error}</p>}
        {!data && !error && <div className="space-y-3" aria-busy="true" aria-label="Loading run details"><Skeleton className="h-8 w-1/3" /><Skeleton className="h-40" /><Skeleton className="h-40" /></div>}
        {data && current === 'overview' && <div className="space-y-4">
          <section className="rounded-xl border hairline bg-[#171c25] p-4">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold"><ListChecks size={15} className="text-[#7bdcb2]" /> What this run used</h3>
            {facts.length ? <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">{facts.map(fact => <div key={fact.label} className={`min-w-0 ${fact.label === 'Additional context' ? 'sm:col-span-2 lg:col-span-3' : ''}`}><dt className="text-[10px] font-semibold uppercase tracking-wider text-[#929aa9]">{fact.label}</dt><dd className="mt-0.5 break-words text-sm text-[#e8ebf1]">{fact.value}</dd></div>)}</dl> : <p className="muted text-xs">No settings were recorded for this run.</p>}
          </section>
          <section className="grid gap-3 sm:grid-cols-3">
            {([['Outputs', data.results.length, 'outputs', 'green'], ['Video windows', data.video_windows.length, 'windows', 'blue'], ['Errors', errorCount, 'errors', 'red']] as const).map(([label, count, target, tone]) => <button key={label} type="button" disabled={!count} onClick={() => setTab(target)} className="rounded-xl border hairline bg-[#171c25] p-4 text-left enabled:hover:border-[#4a4363] disabled:opacity-60"><span className="text-[10px] font-semibold uppercase tracking-wider text-[#929aa9]">{label}</span><span className="mt-1 flex items-center gap-2 text-2xl font-semibold">{count}{count > 0 && <StatusChip tone={tone}>{target === 'errors' ? 'needs attention' : 'view'}</StatusChip>}</span></button>)}
          </section>
          <p className="flex items-start gap-1.5 text-[11px] leading-5 text-[#929aa9]"><Info size={13} className="mt-1 shrink-0" aria-hidden />Inputs show the saved queue snapshot and each result's prompt and context. Some model requests, temporary video clips and raw successful responses are not saved, so these records may not contain every byte sent to a model.</p>
        </div>}
        {data && current === 'input' && <section><h3 className="mb-2 flex items-center gap-2 text-sm font-semibold"><Settings2 size={15} className="text-[#8cbcff]" /> Queued input and settings</h3><DataView value={data.input} maxHeight="max-h-[calc(90vh-16rem)]" /></section>}
        {data && current === 'outputs' && <section className="space-y-3">{data.results.map(result => <ResultCard key={result.id} result={result} job={job} />)}</section>}
        {data && current === 'prompts' && <section className="space-y-4">{prompts.map(result => { const accent = resultAccent(result.settings, result.title); return <div key={result.id}><h3 className="mb-2 flex items-center gap-2 text-sm font-semibold" style={{ color: accent.color }}><FileText size={15} />{result.title || accent.label}</h3><DataView value={result.prompt} variant="prompt" maxHeight="max-h-[calc(90vh-17rem)]" /></div> })}</section>}
        {data && current === 'windows' && data.video_windows.length > 0 && <Windows windows={data.video_windows} />}
        {data && current === 'errors' && <section className="space-y-4">
          {data.error && <div><h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-[#ff9fa7]"><AlertTriangle size={15} /> Run error</h3><DataView value={data.error} variant="error" maxHeight="max-h-[40vh]" /></div>}
          {data.rejected_responses.map((item, index) => <div key={index} className="rounded-xl border border-[#63363e] bg-[#1d1215] p-3"><h3 className="flex flex-wrap items-center gap-2 text-sm font-semibold text-[#ff9fa7]"><AlertTriangle size={15} /> Rejected video response <span className="font-mono text-xs font-normal">{item.start_time.toFixed(3)}s → {item.end_time.toFixed(3)}s</span></h3><p className="my-2 whitespace-pre-wrap break-words text-xs text-[#ffb6bc]">{item.error}</p><DataView value={item.response} maxHeight="max-h-[34vh]" /></div>)}
        </section>}
      </div>
    </div>
  </div>
}
