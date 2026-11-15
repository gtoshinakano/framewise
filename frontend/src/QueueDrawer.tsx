import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Ban, CheckCircle2, ChevronDown, ChevronRight, Circle, Clock, Cpu, ExternalLink, History, ListChecks, Loader2, MessageSquare, MoreHorizontal, PlayCircle, RotateCcw, Trash2, X, XCircle } from 'lucide-react'
import { queueApi } from './queueApi'
import {
  STATUS_TEXT, STATUS_TONE, counterText, counterTitle, cutText, formatRunTime, localTime, mergeFirstPage, runSeconds, stageText, targetText, unitsText,
  type ChatRun, type HistoryFilter, type QueueState, type QueueSummary, type Run, type RunPage, type RunStatus,
} from './queueView'
import { ActionButton, StatusChip } from './RunUi'
import { Section, SegmentedControl } from './EditorUi'
import { Menu, type MenuEntry } from './Popover'
import { useToast } from './Toasts'

const POLL_MS = 2000
const STATUS_ICON: Record<RunStatus, ReactNode> = {
  queued: <Clock size={11} aria-hidden />, running: <Loader2 size={11} className="animate-spin motion-reduce:animate-none" aria-hidden />,
  stopping: <Loader2 size={11} className="animate-spin motion-reduce:animate-none" aria-hidden />, done: <CheckCircle2 size={11} aria-hidden />,
  failed: <XCircle size={11} aria-hidden />, cancelled: <Circle size={11} aria-hidden />,
}
const LINK = 'max-w-full truncate text-left text-[#c3aeff] hover:underline'

type Links = { onOpenCut: (cutId: string) => void; onOpenSequence: (sequenceId: string) => void }

/**
 * The episode's queue: what runs now, what waits, failures that need the user, and the run history,
 * all for this episode only (backend `runs.py`). GPU work of other episodes shows as one notice.
 */
export function QueueDrawer({ episodeId, focus, onClose, onOpenCut, onOpenSequence, onChanged, onSummary }: Links & {
  episodeId: string
  /** Scroll to this section when the drawer opens (the header's failure chip opens it at Needs attention). */
  focus?: 'attention' | null
  onClose: () => void
  onChanged: () => Promise<void> | void
  /** The counter the drawer just read, so the header updates at once. */
  onSummary: (summary: QueueSummary) => void
}) {
  const toast = useToast()
  const panel = useRef<HTMLDivElement>(null)
  const attentionRef = useRef<HTMLDivElement>(null)
  const summaryRef = useRef(onSummary)
  summaryRef.current = onSummary
  const [state, setState] = useState<QueueState | null>(null)
  const [loadError, setLoadError] = useState('')
  const [filter, setFilter] = useState<HistoryFilter>('all')
  const [history, setHistory] = useState<(RunPage & { filter: HistoryFilter }) | null>(null)
  const [loadingMore, setLoadingMore] = useState(false)
  const [working, setWorking] = useState<string | null>(null)
  const [openGroups, setOpenGroups] = useState<Set<string>>(new Set())
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape' && !document.querySelector('.popover')) onClose() }
    window.addEventListener('keydown', key)
    return () => { window.removeEventListener('keydown', key); opener?.focus() }
  }, [onClose])
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer) }, [])

  const reload = useCallback(async () => {
    try {
      const value = await queueApi.state(episodeId)
      setState(value); summaryRef.current(value.summary); setLoadError('')
    } catch (err) { setLoadError((err as Error).message) }
  }, [episodeId])
  useEffect(() => { void reload(); const timer = setInterval(() => void reload(), POLL_MS); return () => clearInterval(timer) }, [reload])

  // Scroll to Needs attention once it has loaded, when the drawer was opened from the failure chip.
  const focused = useRef(false)
  useEffect(() => {
    if (focus === 'attention' && state?.attention.length && !focused.current) { focused.current = true; attentionRef.current?.scrollIntoView({ block: 'start' }) }
  }, [focus, state])

  // The history's first page follows what finishes; older pages the user loaded stay below it.
  const historyKey = state ? [state.running.map(run => run.id + run.status).join(','), state.waiting_total, state.attention.length, state.summary.finished, state.summary.cancelled].join('|') : ''
  useEffect(() => {
    if (!historyKey) return
    let alive = true
    queueApi.history(episodeId, filter).then(page => {
      if (!alive) return
      setHistory(current => {
        if (!current || current.filter !== filter) return { ...page, filter }
        const runs = mergeFirstPage(current.runs, page.runs)
        return { ...page, filter, runs, next: runs.length > page.runs.length ? current.next : page.next }
      })
    }).catch(err => { if (alive) setLoadError((err as Error).message) })
    return () => { alive = false }
  }, [episodeId, filter, historyKey])

  async function loadMore() {
    if (!history?.next) return
    setLoadingMore(true)
    try {
      const page = await queueApi.history(episodeId, filter, history.next)
      setHistory(current => current && current.filter === filter ? { ...page, filter, runs: [...current.runs, ...page.runs.filter(run => !current.runs.some(shown => shown.id === run.id))] } : current)
    } catch (err) { toast({ tone: 'error', message: (err as Error).message }) } finally { setLoadingMore(false) }
  }

  async function act(key: string, work: () => Promise<unknown>, done?: string) {
    setWorking(key)
    try {
      await work()
      if (done) toast({ tone: 'success', message: done })
      await Promise.all([reload(), onChanged()])
    } catch (err) { toast({ tone: 'error', message: (err as Error).message }) } finally { setWorking(null) }
  }
  const cancelRun = (run: Run) => act(run.id, () => queueApi.cancel(run.id), run.status === 'running' ? `Cancelling ${run.label}. It stops after the current step.` : `Cancelled ${run.label}.`)
  function cancelEpisode(includeRunning: boolean) {
    const question = includeRunning ? 'Cancel all waiting and running GPU work in this episode? Running steps stop after the step they are on; finished results stay.' : 'Cancel all waiting GPU work in this episode? Running work continues.'
    if (!window.confirm(question)) return
    void act('all', async () => {
      const result = await queueApi.cancelEpisode(episodeId, { include_running: includeRunning })
      toast({ tone: 'success', message: result.cancelled ? `Cancelled ${result.cancelled} ${result.cancelled === 1 ? 'run' : 'runs'} in this episode.` : 'Nothing in this episode could be cancelled.' })
    })
  }
  function remove(run: Run) {
    if (!window.confirm('Delete this run from the history? Its saved settings, errors and retry links are removed. Results and transcripts it made stay. This cannot be undone.')) return
    void act(run.id, async () => {
      await queueApi.remove(run.id)
      setHistory(current => current && { ...current, runs: current.runs.filter(item => item.id !== run.id) })
    })
  }

  const summary = state?.summary
  const cancellableRunning = !!state?.running.some(run => run.actions.cancel)
  const cancellableWaiting = !!state?.waiting_groups.some(group => group.cancel)
  const cancelItems: MenuEntry[] = [
    { key: 'waiting', icon: Ban, tone: 'caution', title: 'Cancel waiting work', hint: 'Every waiting GPU run in this episode. Running work continues.', disabled: !cancellableWaiting, disabledReason: 'Nothing is waiting.', onSelect: () => cancelEpisode(false) },
    { key: 'all', icon: Ban, tone: 'danger', title: 'Cancel waiting and running work', hint: 'Running steps stop after the step they are on. Finished results stay.', disabled: !cancellableWaiting && !cancellableRunning, disabledReason: 'Nothing can be cancelled.', onSelect: () => cancelEpisode(true) },
  ]
  const links: Links = { onOpenCut: cutId => { onOpenCut(cutId); onClose() }, onOpenSequence: id => { onOpenSequence(id); onClose() } }
  const elsewhere = state?.elsewhere
  const runningCount = (state?.running.length ?? 0) + (state?.chat.length ?? 0)

  return <>
    <div className="drawer-backdrop" onMouseDown={onClose} aria-hidden />
    <div ref={panel} role="dialog" aria-modal="true" aria-labelledby="queue-drawer-title" tabIndex={-1} className="drawer outline-none">
      <header className="shrink-0 border-b hairline px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <h2 id="queue-drawer-title" className="flex items-center gap-2 text-sm font-semibold"><ListChecks size={16} className="text-[#bca7ff]" aria-hidden /> Queue &amp; history</h2>
          <div className="flex items-center gap-1.5">
            <Menu label="Cancel work in this episode" items={cancelItems} align="end" trigger={props => <button type="button" {...props} disabled={working === 'all'} className="icon-btn icon-btn-md icon-btn-text tone-danger"><Ban size={15} aria-hidden /><span>Cancel</span><ChevronDown size={13} aria-hidden /></button>} />
            <button type="button" className="icon-btn icon-btn-md ghost" onClick={onClose} aria-label="Close the queue"><X size={17} /></button>
          </div>
        </div>
        <p className="mt-1.5 text-xs text-[#b8bfcb]" aria-live="polite">{summary ? counterTitle(summary) : 'Loading…'}</p>
        {summary?.busy && summary.total > 0 && <div className="mt-2 flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#232a36]" role="progressbar" aria-label="GPU steps finished" aria-valuemin={0} aria-valuemax={summary.total} aria-valuenow={summary.finished}>
            <div className="h-full rounded-full bg-[#7bdcb2]" style={{ width: `${Math.round(summary.done / summary.total * 100)}%` }} />
          </div>
          <span className="font-mono text-xs tabular-nums text-[#d5dbe5]">{counterText(summary)}</span>
        </div>}
        {loadError && <p role="alert" className="mt-1.5 text-xs text-[#ff9fa7]">{loadError}</p>}
      </header>

      <div className="scrollbar min-h-0 flex-1 overflow-y-auto pb-6">
        {elsewhere && (elsewhere.episodes.length > 0 || elsewhere.chat_api.length > 0) && <div className="px-4 pt-3">
          <section aria-label="GPU work outside this episode" className="rounded-lg border border-[#2f4a72] bg-[#141f33] px-3 py-2.5 text-xs text-[#cfe0ff]">
            <p className="flex items-center gap-1.5 font-semibold"><Cpu size={14} aria-hidden /> The GPU is also working outside this episode</p>
            <p className="mt-0.5 text-[#9ec5ff]">Waiting work here starts when the GPU is free. It is not counted above.</p>
            <ul className="mt-2 space-y-2">
              {elsewhere.episodes.map(item => <li key={item.episode_id} className="flex flex-wrap items-center justify-between gap-2">
                <span className="min-w-0"><strong className="text-white">{item.title}</strong> <span className="text-[#9ec5ff]">· {item.project_title}</span>
                  <span className="block text-[#b9cff0]">{[item.current ? `${item.current.label} running` : item.running ? `${item.running} running` : '', item.waiting ? `${item.waiting} waiting` : ''].filter(Boolean).join(' · ')}</span></span>
                <span className="flex shrink-0 gap-1.5">
                  <ActionButton compact icon={<ExternalLink size={13} />} onClick={() => { window.location.hash = `/episode/${item.episode_id}` }}>Open</ActionButton>
                  <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === item.episode_id} disabled={working !== null}
                    onClick={() => { if (window.confirm(`Cancel all waiting and running GPU work in “${item.title}”? Running steps stop after the step they are on.`)) void act(item.episode_id, () => queueApi.cancelEpisode(item.episode_id, { include_running: true }), `Cancelled the GPU work of ${item.title}.`) }}>Cancel</ActionButton>
                </span>
              </li>)}
              {elsewhere.chat_api.map(request => <li key={request.id} className="flex flex-wrap items-center justify-between gap-2">
                <span className="min-w-0"><strong className="text-white">Chat API request</strong> <span className="text-[#9ec5ff]">· {request.status === 'queued' ? 'waiting' : request.stage}{request.duration_seconds != null ? ` · ${formatRunTime(request.duration_seconds)}` : ''}</span></span>
                <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === request.id} disabled={working !== null || request.status === 'stopping'} onClick={() => void act(request.id, () => queueApi.stopChatApi(request.id))}>{request.status === 'stopping' ? 'Stopping' : 'Stop'}</ActionButton>
              </li>)}
            </ul>
          </section>
        </div>}

        <Section id="queue.running" title="Running now" icon={PlayCircle} accent="#c3aeff" badge={<Count value={runningCount} />}>
          {!state ? <p className="muted text-xs">Loading…</p> : runningCount === 0 ? <p className="muted text-xs">Nothing is running in this episode.</p> :
            <ul className="space-y-2">
              {state.running.map(run => <RunItem key={run.id} run={run} now={now} links={links}>
                {run.actions.cancel && <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === run.id} disabled={working !== null} onClick={() => void cancelRun(run)} title="Stops after the current step; finished results stay">Cancel</ActionButton>}
              </RunItem>)}
              {state.chat.map(turn => <ChatItem key={turn.id} turn={turn} now={now} links={links}>
                <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === turn.id} disabled={working !== null || turn.status === 'stopping'} onClick={() => void act(turn.id, () => queueApi.stopChat(turn.id))}>{turn.status === 'stopping' ? 'Stopping' : 'Stop'}</ActionButton>
              </ChatItem>)}
            </ul>}
        </Section>

        <Section id="queue.waiting" title="Waiting" icon={Clock} accent="#e6c58b" badge={<Count value={state?.waiting_total ?? 0} />}>
          {!state ? <p className="muted text-xs">Loading…</p> : state.waiting_total === 0 ? <p className="muted text-xs">Nothing is waiting in this episode.</p> :
            <ul className="space-y-1.5">{state.waiting_groups.map(group => {
              const open = openGroups.has(group.group)
              const rows = state.waiting.filter(run => run.group === group.group)
              return <li key={group.group} className="rounded-lg border hairline bg-[#171c25] text-xs">
                <div className="flex items-center gap-2 px-3 py-2">
                  <button type="button" aria-expanded={open} onClick={() => setOpenGroups(current => { const next = new Set(current); if (open) next.delete(group.group); else next.add(group.group); return next })} className="flex min-w-0 flex-1 items-center gap-2 text-left">
                    <ChevronRight size={14} aria-hidden className={`shrink-0 transition-transform ${open ? 'rotate-90' : ''}`} />
                    <span className="truncate font-semibold text-[#e3e6ed]">{group.label}</span>
                    <span className="rounded-full bg-[#232a36] px-1.5 tabular-nums text-[#c9ced8]">{group.count}</span>
                    {!group.gpu && <CpuTag />}
                  </button>
                  {group.cancel && <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === group.group} disabled={working !== null}
                    onClick={() => void act(group.group, () => queueApi.cancelEpisode(episodeId, { group: group.group }), `Cancelled ${group.count} waiting ${group.label.toLowerCase()} ${group.count === 1 ? 'run' : 'runs'}.`)}>{group.count > 1 ? `Cancel ${group.count}` : 'Cancel'}</ActionButton>}
                </div>
                {open && <ul className="space-y-1.5 border-t hairline p-2">
                  {rows.map(run => <RunItem key={run.id} run={run} now={now} links={links}>
                    {run.actions.cancel && <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === run.id} disabled={working !== null} onClick={() => void cancelRun(run)}>Cancel</ActionButton>}
                  </RunItem>)}
                  {rows.length < group.count && <li className="muted px-1">Showing the first {rows.length} of {group.count}.</li>}
                </ul>}
              </li>
            })}</ul>}
        </Section>

        {state && state.attention.length > 0 && <div ref={attentionRef} className="scroll-mt-2">
          <Section id="queue.attention" title="Needs attention" icon={AlertTriangle} accent="#ff9fa7" badge={<Count value={state.attention.length} tone="danger" />} openSignal={focus === 'attention' ? 'attention' : undefined}
            actions={<button type="button" className="text-[11px] text-[#c3aeff] hover:underline disabled:opacity-50" disabled={working !== null} onClick={() => void act('dismiss-all', () => queueApi.dismissAll(episodeId), 'Dismissed every failure.')}>Dismiss all</button>}>
            <p className="muted mb-2 text-xs">These runs failed and have not been run again. Retry them, open their cut to run the step again, or dismiss them once you have dealt with them.</p>
            <ul className="space-y-2">{state.attention.map(run => <RunItem key={run.id} run={run} now={now} links={links} showTime>
              {run.actions.retry && <ActionButton compact icon={<RotateCcw size={13} />} pending={working === run.id} disabled={working !== null} onClick={() => void act(run.id, () => queueApi.retry(run.id), `Queued ${run.label} again.`)} title="Run the unfinished part again with the saved settings">Retry</ActionButton>}
              {run.actions.dismiss && <ActionButton compact icon={<CheckCircle2 size={13} />} disabled={working !== null} onClick={() => void act(`dismiss-${run.id}`, () => queueApi.dismiss(run.id))} title="Stop listing this failure here; it stays in the history">Dismiss</ActionButton>}
            </RunItem>)}</ul>
          </Section>
        </div>}

        <Section id="queue.history" title="History" icon={History} accent="#8b95a7" badge={history ? <Count value={history.counts.done + history.counts.failed + history.counts.cancelled} /> : undefined}>
          <SegmentedControl label="Show runs" value={filter} onChange={setFilter} className="mb-3 w-full" options={[
            { value: 'all', label: 'All' },
            { value: 'done', label: 'Done', badge: history ? <Count value={history.counts.done} /> : undefined },
            { value: 'failed', label: 'Failed', badge: history ? <Count value={history.counts.failed} tone={history.counts.failed ? 'danger' : undefined} /> : undefined },
            { value: 'cancelled', label: 'Cancelled', badge: history ? <Count value={history.counts.cancelled} /> : undefined },
          ]} />
          {!history || history.filter !== filter ? <p className="muted text-xs">Loading…</p> : history.runs.length === 0 ? <p className="muted text-xs">{filter === 'all' ? 'No runs have finished in this episode yet.' : `No ${filter} runs in this episode.`}</p> :
            <ul className="space-y-2">{history.runs.map(run => <RunItem key={run.id} run={run} now={now} links={links} showTime>
              {run.actions.retry && <ActionButton compact icon={<RotateCcw size={13} />} pending={working === run.id} disabled={working !== null} onClick={() => void act(run.id, () => queueApi.retry(run.id), `Queued ${run.label} again.`)}>Retry</ActionButton>}
              {run.actions.delete && <Menu label="Run actions" align="end" width="w-64" items={[{ key: 'delete', icon: Trash2, tone: 'danger', title: 'Delete from history…', hint: 'Results and transcripts it made stay.', onSelect: () => remove(run) }]}
                trigger={props => <button type="button" {...props} className="icon-btn icon-btn-sm" aria-label={`More actions for ${run.label}`} title="More actions"><MoreHorizontal size={14} aria-hidden /></button>} />}
            </RunItem>)}</ul>}
          {history?.next != null && history.filter === filter && <div className="mt-3 flex justify-center"><ActionButton compact pending={loadingMore} onClick={() => void loadMore()}>Load older runs</ActionButton></div>}
        </Section>
      </div>
    </div>
  </>
}

function Count({ value, tone }: { value: number; tone?: 'danger' }) {
  return <span className={`ml-1 rounded-full px-1.5 text-[10px] font-semibold tabular-nums tracking-normal ${tone === 'danger' ? 'bg-[#4a2229] text-[#ffb6bc]' : 'bg-[#232a36] text-[#c9ced8]'}`}>{value}</span>
}

function CpuTag() {
  return <span className="rounded border border-[#353c48] px-1 text-[10px] font-semibold uppercase tracking-wider text-[#929aa9]" title="Media work on the CPU; not counted in the GPU queue">CPU</span>
}

function firstLine(text: string) {
  const line = text.split('\n').find(part => part.trim()) || text
  return line.length > 180 ? `${line.slice(0, 180)}…` : line
}

function ErrorText({ text }: { text: string }) {
  const long = text.length > 180 || text.trim().includes('\n')
  const cls = 'mt-2 rounded-md border border-[#63363e] bg-[#2b191f] px-2 py-1.5 text-[#ffb6bc]'
  return long
    ? <details className={cls}><summary className="cursor-pointer break-words">{firstLine(text)}</summary><pre className="mt-1.5 max-h-60 overflow-auto whitespace-pre-wrap break-words font-sans">{text}</pre></details>
    : <p className={`${cls} whitespace-pre-wrap break-words`}>{text}</p>
}

/** What the run works on, as a link where there is something to open. */
function Target({ run, links }: { run: Run; links: Links }) {
  const text = targetText(run)
  const cut = run.cuts[0]
  if (run.sequence) return <button type="button" className={LINK} onClick={() => links.onOpenSequence(run.sequence!.id)} title="Open the sequence">{text}</button>
  if (run.cut_count === 1 && cut && !cut.missing) return <button type="button" className={LINK} onClick={() => links.onOpenCut(cut.id)} title="Open the cut's analysis">{text}</button>
  return <span className="truncate text-[#c9ced8]" title={run.cut_count > 1 ? run.cuts.map(item => cutText(item, false)).join(', ') + (run.cut_count > run.cuts.length ? ', …' : '') : undefined}>{text}</span>
}

function RunItem({ run, now, links, showTime, children }: { run: Run; now: number; links: Links; showTime?: boolean; children?: ReactNode }) {
  const seconds = runSeconds(run, now)
  const stage = stageText(run)
  const units = unitsText(run)
  const when = run.started_at || run.created_at
  return <li className="rounded-lg border hairline bg-[#171c25] px-3 py-2.5 text-xs">
    <div className="flex items-start gap-3">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-semibold text-[#e3e6ed]">{run.label}</span>
          <StatusChip tone={STATUS_TONE[run.status]} icon={STATUS_ICON[run.status]}>{STATUS_TEXT[run.status]}</StatusChip>
          {!run.gpu && <CpuTag />}
        </div>
        <div className="mt-0.5 flex min-w-0"><Target run={run} links={links} /></div>
        {stage && run.status !== 'done' && <p className="muted mt-0.5 break-words">{stage}</p>}
      </div>
      <div className="shrink-0 text-right text-[11px] leading-4 tabular-nums text-[#aeb5c1]">
        {seconds != null && <span className="block font-semibold text-[#d5dbe5]" title={run.status === 'running' || run.status === 'stopping' ? 'Running for' : 'Took'}>{formatRunTime(seconds)}</span>}
        {units && <span className="block">{units}</span>}
        {showTime && when && <time className="muted block" dateTime={when} title={`${run.started_at ? 'Started' : 'Queued'} ${when.replace('T', ' ').replace('Z', '')} UTC`}>{localTime(when, now)}</time>}
      </div>
    </div>
    {run.error && run.status === 'failed' && <ErrorText text={run.error} />}
    {children && <div className="mt-2 flex flex-wrap items-center justify-end gap-1.5">{children}</div>}
  </li>
}

function ChatItem({ turn, now, links, children }: { turn: ChatRun; now: number; links: Links; children?: ReactNode }) {
  const status: RunStatus = turn.status
  const seconds = runSeconds({ status, started_at: turn.started_at, duration_seconds: turn.duration_seconds }, now)
  return <li className="rounded-lg border hairline bg-[#171c25] px-3 py-2.5 text-xs">
    <div className="flex items-start gap-3">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="flex items-center gap-1.5 font-semibold text-[#e3e6ed]"><MessageSquare size={13} className="text-[#7bdcb2]" aria-hidden />{turn.label}</span>
          <StatusChip tone={STATUS_TONE[status]} icon={STATUS_ICON[status]}>{STATUS_TEXT[status]}</StatusChip>
          <span className="muted" title="Chat replies are listed here but not counted in the queue">not counted</span>
        </div>
        <div className="mt-0.5 flex min-w-0">
          {turn.sequence
            ? <button type="button" className={LINK} onClick={() => links.onOpenSequence(turn.sequence!.id)}>Sequence “{turn.sequence.title || 'Untitled'}”</button>
            : <button type="button" className={LINK} onClick={() => links.onOpenCut(turn.cut.id)}>{cutText({ ...turn.cut, missing: false })}</button>}
        </div>
        <p className="muted mt-0.5 break-words">{[turn.stage, turn.provider === 'local' ? 'local model' : turn.provider].filter(Boolean).join(' · ')}</p>
      </div>
      {seconds != null && <span className="shrink-0 text-[11px] font-semibold tabular-nums text-[#d5dbe5]">{formatRunTime(seconds)}</span>}
    </div>
    {children && <div className="mt-2 flex justify-end">{children}</div>}
  </li>
}
