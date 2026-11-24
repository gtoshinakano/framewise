import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Bot, CheckCheck, CheckCircle2, ExternalLink, Image as ImageIcon, Inbox, RotateCcw, Undo2, Wrench, X } from 'lucide-react'
import { api } from './api'
import { editorPath } from './cutRoute'
import { ActionButton } from './RunUi'
import { SegmentedControl } from './EditorUi'
import { useEpisodeEvents } from './episodeEvents'
import { useToast } from './Toasts'
import { queueApi } from './queueApi'
import { answerEscalation } from './inboxActions'
import {
  choicesFor, decisionCounts, DECISION_LABEL, flatOrder, groupByKind, kindLabel, placeFor, sampleDecisions, sheetPath, TIER_LABEL,
  type AgentRun, type Choice, type Decision, type Escalation, type TriageItem,
} from './inboxView'

type View = 'mine' | 'claude' | 'applied' | 'problems'
const LINK = 'text-[#c3aeff] hover:underline'

/**
 * The autopilot inbox (docs/autopilot/HARNESS.md M7): the questions the agents could not settle, grouped so one answer
 * settles many items; what they applied (spot-check and undo); failed runs with their cause and fix.
 * Keyboard: j/k (or ↓/↑) move between questions, Enter takes the proposal, 1–9 pick an answer, o opens the cut.
 */
export function InboxDrawer({ episodeId, onClose, onChanged }: { episodeId: string; onClose: () => void; onChanged: () => void }) {
  const toast = useToast()
  const panel = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>('mine')
  const [items, setItems] = useState<Escalation[] | null>(null)
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [runs, setRuns] = useState<AgentRun[]>([])
  const [problems, setProblems] = useState<TriageItem[]>([])
  const [error, setError] = useState('')
  const [working, setWorking] = useState<string | null>(null)
  const [focus, setFocus] = useState(0)
  const [sampleOnly, setSampleOnly] = useState(true)
  const [kindFilter, setKindFilter] = useState('')

  const reload = useCallback(async () => {
    try {
      const [open, applied, agentRuns, triage] = await Promise.all([
        api<{ items: Escalation[] }>(`/episodes/${episodeId}/escalations`),
        api<Decision[]>(`/episodes/${episodeId}/agent-decisions?limit=2000`),
        api<AgentRun[]>(`/episodes/${episodeId}/agent-runs`),
        api<{ items: TriageItem[] }>(`/episodes/${episodeId}/triage`),
      ])
      setItems(open.items)
      setDecisions(applied)
      setRuns(agentRuns)
      setProblems(triage.items)
      setError('')
    } catch (err) {
      setError((err as Error).message)
    }
  }, [episodeId])
  useEffect(() => { void reload() }, [reload])
  useEpisodeEvents(change => { if (change.data || change.jobs) void reload() })
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    return () => opener?.focus()
  }, [])

  const mine = useMemo(() => flatOrder((items ?? []).filter(item => item.reviewer === 'human')), [items])
  const forClaude = useMemo(() => flatOrder((items ?? []).filter(item => item.reviewer === 'claude')), [items])
  const shown = view === 'claude' ? forClaude : mine
  useEffect(() => { setFocus(index => Math.min(index, Math.max(0, shown.length - 1))) }, [shown.length])

  function openCut(item: Escalation, cutId = item.cut_ids[0]) {
    if (!cutId) return
    const place = placeFor(item.kind)
    window.location.hash = `#${editorPath({ episodeId, cutId, analysis: true, tab: place.tab, section: place.section })}`
    onClose()
  }

  async function run(key: string, action: () => Promise<unknown>, message: string) {
    setWorking(key)
    try {
      // An action that returns false was cancelled by the person (no name for a new character): nothing changed.
      if (await action() === false) return
      toast({ message })
      await reload()
      onChanged()
    } catch (err) {
      toast({ tone: 'error', message: (err as Error).message })
    } finally {
      setWorking(null)
    }
  }

  /** Close every question waiting for Claude as dismissed (the person says they are already settled). Nothing in the data changes. */
  async function dismissAllForClaude() {
    if (!forClaude.length || !window.confirm(`Dismiss all ${forClaude.length} questions waiting for Claude? Nothing else changes; a later agent run can ask again if something is still unsettled.`)) return
    await run('dismiss-claude', async () => {
      const results = await Promise.allSettled(forClaude.map(item => api(`/escalations/${item.id}/close`, {
        method: 'POST', body: JSON.stringify({ status: 'dismissed', resolution: { by: 'inbox', choice: 'dismiss_all_claude' } }),
      })))
      const failed = results.filter(result => result.status === 'rejected').length
      if (failed) throw new Error(`${failed} of ${results.length} could not be dismissed (already closed?). The rest were.`)
    }, 'Dismissed every question for Claude')
  }

  async function choose(item: Escalation, choice: Choice) {
    if (choice.key === 'open') return openCut(item)
    await run(`${item.id}:${choice.key}`, () => answerEscalation(item, choice), `${kindLabel(item.kind)}: ${choice.label}`)
  }

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !document.querySelector('.popover')) { onClose(); return }
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)) return
      if (view !== 'mine' && view !== 'claude') return
      const item = shown[focus]
      if (event.key === 'j' || event.key === 'ArrowDown') { event.preventDefault(); setFocus(index => Math.min(shown.length - 1, index + 1)) }
      else if (event.key === 'k' || event.key === 'ArrowUp') { event.preventDefault(); setFocus(index => Math.max(0, index - 1)) }
      else if (!item || working) return
      else if (event.key === 'o') openCut(item)
      else if (event.key === 'Enter') {
        const proposed = choicesFor(item).find(choice => choice.proposed) ?? choicesFor(item).find(choice => choice.tone === 'primary' && choice.request)
        if (proposed) { event.preventDefault(); void choose(item, proposed) }
      } else if (/^[1-9]$/.test(event.key)) {
        const choice = choicesFor(item)[Number(event.key) - 1]
        if (choice) { event.preventDefault(); void choose(item, choice) }
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })
  useEffect(() => { document.getElementById(`inbox-${shown[focus]?.id}`)?.scrollIntoView({ block: 'nearest' }) }, [focus, shown])

  const counts = decisionCounts(decisions)
  const visibleDecisions = (sampleOnly ? sampleDecisions(decisions) : decisions.filter(item => !item.undone_at)).filter(item => !kindFilter || item.kind === kindFilter)
  const fixable = problems.filter(item => !['escalate', 'leave'].includes(item.action))

  return <>
    <div className="drawer-backdrop" onMouseDown={onClose} aria-hidden />
    <div ref={panel} role="dialog" aria-modal="true" data-shortcut-view={view === 'mine' || view === 'claude' ? 'inbox' : 'inbox:other'} aria-labelledby="inbox-title" tabIndex={-1} className="drawer outline-none">
      <header className="shrink-0 border-b hairline px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <h2 id="inbox-title" className="flex items-center gap-2 text-sm font-semibold"><Inbox size={16} className="text-[#7bdcb2]" aria-hidden /> Autopilot inbox</h2>
          <button type="button" className="icon-btn icon-btn-md ghost" onClick={onClose} aria-label="Close the inbox"><X size={17} /></button>
        </div>
        <p className="muted mt-1 text-xs">Agents applied what they could; these are the questions left for you. One answer settles every item of a group. Keys: j/k move, Enter takes the proposal, 1–9 pick, o opens the cut.</p>
        <SegmentedControl label="Inbox view" value={view} onChange={setView} className="mt-2 w-full" options={[
          { value: 'mine', label: 'Your decisions', badge: <Count value={mine.length} tone={mine.length ? 'warn' : undefined} /> },
          { value: 'claude', label: 'For Claude', badge: <Count value={forClaude.length} /> },
          { value: 'applied', label: 'Applied', badge: <Count value={decisions.filter(item => !item.undone_at).length} /> },
          { value: 'problems', label: 'Problems', badge: <Count value={problems.length} tone={problems.length ? 'danger' : undefined} /> },
        ]} />
        {error && <p role="alert" className="mt-1.5 text-xs text-[#ff9fa7]">{error}</p>}
      </header>

      <div className="scrollbar min-h-0 flex-1 overflow-y-auto px-4 pb-6 pt-3 text-xs">
        {items === null ? <p className="muted">Loading…</p> : (view === 'mine' || view === 'claude') && <>
          {view === 'claude' && <div className="mb-2 flex items-start justify-between gap-2">
            <p className="muted">These wait for Claude, the second reviewer (a Claude Code session running the autopilot answers them). You can answer them yourself too.</p>
            {forClaude.length > 0 && <ActionButton compact icon={<CheckCheck size={13} />} pending={working === 'dismiss-claude'} disabled={working !== null} onClick={() => void dismissAllForClaude()}>Dismiss all</ActionButton>}
          </div>}
          {shown.length === 0 ? <p className="muted flex items-center gap-1.5"><CheckCircle2 size={14} className="text-[#7bdcb2]" aria-hidden />{view === 'mine' ? 'Nothing waits for you in this episode.' : 'Nothing waits for Claude.'}</p> :
            groupByKind(shown).map(group => <section key={group.kind} className="mb-4">
              <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#929aa9]">{group.label} <Count value={group.items.length} /></h3>
              <ul className="space-y-2">{group.items.map(item => <QuestionCard key={item.id} item={item} focused={shown[focus]?.id === item.id} working={working}
                onFocus={() => setFocus(shown.findIndex(entry => entry.id === item.id))} onChoose={choice => void choose(item, choice)} onOpen={cutId => openCut(item, cutId)}
                onBoxes={(confirm, reject) => void run(`${item.id}:boxes`, () => api(`/escalations/${item.id}/box-labels`, { method: 'POST', body: JSON.stringify({ tier: 'human', confirm, reject }) }), 'Box names saved')} />)}</ul>
            </section>)}
        </>}

        {view === 'applied' && <>
          <div className="mb-3 grid grid-cols-2 gap-2">
            {Object.entries(counts).map(([kind, entry]) => <button key={kind} type="button" onClick={() => setKindFilter(current => current === kind ? '' : kind)}
              className={`rounded-lg border px-2.5 py-2 text-left ${kindFilter === kind ? 'border-[#7bdcb2] bg-[#16261f]' : 'hairline bg-[#171c25]'}`}>
              <span className="block font-semibold text-[#e3e6ed]">{DECISION_LABEL[kind] ?? kind}</span>
              <span className="muted">{entry.total} applied{entry.overridden ? ` · ${entry.overridden} changed by you` : ''} · {Object.entries(entry.tiers).map(([tier, value]) => `${TIER_LABEL[tier as Decision['tier']] ?? tier} ${value}`).join(', ')}</span>
            </button>)}
          </div>
          <label className="mb-2 flex items-center gap-2"><input type="checkbox" checked={sampleOnly} onChange={event => setSampleOnly(event.target.checked)} /> Spot-check sample (5 per kind)</label>
          <ul className="space-y-1.5">{visibleDecisions.map(decision => <li key={decision.id} className="flex items-start gap-2 rounded-lg border hairline bg-[#171c25] px-3 py-2">
            <div className="min-w-0 flex-1">
              <p className="break-words text-[#e3e6ed]">{decision.summary}</p>
              <p className="muted">{TIER_LABEL[decision.tier]}{decision.confidence != null ? ` · ${decision.confidence.toFixed(2)}` : ''}{decision.overridden_at ? ' · changed by you since' : ''}
                {decision.cut_id && <> · <button type="button" className={LINK} onClick={() => { window.location.hash = `#${editorPath({ episodeId, cutId: decision.cut_id, analysis: true, tab: decision.kind === 'box_names' ? 'run' : decision.kind === 'target_frames' ? 'run' : 'results', section: decision.kind === 'target_frames' ? 'targets' : 'characters' })}`; onClose() }}>open the cut</button></>}</p>
            </div>
            <ActionButton compact icon={<Undo2 size={13} />} pending={working === decision.id} disabled={working !== null}
              onClick={() => void run(decision.id, () => api(`/agent-decisions/${decision.id}/undo`, { method: 'POST' }), 'Undone')}>Undo</ActionButton>
          </li>)}</ul>
          {runs.length > 0 && <>
            <h3 className="mb-1.5 mt-4 text-[11px] font-semibold uppercase tracking-wider text-[#929aa9]">Agent runs</h3>
            <ul className="space-y-1.5">{runs.map(agentRun => <li key={agentRun.id} className="flex items-center gap-2 rounded-lg border hairline bg-[#171c25] px-3 py-2">
              <Bot size={14} className="shrink-0 text-[#7bdcb2]" aria-hidden />
              <div className="min-w-0 flex-1"><p className="truncate text-[#e3e6ed]">{agentRun.purpose || agentRun.agent || 'Agent run'} · {agentRun.status}</p>
                <p className="muted">{agentRun.started_at} · {agentRun.decisions.total} decisions{agentRun.decisions.undone ? `, ${agentRun.decisions.undone} undone` : ''}{agentRun.claude_items ? ` · ${agentRun.claude_items} Claude items` : ''}</p></div>
              {agentRun.status !== 'undone' && agentRun.decisions.total > agentRun.decisions.undone && <ActionButton compact variant="danger" icon={<Undo2 size={13} />} pending={working === agentRun.id} disabled={working !== null}
                onClick={() => { if (window.confirm('Undo every decision of this run that nobody changed since?')) void run(agentRun.id, () => api(`/agent-runs/${agentRun.id}/undo`, { method: 'POST' }), 'Run undone') }}>Undo run</ActionButton>}
            </li>)}</ul>
          </>}
        </>}

        {view === 'problems' && <>
          {problems.length === 0 ? <p className="muted flex items-center gap-1.5"><CheckCircle2 size={14} className="text-[#7bdcb2]" aria-hidden />No failed runs wait for anyone.</p> : <>
            {fixable.length > 0 && <div className="mb-3 flex items-center justify-between gap-2 rounded-lg border border-[#2f4a72] bg-[#141f33] px-3 py-2">
              <span>{fixable.length} failure{fixable.length === 1 ? ' has' : 's have'} a fix rule (retry, queue the missing step first, wait for the GPU).</span>
              <ActionButton compact icon={<Wrench size={13} />} pending={working === 'triage'} disabled={working !== null}
                onClick={() => void run('triage', () => api(`/episodes/${episodeId}/triage`, { method: 'POST', body: JSON.stringify({}) }), 'Fixes applied')}>Apply fixes</ActionButton>
            </div>}
            <ul className="space-y-2">{problems.map(problem => <li key={problem.job_id} className="rounded-lg border hairline bg-[#171c25] px-3 py-2.5">
              <p className="flex items-center gap-1.5 font-semibold text-[#e3e6ed]"><AlertTriangle size={13} className="text-[#ff9fa7]" aria-hidden />{problem.label}{problem.cuts.length ? ` · cut ${problem.cuts.slice(0, 6).join(', ')}${problem.cuts.length > 6 ? '…' : ''}` : ''}</p>
              <p className="mt-0.5 text-[#c9ced8]">{problem.cause_text}</p>
              <p className="muted mt-0.5">Fix: {problem.fix}{problem.limit ? ` (try ${problem.attempts + 1} of ${problem.limit})` : ''}</p>
              <p className="mt-1 break-words rounded border border-[#63363e] bg-[#2b191f] px-2 py-1 text-[#ffb6bc]">{problem.error.slice(0, 300)}</p>
              <div className="mt-2 flex justify-end gap-1.5">
                {problem.cut_ids[0] && <ActionButton compact icon={<ExternalLink size={13} />} onClick={() => { window.location.hash = `#${editorPath({ episodeId, cutId: problem.cut_ids[0], analysis: true, tab: 'results' })}`; onClose() }}>Open</ActionButton>}
                <ActionButton compact icon={<RotateCcw size={13} />} pending={working === problem.job_id} disabled={working !== null}
                  onClick={() => void run(problem.job_id, () => queueApi.retry(problem.job_id), `Queued ${problem.label} again`)}>Retry</ActionButton>
                <ActionButton compact disabled={working !== null} onClick={() => void run(`d${problem.job_id}`, () => queueApi.dismiss(problem.job_id), 'Dismissed')}>Dismiss</ActionButton>
              </div>
            </li>)}</ul>
          </>}
        </>}
      </div>
    </div>
  </>
}

function QuestionCard({ item, focused, working, onFocus, onChoose, onOpen, onBoxes }: {
  item: Escalation; focused: boolean; working: string | null
  onFocus: () => void; onChoose: (choice: Choice) => void; onOpen: (cutId?: string) => void
  onBoxes: (confirm: string[], reject: string[]) => void
}) {
  const choices = choicesFor(item)
  const sheet = sheetPath(item)
  const [rejected, setRejected] = useState<Set<string>>(new Set())
  const evidence = item.evidence ?? {}
  const busy = !!working
  return <li id={`inbox-${item.id}`} onClick={onFocus}
    className={`rounded-lg border bg-[#171c25] px-3 py-2.5 ${focused ? 'border-[#7bdcb2] ring-1 ring-[#7bdcb2]/40' : 'hairline'}`}>
    <p className="break-words text-[#e3e6ed]">{item.question}</p>
    {evidence.claude_note && <p className="mt-1 rounded border border-[#2f4a72] bg-[#141f33] px-2 py-1 text-[#cfe0ff]"><Bot size={12} className="mr-1 inline" aria-hidden />Claude: {evidence.claude_note}</p>}
    {item.kind === 'voice' && (evidence.phrases ?? []).slice(0, 4).map((phrase: { text: string; cut?: number }, index: number) =>
      <p key={index} className="muted mt-0.5 truncate">P{index + 1}{phrase.cut ? ` · cut ${phrase.cut}` : ''}: “{phrase.text}”</p>)}
    {item.kind === 'target_frame' && (evidence.frames ?? []).map((frame: { frame: string; why: string; ordinal: number }) =>
      <p key={frame.ordinal} className="muted mt-0.5">{frame.frame}: {frame.why}</p>)}
    {item.kind === 'failed_run' && (evidence.jobs ?? []).slice(0, 3).map((job: { job_id: string; error: string; cuts?: number[] }) =>
      <p key={job.job_id} className="mt-0.5 break-words text-[#ffb6bc]">{job.cuts?.length ? `cut ${job.cuts.join(', ')}: ` : ''}{job.error}</p>)}
    {item.kind === 'box_label' && <ul className="mt-1.5 flex flex-wrap gap-1.5">{(evidence.boxes ?? []).map((box: { id: string; cut: number }, index: number) =>
      <li key={box.id}><label className={`flex cursor-pointer items-center gap-1 rounded border px-1.5 py-0.5 ${rejected.has(box.id) ? 'border-[#63363e] text-[#ffb6bc] line-through' : 'hairline'}`}>
        <input type="checkbox" checked={!rejected.has(box.id)} onChange={() => setRejected(current => { const next = new Set(current); if (next.has(box.id)) next.delete(box.id); else next.add(box.id); return next })} />
        B{index + 1} · cut {box.cut}</label></li>)}</ul>}
    {sheet && <details className="mt-1.5" open={focused && (item.kind === 'box_label' || item.kind === 'new_person')}>
      <summary className="cursor-pointer text-[#c3aeff]"><ImageIcon size={12} className="mr-1 inline" aria-hidden />Sheet</summary>
      <img src={`/api${sheet}`} alt={`Sheet for: ${item.question}`} loading="lazy" className="mt-1 max-w-full rounded border hairline" />
    </details>}
    <div className="mt-2 flex flex-wrap items-center justify-end gap-1.5">
      {item.cut_ids.length > 0 && <button type="button" className={`${LINK} mr-auto`} onClick={() => onOpen()}>
        Open cut{item.cut_ids.length > 1 ? ` (1 of ${item.cut_ids.length})` : ''}</button>}
      {item.kind === 'box_label' && rejected.size > 0 && item.proposal?.character_context_id && <ActionButton compact variant="primary" pending={working === `${item.id}:boxes`} disabled={busy}
        onClick={() => { const boxes: string[] = (evidence.boxes ?? []).map((box: { id: string }) => box.id); onBoxes(boxes.filter(id => !rejected.has(id)), boxes.filter(id => rejected.has(id))) }}>
        Save ticked as {item.proposal.name}</ActionButton>}
      {choices.map((choice, index) => <ActionButton key={choice.key} compact variant={choice.tone === 'primary' ? 'primary' : choice.tone === 'danger' ? 'danger' : 'quiet'}
        title={`${choice.hint ?? ''}${index < 9 ? ` (key ${index + 1})` : ''}`.trim()} pending={working === `${item.id}:${choice.key}`} disabled={busy} onClick={() => onChoose(choice)}>
        {choice.proposed ? '★ ' : ''}{choice.label}</ActionButton>)}
    </div>
  </li>
}

function Count({ value, tone }: { value: number; tone?: 'danger' | 'warn' }) {
  const colour = tone === 'danger' ? 'bg-[#4a2229] text-[#ffb6bc]' : tone === 'warn' ? 'bg-[#3d3220] text-[#f5c678]' : 'bg-[#232a36] text-[#c9ced8]'
  return <span className={`ml-1 rounded-full px-1.5 text-[10px] font-semibold tabular-nums tracking-normal ${colour}`}>{value}</span>
}
