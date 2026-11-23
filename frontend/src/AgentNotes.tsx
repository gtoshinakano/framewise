import { useState, type ReactNode } from 'react'
import { Bot, CircleCheck, Cpu, ListChecks, Undo2, UserRoundCheck } from 'lucide-react'
import { api } from './api'
import { StatusChip } from './RunUi'
import { useToast } from './Toasts'
import { Disclosure } from './ReviewUi'
import { DECISION_LABEL, kindLabel, type Decision, type Escalation } from './inboxView'
import { NEED_LABEL, TIER_NAME, type CutNeeds, type NeedKey } from './cutNeedsView'

const TIER_ICON: Record<string, typeof Bot> = { claude: Bot, local: Cpu, rules: ListChecks, human: UserRoundCheck }

/**
 * "Claude answered B — note" under whatever an agent decided. Claude's strips stand out; the local model's and the rules' are
 * quieter. Undo puts the target back while it still holds what the agent wrote (the server answers 409 otherwise).
 */
export function DecisionStrip({ decision, children, note, onChanged }: { decision: Decision; children: ReactNode; note?: string; onChanged: () => void }) {
  const toast = useToast()
  const [working, setWorking] = useState(false)
  const claude = decision.tier === 'claude'
  const Icon = TIER_ICON[decision.tier] ?? Bot
  async function undo() {
    setWorking(true)
    try { await api(`/agent-decisions/${decision.id}/undo`, { method: 'POST' }); toast({ message: 'Undone.' }); onChanged() }
    catch (err) { toast({ tone: 'error', message: (err as Error).message }) } finally { setWorking(false) }
  }
  return <div className={`flex items-start gap-2 rounded-md border px-2.5 py-1.5 text-[12px] leading-5 ${claude ? 'border-[#4a4363] bg-[#221c33] text-[#e0d6ff]' : 'border-[#353c48] bg-[#1b202a] text-[#c8cdd8]'}`}>
    <Icon size={14} className={`mt-0.5 shrink-0 ${claude ? 'text-[#c3aeff]' : 'text-[#929aa9]'}`} aria-hidden />
    <p className="min-w-0 flex-1 break-words"><span className="font-semibold">{TIER_NAME[decision.tier] ?? decision.tier}</span> {children}
      {note && <span className="block text-[11.5px] text-[#aeb5c1]">Note: {note}</span>}
      {decision.overridden_at && <span className="block text-[11px] text-[#e6c58b]">You changed it since.</span>}</p>
    {!decision.undone_at && !decision.overridden_at && <button type="button" disabled={working} onClick={() => void undo()} className="inline-flex shrink-0 items-center gap-1 text-[11px] text-[#c3aeff] hover:underline disabled:opacity-50" title="Put back what was there before the agent's answer"><Undo2 size={12} aria-hidden />Undo</button>}
  </div>
}

/**
 * The top of the Results tab: what on this cut waits for the person (each chip goes to where it is settled), what waits for
 * Claude, and what the agents already applied here.
 */
export function CutNeedsBar({ needs, escalations, decisions, onJump, onChanged }: { needs?: CutNeeds; escalations: Escalation[]; decisions: Decision[]; onJump: (key: NeedKey) => void; onChanged: () => void }) {
  if (!needs) return null
  const forClaude = escalations.filter(item => item.reviewer === 'claude')
  const unsure = escalations.filter(item => item.reviewer === 'human' && item.evidence?.claude_note)
  const applied = decisions.filter(decision => !decision.undone_at)
  const claudeKinds = [...new Set(forClaude.map(item => item.kind))]
  return <div role="region" aria-label="What this cut needs" className="mb-1 shrink-0 space-y-2 rounded-xl border hairline bg-[#171c25] px-3 py-2">
    <div className="flex flex-wrap items-center gap-1.5">
      {needs.people.length ? <>
        <span className="text-[11px] font-bold uppercase tracking-[.1em] text-[#f0d9a5]">Needs you</span>
        {needs.people.map(item => <button key={item.key} type="button" onClick={() => onJump(item.key)} title="Show where this is settled"
          className="rounded-full border border-[#6b5a2f] bg-[#2a2314] px-2 py-0.5 text-[11px] font-semibold text-[#f0c979] hover:border-[#f0a64a] hover:text-white">{NEED_LABEL[item.key]}{item.count > 1 ? ` · ${item.count}` : ''}</button>)}
      </> : <StatusChip tone="green" icon={<CircleCheck size={11} aria-hidden />}>Nothing waits for you on this cut</StatusChip>}
      {claudeKinds.length > 0 && <>
        <span className="ml-2 text-[11px] font-bold uppercase tracking-[.1em] text-[#c3aeff]">Claude will check</span>
        {claudeKinds.map(kind => <StatusChip key={kind} tone="purple" icon={<Bot size={11} aria-hidden />} title="Claude (the second reviewer) answers these next; you see them only if Claude is unsure">{kindLabel(kind)}</StatusChip>)}
      </>}
    </div>
    {unsure.map(item => <p key={item.id} className="flex items-start gap-1.5 text-[12px] leading-5 text-[#d6c9ff]"><Bot size={13} className="mt-1 shrink-0" aria-hidden />
      <span className="min-w-0"><span className="font-semibold">Claude was unsure</span> ({kindLabel(item.kind)}): {String(item.evidence.claude_note)}</span></p>)}
    {applied.length > 0 && <Disclosure label={`What agents did here (${applied.length})`}>
      <div className="space-y-1.5">{applied.map(decision => <DecisionStrip key={decision.id} decision={decision} note={decision.evidence?.note} onChanged={onChanged}>
        <span className="text-[#aeb5c1]">· {DECISION_LABEL[decision.kind] ?? kindLabel(decision.kind)}:</span> {decision.summary}
      </DecisionStrip>)}</div>
    </Disclosure>}
  </div>
}
