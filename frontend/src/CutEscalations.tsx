import { useState } from 'react'
import { Bot, CircleHelp } from 'lucide-react'
import { ActionButton } from './RunUi'
import { useToast } from './Toasts'
import { answerEscalation } from './inboxActions'
import { choicesFor, kindLabel, sheetPath, type Choice, type Escalation } from './inboxView'

/** Questions about this cut that nothing else in the analysis dialog answers: a failed run. (Missed cuts and lines across a
 * hard cut are never asked since HARNESS.md D37.) */
export const OTHER_KINDS = new Set(['failed_run'])

/** "Other decisions for this cut", in the Your answers column: the Inbox's one-click answers for those questions. */
export function CutEscalations({ escalations, onChanged }: { escalations: Escalation[]; onChanged: () => void }) {
  const toast = useToast()
  const [working, setWorking] = useState<string | null>(null)
  const items = escalations.filter(item => item.reviewer === 'human' && OTHER_KINDS.has(item.kind))
  if (!items.length) return null
  async function choose(item: Escalation, choice: Choice) {
    setWorking(`${item.id}:${choice.key}`)
    try { if (await answerEscalation(item, choice)) { toast({ message: `${kindLabel(item.kind)}: ${choice.label}` }); onChanged() } }
    catch (err) { toast({ tone: 'error', message: (err as Error).message }) } finally { setWorking(null) }
  }
  return <section data-need="other-decisions" aria-label="Other decisions for this cut" className="shrink-0 space-y-2 rounded-lg border-l-[3px] border-l-[#f0a64a] bg-[#2a2314] px-3 py-2.5">
    <h4 className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[.1em] text-[#f0d9a5]"><CircleHelp size={14} aria-hidden />Other decisions for this cut</h4>
    {items.map(item => {
      const sheet = sheetPath(item)
      return <div key={item.id} className="space-y-1.5 rounded-md border border-[#4a3f24] bg-[#1d1a12] p-2">
        <p className="text-[11px] font-semibold text-[#e6c58b]">{kindLabel(item.kind)}</p>
        <p className="break-words text-[12.5px] leading-5 text-[#f2f4f8]">{item.question}</p>
        {sheet && <a href={`/api${sheet}`} target="_blank" rel="noreferrer" title="Open the picture full size"><img src={`/api${sheet}`} alt={`Evidence for: ${item.question}`} loading="lazy" className="max-h-40 w-full rounded border border-[#353c48] object-contain" /></a>}
        {item.evidence?.claude_note && <p className="flex items-start gap-1 text-[11.5px] text-[#d6c9ff]"><Bot size={12} className="mt-0.5 shrink-0" aria-hidden />Claude was unsure: {String(item.evidence.claude_note)}</p>}
        <div className="flex flex-wrap gap-1.5">{choicesFor(item).filter(choice => choice.key !== 'open').map(choice => <ActionButton key={choice.key} compact
          variant={choice.tone === 'primary' ? 'primary' : 'quiet'} pending={working === `${item.id}:${choice.key}`} disabled={working !== null}
          title={choice.hint ?? (choice.proposed ? 'The agent proposes this' : undefined)} onClick={() => void choose(item, choice)}>{choice.label}{choice.proposed ? ' (proposed)' : ''}</ActionButton>)}</div>
      </div>
    })}
  </section>
}
