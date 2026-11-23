import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { useEpisodeEvents } from './episodeEvents'
import type { Decision, Escalation } from './inboxView'

export type CutAgentWork = { escalations: Escalation[]; decisions: Decision[]; reload: () => void }

const EMPTY: Escalation[] = []
const NO_DECISIONS: Decision[] = []

/** The open autopilot questions that name this cut (the person's and Claude's) and what agents applied on it. */
export function useCutAgentWork(episodeId: string, cutId: string): CutAgentWork {
  const [escalations, setEscalations] = useState<Escalation[]>(EMPTY)
  const [decisions, setDecisions] = useState<Decision[]>(NO_DECISIONS)
  const reload = useCallback(() => {
    api<{ items: Escalation[] }>(`/episodes/${episodeId}/escalations`)
      .then(payload => setEscalations(payload.items.filter(item => item.cut_ids.includes(cutId)))).catch(() => setEscalations(EMPTY))
    api<Decision[]>(`/episodes/${episodeId}/agent-decisions?cut_id=${encodeURIComponent(cutId)}`).then(setDecisions).catch(() => setDecisions(NO_DECISIONS))
  }, [episodeId, cutId])
  useEffect(() => { reload() }, [reload])
  useEpisodeEvents(change => { if (change.data) reload() }, episodeId)
  return { escalations, decisions, reload }
}
