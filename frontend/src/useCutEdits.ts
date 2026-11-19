import { useRef, useState } from 'react'
import { api, type CutHistory, type Episode } from './api'
import type { ToastInput } from './Toasts'

export type CutEditAction = 'timing' | 'merge-next' | 'delete' | 'split'

/**
 * Split, merge, delete and timing edits of the selected cut, plus Undo and Redo of any cut edit
 * (backend cut_history.py). Every edit and every reversal offers the opposite action in a toast.
 */
export function useCutEdits({ episodeId, onEpisode, onSelect, toast }: {
  episodeId: string
  /** Receives the reloaded episode after an edit. */
  onEpisode: (episode: Episode) => void
  /** Selects the cut the server named, once the reloaded episode is in place. */
  onSelect: (cutId: string, episode: Episode) => void
  toast: (toast: ToastInput) => void
}) {
  const [working, setWorking] = useState(false)
  const busy = useRef(false)

  async function reload(selectedId: string | null) {
    const value = await api<Episode>(`/episodes/${episodeId}`)
    onEpisode(value)
    if (selectedId) onSelect(selectedId, value)
    return value
  }

  async function run(work: () => Promise<void>) {
    if (busy.current) return
    busy.current = true; setWorking(true)
    try { await work() } catch (err) { toast({ message: (err as Error).message, tone: 'error' }) } finally { busy.current = false; setWorking(false) }
  }

  function edit(action: CutEditAction, cutId: string, values: { start?: number; end?: number; time?: number } = {}) {
    return run(async () => {
      const path = action === 'delete' ? `/cuts/${cutId}` : `/cuts/${cutId}/${action}`
      const init: RequestInit = action === 'timing' ? { method: 'PUT', body: JSON.stringify({ start_time: values.start, end_time: values.end }) }
        : action === 'split' ? { method: 'POST', body: JSON.stringify({ time: Math.round((values.time ?? 0) * 1000) / 1000 }) }
        : { method: action === 'delete' ? 'DELETE' : 'POST' }
      const result = await api<{ selected_id: string }>(path, init)
      const value = await reload(result.selected_id)
      const label = value.cut_history?.undo?.label
      if (label) toast({ message: label, tone: 'success', action: { label: 'Undo', run: () => void undo() } })
    })
  }

  function reverse(direction: 'undo' | 'redo') {
    return run(async () => {
      const result = await api<{ label: string; selected_id: string | null; cut_history: CutHistory }>(`/episodes/${episodeId}/cut-edits/${direction}`, { method: 'POST' })
      await reload(result.selected_id)
      toast(direction === 'undo'
        ? { message: `Undone: ${result.label}`, action: { label: 'Redo', run: () => void redo() } }
        : { message: `Redone: ${result.label}`, action: { label: 'Undo', run: () => void undo() } })
    })
  }
  const undo = () => reverse('undo')
  const redo = () => reverse('redo')
  return { edit, undo, redo, working }
}
