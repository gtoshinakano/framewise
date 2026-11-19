import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { ListChecks, X } from 'lucide-react'
import { api, jsonBody } from './api'
import { SUBTITLE_LANGUAGES } from './SubtitlePanel'

export function PrepareEpisodeDialog({ episodeId, savedLanguage, onClose, onQueued }: {
  episodeId: string
  savedLanguage: string | null
  onClose: () => void
  onQueued: (queued: boolean) => void
}) {
  const [language, setLanguage] = useState(savedLanguage || '')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const field = useRef<HTMLInputElement>(null)
  const suggestions = useId()
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    field.current?.focus()
    return () => opener?.focus()
  }, [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (submitting || !language.trim()) return
    setSubmitting(true)
    setError('')
    try {
      const result = await api<{ job_ids: string[] }>(`/episodes/${episodeId}/prepare`, jsonBody({ language: language.trim() }))
      onQueued(result.job_ids.length > 0)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4"
    onMouseDown={event => { if (event.target === event.currentTarget && !submitting) onClose() }}>
    <form onSubmit={submit} className="w-full max-w-md space-y-4 rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl"
      role="dialog" aria-modal="true" aria-labelledby="prepare-episode-title"
      onKeyDown={event => { if (event.key === 'Escape' && !submitting) onClose() }}>
      <div className="flex items-center justify-between">
        <h2 id="prepare-episode-title" className="text-base font-semibold">Prepare episode</h2>
        <button type="button" onClick={onClose} disabled={submitting} aria-label="Close Prepare episode"
          className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button>
      </div>
      <p className="muted text-xs leading-5">Queues hard cuts, transcription, subtitle extraction, character boxes and character recognition. Finished steps are skipped.</p>
      <label className="block"><span className="label">Burned-in subtitle language</span>
        <input ref={field} required minLength={2} maxLength={60} list={suggestions} value={language}
          onChange={event => setLanguage(event.target.value)} placeholder="e.g. Portuguese" className="field" />
      </label>
      <datalist id={suggestions}>{SUBTITLE_LANGUAGES.map(name => <option key={name} value={name} />)}</datalist>
      {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
      <div className="flex justify-end gap-2">
        <button type="button" onClick={onClose} disabled={submitting} className="btn btn-quiet">Cancel</button>
        <button type="submit" disabled={submitting || !language.trim()} className="btn btn-primary">
          <ListChecks size={15} /> {submitting ? 'Queuing…' : 'Queue preparation'}
        </button>
      </div>
    </form>
  </div>
}
