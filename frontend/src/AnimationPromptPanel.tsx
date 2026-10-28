import { useState } from 'react'
import { RefreshCw, Sparkles } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'

export const ANIMATION_TITLE = 'MiniMax animation prompt'

export function AnimationPromptPanel({ cutId, current, job, disabled, onStarted }: { cutId: string; current?: AnalysisResult; job?: Job; disabled: boolean; onStarted: () => Promise<void> }) {
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const running = job?.pipeline === 'regen' && (job.status === 'queued' || job.status === 'running')
  async function start(mode: 'edit' | 'regenerate') {
    setWorking(true); setError('')
    try {
      await api(`/cuts/${cutId}/regenerate-prompt`, { method: 'POST', body: JSON.stringify({ mode, instruction }) })
      if (mode === 'edit') setInstruction('')
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const failure = job?.pipeline === 'regen' && job.status === 'failed' ? job.error : null
  return <section className="rounded-xl border hairline bg-[#1b202a] p-4 space-y-3" aria-label="Animation prompt">
    <div><h3 className="flex items-center gap-2 text-xs font-semibold"><Sparkles size={14} className="text-[#bca7ff]" /> Animation prompt</h3><p className="muted mt-2 text-[11px] leading-5">Rewrite only the MiniMax animation prompt. Frames are not analyzed again, and the previous version stays in history. Subtitles in the image are used as context only and are kept out of the prompt.</p></div>
    {current ? <p className="max-h-28 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[#333a45] bg-[#151a22] p-2 text-[11px] leading-5">{current.content}</p> : <p className="muted text-[11px]">No animation prompt yet. Regenerate builds one from the available results and context.</p>}
    <label className="block"><span className="label">Instruction</span><textarea className="field min-h-16 resize-y" maxLength={5000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: slower dolly-in, keep the rain, shorten to 6 seconds" /></label>
    {running && <p role="status" className="text-xs text-[#c3aeff]">{job?.stage || 'Waiting for the inference worker'}…</p>}
    {(error || failure) && <p role="alert" className="text-xs text-[#ff9fa7]">{error || failure}</p>}
    <div className="flex gap-2">
      <button className="btn btn-primary flex-1" disabled={working || disabled || running || !current || !instruction.trim()} onClick={() => start('edit')} title="Edit the current prompt following your instruction"><Sparkles size={14} /> Edit current</button>
      <button className="btn btn-quiet flex-1" disabled={working || disabled || running} onClick={() => start('regenerate')} title="Write a new prompt from the available results and context; your instruction is used as direction"><RefreshCw size={14} /> Regenerate</button>
    </div>
    <p className="muted text-[11px]">Sends the cut's results and selected context to the OpenRouter model chosen in Run settings.</p>
  </section>
}
