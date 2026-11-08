import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisConfig, type Cut, type VisualStage } from './api'
import { estimatedVideoFrames } from './analysisFrames'
import { CharacterDetectionPanel } from './CharacterDetectionPanel'
import { ANSWERS_BATCH_REASONS, SUMMARY_BATCH_REASONS, batchQueueMessage, type BatchQueueResult as TextBatchResult } from './batchQueueMessage'
import type { Job } from './api'

function validVideoSettings(config: AnalysisConfig) {
  return Number.isInteger(config.video_width) && config.video_width >= 128 && config.video_width <= 1920 && config.video_width % 2 === 0 &&
    Number.isInteger(config.video_height) && config.video_height >= 128 && config.video_height <= 1080 && config.video_height % 2 === 0 &&
    Number.isFinite(config.video_fps) && config.video_fps > 0 && config.video_fps <= 30
}

type BatchQueueResult = { queued?: number; skipped?: number; blocked?: { unknown_characters: number[]; not_detected: number[] } }

function cutList(numbers: number[]) {
  return numbers.slice(0, 8).join(', ') + (numbers.length > 8 ? ` and ${numbers.length - 8} more` : '')
}

/** Says what was queued and, for the cuts that were held back, which ones need a person first. */
function fromHereMessage(firstCut: number, result: BatchQueueResult) {
  const unknown = result.blocked?.unknown_characters ?? []
  const undetected = result.blocked?.not_detected ?? []
  const parts = [`Queued ${result.queued ?? 0} cuts from cut ${firstCut}; ${result.skipped ?? 0} already complete were skipped.`]
  if (unknown.length) parts.push(`Not queued, unknown characters: cut ${cutList(unknown)}.`)
  if (undetected.length) parts.push(`Not queued, characters not detected yet: cut ${cutList(undetected)}.`)
  return parts.join(' ')
}

export default function AnalysisControls({ cut, episodeId, busy, enabled, onStarted, onConfigChange, onPrevious, onNext, jobs }: { jobs: Job[]; cut: Cut; episodeId: string; busy: boolean; enabled: boolean; onStarted: () => Promise<void>; onConfigChange: (config: AnalysisConfig | null) => void; onPrevious?: () => void; onNext?: () => void }) {
  const cutId = cut.id
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [selectedStage, setSelectedStage] = useState<VisualStage>('video_analysis')
  useEffect(() => {
    let active = true
    setConfig(null); setContext(null); setError(''); setMessage('')
    Promise.all([api<AnalysisConfig>(`/cuts/${cutId}/analysis-config`), api<AnalysisContext>(`/cuts/${cutId}/analysis-context`)])
      .then(([nextConfig, nextContext]) => { if (active) { setConfig(nextConfig); setContext(nextContext) } })
      .catch(err => { if (active) setError((err as Error).message) })
    return () => { active = false }
  }, [cutId])
  useEffect(() => { onConfigChange(config) }, [config, onConfigChange])
  async function save(run: 'none' | 'cut' | 'next' | 'batch' | 'from-here') {
    if (!config || working) return
    if (!validVideoSettings(config)) { setError('Enter even dimensions within the listed limits and a frame rate greater than 0 and at most 30.'); return }
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      if (run !== 'none') {
        const url = run === 'batch' ? `/episodes/${episodeId}/visual-analysis`
          : run === 'from-here' ? `/episodes/${episodeId}/visual-analysis?from_cut=${cut.cut_index + 1}&characters_resolved=true`
          : `/cuts/${cutId}/analyze`
        const result = await api<BatchQueueResult>(url, { method: 'POST' })
        setMessage(run === 'batch' ? `Visual analysis queued for ${result.queued ?? 0} cuts; ${result.skipped ?? 0} already complete were skipped.`
          : run === 'from-here' ? fromHereMessage(cut.cut_index + 1, result) : 'Visual analysis queued.')
        try { await onStarted() } catch { setMessage('Analysis queued, but the episode could not refresh. Refresh the page to see its status.') }
        if (run === 'next') onNext?.()
      } else setMessage('Settings saved for this cut.')
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function queueConsistencySummary() {
    if (working) return
    setWorking(true); setError(''); setMessage('')
    try {
      if (config) await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      await api(`/cuts/${cutId}/consistency-summary`, { method: 'POST' })
      setMessage('Consistency summary queued.')
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function queueSummariesFromHere() {
    if (working) return
    setWorking(true); setError(''); setMessage('')
    try {
      if (config) await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      const result = await api<{ queued: number }>(`/episodes/${episodeId}/consistency-summaries?from_cut=${cut.cut_index + 1}`, { method: 'POST' })
      setMessage(`Consistency summary queued for ${result.queued} cuts from cut ${cut.cut_index + 1}.`)
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  /** Queues production-decision drafts (saved when finished) or Cut summaries for this cut and every later ready one. */
  async function queueTextFromHere(kind: 'answers' | 'summaries') {
    if (working) return
    setWorking(true); setError(''); setMessage('')
    try {
      if (config) await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      const first = cut.cut_index + 1
      const result = await api<TextBatchResult>(`/episodes/${episodeId}/${kind === 'answers' ? 'review-answers' : 'cut-summaries'}?from_cut=${first}`, { method: 'POST' })
      setMessage(kind === 'answers'
        ? batchQueueMessage('Production decisions draft', first, result, ANSWERS_BATCH_REASONS)
        : batchQueueMessage('Cut summary', first, result, SUMMARY_BATCH_REASONS, 'they already have a Cut summary'))
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function queueStage() {
    if (!config || working) return
    if (!validVideoSettings(config)) { setError('Enter valid video settings before queueing.'); return }
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      await api(`/cuts/${cutId}/analysis-stages/run`, { method: 'POST', body: JSON.stringify({ stage: selectedStage }) })
      setMessage('Analysis stage queued.')
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function queueAllText() {
    if (!config || working) return
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      const settings = await api<Pick<AnalysisConfig, 'text_provider' | 'local_thinking' | 'openrouter_model'>>('/text-generation-settings')
      const result = await api<{ cuts_queued: number; skipped: number }>(`/episodes/${episodeId}/text-processing`, { method: 'POST', body: JSON.stringify(settings) })
      setMessage(`Text processing queued for ${result.cuts_queued} cuts; ${result.skipped} not ready or already queued.`)
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const unavailable = !config || !context || working || busy || !enabled || !validVideoSettings(config)
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4 space-y-4">
    <div><h3 className="text-sm font-semibold">Prepare this cut</h3><p className="muted mt-1 text-[11px]">Review the frames, select the context, then queue and continue.</p></div>
    {config && context && <>
      <fieldset><legend className="label">Video analysis</legend><div className="grid grid-cols-3 gap-2">
        <label><span className="muted text-[11px]">Width · px</span><input className="field" type="number" min={128} max={1920} step={2} value={config.video_width} onChange={event => setConfig({ ...config, video_width: event.target.valueAsNumber })} /></label>
        <label><span className="muted text-[11px]">Height · px</span><input className="field" type="number" min={128} max={1080} step={2} value={config.video_height} onChange={event => setConfig({ ...config, video_height: event.target.valueAsNumber })} /></label>
        <label><span className="muted text-[11px]">Frames per second</span><input className="field" type="number" min={0.000001} max={30} step="any" value={config.video_fps} onChange={event => setConfig({ ...config, video_fps: event.target.valueAsNumber, video_frame_budget: null })} /></label>
      </div><p className="muted mt-2 text-[11px]" aria-live="polite">Estimated video frames: {estimatedVideoFrames(config, cut)}{config.video_frame_budget ? ` (maximum ${config.video_frame_budget}, including padding)` : ' (including padding; actual count may vary)'}</p><button type="button" className="btn btn-quiet mt-2" onClick={() => setConfig({ ...config, video_fps: Math.min(30, 56 / Math.max(.000001, cut.end_time - cut.start_time)), video_frame_budget: 60 })}>Reset to 60-frame maximum</button><p className="muted mt-1 text-[11px]">The cut is scaled to fit this canvas. These settings apply when queuing one cut.</p></fieldset>
    </>}
    {context && <CharacterDetectionPanel cutId={cutId} projectId={context.project.id} jobs={jobs} onStarted={onStarted} noCharacters={!!cut.no_characters} />}
    {config && <label className="block"><span className="label">Additional context for this cut</span><textarea className="field min-h-24 resize-y" maxLength={20000} value={config.additional_context} onChange={event => setConfig({ ...config, additional_context: event.target.value })} onKeyDown={event => {
        if (event.key !== 'Enter' || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.nativeEvent.isComposing) return
        event.preventDefault()
        if (!event.repeat && !unavailable) void save('cut')
      }} /><span className="muted mt-1 block text-[11px]">Local visual models use the first 2,000 characters here and 500 characters per selected context item. Press Ctrl+Enter to queue visual analysis for this cut.</span></label>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}{message && <p role="status" className="text-xs text-[#7bdcb2]">{message}</p>}
    <p className="muted text-[11px]">Queueing this cut confirms its displayed named character boxes for analysis. Correct or clear any uncertain names first.</p>
    {!enabled && <p className="muted text-xs">This cut is disabled for analysis. Enable it in the cut editor to queue it.</p>}
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn btn-primary" disabled={unavailable || !onNext} onClick={() => void save('next')}>Queue visual &amp; next <ArrowRight size={14} /></button>
      <button className="btn btn-quiet" disabled={unavailable} onClick={() => void save('cut')}>Queue visual cut</button>
      <button className="btn btn-quiet" disabled={working || !enabled} onClick={() => void queueConsistencySummary()}>Queue consistency summary</button>
      <button className="btn btn-quiet" disabled={working || !onPrevious} onClick={onPrevious}><ArrowLeft size={14} /> Previous</button>
      <button className="btn btn-quiet" disabled={working || !onNext} onClick={onNext}>Next <ArrowRight size={14} /></button>
    </div>
    <div className="space-y-2 rounded border hairline p-3"><label className="block"><span className="label">Run one visual stage</span><select className="field" value={selectedStage} onChange={event => setSelectedStage(event.target.value as VisualStage)}><option value="initial_frame">Initial frame</option><option value="video_analysis">Video analysis</option><option value="last_frame">Last frame</option></select></label><p className="muted text-[11px]">Runs only this stage. Subtitles are not a cut stage: extract them once for the whole video from the audio tab, and every stage is told their language and where they sit.</p><button className="btn btn-quiet" disabled={unavailable} onClick={() => void queueStage()}>Queue selected stage</button></div>
    {config && <div className="space-y-2 rounded border hairline p-3"><strong className="text-xs">Episode processing</strong><p className="muted text-[11px]">First queue visual analysis, then queue text processing after visual jobs finish. Text processing creates a consistency summary and then a cut summary for each ready cut.</p><div className="flex flex-wrap gap-2"><button className="btn btn-quiet" disabled={working || busy || !context} onClick={() => void save('batch')}>Queue all visual analysis</button><button className="btn btn-quiet" disabled={working || busy || !context || !cut.active} title="Queues first and last frame and video analysis for this cut and every later one whose characters are settled (marked as no characters, or every box assigned). Cuts with unknown or undetected characters are skipped and listed." onClick={() => void save('from-here')}>Queue all possible visual analysis from this cut</button><button className="btn btn-quiet" disabled={working || !context || !cut.active} title="Queues a consistency summary for this cut and every later enabled cut. Nothing is queued unless every one of them already has its initial frame, video, and last frame results." onClick={() => void queueSummariesFromHere()}>Queue all consistency summary from this cut</button><button className="btn btn-quiet" disabled={working || !context || !cut.active} title="Drafts answers to the consistency summary's human review and suggested checks for this cut and every later cut that has a consistency summary and something to decide, and saves each draft as that cut's answers. Cuts that are not ready are skipped and listed. Saved answers are kept in the draft, and answers you edit while a draft waits are not overwritten." onClick={() => void queueTextFromHere('answers')}>Queue production decisions draft from this cut</button><button className="btn btn-quiet" disabled={working || !context || !cut.active} title="Queues a Cut summary for this cut and every later cut that has a consistency summary but no Cut summary yet. Cuts whose production decisions are still being drafted, or that are not ready, are skipped and listed." onClick={() => void queueTextFromHere('summaries')}>Queue Cut Summary from this cut</button><button className="btn btn-quiet" disabled={working || busy || !context} onClick={() => void queueAllText()}>Queue all text processing</button></div></div>}
    <button className="btn btn-quiet" disabled={working || !context} onClick={() => void save('none')}>Save preset</button>
  </div>
}
