import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, Minus, Plus } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisConfig, type Cut, type VisualStage } from './api'
import { analysisFrameCount, estimatedVideoFrames } from './analysisFrames'
import { CharacterDetectionPanel } from './CharacterDetectionPanel'
import type { Job } from './api'

function validVideoSettings(config: AnalysisConfig) {
  return Number.isInteger(config.video_width) && config.video_width >= 128 && config.video_width <= 1920 && config.video_width % 2 === 0 &&
    Number.isInteger(config.video_height) && config.video_height >= 128 && config.video_height <= 1080 && config.video_height % 2 === 0 &&
    Number.isFinite(config.video_fps) && config.video_fps > 0 && config.video_fps <= 30
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
  function setFrameCount(count: number) {
    setConfig(current => current && { ...current, frame_count: count, frame_mode: 'uniform' })
  }
  async function save(run: 'none' | 'cut' | 'next' | 'batch') {
    if (!config || working) return
    if (!validVideoSettings(config)) { setError('Enter even dimensions within the listed limits and a frame rate greater than 0 and at most 30.'); return }
    setWorking(true); setError(''); setMessage('')
    try {
      await api(`/cuts/${cutId}/analysis-config`, { method: 'PUT', body: JSON.stringify(config) })
      if (run !== 'none') {
        await api(run === 'batch' ? `/episodes/${episodeId}/visual-analysis` : `/cuts/${cutId}/analyze`, { method: 'POST' })
        setMessage(run === 'batch' ? 'Visual analysis queued for enabled cuts.' : 'Visual analysis queued.')
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
  const count = config ? analysisFrameCount(config, cut) : 1
  const unavailable = !config || !context || working || busy || !enabled || !validVideoSettings(config)
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4 space-y-4">
    <div><h3 className="text-sm font-semibold">Prepare this cut</h3><p className="muted mt-1 text-[11px]">Review the frames, select the context, then queue and continue.</p></div>
    {config && context && <>
      <div>
        <span className="label" id="frame-count-label">Subtitle reference frames</span>
        <div className="flex flex-wrap items-center gap-2" aria-labelledby="frame-count-label">
          <button type="button" className="btn btn-quiet px-3 py-2" aria-label="Remove one frame" disabled={count <= 1} onClick={() => setFrameCount(count - 1)}><Minus size={15} /></button>
          <output className="min-w-8 text-center text-sm font-semibold tabular-nums" aria-live="polite">{count}</output>
          <button type="button" className="btn btn-quiet px-3 py-2" aria-label="Add one frame" onClick={() => setFrameCount(count + 1)}><Plus size={15} /></button>
          <button type="button" className="btn btn-quiet text-xs" disabled={config.frame_mode === 'per_second'} onClick={() => setConfig({ ...config, frame_mode: 'per_second' })}>Reset to 1 per second</button>
        </div>
        <p className="muted mt-2 text-[11px]">{config.frame_mode === 'per_second' ? 'Default: one frame per second of cut duration, rounded up. ' : 'Custom frame count. '}{count === 1 ? 'Selects the first frame.' : 'Selected frames are evenly spaced from first to last.'} The first and last frames are also checked for subtitles.</p>
      </div>

      <fieldset><legend className="label">Video analysis</legend><div className="grid grid-cols-3 gap-2">
        <label><span className="muted text-[11px]">Width · px</span><input className="field" type="number" min={128} max={1920} step={2} value={config.video_width} onChange={event => setConfig({ ...config, video_width: event.target.valueAsNumber })} /></label>
        <label><span className="muted text-[11px]">Height · px</span><input className="field" type="number" min={128} max={1080} step={2} value={config.video_height} onChange={event => setConfig({ ...config, video_height: event.target.valueAsNumber })} /></label>
        <label><span className="muted text-[11px]">Frames per second</span><input className="field" type="number" min={0.000001} max={30} step="any" value={config.video_fps} onChange={event => setConfig({ ...config, video_fps: event.target.valueAsNumber, video_frame_budget: null })} /></label>
      </div><p className="muted mt-2 text-[11px]" aria-live="polite">Estimated video frames: {estimatedVideoFrames(config, cut)}{config.video_frame_budget ? ` (maximum ${config.video_frame_budget}, including padding)` : ' (including padding; actual count may vary)'}</p><button type="button" className="btn btn-quiet mt-2" onClick={() => setConfig({ ...config, video_fps: Math.min(30, 56 / Math.max(.000001, cut.end_time - cut.start_time)), video_frame_budget: 60 })}>Reset to 60-frame maximum</button><p className="muted mt-1 text-[11px]">The cut is scaled to fit this canvas. These settings apply when queuing one cut.</p></fieldset>
    </>}
    {context && <CharacterDetectionPanel cutId={cutId} projectId={context.project.id} jobs={jobs} onStarted={onStarted} />}
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
    <div className="space-y-2 rounded border hairline p-3"><label className="block"><span className="label">Run one visual stage</span><select className="field" value={selectedStage} onChange={event => setSelectedStage(event.target.value as VisualStage)}><option value="subtitle_ocr">Subtitle detection and OCR</option><option value="initial_frame">Initial frame</option><option value="video_analysis">Video analysis</option><option value="last_frame">Last frame</option></select></label><p className="muted text-[11px]">Runs only this stage. Frame and video stages use subtitle regions saved for the current cut timing when queued; otherwise they use none. Subtitle-only runs do not classify language.</p><button className="btn btn-quiet" disabled={unavailable} onClick={() => void queueStage()}>Queue selected stage</button></div>
    {config && <div className="space-y-2 rounded border hairline p-3"><strong className="text-xs">Episode processing</strong><p className="muted text-[11px]">First queue visual analysis, then queue text processing after visual jobs finish. Text processing creates a consistency summary and then a cut summary for each ready cut.</p><div className="flex flex-wrap gap-2"><button className="btn btn-quiet" disabled={working || busy || !context} onClick={() => void save('batch')}>Queue all visual analysis</button><button className="btn btn-quiet" disabled={working || busy || !context} onClick={() => void queueAllText()}>Queue all text processing</button></div></div>}
    <button className="btn btn-quiet" disabled={working || !context} onClick={() => void save('none')}>Save preset</button>
  </div>
}
