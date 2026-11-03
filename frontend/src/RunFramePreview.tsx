import { FrameImage } from './FrameImage'
import { useRef, useState } from 'react'
import { Download } from 'lucide-react'
import { mediaUrl, type AnalysisConfig, type Cut } from './api'

export function RunFramePreview({ cut, episodeId, config, frameVersion }: { cut: Cut; episodeId: string; config: AnalysisConfig | null; frameVersion: string }) {
  const [index, setIndex] = useState(0)
  const [readyFrame, setReadyFrame] = useState('')
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  if (!config) return <p className="muted text-xs">Loading frame settings…</p>
  const count = config.frame_mode === 'endpoints' ? 2 : Math.max(2, Math.min(32, Math.floor(config.frame_count) || 2))
  const selected = Math.min(index, count - 1)
  // Match inference_worker.sample_times, including its end-of-cut adjustment.
  const first = cut.start_time
  const last = Math.max(first, cut.end_time - Math.min(0.000001, (cut.end_time - first) / 4))
  const time = first + (last - first) * selected / (count - 1)
  const endpoint = selected === 0 ? 'first' : selected === count - 1 ? 'last' : null
  const imageUrl = endpoint ? `/api/cuts/${cut.id}/frame/${endpoint}?v=${encodeURIComponent(frameVersion)}` : null
  const frameKey = `${time}-${frameVersion}`
  const ready = readyFrame === frameKey
  const filename = `cut-${cut.cut_index + 1}-frame-${time.toFixed(3)}s.jpg`
  function downloadVideoFrame() {
    const element = video.current
    if (!element || !element.videoWidth || !ready) return
    const canvas = document.createElement('canvas')
    canvas.width = element.videoWidth; canvas.height = element.videoHeight
    const context = canvas.getContext('2d')
    if (!context) { setError('Could not download this frame.'); return }
    context.drawImage(element, 0, 0)
    canvas.toBlob(blob => {
      if (!blob) { setError('Could not download this frame.'); return }
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }, 'image/jpeg', .95)
  }
  return <div className="space-y-3">
    <div className="flex flex-wrap gap-1" role="tablist" aria-label="Frames to analyze">{Array.from({ length: count }, (_, position) => {
      const timestamp = first + (last - first) * position / (count - 1)
      return <button key={position} role="tab" aria-selected={position === selected} onClick={() => { setIndex(position); setError('') }} className={`rounded-md border px-2 py-1 text-[10px] ${position === selected ? 'border-[#a58ae8] bg-[#322a48]' : 'border-[#333a45] hover:border-[#716585]'}`}>{position === 0 ? 'First · ' : position === count - 1 ? 'Last · ' : `Frame ${position + 1} · `}{timestamp.toFixed(2)}s</button>
    })}</div>
    <div className="overflow-hidden rounded-lg border border-[#343b49] bg-black">{imageUrl ? <FrameImage src={imageUrl} alt={`${endpoint === 'first' ? 'First' : 'Last'} frame of the cut`} className="aspect-video w-full object-contain" /> : <video key={frameKey} ref={video} src={mediaUrl(episodeId, 'proxy')} muted playsInline preload="auto" className="aspect-video w-full object-contain" aria-label={`Preview frame at ${time.toFixed(2)} seconds`} onLoadedMetadata={event => { setReadyFrame(''); event.currentTarget.currentTime = time }} onSeeked={() => setReadyFrame(frameKey)} onError={() => setError('Video preview is unavailable. Run Prepare to create it.')} />}</div>
    <div className="flex flex-wrap items-center justify-between gap-2"><p className="muted text-[11px]">{imageUrl ? 'Boundary frame' : 'Preview from the prepared video; the analyzed frame may differ slightly.'}</p>{imageUrl ? <a href={imageUrl} download={filename} className="btn btn-quiet"><Download size={14} /> Download image</a> : <button type="button" disabled={!ready} onClick={downloadVideoFrame} className="btn btn-quiet"><Download size={14} /> Download image</button>}</div>
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
  </div>
}
