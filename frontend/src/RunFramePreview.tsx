import { FrameImage } from './FrameImage'
import { useRef, useState } from 'react'
import { Download } from 'lucide-react'
import { mediaUrl, type AnalysisConfig, type Cut } from './api'

function PreviewFrame({ cut, episodeId, position, count, frameVersion }: { cut: Cut; episodeId: string; position: number; count: number; frameVersion: string }) {
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  // Match inference_worker.sample_times, including its end-of-cut adjustment.
  const first = cut.start_time
  const last = Math.max(first, cut.end_time - Math.min(0.000001, (cut.end_time - first) / 4))
  const time = first + (last - first) * position / (count - 1)
  const endpoint = position === 0 ? 'first' : position === count - 1 ? 'last' : null
  const imageUrl = endpoint ? `/api/cuts/${cut.id}/frame/${endpoint}?v=${encodeURIComponent(frameVersion)}` : null
  const label = position === 0 ? 'First' : position === count - 1 ? 'Last' : `Frame ${position + 1}`
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
  return <figure className="min-w-0 overflow-hidden rounded-lg border border-[#343b49] bg-[#1b202a]">
    <div className="bg-black">{imageUrl ? <FrameImage src={imageUrl} alt={`${label} frame of the cut`} className="aspect-video w-full object-contain" /> : <video ref={video} src={mediaUrl(episodeId, 'proxy')} muted playsInline preload="auto" className="aspect-video w-full object-contain" aria-label={`${label} preview at ${time.toFixed(2)} seconds`} onLoadedMetadata={event => { event.currentTarget.currentTime = time }} onSeeked={() => setReady(true)} onError={() => setError('Preview unavailable. Run Prepare.')} />}</div>
    <figcaption className="flex items-center justify-between gap-1 px-2 py-1.5 text-[11px]"><span className="min-w-0 truncate" title={`${label} · ${time.toFixed(2)}s`}>{label} · {time.toFixed(2)}s</span>{imageUrl ? <a href={imageUrl} download={filename} className="shrink-0 rounded p-1 text-[#c3aeff] hover:bg-[#343047]" aria-label={`Download ${label.toLowerCase()} frame`}><Download size={14} /></a> : <button type="button" disabled={!ready} onClick={downloadVideoFrame} className="shrink-0 rounded p-1 text-[#c3aeff] hover:bg-[#343047] disabled:opacity-40" aria-label={`Download ${label.toLowerCase()}`}><Download size={14} /></button>}</figcaption>
    {error && <p role="alert" className="px-2 pb-2 text-xs text-[#ff9fa7]">{error}</p>}
  </figure>
}

export function RunFramePreview({ cut, episodeId, config, frameVersion }: { cut: Cut; episodeId: string; config: AnalysisConfig | null; frameVersion: string }) {
  if (!config) return <p className="muted text-xs">Loading frame settings…</p>
  const count = config.frame_mode === 'endpoints' ? 2 : Math.max(2, Math.min(32, Math.floor(config.frame_count) || 2))
  const positions = Array.from({ length: count }, (_, position) => position)
  return <div className="space-y-2">
    <h3 className="text-xs font-semibold">Frames to analyze</h3>
    <div className="grid grid-cols-2 gap-2">{positions.map(position => <PreviewFrame key={`${position}-${count}-${frameVersion}`} cut={cut} episodeId={episodeId} position={position} count={count} frameVersion={frameVersion} />)}</div>
    <p className="muted text-[11px]">Middle frames are previews from the prepared video; analyzed frames may differ slightly.</p>
  </div>
}
