import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react'
import { api } from './api'
import { NOTHING_FOUND, type Point } from './pointDetect'
import type { Box } from './targetView'

export type CapturedFrame = { canvas: HTMLCanvasElement; blob: Blob; time: number }
export type RegistrationFrame = { url: string; size: [number, number]; box?: Box }

function jpeg(canvas: HTMLCanvasElement, quality = 0.88): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob)
    else reject(new Error('Could not capture the video frame.'))
  }, 'image/jpeg', quality))
}

export async function captureFrame(video: HTMLVideoElement): Promise<CapturedFrame> {
  video.pause()
  if (video.seeking || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => finish(new Error('The video frame is still loading.')), 4000)
      function finish(error?: Error) {
        window.clearTimeout(timer)
        video.removeEventListener('seeked', check)
        video.removeEventListener('loadeddata', check)
        if (error) reject(error)
        else resolve()
      }
      function check() {
        if (!video.seeking && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) finish()
      }
      video.addEventListener('seeked', check)
      video.addEventListener('loadeddata', check)
    })
  }
  const canvas = document.createElement('canvas')
  const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight))
  canvas.width = Math.round(video.videoWidth * scale)
  canvas.height = Math.round(video.videoHeight * scale)
  const context = canvas.getContext('2d')
  if (!context || !canvas.width || !canvas.height) throw new Error('The video preview is unavailable.')
  const time = video.currentTime
  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  const blob = await jpeg(canvas)
  return { canvas, blob, time }
}

export async function cropFrame(frame: CapturedFrame, box: Box) {
  const [x1, y1, x2, y2] = box
  const canvas = document.createElement('canvas')
  const scale = Math.min(1, 512 / Math.max(x2 - x1, y2 - y1))
  canvas.width = Math.max(1, Math.round((x2 - x1) * scale))
  canvas.height = Math.max(1, Math.round((y2 - y1) * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Could not crop the selected character.')
  context.drawImage(frame.canvas, x1, y1, x2 - x1, y2 - y1, 0, 0, canvas.width, canvas.height)
  return jpeg(canvas)
}

export function useFrameUrl(frame?: CapturedFrame) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    if (!frame) return
    const next = URL.createObjectURL(frame.blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [frame])
  return url
}

const bounds = (a: Point, b: Point): Box => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]

export function CharacterRegistrationFrame({ frame, episodeId, image, box, onBox }: {
  frame?: CapturedFrame; episodeId?: string; image: RegistrationFrame; box: Box | null; onBox: (box: Box | null) => void
}) {
  const start = useRef<Point | null>(null)
  const pending = useRef<AbortController | null>(null)
  const [draw, setDraw] = useState<Box | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [error, setError] = useState('')
  const [width, height] = image.size
  const shown = draw || box
  useEffect(() => () => pending.current?.abort(), [])

  function pointAt(event: MouseEvent<SVGSVGElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect()
    return [Math.round(Math.max(0, Math.min(width, (event.clientX - rect.left) / rect.width * width))),
      Math.round(Math.max(0, Math.min(height, (event.clientY - rect.top) / rect.height * height)))]
  }
  function finish(event: PointerEvent<SVGSVGElement>) {
    if (!start.current) return
    const next = bounds(start.current, pointAt(event))
    start.current = null; setDraw(null)
    if (next[2] - next[0] >= 4 && next[3] - next[1] >= 4) {
      pending.current?.abort(); setDetecting(false); setError(''); onBox(next)
    }
  }
  async function detect(event: MouseEvent<SVGSVGElement>) {
    if (!frame || !episodeId) return
    pending.current?.abort()
    const request = new AbortController()
    pending.current = request
    const [x, y] = pointAt(event)
    const body = new FormData()
    body.set('frame', frame.blob, 'frame.jpg')
    body.set('x', String(x)); body.set('y', String(y))
    setDetecting(true); setError('')
    try {
      const result = await api<{ box: Box | null }>(`/episodes/${episodeId}/registration-point`, { method: 'POST', body, signal: request.signal })
      if (!request.signal.aborted) {
        if (result.box) onBox(result.box)
        else setError(NOTHING_FOUND)
      }
    } catch (reason) { if (!request.signal.aborted) setError((reason as Error).message) }
    finally { if (!request.signal.aborted) setDetecting(false) }
  }

  return <div className="space-y-2">
    <div className="relative overflow-hidden rounded-lg bg-black" style={{ aspectRatio: `${width}/${height}` }}>
      <img src={image.url} alt="Frame selected for character registration" draggable={false} className="absolute inset-0 h-full w-full" />
      <svg viewBox={`0 0 ${width} ${height}`} className={`absolute inset-0 h-full w-full ${frame ? 'cursor-crosshair touch-none' : 'pointer-events-none'}`}
        aria-label="Character box" onDoubleClick={event => void detect(event)}
        onPointerDown={event => { if (!frame || event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); start.current = pointAt(event) }}
        onPointerMove={event => { if (start.current) setDraw(bounds(start.current, pointAt(event))) }}
        onPointerUp={finish} onPointerCancel={() => { start.current = null; setDraw(null) }}>
        {shown && <rect x={shown[0]} y={shown[1]} width={shown[2] - shown[0]} height={shown[3] - shown[1]}
          fill="#bca7ff" fillOpacity=".15" stroke="#bca7ff" vectorEffect="non-scaling-stroke" strokeWidth="2" />}
      </svg>
    </div>
    {frame && <div className="flex justify-between gap-2 text-xs"><p className="muted">Drag a box or double-click the character.</p>
      {box && <button type="button" className="text-[#cbb9ff]" onClick={() => { pending.current?.abort(); setDetecting(false); onBox(null) }}>Clear box</button>}</div>}
    {detecting && <p role="status" className="muted text-xs">Finding the person…</p>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
  </div>
}
