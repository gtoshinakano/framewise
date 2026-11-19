import { useEffect, useMemo, useRef, useState } from 'react'
import { FrameImage } from './FrameImage'
import { Download, Eye, EyeOff, ScanEye, X } from 'lucide-react'
import { cutFrameUrl, type AnalysisResult, type Cut } from './api'

type Detection = { type: string; coords: number[] }
type Frame = { id: string; time: string; size: [number, number]; detections: Record<string, Detection[]> }

const palette = ['#ff6b8b', '#4cc9f0', '#ffd166', '#7bdcb2', '#c3aeff', '#ff9f5a', '#8ecae6', '#f28482']

function parseFrames(results: AnalysisResult[]): Frame[] {
  const frames: Frame[] = []
  for (const result of results) {
    const match = /^Object detections at ([\d.]+)s$/.exec(result.title)
    if (!match) continue
    try {
      const value = JSON.parse(result.content)
      if (Array.isArray(value.image_size) && value.detections && typeof value.detections === 'object') frames.push({ id: result.id, time: match[1], size: value.image_size, detections: value.detections })
    } catch { /* the result was hand-edited into something unreadable; skip its layers */ }
  }
  return frames.sort((a, b) => Number(a.time) - Number(b.time))
}

export function DetectionPreview({ cut, results, frameVersion }: { cut: Cut; results: AnalysisResult[]; frameVersion: string }) {
  const frames = useMemo(() => parseFrames(results), [results])
  const [index, setIndex] = useState(0)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [focus, setFocus] = useState<string | null>(null)
  const [failed, setFailed] = useState<Set<string>>(new Set())
  useEffect(() => setFailed(new Set()), [cut.id, frameVersion])
  const selected = Math.min(index, frames.length - 1)
  const frame = frames[selected]
  const labels = frame ? Object.keys(frame.detections) : []
  const color = (label: string) => palette[labels.indexOf(label) % palette.length]
  const toggle = (label: string) => setHidden(current => { const next = new Set(current); if (next.has(label)) next.delete(label); else next.add(label); return next })
  if (!frame) return <div className="mx-auto w-full max-w-3xl space-y-2"><div className="overflow-hidden rounded-lg border border-[#343b49] bg-black"><FrameImage src={cutFrameUrl(cut, 'first', frameVersion)} alt="First frame of the cut" className="aspect-video w-full object-contain" /></div><a href={cutFrameUrl(cut, 'first', frameVersion)} download={`cut-${cut.cut_index + 1}-first.jpg`} className="btn btn-quiet"><Download size={14} /> Download image</a><p className="muted text-[11px] leading-5">No detections yet. Run the analysis to see detected objects as layers on the frame.</p></div>
  const [width, height] = frame.size
  const stroke = Math.max(2, width / 400)
  const font = Math.max(12, width / 55)
  return <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_17rem] lg:items-start">
    {frames.length > 1 && <div className="flex flex-wrap gap-1 lg:col-span-2" role="tablist" aria-label="Analyzed frames">{frames.map((item, position) => <button key={item.id} role="tab" aria-selected={position === selected} onClick={() => setIndex(position)} className={`rounded-md border px-2 py-1 text-[10px] ${position === selected ? 'border-[#a58ae8] bg-[#322a48]' : 'border-[#333a45] hover:border-[#716585]'}`}>{Number(item.time).toFixed(2)}s</button>)}</div>}
    <div className="min-w-0 space-y-3"><div className="relative overflow-hidden rounded-lg border border-[#343b49] bg-black" style={{ aspectRatio: `${width} / ${height}` }}>
      {failed.has(frame.id) ? <p className="flex h-full items-center justify-center p-4 text-center text-[11px] text-[#98a1af]">Frame unavailable. Check that the original video is available, then try again.</p> : <img src={`/api/analysis-results/${frame.id}/frame?v=${encodeURIComponent(frameVersion)}`} alt={`Analyzed frame at ${frame.time} seconds`} onError={() => setFailed(current => new Set(current).add(frame.id))} className="absolute inset-0 h-full w-full" />}
      <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" aria-label="Detected object layers">
        {labels.filter(label => !hidden.has(label)).map(label => <g key={label} opacity={focus && focus !== label ? .25 : 1}>{frame.detections[label].map((item, position) => item.type === 'point'
          ? <circle key={position} cx={item.coords[0]} cy={item.coords[1]} r={stroke * 3} fill={color(label)} stroke="#000" strokeWidth={stroke / 2} />
          : <g key={position}><rect x={item.coords[0]} y={item.coords[1]} width={item.coords[2] - item.coords[0]} height={item.coords[3] - item.coords[1]} fill={color(label)} fillOpacity={focus === label ? .28 : .12} stroke={color(label)} strokeWidth={stroke} />
            <text x={item.coords[0] + stroke} y={Math.max(font, item.coords[1] - stroke)} fontSize={font} fontWeight="700" fill={color(label)} stroke="#000" strokeWidth={font / 6} paintOrder="stroke">{label}</text></g>)}</g>)}
      </svg>
    </div>
    {!failed.has(frame.id) && <a href={`/api/analysis-results/${frame.id}/frame?v=${encodeURIComponent(frameVersion)}`} download={`cut-${cut.cut_index + 1}-frame-${frame.time}s.jpg`} className="btn btn-quiet"><Download size={14} /> Download image</a>}</div>
    <div>
      <div className="mb-2 flex items-center justify-between"><span className="label !mb-0">Detected layers</span>{labels.length > 0 && <button className="muted text-[11px] hover:text-white" onClick={() => setHidden(hidden.size ? new Set() : new Set(labels))}>{hidden.size ? 'Show all' : 'Hide all'}</button>}</div>
      {labels.length ? <ul className="space-y-1">{labels.map(label => <li key={label}><button onClick={() => toggle(label)} onMouseEnter={() => setFocus(label)} onMouseLeave={() => setFocus(null)} onFocus={() => setFocus(label)} onBlur={() => setFocus(null)} aria-pressed={!hidden.has(label)} className={`flex w-full items-center gap-2 rounded-md border border-[#333a45] bg-[#1b202a] px-2 py-1.5 text-left text-xs hover:border-[#716585] ${hidden.has(label) ? 'opacity-50' : ''}`}><span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: color(label) }} /><span className="min-w-0 flex-1 truncate">{label}</span><span className="muted text-[10px]">{frame.detections[label].length}</span>{hidden.has(label) ? <EyeOff size={13} /> : <Eye size={13} />}</button></li>)}</ul> : <p className="muted text-[11px]">Nothing was detected in this frame. An absent detection does not prove an object is absent.</p>}
    </div>
  </div>
}

/** The file the quick download in the video corner saves: the first analyzed frame, or the cut's first frame before any analysis. */
export function frameDownload(cut: Cut, results: AnalysisResult[], frameVersion: string): { href: string; filename: string } {
  const frame = parseFrames(results)[0]
  const version = encodeURIComponent(frameVersion)
  return frame
    ? { href: `/api/analysis-results/${frame.id}/frame?v=${version}`, filename: `cut-${cut.cut_index + 1}-frame-${frame.time}s.jpg` }
    : { href: cutFrameUrl(cut, 'first', frameVersion), filename: `cut-${cut.cut_index + 1}-first.jpg` }
}

/** Two small buttons laid over a corner of the video: preview the analyzed frames in a dialog, or download one. */
export function FrameTools({ cut, results, frameVersion, onPreview }: { cut: Cut; results: AnalysisResult[]; frameVersion: string; onPreview: () => void }) {
  const target = useMemo(() => frameDownload(cut, results, frameVersion), [cut.id, cut.cut_index, results, frameVersion])
  const button = 'inline-flex rounded p-1.5 text-white/85 hover:bg-white/15 hover:text-white'
  return <div className="absolute right-2 top-2 z-10 flex gap-0.5 rounded-md bg-black/70 p-0.5">
    <button type="button" onClick={onPreview} className={button} title="Preview analyzed frames and detections" aria-label="Preview analyzed frames and detections"><ScanEye size={15} aria-hidden /></button>
    <a href={target.href} download={target.filename} className={button} title="Download analyzed frame" aria-label="Download analyzed frame"><Download size={15} aria-hidden /></a>
  </div>
}

/** Same nesting rules as the run-details dialog: Escape and outside clicks close only this one. */
export function FramesDialog({ cut, results, frameVersion, onClose }: { cut: Cut; results: AnalysisResult[]; frameVersion: string; onClose: () => void }) {
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    return () => opener?.focus()
  }, [])
  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="frames-title" className="flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b hairline bg-[#1b202a] px-5 py-3">
        <div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">{cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut'}</p><h2 id="frames-title" className="text-base font-semibold">Analyzed frames and detections</h2></div>
        <button type="button" onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close frame preview"><X size={18} aria-hidden /></button>
      </header>
      <div className="scrollbar min-h-0 flex-1 overflow-y-auto p-5"><DetectionPreview cut={cut} results={results} frameVersion={frameVersion} /></div>
    </div>
  </div>
}
