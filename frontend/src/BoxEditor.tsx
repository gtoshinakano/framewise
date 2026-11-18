import { useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import type { Box } from './targetView'

export type EditorBox = { id: string; box: Box; color: string; label: string }
type Drag = { id: string; handle: string; box: Box; from?: [number, number]; orig?: Box }

const HANDLES: [string, number, number][] = [['nw', 0, 0], ['n', .5, 0], ['ne', 1, 0], ['e', 1, .5], ['se', 1, 1], ['s', .5, 1], ['sw', 0, 1], ['w', 0, .5]]
const CURSORS: Record<string, string> = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' }
const MIN_SIDE = 8

/**
 * The SVG layer over one frame: boxes you can move, resize and remove, and a drawing mode for a person the
 * detector missed. Coordinates are image pixels (`size`), whatever size the frame is shown at. Changes are
 * reported when the pointer is released; the caller saves them.
 */
export function BoxEditor({ size, boxes, editable, drawing, onDrawn, onChange, onRemove, label }: {
  size: [number, number]; boxes: EditorBox[]; editable: boolean; drawing: boolean
  onDrawn: (box: Box) => void; onChange: (id: string, box: Box) => void; onRemove: (id: string) => void; label: string
}) {
  const svg = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [start, setStart] = useState<[number, number] | null>(null)
  const [current, setCurrent] = useState<[number, number] | null>(null)
  const handleSize = Math.max(10, size[0] / 80)

  function point(event: PointerEvent): [number, number] {
    const rect = svg.current!.getBoundingClientRect()
    return [Math.max(0, Math.min(size[0], (event.clientX - rect.left) * size[0] / rect.width)), Math.max(0, Math.min(size[1], (event.clientY - rect.top) * size[1] / rect.height))]
  }
  function moveDrag(event: PointerEvent) {
    if (!drag) return
    const [x, y] = point(event)
    if (drag.handle === 'move' && drag.from && drag.orig) {
      const [x1, y1, x2, y2] = drag.orig
      const dx = Math.max(-x1, Math.min(size[0] - x2, x - drag.from[0]))
      const dy = Math.max(-y1, Math.min(size[1] - y2, y - drag.from[1]))
      setDrag({ ...drag, box: [x1 + dx, y1 + dy, x2 + dx, y2 + dy] })
      return
    }
    const box: Box = [...drag.box]
    if (drag.handle.includes('w')) box[0] = Math.min(x, box[2] - MIN_SIDE)
    if (drag.handle.includes('e')) box[2] = Math.max(x, box[0] + MIN_SIDE)
    if (drag.handle.includes('n')) box[1] = Math.min(y, box[3] - MIN_SIDE)
    if (drag.handle.includes('s')) box[3] = Math.max(y, box[1] + MIN_SIDE)
    setDrag({ ...drag, box })
  }
  function endDrag() {
    const value = drag
    setDrag(null)
    if (value) onChange(value.id, value.box.map(Math.round) as Box)
  }
  function endDrawing(event: PointerEvent) {
    if (!start) return
    const end = point(event)
    setStart(null); setCurrent(null)
    const box: Box = [Math.round(Math.min(start[0], end[0])), Math.round(Math.min(start[1], end[1])), Math.round(Math.max(start[0], end[0])), Math.round(Math.max(start[1], end[1]))]
    if (box[2] - box[0] >= 4 && box[3] - box[1] >= 4) onDrawn(box)
  }
  const grab = (event: PointerEvent<SVGElement>, next: Drag) => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setDrag(next) }

  return <svg ref={svg} viewBox={`0 0 ${size[0]} ${size[1]}`} aria-label={label} className={`absolute inset-0 h-full w-full ${drawing ? 'touch-none cursor-crosshair' : 'pointer-events-none'}`}
    onPointerDown={event => { if (!drawing) return; event.currentTarget.setPointerCapture(event.pointerId); const p = point(event); setStart(p); setCurrent(p) }}
    onPointerMove={event => { if (drawing && start) setCurrent(point(event)) }} onPointerUp={endDrawing} onPointerCancel={() => { setStart(null); setCurrent(null) }}>
    {boxes.map(item => {
      const box = drag?.id === item.id ? drag.box : item.box
      const movable = editable && !drawing
      const r = Math.max(9, size[0] / 70)
      return <g key={item.id}>
        <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill={item.color} fillOpacity=".12" stroke={item.color} strokeWidth="3"
          style={movable ? { pointerEvents: 'all', cursor: 'move', touchAction: 'none' } : undefined}
          onPointerDown={movable ? event => grab(event, { id: item.id, handle: 'move', box: [...item.box] as Box, from: point(event), orig: [...item.box] as Box }) : undefined}
          onPointerMove={movable ? moveDrag : undefined} onPointerUp={movable ? endDrag : undefined} onPointerCancel={movable ? () => setDrag(null) : undefined} />
        <text x={box[0] + 4} y={Math.max(20, box[1] - 4)} fill={item.color} stroke="black" strokeWidth="3" paintOrder="stroke" fontSize="20" fontWeight="700">{item.label}</text>
        {movable && HANDLES.map(([handle, fx, fy]) => <rect key={handle} x={box[0] + (box[2] - box[0]) * fx - handleSize / 2} y={box[1] + (box[3] - box[1]) * fy - handleSize / 2} width={handleSize} height={handleSize}
          fill={item.color} stroke="black" strokeWidth="1.5" style={{ pointerEvents: 'all', cursor: CURSORS[handle], touchAction: 'none' }}
          onPointerDown={event => grab(event, { id: item.id, handle, box: [...item.box] as Box })} onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={() => setDrag(null)} />)}
        {movable && !drag && <g role="button" tabIndex={0} aria-label={`Remove ${item.label}`} style={{ pointerEvents: 'all', cursor: 'pointer', touchAction: 'none' }} onPointerDown={event => event.stopPropagation()}
          onClick={event => { event.stopPropagation(); onRemove(item.id) }} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onRemove(item.id) } }}>
          <title>Remove this box</title>
          <circle cx={Math.min(size[0] - r, box[2] - r)} cy={Math.max(r, box[1] + r)} r={r} fill="#111" fillOpacity=".8" stroke={item.color} strokeWidth="2" />
          <path d={`M${Math.min(size[0] - r, box[2] - r) - r * .42} ${Math.max(r, box[1] + r) - r * .42} l${r * .84} ${r * .84} m0 ${-r * .84} l${-r * .84} ${r * .84}`} stroke="#fff" strokeWidth="2.5" strokeLinecap="round" fill="none" />
        </g>}
      </g>
    })}
    {drawing && start && current && <rect x={Math.min(start[0], current[0])} y={Math.min(start[1], current[1])} width={Math.abs(start[0] - current[0])} height={Math.abs(start[1] - current[1])} fill="none" stroke="white" strokeWidth="3" />}
  </svg>
}
