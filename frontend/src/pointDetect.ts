import { api } from './api'
import type { Box } from './targetView'

export type Point = [number, number]
/** Which frame was double-clicked: a first-frame run's `position`, or a Target frames run's `frame_ordinal`. */
export type PointTarget = { surface: 'first'; run_id: string; position: 'first' | 'last' } | { surface: 'target'; run_id: string; frame_ordinal: number }

export const NOTHING_FOUND = 'No character found there. Draw the box instead.'

/** The box of the person under `point` (frame pixels), or null when the detector sees nobody there. Nothing is saved. */
export async function detectAt(target: PointTarget, point: Point): Promise<Box | null> {
  const result = await api<{ box: Box | null }>('/point-detections', { method: 'POST', body: JSON.stringify({ ...target, point: point.map(Math.round) }) })
  return result.box
}
