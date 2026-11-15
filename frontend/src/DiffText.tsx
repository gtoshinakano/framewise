import { useMemo } from 'react'
import type { DiffSegment } from './assistantApi'
import { compactDiff } from './assistantFormat'

/** A text change shown the way every AI change is shown: insertions in green, deletions struck through in red, long unchanged runs shortened. */
export function DiffText({ segments }: { segments: DiffSegment[] }) {
  const parts = useMemo(() => compactDiff(segments), [segments])
  return <>{parts.map((segment, index) => <span key={index} className={segment.op === 'insert' ? 'diff-insert' : segment.op === 'delete' ? 'diff-delete' : ''}>{segment.text}</span>)}</>
}
