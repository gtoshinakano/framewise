import { useEffect, useState } from 'react'
import { Layers3, Play, Save } from 'lucide-react'
import type { Chunk } from './api'
import { PANEL_ACCENT } from './InspectorRail'
import { formatDuration, formatTimecode } from './timecode'

/** The Chunk panel: a working segment of a long video with its own notes. */
export function ChunkInspector({ chunk, onPlay, onSave }: { chunk?: Chunk; onPlay: (start: number, end: number) => void; onSave: (chunk: Chunk) => Promise<void> }) {
  const [draft, setDraft] = useState<Chunk | undefined>(chunk)
  const [saving, setSaving] = useState(false)
  useEffect(() => { setDraft(chunk) }, [chunk?.id, chunk?.title, chunk?.notes])
  if (!chunk || !draft) return <div className="p-8 text-center">
    <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[#1f3236]" style={{ color: PANEL_ACCENT.chunk }}><Layers3 size={21} /></div>
    <h2 className="text-sm font-semibold">Select a chunk</h2><p className="muted mt-2 text-xs leading-5">Click a block in the Chunks lane of the timeline to add notes to that working segment.</p>
  </div>
  const dirty = draft.title !== chunk.title || draft.notes !== chunk.notes
  async function save() { if (!draft) return; setSaving(true); try { await onSave(draft) } finally { setSaving(false) } }
  return <div className="flex h-full min-h-0 flex-col">
    <div className="scrollbar min-h-0 flex-1 overflow-y-auto p-4">
      <p className="text-xs font-semibold uppercase tracking-[.14em]" style={{ color: PANEL_ACCENT.chunk }}>Chunk {chunk.chunk_index + 1}</p>
      <h2 className="mt-1 font-mono text-[15px] font-semibold tabular-nums">{formatTimecode(chunk.start_time)} – {formatTimecode(chunk.end_time)} <span className="ml-1 rounded-md bg-[#1f3236] px-1.5 py-0.5 align-middle font-sans text-[10px] text-[#a6e0e4]">{formatDuration(chunk.end_time - chunk.start_time)}</span></h2>
      <p className="muted mt-1 text-xs">A working segment for long-video analysis.</p>
      <label className="mt-5 block"><span className="label">Title</span><input className="field !py-2" value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="Name this chunk" /></label>
      <label className="mt-4 block"><span className="label">Working notes</span><textarea className="field min-h-36 resize-y" value={draft.notes} onChange={event => setDraft({ ...draft, notes: event.target.value })} placeholder="Chapter happenings, context, or observations…" /></label>
    </div>
    <div className="flex gap-2 border-t hairline bg-[#171b24] px-4 py-2.5">
      <button onClick={() => onPlay(chunk.start_time, chunk.end_time)} className="btn btn-quiet flex-1 !py-2"><Play size={14} /> Play chunk</button>
      <button onClick={() => void save()} disabled={!dirty || saving} className="btn btn-primary flex-1 !py-2"><Save size={14} /> Save</button>
    </div>
  </div>
}
