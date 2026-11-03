import { useEffect, useState } from 'react'
import type { Cut } from './api'

export function CutFramePair({ cut, onSeek, frameVersion }: { cut: Cut; onSeek: (time: number) => void; frameVersion: string }) {
  const [missing, setMissing] = useState<Record<string, boolean>>({})
  useEffect(() => setMissing({}), [cut.id, frameVersion])
  return <div className="grid grid-cols-2 gap-2">
    {(['first', 'last'] as const).map(position => {
      const time = position === 'first' ? cut.first_frame_time : cut.last_frame_time
      return <button key={position} type="button" onClick={() => onSeek(time)} className="overflow-hidden rounded-lg border border-[#343b49] bg-[#1b202b] text-left hover:border-[#a58ae8]" title={`Seek to ${time.toFixed(2)} seconds`}>
        {missing[position] ? <span className="flex aspect-video items-center justify-center px-2 text-center text-[10px] text-[#98a1af]">Frame unavailable. Check the source video and try again.</span> : <img src={`/api/cuts/${cut.id}/frame/${position}?v=${encodeURIComponent(frameVersion)}`} onError={() => setMissing(current => ({ ...current, [position]: true }))} alt={`${position === 'first' ? 'First' : 'Last'} frame of ${cut.active ? `cut ${cut.cut_index + 1}` : 'archived cut'}`} className="aspect-video w-full object-cover" />}
        <span className="flex justify-between px-2 py-1.5 text-[10px]"><span className="capitalize">{position} frame</span><span className="text-[#98a1af]">{time.toFixed(2)}s</span></span>
      </button>
    })}
  </div>
}
