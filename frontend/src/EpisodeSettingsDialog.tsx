import { useEffect, useRef, useState } from 'react'
import { Save, Settings2, Trash2, X } from 'lucide-react'
import type { Episode } from './api'
import { formatTimecode } from './timecode'

type Settings = Episode['settings']

/** The episode's processing settings (formerly the Setup tab), opened from the episode menu. */
export function EpisodeSettingsDialog({ episode, initial, onSave, onDelete, onClose }: {
  episode: Episode
  initial: Settings
  onSave: (settings: Settings) => Promise<boolean>
  onDelete: () => void
  onClose: () => void
}) {
  const [settings, setSettings] = useState<Settings>(initial)
  const [saving, setSaving] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', key)
    return () => { window.removeEventListener('keydown', key); opener?.focus() }
  }, [onClose])
  async function save() { setSaving(true); try { if (await onSave(settings)) onClose() } finally { setSaving(false) } }
  const number = (key: keyof Settings) => (event: React.ChangeEvent<HTMLInputElement>) => setSettings({ ...settings, [key]: Number(event.target.value) })
  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="episode-settings-title" className="max-h-[92vh] w-full max-w-2xl overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl outline-none">
      <div className="flex items-start justify-between gap-4"><div><h2 id="episode-settings-title" className="flex items-center gap-2 text-base font-semibold"><Settings2 size={17} className="text-[#bca7ff]" aria-hidden /> Processing settings</h2><p className="muted mt-1 text-xs leading-5">Choose a balance of speed and detail. Run Prepare or Detect hard cuts again to apply changes.</p></div><button type="button" onClick={onClose} aria-label="Close processing settings" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button></div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className="block"><span className="label">Preview max width · pixels</span><input className="field" type="number" min="320" max="1920" value={settings.proxy_width} onChange={number('proxy_width')} /><p className="muted mt-1 text-[11px]">Lower values make preparation and playback lighter.</p></label>
        <label className="block"><span className="label">Preview quality · CRF</span><input className="field" type="number" min="18" max="35" value={settings.proxy_crf} onChange={number('proxy_crf')} /><p className="muted mt-1 text-[11px]">Lower keeps more detail and uses more disk. 18–35.</p></label>
        <label className="block"><span className="label">Encoding speed</span><select className="field" value={settings.proxy_preset} onChange={event => setSettings({ ...settings, proxy_preset: event.target.value as Settings['proxy_preset'] })}>{['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'].map(value => <option key={value} value={value}>{value}</option>)}</select><p className="muted mt-1 text-[11px]">Faster presets prepare quickly; slower ones make smaller files.</p></label>
        <label className="block"><span className="label">Frame extraction interval · seconds</span><input className="field" type="number" min="0.5" max="60" step="0.5" value={settings.frame_interval} onChange={number('frame_interval')} /></label>
        <label className="block"><span className="label">Scene change threshold</span><input className="field" type="number" min="0.05" max="0.9" step="0.01" value={settings.scene_threshold} onChange={number('scene_threshold')} /><p className="muted mt-1 text-[11px]">Lower detects more candidate hard cuts. 0.05–0.9.</p></label>
        <label className="block"><span className="label">Chunk duration · seconds</span><input className="field" type="number" min="10" max="3600" step="10" value={settings.chunk_duration} onChange={number('chunk_duration')} /><p className="muted mt-1 text-[11px]">Timeline working segments. 10–3600 seconds.</p></label>
      </div>
      <div className="mt-6 grid gap-4 border-t hairline pt-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <dl className="grid grid-cols-3 gap-3 text-xs"><div><dt className="muted">Duration</dt><dd className="mt-1 font-mono">{formatTimecode(episode.duration, { precision: 0 })}</dd></div><div><dt className="muted">Audio track</dt><dd className="mt-1">{episode.has_audio ? 'Available' : 'None'}</dd></div><div><dt className="muted">Frames</dt><dd className="mt-1">Every {episode.settings.frame_interval}s</dd></div></dl>
        <div className="flex justify-end gap-2"><button className="btn btn-quiet" onClick={onClose}>Cancel</button><button className="btn btn-primary" disabled={saving} onClick={() => void save()}><Save size={15} /> Save settings</button></div>
      </div>
      <div className="mt-6 flex items-center justify-between gap-3 rounded-lg border border-[#5b3439] bg-[#2b191d55] p-3 text-xs"><span><strong className="text-[#ffaaa7]">Danger zone</strong><span className="muted ml-2">Deletes this episode with all its media and analysis.</span></span><button type="button" onClick={onDelete} className="btn btn-danger !px-3 !py-1.5 !text-xs"><Trash2 size={14} /> Delete episode</button></div>
    </div>
  </div>
}
