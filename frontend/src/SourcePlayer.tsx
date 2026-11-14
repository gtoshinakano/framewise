import { createContext, useContext, useLayoutEffect, useRef, type ReactNode } from 'react'

export type PlayRange = { start: number; end: number }

/**
 * One `<video>` of the episode's proxy that the whole sequence workspace shares.
 *
 * The element is created once and never re-created: switching shots only moves the playhead and the
 * allowed range, and switching tabs moves the element between slots without reloading it (the source
 * is never reset, so the browser keeps what it has buffered). Times are episode seconds.
 */
export class SourcePlayer {
  readonly host = document.createElement('div')
  private readonly video = document.createElement('video')
  private range: PlayRange | null = null
  private stopAt: number | null = null
  private src = ''

  constructor() {
    const video = this.video
    video.controls = true
    video.preload = 'auto'
    video.playsInline = true
    video.setAttribute('aria-label', 'Source video of this sequence')
    video.className = 'h-full w-full bg-black'
    this.host.className = 'h-full w-full'
    this.host.append(video)
    video.addEventListener('seeking', () => {
      const range = this.range
      if (!range) return
      if (video.currentTime < range.start) video.currentTime = range.start
      else if (video.currentTime > range.end) video.currentTime = range.end
    })
    video.addEventListener('play', () => {
      const range = this.range
      if (range && video.currentTime >= range.end - 0.05) video.currentTime = range.start
    })
    video.addEventListener('pause', () => { this.stopAt = null })
    video.addEventListener('timeupdate', () => {
      const limit = this.stopAt ?? this.range?.end
      if (limit !== undefined && limit !== null && video.currentTime >= limit && !video.paused) {
        video.pause()
        if (video.currentTime > limit) video.currentTime = limit
      }
    })
  }

  /** Point the player at the proxy once; later calls with the same address do nothing, so nothing reloads. */
  load(src: string) {
    if (!src || src === this.src) return
    this.src = src
    this.video.src = src
  }

  get loaded() { return !!this.src }

  /** Limit seeking and playing to an interval (a shot, or the whole sequence) and put the playhead at its start when outside it. */
  setRange(range: PlayRange | null, moveToStart = false) {
    this.range = range
    this.stopAt = null
    if (range && (moveToStart || this.video.currentTime < range.start || this.video.currentTime > range.end)) this.seek(range.start)
  }

  seek(time: number) {
    const range = this.range
    this.video.currentTime = range ? Math.min(range.end, Math.max(range.start, time)) : Math.max(0, time)
  }

  /** Play from `start` and stop at `end` (both inside the range). */
  playRange(start: number, end: number) {
    this.seek(start)
    this.stopAt = Math.min(end, this.range?.end ?? end)
    void this.video.play().catch(() => undefined)
  }

  /** Stop and release the video when the workspace closes. */
  dispose() {
    this.video.pause()
    this.host.remove()
    this.video.removeAttribute('src')
    this.video.load()
    this.src = ''  // so a later load() (React re-runs effects in development) attaches the source again
  }

  pause() { this.video.pause() }
  time() { return this.video.currentTime }
}

const PlayerContext = createContext<SourcePlayer | null>(null)

export function PlayerProvider({ player, children }: { player: SourcePlayer; children: ReactNode }) {
  return <PlayerContext.Provider value={player}>{children}</PlayerContext.Provider>
}

export function usePlayer() {
  const player = useContext(PlayerContext)
  if (!player) throw new Error('usePlayer needs a PlayerProvider')
  return player
}

/** Where the shared video is shown. Only one slot is mounted at a time; mounting a slot moves the element into it. */
export function PlayerSlot({ className = '', children }: { className?: string; children?: ReactNode }) {
  const player = usePlayer()
  const slot = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const element = slot.current
    if (element && player.host.parentElement !== element) element.prepend(player.host)
  }, [player])
  return <div ref={slot} className={`relative overflow-hidden rounded-xl border hairline bg-black ${className}`}>{children}</div>
}
