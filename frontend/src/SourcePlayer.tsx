import { limitPlayback } from './playbackLimit'
import { createContext, useCallback, useContext, useLayoutEffect, useRef, useSyncExternalStore, type ReactNode } from 'react'

export type PlayRange = { start: number; end: number }
export type SourcePlayerOptions = { /** The video's accessible name. */ label?: string; /** How much the browser fetches before play; the sequence workspace wants all of it, a cut preview only the start. */ preload?: 'none' | 'metadata' | 'auto' }

/**
 * One `<video>` of the episode's proxy that the whole sequence workspace shares.
 *
 * The element is created once and never re-created: switching shots only moves the playhead and the
 * allowed range, and switching tabs moves the element between slots without reloading it (the source
 * is never reset, so the browser keeps what it has buffered). Times are episode seconds.
 * Readouts follow it with `subscribe` (every animation frame while it plays) instead of polling.
 */
export class SourcePlayer {
  readonly host = document.createElement('div')
  private readonly video = document.createElement('video')
  private range: PlayRange | null = null
  private stopAt: number | null = null
  private src = ''
  private readonly listeners = new Set<() => void>()
  private frame = 0

  constructor({ label = 'Source video of this sequence', preload = 'auto' }: SourcePlayerOptions = {}) {
    const video = this.video
    video.controls = true
    video.preload = preload
    video.playsInline = true
    video.setAttribute('aria-label', label)
    video.className = 'h-full w-full bg-black'
    this.host.className = 'h-full w-full'
    this.host.append(video)
    video.addEventListener('seeking', () => {
      const range = this.range
      if (!range) return
      if (video.currentTime < range.start - 0.002) video.currentTime = range.start
      else if (video.currentTime > range.end + 0.002) video.currentTime = range.end
    })
    // Stops on the last frame before the end of the playing interval, not a few frames after it.
    limitPlayback(video, () => this.stopAt ?? this.range?.end ?? null, { restartAt: () => this.range?.start ?? null })
    video.addEventListener('pause', () => { this.stopAt = null })
    // While it plays, listeners hear about every frame; otherwise only when the playhead or the state changes.
    const tick = () => {
      this.notify()
      this.frame = !video.paused && !video.ended ? requestAnimationFrame(tick) : 0
    }
    const wake = () => { cancelAnimationFrame(this.frame); tick() }
    for (const name of ['play', 'playing', 'pause', 'ended', 'seeking', 'seeked', 'loadedmetadata', 'emptied'] as const) video.addEventListener(name, wake)
  }

  /** Point the player at the proxy once; later calls with the same address do nothing, so nothing reloads. */
  load(src: string) {
    if (!src || src === this.src) return
    this.src = src
    this.video.src = src
  }

  get loaded() { return !!this.src }

  /** The `<video>` itself, for code that listens to its events or reads it through a ref. */
  get element() { return this.video }

  /** Limit seeking and playing to an interval (a shot, or the whole sequence) and put the playhead at its start when outside it. */
  setRange(range: PlayRange | null, moveToStart = false) {
    this.range = range
    this.stopAt = null
    if (range && (moveToStart || this.video.currentTime < range.start - 0.002 || this.video.currentTime > range.end + 0.002)) this.seek(range.start)
  }

  seek(time: number) {
    const range = this.range
    this.video.currentTime = range ? Math.min(range.end, Math.max(range.start, time)) : Math.max(0, time)
  }

  /** Play from `time` until the end of the range (or of the video). */
  playFrom(time: number) {
    this.seek(time)
    this.stopAt = null
    void this.video.play().catch(() => undefined)
  }

  /** Play from `start` and stop at `end` (both inside the range). */
  playRange(start: number, end: number) {
    this.seek(start)
    this.stopAt = Math.min(end, this.range?.end ?? end)
    void this.video.play().catch(() => undefined)
  }

  /** Stop and release the video when the workspace closes. */
  dispose() {
    cancelAnimationFrame(this.frame)
    this.video.pause()
    this.host.remove()
    this.video.removeAttribute('src')
    this.video.load()
    this.src = ''  // so a later load() (React re-runs effects in development) attaches the source again
  }

  pause() { this.video.pause() }
  time() { return this.video.currentTime }
  get paused() { return this.video.paused }

  /** Call `listener` whenever the playhead or the play state changes; returns the unsubscribe function. */
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private notify() { for (const listener of this.listeners) listener() }
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

/**
 * A value derived from the playhead (time in episode seconds) and the play state. The component re-renders only
 * when the value changes, so `select` should return a primitive: a rounded number, a string, an id.
 */
export function usePlayerValue<T>(select: (time: number, paused: boolean) => T) {
  const player = usePlayer()
  const latest = useRef(select)
  latest.current = select
  const snapshot = useCallback(() => latest.current(player.time(), player.paused), [player])
  return useSyncExternalStore(player.subscribe, snapshot)
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
