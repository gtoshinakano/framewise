/**
 * Stops a `<video>` on the last frame before a time limit, such as the end of a cut.
 *
 * `timeupdate` fires only every 150–250 ms, so pausing from it let playback run on for a few frames
 * (into the next cut). `requestVideoFrameCallback` runs once for every frame that is shown and reports
 * that frame's own time, so the pause lands on the frame whose successor would start at or after the limit.
 */

/** Frame length assumed until two frames have been seen. */
export const DEFAULT_FRAME = 1 / 30
/** Slack for the rounding of presentation times. */
const SLACK = 0.004

export type Verdict = 'continue' | 'last' | 'over'

/** `last`: the frame shown at `mediaTime` is the final one before `limit`; `over`: it already starts at or after it. */
export function judgeFrame(mediaTime: number, frame: number, limit: number): Verdict {
  if (mediaTime >= limit - 0.001) return 'over'
  if (mediaTime + frame >= limit - SLACK) return 'last'
  return 'continue'
}

/** Where to rest the playhead when playback ended past `limit`: inside the last frame before it. */
export function lastFrameTime(limit: number, frame: number): number {
  return Math.max(0, limit - frame / 2)
}

/** Smallest plausible gap between two shown frames. Callbacks can skip frames, so the minimum is kept, never the latest. */
export function nextFrameEstimate(current: number | null, previous: number | null, mediaTime: number): number | null {
  if (previous === null) return current
  const gap = mediaTime - previous
  if (!(gap > 1 / 240 && gap < 0.25)) return current
  return current === null ? gap : Math.min(current, gap)
}

export type PlaybackLimiter = {
  /** The playhead is on the final frame before the limit, or past it. */
  atEnd(): boolean
  dispose(): void
}

type FrameVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number, metadata: { mediaTime: number }) => void) => number
  cancelVideoFrameCallback?: (handle: number) => void
}

/**
 * Pause `video` on the last frame before `limit()` while it plays (a `null` limit does nothing).
 * `restartAt()` is where Play goes when pressed at the end; `onStop` runs after the pause.
 */
export function limitPlayback(
  video: HTMLVideoElement,
  limit: () => number | null,
  options: { restartAt?: () => number | null; onStop?: () => void } = {},
): PlaybackLimiter {
  const source = video as FrameVideo
  let frame: number | null = null
  let previous: number | null = null
  let handle = 0
  const length = () => frame ?? DEFAULT_FRAME

  const onFrame = (_now: number, metadata: { mediaTime: number }) => {
    handle = 0
    frame = nextFrameEstimate(frame, previous, metadata.mediaTime)
    previous = metadata.mediaTime
    const end = limit()
    if (end !== null && !video.paused && judgeFrame(metadata.mediaTime, length(), end) !== 'continue') {
      video.pause()
      options.onStop?.()
      return
    }
    arm()
  }
  const arm = () => {
    if (handle || video.paused || !source.requestVideoFrameCallback) return
    handle = source.requestVideoFrameCallback(onFrame)
  }
  const atEnd = () => {
    const end = limit()
    return end !== null && video.currentTime >= end - length() * 1.5
  }

  const onPlay = () => {
    const restart = options.restartAt?.() ?? null
    if (restart !== null && atEnd()) video.currentTime = restart
    previous = null
    arm()
  }
  // A pause that came too late leaves the video on the next cut's first frame: step back into the last one.
  const onPause = () => {
    const end = limit()
    if (end !== null && video.currentTime >= end - 0.001) video.currentTime = lastFrameTime(end, length())
  }
  // Without frame callbacks, or as a backstop when the tab is throttled, the coarse clock still stops it.
  const onTimeUpdate = () => {
    const end = limit()
    if (end !== null && !video.paused && !video.seeking && video.currentTime >= end) {
      video.pause()
      options.onStop?.()
    }
  }
  const onSeeking = () => { previous = null }

  video.addEventListener('play', onPlay)
  video.addEventListener('playing', arm)
  video.addEventListener('pause', onPause)
  video.addEventListener('timeupdate', onTimeUpdate)
  video.addEventListener('seeking', onSeeking)
  arm()
  return {
    atEnd,
    dispose() {
      if (handle) source.cancelVideoFrameCallback?.(handle)
      handle = 0
      video.removeEventListener('play', onPlay)
      video.removeEventListener('playing', arm)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('timeupdate', onTimeUpdate)
      video.removeEventListener('seeking', onSeeking)
    },
  }
}
