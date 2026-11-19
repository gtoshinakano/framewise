import { createContext, useContext, useEffect, useRef, useState } from 'react'

export const EpisodeEventsContext = createContext<string | undefined>(undefined)

type Revisions = { version: string; queue: string; jobs?: number; data?: number; stale?: number }
type Change = { queue: boolean; jobs: boolean; data: boolean; stale: boolean }
type Listener = (change: Change) => void

const RECONNECT_REFRESH_MS = 12000
const ALL_CHANGED: Change = { queue: true, jobs: true, data: true, stale: true }
const feeds = new Map<string | null, { listeners: Set<Listener>; close: () => void }>()

function openFeed(episodeId: string | null) {
  const path = episodeId === null ? '/api/events' : `/api/episodes/${encodeURIComponent(episodeId)}/events`
  const source = new EventSource(path)
  const listeners = new Set<Listener>()
  let previous: Revisions | undefined
  let fallback: number | undefined
  const stopFallback = () => {
    window.clearTimeout(fallback)
    fallback = undefined
  }
  const refreshWhileDisconnected = () => {
    fallback = undefined
    if (source.readyState === EventSource.OPEN) return
    for (const listener of listeners) listener(ALL_CHANGED)
    fallback = window.setTimeout(refreshWhileDisconnected, RECONNECT_REFRESH_MS)
  }
  source.onopen = stopFallback
  source.onerror = () => {
    if (fallback === undefined) fallback = window.setTimeout(refreshWhileDisconnected, RECONNECT_REFRESH_MS)
  }
  source.addEventListener('changed', event => {
    const next: Revisions = JSON.parse(event.data)
    if (next.version === previous?.version) return
    const change = previous ? {
      queue: next.queue !== previous.queue, jobs: next.jobs !== previous.jobs,
      data: next.data !== previous.data, stale: next.stale !== previous.stale,
    } : ALL_CHANGED
    previous = next
    for (const listener of listeners) listener(change)
  })
  return { listeners, close() { stopFallback(); source.close() } }
}

/** One EventSource for all components in an episode, or one host-wide feed for overview screens. */
export function subscribeEpisodeEvents(scope: string | null, listener: Listener) {
  const feed = feeds.get(scope) || openFeed(scope)
  feeds.set(scope, feed)
  feed.listeners.add(listener)
  return () => {
    feed.listeners.delete(listener)
    if (!feed.listeners.size) {
      feed.close()
      feeds.delete(scope)
    }
  }
}

export function useEpisodeEvents(onChange: Listener, episodeId?: string | null) {
  const inherited = useContext(EpisodeEventsContext)
  const scope = episodeId === undefined ? inherited ?? null : episodeId
  const latest = useRef(onChange)
  latest.current = onChange
  useEffect(() => subscribeEpisodeEvents(scope, change => latest.current(change)), [scope])
}

export function useEpisodeRevision(fields: (keyof Change)[], enabled = true) {
  const [revision, setRevision] = useState(0)
  useEpisodeEvents(change => {
    if (enabled && fields.some(field => change[field])) setRevision(value => value + 1)
  })
  return revision
}
