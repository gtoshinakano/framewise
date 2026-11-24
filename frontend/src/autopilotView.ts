// Pure helpers of the Autopilot profile dialog (backend autopilot_profile.py, docs/autopilot/HARNESS.md).

export type Mode = 'apply' | 'ask' | 'off'
export type WhenUnsure = 'evidence' | 'recommended' | 'ask'
export type Relevance = 'auto' | 'always' | 'background'
export type TopicRule = { rule: string; when_unsure: WhenUnsure }
export type TopicStat = { asked: number; answered: number; recommended: number }
export type AutopilotProfile = {
  version: number
  modes: Record<string, Mode>
  claude: { enabled: boolean; max_items_per_run: number }
  topics: Record<string, TopicRule>
  characters: Record<string, { relevance: Relevance }>
  look_alikes: [string, string][]
  notes: string
}
export type ProfileResponse = {
  profile: AutopilotProfile
  updated_at: string | null
  decisions: Record<string, string>
  topics: string[]
  topic_stats: Record<string, TopicStat>
}

/** What each consistency-question topic covers, and an example of a standing rule the user might write for it. */
export const TOPIC_HELP: Record<string, { label: string; covers: string; example: string }> = {
  speaking: { label: 'Speaking', covers: 'who speaks, whether the mouth moves with the line, when a line starts', example: 'A line said with the face turned away is inner voice: no lip sync.' },
  pose: { label: 'Pose', covers: 'body pose, posture, limbs', example: 'Someone behind a desk counts as sitting unless they visibly stand up.' },
  state: { label: 'State', covers: 'sitting, standing, walking at the start or end of the cut', example: 'When the cut starts mid-step, call it walking, not standing.' },
  gaze: { label: 'Gaze', covers: 'where a character looks', example: 'Eyes hidden by hair: follow the head direction.' },
  emotion: { label: 'Emotion', covers: 'expression and feeling', example: 'Prefer the milder emotion when the face is small or blurred.' },
  identity: { label: 'Identity', covers: 'who a person is', example: 'A small child at the Azusagawa home is Kaede unless the cast says otherwise.' },
  position: { label: 'Position', covers: 'where someone is in the frame or the room', example: '' },
  camera: { label: 'Camera', covers: 'framing and camera moves', example: 'A slow blur at the edges marks a flashback, not a camera move.' },
  objects: { label: 'Objects', covers: 'props and what characters hold', example: '' },
  other: { label: 'Other', covers: 'anything else', example: '' },
}

export const MODE_OPTIONS: { value: Mode; label: string }[] = [
  { value: 'apply', label: 'Apply' }, { value: 'ask', label: 'Ask me' }, { value: 'off', label: 'Off' },
]
/** Decisions that never ask (backend autopilot_profile.NEVER_ASK): the missed-cut checker and the line splitter apply or leave. */
export const NEVER_ASK = ['missed_cuts', 'line_splits']

/** The modes one decision kind offers; a stored "ask" of a never-ask kind shows as Apply, as the server reads it. */
export function modesFor(kind: string) {
  return NEVER_ASK.includes(kind) ? MODE_OPTIONS.filter(option => option.value !== 'ask') : MODE_OPTIONS
}

export function modeOf(kind: string, mode: Mode | undefined): Mode {
  return mode === 'ask' && NEVER_ASK.includes(kind) ? 'apply' : mode ?? 'apply'
}
export const UNSURE_OPTIONS: { value: WhenUnsure; label: string }[] = [
  { value: 'evidence', label: 'Follow the evidence' }, { value: 'recommended', label: 'Take the recommended option' }, { value: 'ask', label: 'Ask me' },
]
export const RELEVANCE_OPTIONS: { value: Relevance; label: string }[] = [
  { value: 'auto', label: 'Auto' }, { value: 'always', label: 'Always matters' }, { value: 'background', label: 'Background' },
]

/** "181 asked · 57 answered · you took the recommended option 68% of the time" (only what is known). */
export function statText(stat: TopicStat | undefined): string {
  if (!stat || !stat.asked) return 'no questions yet'
  const parts = [`${stat.asked} asked`, `${stat.answered} answered`]
  if (stat.answered) parts.push(`you took the recommended option ${Math.round((100 * stat.recommended) / stat.answered)}% of the time`)
  return parts.join(' · ')
}

export function relevanceOf(profile: AutopilotProfile, characterId: string): Relevance {
  return profile.characters[characterId]?.relevance ?? 'auto'
}

/** The profile with one character's relevance set; `auto` is the default and is left out, as the backend stores it. */
export function withRelevance(profile: AutopilotProfile, characterId: string, relevance: Relevance): AutopilotProfile {
  const characters = { ...profile.characters }
  if (relevance === 'auto') delete characters[characterId]
  else characters[characterId] = { relevance }
  return { ...profile, characters }
}

/** Pairs are unordered and unique; a pair of one character with itself is ignored. */
export function addPair(pairs: [string, string][], first: string, second: string): [string, string][] {
  if (!first || !second || first === second) return pairs
  const pair = [first, second].sort() as [string, string]
  return pairs.some(([a, b]) => a === pair[0] && b === pair[1]) ? pairs : [...pairs, pair]
}

export function removePair(pairs: [string, string][], index: number): [string, string][] {
  return pairs.filter((_, i) => i !== index)
}

/** Characters for the list: main characters first, then by name. */
export function orderCharacters<T extends { title: string; is_main?: number }>(items: T[]): T[] {
  return [...items].sort((a, b) => (b.is_main ?? 0) - (a.is_main ?? 0) || a.title.localeCompare(b.title, undefined, { sensitivity: 'base' }))
}

export function sameProfile(a: AutopilotProfile | null, b: AutopilotProfile | null): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}
