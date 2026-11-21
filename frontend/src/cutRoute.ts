// The editor's address: one URL per episode and per cut, so a link can open a cut (and the place on it) directly.
//
//   #/episode/<episode id>
//   #/episode/<episode id>/cut/<cut id>                         the cut selected, video at its first frame
//   #/episode/<episode id>/cut/<cut id>/analysis                Analysis dialog, Results tab (review, answers, summary)
//   #/episode/<episode id>/cut/<cut id>/analysis/run            Run tab
//   #/episode/<episode id>/cut/<cut id>/analysis/characters     Run tab, Characters in this cut (first-frame boxes)
//   #/episode/<episode id>/cut/<cut id>/analysis/target-frames  Run tab, Target frames
//   #/episode/<episode id>/cut/<cut id>/analysis/assistant      Cut Assistance
//
// Cut ids are UUIDs and never change (redetection keeps them), so a link stays valid. scripts/fw builds the same
// addresses (cut_url); keep the two in step.

export type DialogTab = 'results' | 'run' | 'assistant'
export type RunSection = 'characters' | 'targets'

export type EditorRoute = { episodeId: string; cutId: string | null; analysis: boolean; tab: DialogTab; section: RunSection }

const SEGMENT = /^[A-Za-z0-9_-]+$/
const DEFAULT: Pick<EditorRoute, 'tab' | 'section'> = { tab: 'results', section: 'characters' }

/** `route` is the hash without its `#` (`/episode/…`). Anything that is not an episode address gives null. */
export function parseEditorRoute(route: string): EditorRoute | null {
  const parts = route.split('?')[0].split('/').filter(Boolean)
  if (parts[0] !== 'episode' || !parts[1] || !SEGMENT.test(parts[1])) return null
  const base: EditorRoute = { episodeId: parts[1], cutId: null, analysis: false, ...DEFAULT }
  if (parts[2] !== 'cut' || !parts[3] || !SEGMENT.test(parts[3])) return base
  base.cutId = parts[3]
  if (parts[4] !== 'analysis') return base
  base.analysis = true
  switch (parts[5]) {
    case 'run': return { ...base, tab: 'run' }
    case 'characters': return { ...base, tab: 'run', section: 'characters' }
    case 'target-frames': case 'targets': return { ...base, tab: 'run', section: 'targets' }
    case 'assistant': return { ...base, tab: 'assistant' }
    default: return base
  }
}

/** The canonical route (with its leading slash, without `#`) for the editor's state. */
export function editorPath(state: { episodeId: string; cutId?: string | null; analysis?: boolean; tab?: DialogTab; section?: RunSection }): string {
  const episode = `/episode/${state.episodeId}`
  if (!state.cutId) return episode
  const cut = `${episode}/cut/${state.cutId}`
  if (!state.analysis) return cut
  if (state.tab === 'run') return `${cut}/analysis/${state.section === 'targets' ? 'target-frames' : 'characters'}`
  return state.tab === 'assistant' ? `${cut}/analysis/assistant` : `${cut}/analysis`
}

/** The tab a freshly opened dialog starts on: Run for a cut that has no results yet, otherwise Results. */
export function defaultTab(cut: { active: boolean; has_analysis_results?: boolean }): DialogTab {
  return cut.active && cut.has_analysis_results === false ? 'run' : 'results'
}

/** An old-style path address (`/episode/<id>/cut/<id>`, served by the API for pasted links) as a route. */
export function routeFromLocation(hash: string, pathname: string): string {
  if (hash.length > 1) return hash.slice(1)
  return /^\/(project|episode)\//.test(pathname) ? pathname : '/'
}
