import { WebSearchSources } from './WebSearchSources'
import { AlertTriangle, BookOpen, Bot, Inbox, Shirt, BoxSelect, ChevronDown, ChevronRight, Clapperboard, FolderOpen, ListChecks, LoaderCircle, MoreHorizontal, RefreshCw, RotateCcw, ScanFace, Scissors, Settings2, Trash2, UserSearch, Users } from 'lucide-react'
import { LocalModelControl } from './LocalModelControl'
import { VisualEngineControl } from './VisualEngineControl'
import { Menu, type MenuEntry } from './Popover'
import { attentionText, attentionTitle, counterText, counterTitle, type QueueSummary } from './queueView'

export type CharacterStep = 'character-boxes' | 'character-detection' | 'character-detection-all' | 'character-detection-rerun'

const go = (path: string) => { window.location.hash = path }

/** The episode editor's top bar: where you are, the episode-wide actions grouped in menus, and the queue. */
export function EditorHeader({ projectId, projectTitle, episodeTitle, status, stage, hasCuts, mediaBlocked, characterBlock, onCharacterStep, onDetectCuts, onPrepare, onPrepareEpisode, queue, running, onOpenQueue, inbox, onOpenInbox, onOpenSettings, onOpenGlossary, onOpenOutfits, onOpenMatcher, onEditCharacter, onOpenAutopilot, onDeleteEpisode }: {
  projectId?: string
  projectTitle?: string
  episodeTitle?: string
  status: string
  stage?: string
  hasCuts: boolean
  /** Why media actions are unavailable right now, or empty when they can run. */
  mediaBlocked: string
  /** Why character detection cannot be queued right now, or empty when it can. */
  characterBlock: string
  onCharacterStep: (step: CharacterStep) => void
  onDetectCuts: () => void
  onPrepare: () => void
  onPrepareEpisode: () => void
  /** The episode's queue counter (backend runs.summary); null until the first status response. */
  queue: QueueSummary | null
  /** Anything of this episode running, GPU or media work. */
  running: boolean
  onOpenQueue: (focus?: 'attention') => void
  /** Open autopilot questions: this episode's for the person, and a title naming other episodes with questions. */
  inbox: { mine: number; elsewhere: string }
  onOpenInbox: () => void
  onOpenSettings: () => void
  onOpenGlossary: () => void
  onOpenOutfits: () => void
  onOpenMatcher: () => void
  onEditCharacter: () => void
  onOpenAutopilot: () => void
  onDeleteEpisode: () => void
}) {
  const characterItems: MenuEntry[] = [
    { key: 'all', icon: ScanFace, title: 'Find boxes + recognize', hint: 'Steps 1 and 2 in one pass: boxes for all cuts first, then recognition with the models loaded once.', disabled: !!characterBlock, disabledReason: characterBlock, onSelect: () => onCharacterStep('character-detection-all') },
    'separator',
    { key: 'boxes', icon: BoxSelect, title: '1 · Find character boxes', hint: 'Boxes in the first frame of cuts without detections. Cuts marked as having no characters are skipped.', disabled: !!characterBlock, disabledReason: characterBlock, onSelect: () => onCharacterStep('character-boxes') },
    { key: 'recognize', icon: UserSearch, title: '2 · Recognize unlabeled boxes', hint: 'Compares first-frame boxes with every box you have labelled.', disabled: !!characterBlock, disabledReason: characterBlock, onSelect: () => onCharacterStep('character-detection') },
    'separator',
    { key: 'rerun', icon: RotateCcw, tone: 'caution', title: 'Redo automatic labels…', hint: 'Recognizes every cut again using your confirmed labels as examples. Manual labels and drawn boxes stay as they are.', disabled: !!characterBlock, disabledReason: characterBlock, onSelect: () => onCharacterStep('character-detection-rerun') },
    'separator',
    { key: 'edit', icon: Users, title: 'Edit character…', disabled: !projectId, onSelect: onEditCharacter },
    { key: 'glossary', icon: BookOpen, title: 'Name glossary…', disabled: !projectId, onSelect: onOpenGlossary },
    { key: 'matcher', icon: UserSearch, title: 'Character matcher store…', hint: 'Manage matching images from all episodes.', disabled: !projectId, onSelect: onOpenMatcher },
    { key: 'outfits', icon: Shirt, title: 'Outfits…', hint: 'What each character wears, shared by every episode. The Sequence director fills it.', disabled: !projectId, onSelect: onOpenOutfits },
  ]
  const cutItems: MenuEntry[] = [
    ...(hasCuts ? [{ key: 'prepare-episode', icon: ListChecks, title: 'Prepare episode…', onSelect: onPrepareEpisode }] : []),
    { key: 'redetect', icon: Scissors, tone: hasCuts ? 'caution' : undefined, title: hasCuts ? 'Redetect all cuts…' : 'Detect hard cuts',
      hint: hasCuts ? 'Notes stay on matching cuts. Earlier cut edits can no longer be undone.' : undefined, disabled: !!mediaBlocked, disabledReason: mediaBlocked, onSelect: onDetectCuts },
    { key: 'prepare', icon: RefreshCw, title: 'Rebuild preview & waveform', hint: 'Creates the preview video, frames, audio and waveform again (Prepare).', disabled: !!mediaBlocked, disabledReason: mediaBlocked, onSelect: onPrepare },
  ]
  const episodeItems: MenuEntry[] = [
    { key: 'settings', icon: Settings2, title: 'Processing settings…', hint: 'Preview size, frame interval, scene threshold and chunk length.', onSelect: onOpenSettings },
    { key: 'autopilot', icon: Bot, title: 'Autopilot profile…', hint: 'What agents may apply for this project, how to settle review questions, which characters matter.', disabled: !projectId, onSelect: onOpenAutopilot },
    { key: 'projects', icon: FolderOpen, title: 'All projects', onSelect: () => go('/') },
    'separator',
    { key: 'delete', icon: Trash2, tone: 'danger', title: 'Delete episode…', hint: 'Removes the episode with all its media and analysis.', onSelect: onDeleteEpisode },
  ]
  const statusDot = <i className={`status-dot status-${status}`} aria-hidden />
  return <header className="flex h-12 items-center justify-between gap-3 border-b hairline bg-[#10131a] px-3">
    <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2 text-sm">
      <button onClick={() => go('/')} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#bca7ff] text-[#171224]" aria-label="framewise: all projects" title="All projects"><Clapperboard size={17} strokeWidth={2.4} /></button>
      <button onClick={() => go(`/project/${projectId || ''}`)} className="muted hidden max-w-[22ch] truncate hover:text-white sm:block">{projectTitle || 'Project'}</button>
      <ChevronRight size={14} className="muted hidden shrink-0 sm:block" aria-hidden />
      <span className="truncate font-semibold">{episodeTitle || 'Loading…'}</span>
      <span className="hidden min-w-0 items-center gap-1.5 truncate text-xs text-[#b8bfcb] md:inline-flex" title={stage || status}>{statusDot}<span className="truncate">{stage || status}</span></span>
    </nav>
    <div className="flex shrink-0 items-center gap-1.5">
      <Menu label="Character detection" items={characterItems} align="end" trigger={props => <button type="button" {...props} className="icon-btn icon-btn-md icon-btn-text" title="Character detection"><Users size={16} className="text-[#7bdcb2]" aria-hidden /><span className="hidden lg:inline">Characters</span><ChevronDown size={13} aria-hidden /></button>} />
      {hasCuts
        ? <Menu label="Cuts and media" items={cutItems} align="end" trigger={props => <button type="button" {...props} className="icon-btn icon-btn-md icon-btn-text" title="Cuts and media"><Scissors size={16} className="text-[#bca7ff]" aria-hidden /><span className="hidden lg:inline">Cuts</span><ChevronDown size={13} aria-hidden /></button>} />
        : <>
          <button type="button" onClick={onPrepareEpisode} className="icon-btn icon-btn-md icon-btn-text tone-primary"><ListChecks size={16} aria-hidden /><span>Prepare episode</span></button>
          <Menu label="Media" items={cutItems} align="end" trigger={props => <button type="button" {...props} className="icon-btn icon-btn-md" aria-label="Media actions" title="Media actions"><ChevronDown size={14} aria-hidden /></button>} />
        </>}
      <span className="mx-1 hidden h-5 w-px bg-[#333945] sm:block" aria-hidden />
      <button type="button" onClick={() => onOpenQueue()} className="icon-btn icon-btn-md icon-btn-text" title={counterTitle(queue)} aria-label={`Queue and history. ${counterTitle(queue)}`}>
        {running || queue?.busy ? <LoaderCircle size={16} className="animate-spin text-[#f5c678] motion-reduce:animate-none" aria-hidden /> : <ListChecks size={16} aria-hidden />}
        <span className="hidden sm:inline">Queue</span>
        {counterText(queue) && <span className="font-mono tabular-nums">{counterText(queue)}</span>}
        {!!queue?.stopping && <span className="text-[11px] font-semibold text-[#f5c678]">Stopping</span>}
      </button>
      {!!queue?.attention && <button type="button" onClick={() => onOpenQueue('attention')} title={attentionTitle(queue.attention)} aria-label={attentionTitle(queue.attention)}
        className="inline-flex h-8 items-center gap-1 whitespace-nowrap rounded-lg border border-[#63363e] bg-[#2b191f] px-2 text-xs font-semibold text-[#ff9fa7] hover:bg-[#3a2027]">
        <AlertTriangle size={13} aria-hidden />{attentionText(queue.attention)}
      </button>}
      <button type="button" onClick={onOpenInbox} className="icon-btn icon-btn-md icon-btn-text"
        title={[inbox.mine ? `${inbox.mine} autopilot question${inbox.mine === 1 ? '' : 's'} wait for you` : 'Autopilot inbox: nothing waits for you', inbox.elsewhere].filter(Boolean).join('. ')}
        aria-label={`Autopilot inbox${inbox.mine ? `, ${inbox.mine} waiting` : ''}`}>
        <Inbox size={16} className={inbox.mine ? 'text-[#f5c678]' : undefined} aria-hidden />
        <span className="hidden xl:inline">Inbox</span>
        {inbox.mine > 0 && <span className="font-mono tabular-nums text-[#f5c678]">{inbox.mine}</span>}
        {!inbox.mine && inbox.elsewhere && <i className="h-1.5 w-1.5 rounded-full bg-[#f5c678]" aria-hidden />}
      </button>
      <WebSearchSources compact />
      <VisualEngineControl compact />
      <LocalModelControl compact />
      <Menu label="Episode" items={episodeItems} align="end" width="w-72" trigger={props => <button type="button" {...props} className="icon-btn icon-btn-md" aria-label="Episode menu" title="Episode menu"><MoreHorizontal size={16} aria-hidden /></button>} />
    </div>
  </header>
}
