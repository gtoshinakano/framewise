import { BoxSelect, Circle, CircleCheck, Clock3, FileText, FileX, History, LoaderCircle, ScanFace, UserCheck, UserRound, UserX } from 'lucide-react'
import type { Cut } from './api'
import type { AnalysisQueueState } from './analysisQueueStates'

/** The legend for CutStatusIcons, shown in the timeline's help popover. */
export const CUT_STATUS_LEGEND = [
  { icon: CircleCheck, color: '#7bdcb2', label: 'Has analysis results' },
  { icon: History, color: '#c9733e', label: 'Some results are out of date (an input changed)' },
  { icon: Circle, color: '#c5cad3', label: 'No analysis results' },
  { icon: BoxSelect, color: '#4cc9f0', label: 'Character boxes detected' },
  { icon: UserCheck, color: '#7bdcb2', label: 'Character recognized' },
  { icon: UserRound, color: '#e6c58b', label: 'Unlabeled characters' },
  { icon: ScanFace, color: '#e2709c', label: 'Unknown people in target frames' },
  { icon: UserX, color: '#c5cad3', label: 'Marked as having no characters' },
  { icon: FileText, color: '#c3aeff', label: 'Has a Cut summary' },
  { icon: FileX, color: '#6f7787', label: 'No Cut summary' },
  { icon: LoaderCircle, color: '#f5c678', label: 'Analysis running' },
  { icon: Clock3, color: '#86d1e1', label: 'Analysis queued' },
]

/** One row of small icons that tell what a cut has and what is still missing. Same icons everywhere a cut appears. */
export function CutStatusIcons({ cut, queueState, size = 12 }: { cut: Cut; queueState?: AnalysisQueueState; size?: number }) {
  return <span className="inline-flex shrink-0 items-center gap-1">
    {cut.has_stale_results ? <History size={size} className="shrink-0 text-[#c9733e]" aria-label="Some results are out of date" />
      : cut.has_analysis_results ? <CircleCheck size={size} className="shrink-0 text-[#7bdcb2]" aria-label="Has analysis results" /> : <Circle size={size} className="shrink-0 text-[#c5cad3]" aria-label="No analysis results" />}
    {cut.has_boxes ? <BoxSelect size={size} className="shrink-0 text-[#4cc9f0]" aria-label="Boxes detected" /> : null}
    {cut.no_characters ? <UserX size={size} className="shrink-0 text-[#c5cad3]" aria-label="No characters" />
      : cut.has_characters ? <UserCheck size={size} className="shrink-0 text-[#7bdcb2]" aria-label="Character detected" />
      : cut.has_boxes ? <UserRound size={size} className="shrink-0 text-[#e6c58b]" aria-label="Unlabeled characters" /> : null}
    {cut.has_unlabeled_target_boxes && !cut.no_characters ? <ScanFace size={size} className="shrink-0 text-[#e2709c]" aria-label="Unknown people in target frames" /> : null}
    {cut.has_summary ? <FileText size={size} className="shrink-0 text-[#c3aeff]" aria-label="Has cut summary" /> : <FileX size={size} className="shrink-0 text-[#6f7787]" aria-label="No cut summary" />}
    {queueState === 'running' ? <LoaderCircle size={size} className="shrink-0 animate-spin text-[#f5c678] motion-reduce:animate-none" aria-label="Analysis running" />
      : queueState === 'queued' ? <Clock3 size={size} className="shrink-0 text-[#86d1e1]" aria-label="Analysis queued" /> : null}
  </span>
}
