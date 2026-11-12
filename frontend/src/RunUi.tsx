import { useEffect, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Circle, Clock, Info, Loader2, X, XCircle } from 'lucide-react'
import { STATE_TEXT, stateTone, type StepState } from './runState'

export type Tone = 'green' | 'purple' | 'amber' | 'red' | 'grey' | 'blue'

const TONES: Record<Tone, string> = {
  green: 'border-[#2f6b55] bg-[#12261f] text-[#7bdcb2]',
  purple: 'border-[#4a4363] bg-[#221c33] text-[#c3aeff]',
  amber: 'border-[#6b5a2f] bg-[#2a2314] text-[#e6c58b]',
  red: 'border-[#63363e] bg-[#2b191f] text-[#ff9fa7]',
  grey: 'border-[#353c48] bg-[#1b202a] text-[#929aa9]',
  blue: 'border-[#2f4a72] bg-[#141f33] text-[#8cbcff]',
}

export function StatusChip({ tone, icon, children, title, className = '' }: { tone: Tone; icon?: ReactNode; children: ReactNode; title?: string; className?: string }) {
  return <span title={title} className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold leading-4 ${TONES[tone]} ${className}`}>{icon}{children}</span>
}

const STATE_ICON: Record<StepState, ReactNode> = {
  none: <Circle size={11} aria-hidden />, queued: <Clock size={11} aria-hidden />, running: <Loader2 size={11} className="animate-spin" aria-hidden />,
  done: <CheckCircle2 size={11} aria-hidden />, failed: <XCircle size={11} aria-hidden />,
}

/** A step's state as a coloured chip; the state is always spelled out, never colour alone. */
export function StateChip({ state, label }: { state: StepState; label?: string }) {
  return <StatusChip tone={stateTone(state)} icon={STATE_ICON[state]} title={label ? `${label}: ${STATE_TEXT[state]}` : STATE_TEXT[state]}>{label ?? STATE_TEXT[state]}{label && <span className="sr-only">: {STATE_TEXT[state]}</span>}</StatusChip>
}

/** Job status words used in the history table and the run-details header. */
export function jobTone(status: string): Tone {
  return status === 'Succeeded' || status === 'done' ? 'green' : status === 'Failed' || status === 'failed' ? 'red' : status === 'Running' || status === 'running' ? 'purple' : status === 'Queued' || status === 'queued' || status === 'Waiting' ? 'amber' : status === 'Cancelled' || status === 'Not completed' ? 'grey' : 'grey'
}

export function JobChip({ status }: { status: string }) {
  const tone = jobTone(status)
  const icon = tone === 'green' ? <CheckCircle2 size={11} aria-hidden /> : tone === 'red' ? <XCircle size={11} aria-hidden /> : tone === 'purple' ? <Loader2 size={11} className="animate-spin" aria-hidden /> : tone === 'amber' ? <Clock size={11} aria-hidden /> : <Circle size={11} aria-hidden />
  return <StatusChip tone={tone} icon={icon}>{status}</StatusChip>
}

type ActionProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'type'> & { icon?: ReactNode; pending?: boolean; variant?: 'primary' | 'quiet' | 'danger'; compact?: boolean }

/** Only the button that was clicked spins; the others stay as they are. */
export function ActionButton({ icon, pending = false, variant = 'quiet', compact = false, className = '', children, disabled, ...rest }: ActionProps) {
  return <button type="button" {...rest} disabled={disabled || pending} aria-busy={pending || undefined} className={`btn btn-${variant} ${compact ? '!px-2.5 !py-1.5 !text-xs' : '!px-3 !py-2 !text-xs'} ${className}`}>{pending ? <Loader2 size={14} className="animate-spin" aria-hidden /> : icon}{children}</button>
}

const parseUtc = (value: string) => Date.parse(`${value.replace(' ', 'T')}${/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? '' : 'Z'}`)

export function formatElapsed(seconds: number) {
  const total = Math.max(0, Math.floor(seconds))
  return total >= 3600 ? `${Math.floor(total / 3600)}h ${Math.floor(total % 3600 / 60)}m` : total >= 60 ? `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s` : `${total}s`
}

/** Ticks on its own timer, so only this span re-renders every second. */
export function Elapsed({ since }: { since: string | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer) }, [])
  const start = since ? parseUtc(since) : NaN
  if (!Number.isFinite(start)) return null
  return <span className="tabular-nums" title="Time since this run started">{formatElapsed((now - start) / 1000)}</span>
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-[#232a36] ${className}`} aria-hidden />
}

export type Notice = { tone: 'success' | 'warning' | 'error'; lines: string[] }

const NOTICE = {
  success: { cls: 'border-[#2f6b55] bg-[#12261f] text-[#9be8c8]', icon: <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-[#7bdcb2]" aria-hidden /> },
  warning: { cls: 'border-[#6b5a2f] bg-[#2a2314] text-[#f0d9a5]', icon: <AlertTriangle size={15} className="mt-0.5 shrink-0 text-[#e6c58b]" aria-hidden /> },
  error: { cls: 'border-[#63363e] bg-[#2b191f] text-[#ffb6bc]', icon: <XCircle size={15} className="mt-0.5 shrink-0 text-[#ff9fa7]" aria-hidden /> },
}

/** Says what just happened, in the place the user is looking. Errors are announced as alerts. */
export function NoticeStrip({ notice, onDismiss }: { notice: Notice; onDismiss: () => void }) {
  const style = NOTICE[notice.tone]
  return <div role={notice.tone === 'error' ? 'alert' : 'status'} className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-xs leading-5 ${style.cls}`}>
    {style.icon}
    <div className="min-w-0 flex-1 space-y-0.5">{notice.lines.map((line, index) => <p key={index} className={index === 0 ? 'font-semibold' : 'opacity-90'}>{line}</p>)}</div>
    <button type="button" onClick={onDismiss} aria-label="Dismiss message" className="shrink-0 rounded p-0.5 opacity-70 hover:bg-white/10 hover:opacity-100"><X size={14} /></button>
  </div>
}

export function WarningLine({ level, children }: { level: 'error' | 'warn' | 'info'; children: ReactNode }) {
  const cls = level === 'error' ? 'text-[#ff9fa7]' : level === 'warn' ? 'text-[#e6c58b]' : 'text-[#9ec5ff]'
  return <p className={`flex items-start gap-1.5 text-[11px] leading-4 ${cls}`}>{level === 'info' ? <Info size={13} className="mt-px shrink-0" aria-hidden /> : <AlertTriangle size={13} className="mt-px shrink-0" aria-hidden />}<span>{children}</span></p>
}

/** Colour by kind of saved result, matching the Analysis timeline bar. */
export function resultAccent(settings: Record<string, unknown> | undefined, title: string): { color: string; label: string } {
  const type = String(settings?.structured_type ?? '')
  if (type === 'initial_frame') return { color: '#6aa8ff', label: 'Initial frame' }
  if (type === 'last_frame') return { color: '#8f9bff', label: 'Last frame (earlier analysis)' }
  if (type === 'video_analysis') return { color: '#4cc9c0', label: 'Video analysis' }
  if (type === 'consistency_summary') return { color: '#b496ff', label: 'Consistency summary' }
  if (title === 'Cut summary') return { color: '#e5d25a', label: 'Cut summary' }
  return { color: '#8b95a7', label: title || 'Result' }
}
