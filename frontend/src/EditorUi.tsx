import { forwardRef, useEffect, useId, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { ChevronDown, type LucideIcon } from 'lucide-react'
import { usePref } from './layoutPrefs'

type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  icon: LucideIcon
  label: string
  shortcut?: string
  tone?: 'default' | 'primary' | 'caution' | 'danger'
  /** Shown as text next to the icon; the label is still the accessible name and tooltip. */
  text?: ReactNode
  size?: 'sm' | 'md'
  spin?: boolean
}

/** A compact button whose label lives in its tooltip and accessible name. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton({ icon: Icon, label, shortcut, tone = 'default', text, size = 'md', spin, className = '', title, ...rest }, ref) {
  return <button ref={ref} type="button" aria-label={label} title={title || (shortcut ? `${label} (${shortcut})` : label)} className={`icon-btn icon-btn-${size} tone-${tone} ${text ? 'icon-btn-text' : ''} ${className}`} {...rest}>
    <Icon size={size === 'sm' ? 14 : 16} aria-hidden className={spin ? 'animate-spin motion-reduce:animate-none' : undefined} />{text && <span>{text}</span>}
  </button>
})

/** A collapsible block of the inspector. Its open state is remembered per `id`. */
export function Section({ id, title, icon: Icon, defaultOpen = true, badge, actions, children, accent = '#bca7ff', openSignal }: {
  id: string; title: string; icon?: LucideIcon; defaultOpen?: boolean; badge?: ReactNode; actions?: ReactNode; children: ReactNode; accent?: string
  /** Opens the section whenever this value changes to something truthy (for example a new selection that the section edits). */
  openSignal?: unknown
}) {
  const [open, setOpen] = usePref(`section.${id}`, defaultOpen)
  useEffect(() => { if (openSignal) setOpen(true) }, [openSignal])
  const body = useId()
  return <section className="section">
    <div className="flex items-center gap-2">
      <button type="button" aria-expanded={open} aria-controls={body} onClick={() => setOpen(!open)} className="section-toggle">
        <ChevronDown size={14} aria-hidden className={`transition-transform ${open ? '' : '-rotate-90'}`} />
        {Icon && <Icon size={14} aria-hidden style={{ color: accent }} />}
        <span>{title}</span>{badge}
      </button>
      {actions && <div className="ml-auto flex items-center gap-1">{actions}</div>}
    </div>
    {open && <div id={body} className="pt-3">{children}</div>}
  </section>
}

export type Segment<T extends string> = { value: T; label: string; icon?: LucideIcon; badge?: ReactNode }

/** A small group of mutually exclusive choices, shown as joined buttons. */
export function SegmentedControl<T extends string>({ value, options, onChange, label, className = '' }: { value: T; options: Segment<T>[]; onChange: (value: T) => void; label: string; className?: string }) {
  return <div role="radiogroup" aria-label={label} className={`segmented ${className}`}>
    {options.map(option => <button key={option.value} type="button" role="radio" aria-checked={value === option.value} onClick={() => onChange(option.value)}>
      {option.icon && <option.icon size={14} aria-hidden />}<span>{option.label}</span>{option.badge}
    </button>)}
  </div>
}

/** An on/off switch with its label. */
export function Switch({ checked, onChange, label, disabled, hint }: { checked: boolean; onChange: (value: boolean) => void; label: string; disabled?: boolean; hint?: string }) {
  return <button type="button" role="switch" aria-checked={checked} disabled={disabled} title={hint} onClick={() => onChange(!checked)} className="switch">
    <span className="switch-track" aria-hidden><span className="switch-thumb" /></span><span>{label}</span>
  </button>
}
