import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { LucideIcon } from 'lucide-react'

type TriggerProps = { 'aria-expanded': boolean; 'aria-haspopup': 'dialog' | 'menu'; 'aria-controls': string; onClick: () => void; ref: (element: HTMLButtonElement | null) => void }

const GAP = 6
const MARGIN = 8
const MAX_HEIGHT = 600

/** Fixed coordinates next to the trigger, so a panel is never clipped by a scrolling container. */
function placement(trigger: DOMRect, side: 'top' | 'bottom', align: 'start' | 'end'): CSSProperties {
  const vertical = side === 'top'
    ? { bottom: window.innerHeight - trigger.top + GAP, maxHeight: Math.min(MAX_HEIGHT, trigger.top - GAP - MARGIN) }
    : { top: trigger.bottom + GAP, maxHeight: Math.min(MAX_HEIGHT, window.innerHeight - trigger.bottom - GAP - MARGIN) }
  const horizontal = align === 'end' ? { right: Math.max(MARGIN, window.innerWidth - trigger.right) } : { left: Math.max(MARGIN, trigger.left) }
  return { position: 'fixed', ...vertical, ...horizontal, maxWidth: `calc(100vw - ${2 * MARGIN}px)` }
}

/**
 * A panel anchored to its trigger. It closes on Escape or a click outside and gives focus back to the
 * trigger. `side="top"` opens upwards (used in the timeline, which sits at the bottom of the screen).
 * The panel is rendered at the end of the page with fixed coordinates.
 */
export function Popover({ trigger, children, align = 'start', side = 'bottom', open: controlled, onOpenChange, panelClassName = '', role = 'dialog', label }: {
  trigger: (props: TriggerProps) => ReactNode
  children: ReactNode | ((close: () => void) => ReactNode)
  align?: 'start' | 'end'
  side?: 'top' | 'bottom'
  open?: boolean
  onOpenChange?: (open: boolean) => void
  panelClassName?: string
  role?: 'dialog' | 'menu'
  label: string
}) {
  const [own, setOwn] = useState(false)
  const open = controlled ?? own
  const panel = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement | null>(null)
  const [style, setStyle] = useState<CSSProperties | null>(null)
  const id = useId()
  const setOpen = (value: boolean) => { if (controlled === undefined) setOwn(value); onOpenChange?.(value) }
  const close = () => { setOpen(false); button.current?.focus() }
  useLayoutEffect(() => {
    if (!open) { setStyle(null); return }
    const place = () => { const rect = button.current?.getBoundingClientRect(); if (rect) setStyle(placement(rect, side, align)) }
    place()
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true) }
  }, [open, side, align])
  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent) => {
      const target = event.target as Node
      if (!panel.current?.contains(target) && !button.current?.contains(target)) setOpen(false)
    }
    const key = (event: globalThis.KeyboardEvent) => { if (event.key === 'Escape') { event.stopPropagation(); close() } }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', key, true)
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', key, true) }
  }, [open])
  // Menus focus their first item themselves; a dialog panel takes focus so Tab starts inside it.
  useEffect(() => { if (open && style && role === 'dialog') panel.current?.focus() }, [open, !!style])
  return <>
    {trigger({ 'aria-expanded': open, 'aria-haspopup': role, 'aria-controls': id, onClick: () => setOpen(!open), ref: element => { button.current = element } })}
    {open && style && createPortal(<div ref={panel} id={id} role={role === 'dialog' ? 'dialog' : undefined} aria-label={label} tabIndex={-1} style={style} className={`popover z-[75] ${panelClassName}`}>
      {typeof children === 'function' ? children(close) : children}
    </div>, document.body)}
  </>
}

export type MenuEntry = 'separator' | {
  key: string
  icon: LucideIcon
  title: string
  hint?: string
  tone?: 'default' | 'caution' | 'danger'
  disabled?: boolean
  /** Shown instead of the hint while the item is disabled. */
  disabledReason?: string
  shortcut?: string
  onSelect: () => void
}

/** A dropdown of actions: each item has an icon, a title, an optional hint line and a tone. */
export function Menu({ label, items, trigger, align = 'start', side = 'bottom', width = 'w-80' }: {
  label: string
  items: MenuEntry[]
  trigger: (props: TriggerProps) => ReactNode
  align?: 'start' | 'end'
  side?: 'top' | 'bottom'
  width?: string
}) {
  const list = useRef<HTMLDivElement>(null)
  const focusItem = (step: 1 | -1 | 0) => {
    const buttons = [...(list.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') || [])]
    if (!buttons.length) return
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement)
    const next = step === 0 ? 0 : (current + step + buttons.length) % buttons.length
    buttons[next].focus()
  }
  function onKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'ArrowDown') { event.preventDefault(); focusItem(1) }
    else if (event.key === 'ArrowUp') { event.preventDefault(); focusItem(-1) }
    else if (event.key === 'Home') { event.preventDefault(); focusItem(0) }
  }
  return <Popover label={label} role="menu" align={align} side={side} trigger={trigger} panelClassName={`${width} p-1.5`}>
    {close => <MenuList listRef={list} label={label} onKey={onKey} onMount={() => focusItem(0)}>
      {items.map((item, index) => item === 'separator' ? <div key={`separator-${index}`} role="separator" className="my-1 border-t hairline" /> : <button key={item.key} type="button" role="menuitem" disabled={item.disabled}
        onClick={() => { close(); item.onSelect() }} className={`menu-item tone-${item.tone || 'default'}`}>
        <item.icon size={16} className="menu-icon mt-0.5 shrink-0" aria-hidden />
        <span className="min-w-0 flex-1">
          <span className="flex items-center justify-between gap-3"><span className="menu-title">{item.title}</span>{item.shortcut && <kbd className="kbd">{item.shortcut}</kbd>}</span>
          {(item.disabled ? item.disabledReason || item.hint : item.hint) && <span className="menu-hint">{item.disabled ? item.disabledReason || item.hint : item.hint}</span>}
        </span>
      </button>)}
    </MenuList>}
  </Popover>
}

function MenuList({ children, label, listRef, onKey, onMount }: { children: ReactNode; label: string; listRef: React.RefObject<HTMLDivElement | null>; onKey: (event: KeyboardEvent<HTMLDivElement>) => void; onMount: () => void }) {
  useEffect(() => { onMount() }, [])
  return <div ref={listRef} role="menu" aria-label={label} onKeyDown={onKey}>{children}</div>
}
