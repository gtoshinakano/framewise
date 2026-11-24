import { Fragment, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Keyboard } from 'lucide-react'
import { sheetFor } from './shortcutSheets'

const STORAGE_KEY = 'framewise.shortcuts-panel-open'

// Dialogs provide their own view; unknown dialogs deliberately hide the underlying sheet.
function topDialog(): HTMLElement | undefined {
  const dialogs = Array.from(document.querySelectorAll<HTMLElement>('[aria-modal="true"]')).filter(el => el.getClientRects().length > 0)
  return dialogs.filter(el => el.matches('dialog:modal')).at(-1) ?? dialogs.at(-1)
}
function currentView(): string {
  const dialog = topDialog()
  return dialog?.dataset.shortcutView ?? (dialog ? 'dialog' : 'timeline')
}

export default function ShortcutsPanel() {
  const [view, setView] = useState(currentView)
  const [portal, setPortal] = useState<HTMLElement>(document.body)
  const [open, setOpen] = useState(() => { try { return localStorage.getItem(STORAGE_KEY) === '1' } catch { return false } })
  useEffect(() => {
    const refresh = () => {
      const dialog = topDialog()
      setView(dialog?.dataset.shortcutView ?? (dialog ? 'dialog' : 'timeline'))
      setPortal(dialog?.matches('dialog:modal') ? dialog : document.body)
    }
    const observer = new MutationObserver(refresh)
    observer.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['data-shortcut-view', 'aria-modal', 'hidden', 'open'] })
    refresh()
    return () => observer.disconnect()
  }, [])
  useEffect(() => { try { localStorage.setItem(STORAGE_KEY, open ? '1' : '0') } catch { /* storage may be unavailable */ } }, [open])
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key !== '?' || event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return
      event.preventDefault(); setOpen(value => !value)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])
  const sheet = sheetFor(view)
  const mac = /Mac|iPhone|iPad/.test(navigator.userAgent)
  return createPortal(<div className="fixed bottom-3 right-3 z-[110] max-w-[calc(100vw-1.5rem)]" onMouseDown={event => event.stopPropagation()}>
    <div className={open ? 'flex max-h-[min(70vh,560px)] w-80 max-w-full flex-col overflow-hidden rounded-lg border border-[#454b5b] bg-[#171c25]/95 shadow-2xl backdrop-blur' : ''}>
      <button type="button" aria-expanded={open} aria-controls="shortcuts-panel-content" onClick={() => setOpen(value => !value)} title="Keyboard shortcuts and mouse gestures" className={open ? 'flex shrink-0 items-center gap-1.5 border-b hairline px-3 py-2 text-left text-xs hover:bg-[#272e3a]' : 'flex h-7 items-center gap-1.5 rounded-full border border-[#454b5b] bg-[#171c25]/95 px-2.5 text-[11px] shadow-lg hover:bg-[#272e3a]'}>
        <Keyboard size={13} className="text-[#bba7ef]" /><span className="min-w-0 flex-1 truncate">{sheet.title}</span>{open && <ChevronDown size={14} />}
      </button>
      {open && <div id="shortcuts-panel-content" className="min-h-0 overflow-y-auto px-3 py-2">{sheet.groups.map(group => <section key={group.title} className="mb-2.5 last:mb-0">
        <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[#929aa9]">{group.title}</h3>
        <ul className="flex flex-col gap-1">{group.items.map((item, index) => <li key={index} className="flex items-start gap-2 text-[11px] leading-snug">
          <span className="flex w-28 shrink-0 flex-wrap items-center gap-1">{item.keys.map((combo, i) => <Fragment key={combo}>{i > 0 && <span className="text-[#929aa9]">/</span>}<span className="inline-flex items-center gap-0.5">{(combo === '+' ? ['+'] : combo.split('+')).map((part, j) => <Fragment key={j}>{j > 0 && <span>+</span>}<kbd className="kbd">{mac && part === 'Ctrl' ? '⌘' : mac && part === 'Alt' ? '⌥' : part}</kbd></Fragment>)}</span></Fragment>)}</span>
          <span className="min-w-0 flex-1">{item.does}{item.when && <span className="text-[#929aa9]"> · {item.when}</span>}</span>
        </li>)}</ul>
      </section>)}</div>}
    </div>
  </div>, portal)
}
