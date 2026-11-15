import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { CircleAlert, CircleCheck, Info, X } from 'lucide-react'

export type ToastInput = { message: string; tone?: 'info' | 'success' | 'error'; action?: { label: string; run: () => void }; duration?: number }
type Toast = ToastInput & { id: number }

const ToastContext = createContext<(toast: ToastInput) => void>(() => {})

/** Short notices that float over the editor instead of pushing the layout down. Errors stay until dismissed. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const next = useRef(1)
  const dismiss = useCallback((id: number) => setToasts(current => current.filter(toast => toast.id !== id)), [])
  const show = useCallback((toast: ToastInput) => {
    const id = next.current++
    // The newest notice goes on top; at most four stay visible.
    setToasts(current => [{ ...toast, id }, ...current.filter(item => item.message !== toast.message)].slice(0, 4))
  }, [])
  return <ToastContext.Provider value={show}>
    {children}
    <div aria-live="polite" className="pointer-events-none fixed right-4 top-14 z-[80] flex w-[min(420px,calc(100vw-2rem))] flex-col gap-2">
      {toasts.map(toast => <ToastCard key={toast.id} toast={toast} onDismiss={() => dismiss(toast.id)} />)}
    </div>
  </ToastContext.Provider>
}

function ToastCard({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const tone = toast.tone || 'info'
  useEffect(() => {
    if (tone === 'error') return
    const timer = setTimeout(onDismiss, toast.duration ?? (toast.action ? 8000 : 5000))
    return () => clearTimeout(timer)
  }, [])
  const Icon = tone === 'error' ? CircleAlert : tone === 'success' ? CircleCheck : Info
  return <div role={tone === 'error' ? 'alert' : 'status'} className={`toast toast-${tone} pointer-events-auto`}>
    <Icon size={16} className="mt-0.5 shrink-0" aria-hidden />
    <p className="min-w-0 flex-1 break-words">{toast.message}</p>
    {toast.action && <button type="button" className="toast-action" onClick={() => { toast.action!.run(); onDismiss() }}>{toast.action.label}</button>}
    <button type="button" onClick={onDismiss} aria-label="Dismiss" className="shrink-0 opacity-70 hover:opacity-100"><X size={15} /></button>
  </div>
}

export function useToast() {
  return useContext(ToastContext)
}
