import { memo, useState, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Check, Copy } from 'lucide-react'

/** Copy text, falling back to a hidden textarea when the Clipboard API is unavailable (plain HTTP). */
export async function copyText(text: string) {
  try {
    if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable')
    await navigator.clipboard.writeText(text)
    return true
  } catch { /* fall through */ }
  const previous = document.activeElement as HTMLElement | null
  const field = document.createElement('textarea')
  field.value = text
  field.setAttribute('readonly', '')
  field.style.position = 'fixed'; field.style.left = '-9999px'
  document.body.appendChild(field)
  try { field.focus(); field.select(); return document.execCommand('copy') } catch { return false } finally { field.remove(); previous?.focus() }
}

export function CopyButton({ text, label = 'Copy', className = '' }: { text: string; label?: string; className?: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')
  return <button type="button" className={`inline-flex items-center gap-1 rounded-md px-1.5 py-1 !text-[11px] text-[#aeb5c1] hover:bg-[#303746] hover:text-white ${className}`}
    onClick={async () => { setState(await copyText(text) ? 'copied' : 'failed'); setTimeout(() => setState('idle'), 1600) }} aria-label={label} title={label}>
    {state === 'copied' ? <Check size={13} /> : <Copy size={13} />}<span>{state === 'copied' ? 'Copied' : state === 'failed' ? 'Select to copy' : label}</span>
  </button>
}

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) return textOf((node as { props: { children?: ReactNode } }).props.children)
  return ''
}

const safeHref = (href?: string) => href && /^(https?:|mailto:|\/|#)/i.test(href) ? href : undefined

const components: Components = {
  pre({ children }) {
    const code = textOf(children).replace(/\n$/, '')
    const language = (Array.isArray(children) ? children[0] : children) as { props?: { className?: string } } | undefined
    const name = /language-([\w-]+)/.exec(language?.props?.className || '')?.[1]
    return <div className="md-code group">
      <div className="flex items-center justify-between border-b border-[#2c3340] px-3 py-1"><span className="text-[10px] uppercase tracking-wider text-[#8b93a3]">{name || 'text'}</span><CopyButton text={code} /></div>
      <pre className="scrollbar">{children}</pre>
    </div>
  },
  a({ href, children }) {
    const target = safeHref(href)
    return target ? <a href={target} target="_blank" rel="noreferrer noopener">{children}</a> : <span>{children}</span>
  },
  table({ children }) { return <div className="scrollbar overflow-x-auto"><table>{children}</table></div> },
  img({ alt }) { return <span className="muted">[image: {alt || 'omitted'}]</span> },
}

/** GitHub-flavoured Markdown without raw HTML; safe for model output. */
export const Markdown = memo(function Markdown({ text, className = '' }: { text: string; className?: string }) {
  return <div className={`md ${className}`}><ReactMarkdown remarkPlugins={[remarkGfm]} components={components} skipHtml>{text}</ReactMarkdown></div>
})
