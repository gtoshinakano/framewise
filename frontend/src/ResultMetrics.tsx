import type { AnalysisResult } from './api'

export function ResultMetrics({ result }: { result: AnalysisResult }) {
  const settings = result.settings || {}
  const usage = settings.usage && typeof settings.usage === 'object' ? settings.usage as Record<string, unknown> : {}
  if (result.origin === 'manual_edit' || result.origin === 'manual') return <div className="text-[11px] text-[#aeb5c1]" aria-label="Generation details">Source: <strong className="text-[#e8ebf1]">Manual</strong></div>
  const visual = ['initial_frame', 'last_frame', 'video_analysis', 'subtitle_ocr'].includes(String(settings.structured_type || '')) || /^(Frame|Object detections|Raw OCR) at /.test(result.title)
  const legacyText = ['Cut summary', 'Consistency summary', 'MiniMax animation prompt', 'Visual evidence and confidence', 'On-screen text and confidence'].includes(result.title)
  const provider = settings.provider === 'local' ? 'Local' : settings.provider === 'openrouter' ? 'OpenRouter' : visual ? 'Local (legacy)' : legacyText ? 'OpenRouter (legacy)' : 'Unavailable'
  const duration = visual ? settings.generation_seconds : settings.summary_generation_seconds ?? settings.generation_seconds
  const value = (key: string) => typeof usage[key] === 'number' ? (usage[key] as number).toLocaleString() : 'Unavailable'
  return <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[#aeb5c1]" aria-label="Generation details">
    <span>Source: <strong className="text-[#e8ebf1]">{provider}</strong></span>
    <span>Model: <strong className="text-[#e8ebf1]">{result.model_id || 'Unavailable'}</strong></span>
    <span>Time: <strong className="text-[#e8ebf1]">{typeof duration === 'number' ? `${duration.toFixed(1)}s` : 'Unavailable'}</strong></span>
    <span>Input tokens: {value('prompt_tokens')}</span><span>Output tokens: {value('completion_tokens')}</span><span>Total tokens: {value('total_tokens')}</span>
    {typeof usage.routed_provider === 'string' && <span>Route: {usage.routed_provider}</span>}
    {typeof usage.request_seconds === 'number' && <span>Request: {(usage.request_seconds as number).toFixed(1)}s</span>}
    {typeof usage.reasoning_tokens === 'number' && <span>Thinking tokens: {value('reasoning_tokens')}</span>}
    {typeof usage.load_seconds === 'number' && <span>Model preparation: {(usage.load_seconds as number).toFixed(1)}s</span>}
    {settings.provider === 'local' && <span>Thinking: {settings.thinking ? 'On' : 'Off'}{settings.thinking && typeof settings.thinking_budget_tokens === 'number' ? ` (budget ${settings.thinking_budget_tokens})` : ''}</span>}
  </div>
}
