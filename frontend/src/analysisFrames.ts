import type { AnalysisConfig, Cut } from './api'

export function analysisFrameCount(config: AnalysisConfig, cut: Cut): number {
  if (config.frame_mode === 'per_second') return Math.max(1, Math.ceil(cut.end_time - cut.start_time))
  return config.frame_mode === 'endpoints' ? 2 : config.frame_count
}
