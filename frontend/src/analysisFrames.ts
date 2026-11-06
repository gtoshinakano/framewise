import type { AnalysisConfig, Cut } from './api'

export function analysisFrameCount(config: AnalysisConfig, cut: Cut): number {
  if (config.frame_mode === 'per_second') return Math.max(1, Math.ceil(cut.end_time - cut.start_time))
  return config.frame_mode === 'endpoints' ? 2 : config.frame_count
}

// The encoder rounds sampled frames per window and may add one duplicate.
export function estimatedVideoFrames(config: AnalysisConfig, cut: Cut): number {
  const duration = Math.max(0, cut.end_time - cut.start_time)
  const fps = config.video_fps
  if (!duration || !Number.isFinite(fps) || fps <= 0) return 0
  const window = config.video_frame_budget ? Math.min(Math.max(8, 2 / fps), 32 / fps) : Math.min(8, 32 / fps)
  const full = Math.max(0, Math.floor((duration - 0.000001) / window))
  const last = Math.max(0, duration - full * window)
  const count = (seconds: number) => seconds * fps < 1 ? 2 : Math.max(2, Math.round(seconds * fps) + (seconds * fps < 2 ? 1 : 0))
  const estimate = full * count(window) + count(last)
  return config.video_frame_budget ? Math.min(config.video_frame_budget, estimate) : estimate
}
