import { useEffect, useState, type ImgHTMLAttributes } from 'react'

type Props = ImgHTMLAttributes<HTMLImageElement> & { fallbackClassName?: string }

export function FrameImage({ src, alt, className, fallbackClassName, onError, ...props }: Props) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [src])
  if (failed) return <span role="img" aria-label={`${alt || 'Frame'} unavailable`} className={fallbackClassName || 'flex aspect-video items-center justify-center bg-[#222938] p-2 text-center text-[10px] text-[#98a1af]'}>Frame unavailable</span>
  return <img {...props} src={src} alt={alt} className={className} onError={event => { setFailed(true); onError?.(event) }} />
}
