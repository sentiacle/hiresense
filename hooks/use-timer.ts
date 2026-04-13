"use client"

import { useState, useEffect, useCallback, useRef } from "react"

export function useTimer(
  initialSeconds: number,
  onExpire?: () => void
) {
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds)
  const [isRunning, setIsRunning] = useState(false)
  const onExpireRef = useRef(onExpire)
  onExpireRef.current = onExpire

  useEffect(() => {
    if (!isRunning || secondsLeft <= 0) return

    const interval = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          setIsRunning(false)
          onExpireRef.current?.()
          return 0
        }
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, secondsLeft])

  const start = useCallback(() => setIsRunning(true), [])
  const pause = useCallback(() => setIsRunning(false), [])
  const reset = useCallback(
    (newSeconds?: number) => {
      setSecondsLeft(newSeconds ?? initialSeconds)
      setIsRunning(false)
    },
    [initialSeconds]
  )

  const percentage = initialSeconds > 0 ? (secondsLeft / initialSeconds) * 100 : 0

  return { secondsLeft, isRunning, start, pause, reset, percentage }
}
