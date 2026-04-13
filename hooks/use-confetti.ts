"use client"

import { useCallback } from "react"

export function useConfetti() {
  const fire = useCallback(async () => {
    const confetti = (await import("canvas-confetti")).default
    // First burst
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
      colors: ["#6366f1", "#8b5cf6", "#a78bfa", "#f59e0b", "#10b981"],
    })
    // Second burst slightly delayed
    setTimeout(() => {
      confetti({
        particleCount: 50,
        angle: 60,
        spread: 55,
        origin: { x: 0 },
        colors: ["#6366f1", "#8b5cf6", "#f59e0b"],
      })
      confetti({
        particleCount: 50,
        angle: 120,
        spread: 55,
        origin: { x: 1 },
        colors: ["#6366f1", "#8b5cf6", "#f59e0b"],
      })
    }, 200)
  }, [])

  const fireSmall = useCallback(async () => {
    const confetti = (await import("canvas-confetti")).default
    confetti({
      particleCount: 40,
      spread: 50,
      origin: { y: 0.7 },
      colors: ["#6366f1", "#8b5cf6", "#a78bfa"],
    })
  }, [])

  return { fire, fireSmall }
}
