"use client"

import { useEffect, useState } from "react"
import { motion, useSpring, useTransform } from "framer-motion"

interface AnimatedCounterProps {
  value: number
  duration?: number
  className?: string
  suffix?: string
}

export function AnimatedCounter({
  value,
  duration = 1.5,
  className = "",
  suffix = "",
}: AnimatedCounterProps) {
  const [isClient, setIsClient] = useState(false)
  const spring = useSpring(0, { duration: duration * 1000, bounce: 0 })
  const display = useTransform(spring, (v) => `${Math.round(v)}${suffix}`)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (isClient) {
      spring.set(value)
    }
  }, [spring, value, isClient])

  if (!isClient) return <span className={className}>0{suffix}</span>

  return <motion.span className={className}>{display}</motion.span>
}
