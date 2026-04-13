"use client"

import { motion } from "framer-motion"

interface GaugeChartProps {
  value: number
  size?: number
  strokeWidth?: number
  label?: string
}

export function GaugeChart({
  value,
  size = 180,
  strokeWidth = 14,
  label = "Match",
}: GaugeChartProps) {
  const radius = (size - strokeWidth) / 2
  const circumference = Math.PI * radius // semi-circle
  const progress = (value / 100) * circumference

  const getColor = () => {
    if (value >= 75) return "oklch(0.6 0.18 155)" // success green
    if (value >= 50) return "oklch(0.75 0.15 85)" // amber
    if (value >= 25) return "oklch(0.65 0.2 55)" // orange
    return "oklch(0.577 0.245 27.325)" // red
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size / 2 + strokeWidth}
        viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}
      >
        {/* Background arc */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted"
          strokeLinecap="round"
        />
        {/* Animated progress arc */}
        <motion.path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke={getColor()}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1.5, ease: "easeOut", delay: 0.3 }}
        />
        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2 - 4}
          textAnchor="middle"
          className="fill-foreground text-3xl font-bold"
          style={{ fontSize: size * 0.18 }}
        >
          <motion.tspan
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {value}%
          </motion.tspan>
        </text>
      </svg>
      <span className="text-sm font-medium text-muted-foreground">{label}</span>
    </div>
  )
}
