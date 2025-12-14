import React, { useMemo } from 'react'

interface DotGridProps {
  spacing?: number
  dotSize?: number
  color?: string
}

export const DotGrid: React.FC<DotGridProps> = ({ 
  spacing = 20, 
  dotSize = 2,
  color = 'rgba(0, 0, 0, 0.25)'
}) => {
  // Create SVG pattern as data URI
  const patternUrl = useMemo(() => {
    const svg = `
      <svg width="${spacing}" height="${spacing}" xmlns="http://www.w3.org/2000/svg">
        <circle cx="${spacing / 2}" cy="${spacing / 2}" r="${dotSize}" fill="${color}" />
      </svg>
    `.trim()
    const encoded = encodeURIComponent(svg)
    return `data:image/svg+xml,${encoded}`
  }, [spacing, dotSize, color])

  return (
    <div 
      style={{ 
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: '100%',
        height: '100%',
        backgroundImage: `url("${patternUrl}")`,
        backgroundRepeat: 'repeat',
        pointerEvents: 'none',
        zIndex: 0
      }}
    />
  )
}

