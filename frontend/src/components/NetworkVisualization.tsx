'use client'

/**
 * NetworkVisualization Component
 * 
 * IMPORTANT NOTES:
 * - This component visualizes the STRUCTURE of the neural network, NOT actual weights
 * - All connection lines have the same opacity (0.15) regardless of weight values
 * - Weights are passed as props but currently NOT used for visual differentiation
 * - Forward-pass animation is NOT implemented - visualization is static
 * - Neuron click shows layer/neuron info but does NOT show "top feature influence"
 * 
 * The weights prop is available for future enhancement to visualize actual weight magnitudes.
 */

import { useRef, useMemo, useState, useEffect } from 'react'

interface LayerInfo {
  name: string
  neurons: number
  type: 'input' | 'hidden' | 'output'
  activation?: string
}

interface NetworkVisualizationProps {
  layers: LayerInfo[]
  weights?: Record<string, { weight: number[][]; bias: number[] }>
  activations?: Record<string, number[][]>
  onNeuronClick?: (data: { layer: number; neuron: number } | null) => void
}

export default function NetworkVisualization({
  layers,
  weights,
  activations,
  onNeuronClick,
}: NetworkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredNeuron, setHoveredNeuron] = useState<{ layer: number; neuron: number } | null>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })
  const [isResizing, setIsResizing] = useState(false)
  const [minHeight] = useState(400)
  const [maxHeight] = useState(1200)

  // Calculate neuron positions - show ALL neurons
  const neuronPositions = useMemo(() => {
    const positions: { x: number; y: number; layer: number; neuron: number; totalInLayer: number }[] = []
    const paddingX = 100
    const paddingTop = 70
    const paddingBottom = 60
    const usableWidth = dimensions.width - paddingX * 2
    const usableHeight = dimensions.height - paddingTop - paddingBottom
    const layerSpacing = usableWidth / (layers.length - 1 || 1)
    
    // Find max neurons in any layer
    const maxNeuronsInLayer = Math.max(...layers.map(l => l.neurons))
    
    layers.forEach((layer, layerIndex) => {
      const displayNeurons = layer.neurons // Show ALL neurons
      
      // Calculate neuron size based on available space
      const availableSpacePerNeuron = usableHeight / displayNeurons
      const neuronSpacing = Math.max(8, Math.min(availableSpacePerNeuron, 40))
      const totalLayerHeight = neuronSpacing * (displayNeurons - 1)
      const startY = paddingTop + (usableHeight - totalLayerHeight) / 2
      
      for (let i = 0; i < displayNeurons; i++) {
        positions.push({
          x: paddingX + layerSpacing * layerIndex,
          y: startY + neuronSpacing * i,
          layer: layerIndex,
          neuron: i,
          totalInLayer: layer.neurons,
        })
      }
    })
    
    return positions
  }, [layers, dimensions])

  // Calculate neuron size based on density
  const neuronSize = useMemo(() => {
    const maxNeurons = Math.max(...layers.map(l => l.neurons))
    const availableHeight = dimensions.height - 130
    const spacePerNeuron = availableHeight / maxNeurons
    return Math.max(3, Math.min(9, spacePerNeuron * 0.4))
  }, [layers, dimensions])

  // Handle resize
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    
    const updateDimensions = () => {
      const rect = container.getBoundingClientRect()
      setDimensions(prev => ({
        width: rect.width || prev.width,
        height: rect.height || prev.height,
      }))
    }
    
    updateDimensions()
    
    const resizeObserver = new ResizeObserver(updateDimensions)
    resizeObserver.observe(container)
    return () => resizeObserver.disconnect()
  }, [])

  // Handle manual resize
  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
    
    const startY = e.clientY
    const startHeight = dimensions.height
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaY = moveEvent.clientY - startY
      const newHeight = Math.max(minHeight, Math.min(maxHeight, startHeight + deltaY))
      
      const container = containerRef.current?.parentElement
      if (container) {
        container.style.height = `${newHeight}px`
      }
    }
    
    const handleMouseUp = () => {
      setIsResizing(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
    
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // Draw network
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear with dark background
    ctx.fillStyle = '#0a0a12'
    ctx.fillRect(0, 0, dimensions.width, dimensions.height)

    // Draw subtle grid
    ctx.strokeStyle = 'rgba(40, 40, 60, 0.25)'
    ctx.lineWidth = 1
    const gridSize = 40
    for (let x = 0; x < dimensions.width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, dimensions.height)
      ctx.stroke()
    }
    for (let y = 0; y < dimensions.height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(dimensions.width, y)
      ctx.stroke()
    }

    // Draw connections - green lines with low opacity
    for (let layerIdx = 1; layerIdx < layers.length; layerIdx++) {
      const prevLayerNeurons = neuronPositions.filter((p) => p.layer === layerIdx - 1)
      const currLayerNeurons = neuronPositions.filter((p) => p.layer === layerIdx)

      // Limit connections for performance when many neurons
      const maxConnections = 200
      const totalPossible = prevLayerNeurons.length * currLayerNeurons.length
      const skipFactor = Math.max(1, Math.ceil(totalPossible / maxConnections))
      
      let connCount = 0
      prevLayerNeurons.forEach((prev, prevIdx) => {
        currLayerNeurons.forEach((curr, currIdx) => {
          connCount++
          if (connCount % skipFactor !== 0 && totalPossible > maxConnections) return
          
          ctx.beginPath()
          ctx.strokeStyle = 'rgba(0, 200, 100, 0.15)'
          ctx.lineWidth = 0.5
          ctx.moveTo(prev.x, prev.y)
          ctx.lineTo(curr.x, curr.y)
          ctx.stroke()
        })
      })
    }

    // Draw neurons - simple circles
    neuronPositions.forEach((pos) => {
      const layer = layers[pos.layer]
      const isInput = layer.type === 'input'
      const isOutput = layer.type === 'output'
      const isHovered = hoveredNeuron?.layer === pos.layer && hoveredNeuron?.neuron === pos.neuron

      // Determine colors
      let fillColor: string
      let strokeColor: string
      if (isInput) {
        fillColor = '#00d4ff'
        strokeColor = '#00a8cc'
      } else if (isOutput) {
        fillColor = '#ff6b9d'
        strokeColor = '#cc5580'
      } else {
        fillColor = '#00cc77'
        strokeColor = '#009955'
      }

      const size = isHovered ? neuronSize * 1.5 : neuronSize

      // Draw neuron circle
      ctx.beginPath()
      ctx.fillStyle = fillColor
      ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2)
      ctx.fill()

      // Border
      ctx.strokeStyle = isHovered ? '#ffffff' : strokeColor
      ctx.lineWidth = isHovered ? 2 : 1
      ctx.stroke()
    })

    // Draw layer labels
    ctx.font = 'bold 13px Inter, system-ui, sans-serif'
    ctx.textAlign = 'center'
    
    layers.forEach((layer, idx) => {
      const layerNeurons = neuronPositions.filter(p => p.layer === idx)
      if (layerNeurons.length === 0) return
      
      const x = layerNeurons[0].x
      
      ctx.fillStyle = layer.type === 'input' ? '#00d4ff' : layer.type === 'output' ? '#ff6b9d' : '#00cc77'
      ctx.fillText(
        layer.type === 'input' ? 'INPUT' : layer.type === 'output' ? 'OUTPUT' : `HIDDEN ${idx}`,
        x, 
        25
      )
      
      ctx.font = '11px Inter, system-ui, sans-serif'
      ctx.fillStyle = '#888'
      ctx.fillText(`${layer.neurons} neurons`, x, 43)
      
      ctx.font = 'bold 13px Inter, system-ui, sans-serif'
    })

  }, [layers, neuronPositions, weights, activations, hoveredNeuron, dimensions, neuronSize])

  // Handle mouse interactions
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const hovered = neuronPositions.find((pos) => {
      const dx = pos.x - x
      const dy = pos.y - y
      return Math.sqrt(dx * dx + dy * dy) < neuronSize * 2
    })

    setHoveredNeuron(hovered ? { layer: hovered.layer, neuron: hovered.neuron } : null)
  }

  const handleClick = () => {
    if (hoveredNeuron && onNeuronClick) {
      onNeuronClick(hoveredNeuron)
    }
  }

  return (
    <div ref={containerRef} className="w-full h-full relative rounded-lg overflow-hidden border border-dark-600">
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredNeuron(null)}
        onClick={handleClick}
        className="cursor-pointer"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-5 text-xs bg-dark-900/90 px-4 py-2.5 rounded-lg border border-dark-600">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#00d4ff]" />
          <span className="text-gray-300">Input</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#00cc77]" />
          <span className="text-gray-300">Hidden</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#ff6b9d]" />
          <span className="text-gray-300">Output</span>
        </div>
      </div>
      
      {/* Resize handle */}
      <div 
        className="absolute bottom-0 right-0 w-6 h-6 cursor-se-resize flex items-center justify-center group"
        onMouseDown={handleResizeStart}
      >
        <svg 
          width="12" 
          height="12" 
          viewBox="0 0 12 12" 
          className="text-gray-500 group-hover:text-neon-green transition-colors"
        >
          <path 
            d="M10 2L2 10M10 6L6 10M10 10L10 10" 
            stroke="currentColor" 
            strokeWidth="1.5" 
            strokeLinecap="round"
          />
        </svg>
      </div>
      
      {/* Hovered neuron info */}
      {hoveredNeuron && (
        <div className="absolute top-3 right-3 bg-dark-900/95 border border-dark-500 rounded-lg px-4 py-3 text-sm">
          <p className="text-white font-semibold">
            {layers[hoveredNeuron.layer]?.type === 'input' ? 'Input' : 
             layers[hoveredNeuron.layer]?.type === 'output' ? 'Output' : 
             `Hidden ${hoveredNeuron.layer}`} Layer
          </p>
          <p className="text-neon-green font-mono">Neuron #{hoveredNeuron.neuron + 1}</p>
        </div>
      )}
      
      {/* Resize indicator */}
      {isResizing && (
        <div className="absolute bottom-3 right-12 text-xs text-gray-400 bg-dark-900/90 px-2 py-1 rounded">
          {dimensions.height}px
        </div>
      )}
    </div>
  )
}
