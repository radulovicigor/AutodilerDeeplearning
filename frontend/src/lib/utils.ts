import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number, decimals: number = 2): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toFixed(decimals)
}

export function formatCurrency(num: number): string {
  return new Intl.NumberFormat('de-DE', {
    style: 'currency',
    currency: 'EUR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(num)
}

export function formatPercent(num: number): string {
  return (num * 100).toFixed(1) + '%'
}

export function getMetricColor(metric: string, value: number): string {
  if (metric === 'r2') {
    if (value >= 0.9) return 'text-neon-green'
    if (value >= 0.7) return 'text-yellow-400'
    return 'text-red-400'
  }
  if (metric === 'accuracy' || metric === 'f1_macro') {
    if (value >= 0.85) return 'text-neon-green'
    if (value >= 0.7) return 'text-yellow-400'
    return 'text-red-400'
  }
  return 'text-gray-200'
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'completed':
      return 'bg-neon-green/20 text-neon-green border-neon-green/30'
    case 'running':
    case 'training':
      return 'bg-neon-blue/20 text-neon-blue border-neon-blue/30'
    case 'failed':
      return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'pending':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    default:
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}
