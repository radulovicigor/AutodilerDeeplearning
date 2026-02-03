'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { 
  Car, 
  BarChart3, 
  Brain, 
  Calculator,
  Cpu
} from 'lucide-react'

const navItems = [
  { href: '/', label: 'Home', icon: Car },
  { href: '/predict', label: 'Predict', icon: Calculator },
  { href: '/compare', label: 'Model Comparison', icon: BarChart3 },
  { href: '/network-lab', label: 'Network Lab', icon: Brain },
]

export default function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="border-b border-dark-500 bg-dark-800/80 backdrop-blur-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center group-hover:scale-105 transition-transform">
              <Cpu className="w-6 h-6 text-dark-900" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">
                Auto Diler <span className="text-neon-green">AI</span>
              </h1>
              <p className="text-xs text-gray-500">Duboko Učenje za Automobile</p>
            </div>
          </Link>

          {/* Navigation */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200',
                    isActive
                      ? 'bg-neon-green/10 text-neon-green border border-neon-green/30'
                      : 'text-gray-400 hover:text-white hover:bg-dark-600'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              )
            })}
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
            <span className="text-xs text-gray-400">System Online</span>
          </div>
        </div>
      </div>
    </nav>
  )
}
