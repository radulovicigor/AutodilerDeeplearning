'use client'

import { cn } from '@/lib/utils'
import { SelectHTMLAttributes, forwardRef } from 'react'

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  options: { value: string; label: string }[]
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, options, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm text-gray-400 mb-2">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={cn(
            'w-full border border-dark-500 px-4 py-3 rounded-lg',
            'transition-all duration-200 appearance-none cursor-pointer',
            'focus:outline-none focus:border-neon-green/50 focus:shadow-[0_0_15px_rgba(0,255,136,0.1)]',
            'bg-[url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20fill%3D%27none%27%20viewBox%3D%270%200%2024%2024%27%20stroke%3D%27%2300ff88%27%3E%3Cpath%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%20stroke-width%3D%272%27%20d%3D%27M19%209l-7%207-7-7%27%3E%3C%2Fpath%3E%3C%2Fsvg%3E")] bg-no-repeat bg-[right_1rem_center] bg-[length:1rem]',
            error && 'border-red-500',
            className
          )}
          style={{
            backgroundColor: '#1a1a2e',
            color: '#00ff88',
          }}
          {...props}
        >
          {options.map((option) => (
            <option 
              key={option.value} 
              value={option.value}
              style={{ 
                backgroundColor: '#1a1a2e', 
                color: '#00ff88',
                padding: '10px'
              }}
            >
              {option.label}
            </option>
          ))}
        </select>
        {error && (
          <p className="mt-1 text-sm text-red-400">{error}</p>
        )}
      </div>
    )
  }
)

Select.displayName = 'Select'

export { Select }
