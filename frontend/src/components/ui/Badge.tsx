import React from 'react'
import { cn } from '../../lib/utils'

export type BadgeVariant =
  | 'lavender'
  | 'pink'
  | 'success'
  | 'warning'
  | 'error'
  | 'muted'

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant
  children: React.ReactNode
}

export function Badge({
  variant = 'lavender',
  className,
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn('badge', `badge-${variant}`, className)}
      {...props}
    >
      {children}
    </span>
  )
}
