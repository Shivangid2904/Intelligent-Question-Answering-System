import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  History,
  FlaskConical,
  LayoutDashboard,
  MessageSquareText,
  Sparkles,
  Table2,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { ROUTES } from '../../routes/routes'
import { cn } from '../../lib/utils'

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
  backendReady: boolean | null
}

const NAV_ITEMS = [
  { to: ROUTES.overview, label: 'Overview', Icon: LayoutDashboard },
  { to: ROUTES.library, label: 'Research Library', Icon: BookOpen },
  { to: ROUTES.chat, label: 'Research Chat', Icon: MessageSquareText },
  { to: ROUTES.matrix, label: 'Literature Matrix', Icon: Table2 },
  { to: ROUTES.history, label: 'Query History', Icon: History },
  { to: ROUTES.status, label: 'System Status', Icon: FlaskConical },
]

export default function Sidebar({ collapsed, onToggle, backendReady }: SidebarProps) {
  return (
    <aside
      className={cn(
        'flex flex-col h-screen sticky top-0 overflow-hidden',
        'transition-all duration-300 ease-in-out',
        'border-r shrink-0',
      )}
      style={{
        width: collapsed ? 'var(--space-sidebar-collapsed)' : 'var(--space-sidebar-width)',
        background: 'var(--gradient-sidebar)',
        borderColor: 'var(--color-border)',
        boxShadow: 'var(--shadow-sidebar)',
      }}
    >
      {/* Brand header */}
      <div
        className="flex items-center gap-3 px-4 py-5 shrink-0"
        style={{ borderBottom: '1px solid var(--color-border)' }}
      >
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: 'var(--gradient-btn)' }}
        >
          <Sparkles className="w-5 h-5 text-white sparkle" />
        </div>

        {!collapsed && (
          <div className="overflow-hidden">
            <h1
              className="text-sm font-bold leading-tight tracking-tight truncate"
              style={{ color: 'var(--color-primary)' }}
            >
              IntelliAsk
            </h1>
            <p className="text-xs leading-tight truncate" style={{ color: 'var(--color-text-muted)' }}>
              Your research, understood.
            </p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 flex flex-col gap-1 overflow-y-auto">
        {NAV_ITEMS.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === ROUTES.overview}
            className={({ isActive }) =>
              cn('nav-item', isActive && 'active', collapsed && 'justify-center')
            }
            title={collapsed ? label : undefined}
          >
            <Icon className="nav-item-icon shrink-0" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Backend status */}
      <div
        className="px-3 py-3 shrink-0"
        style={{ borderTop: '1px solid var(--color-border)' }}
      >
        <div
          className={cn(
            'flex items-center gap-2 rounded-lg px-3 py-2',
          )}
          style={{
            background: 'var(--color-bg-card)',
            border: '1px solid var(--color-border)',
          }}
        >
          {backendReady === null ? (
            <div className="status-dot status-dot-muted" />
          ) : backendReady ? (
            <Wifi className="w-4 h-4 shrink-0" style={{ color: 'var(--color-success)' }} />
          ) : (
            <WifiOff className="w-4 h-4 shrink-0" style={{ color: 'var(--color-error)' }} />
          )}

          {!collapsed && (
            <span className="text-xs font-medium truncate" style={{ color: 'var(--color-text-muted)' }}>
              {backendReady === null ? 'Checking…' : backendReady ? 'API connected' : 'API offline'}
            </span>
          )}
        </div>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="btn btn-ghost w-full flex items-center justify-center py-3 rounded-none shrink-0"
        style={{ borderTop: '1px solid var(--color-border)' }}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4" />
        ) : (
          <>
            <ChevronLeft className="w-4 h-4" />
            <span className="text-xs ml-1">Collapse</span>
          </>
        )}
      </button>
    </aside>
  )
}
