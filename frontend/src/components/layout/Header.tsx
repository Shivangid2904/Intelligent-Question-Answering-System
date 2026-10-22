import { Menu } from 'lucide-react'
import { useLocation } from 'react-router-dom'
import { ROUTES } from '../../routes/routes'

const PAGE_TITLES: Record<string, string> = {
  [ROUTES.overview]: 'Overview',
  [ROUTES.library]: 'Research Library',
  [ROUTES.chat]: 'Research Chat',
  [ROUTES.matrix]: 'Literature Matrix',
  [ROUTES.history]: 'Query History',
  [ROUTES.status]: 'System Status',
}

interface HeaderProps {
  onMobileMenuOpen: () => void
}

export default function Header({ onMobileMenuOpen }: HeaderProps) {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'IntelliAsk Research'

  return (
    <header
      className="h-16 flex items-center px-6 gap-4 sticky top-0 z-10 shrink-0"
      style={{
        background: 'rgba(251, 248, 255, 0.90)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid var(--color-border)',
      }}
    >
      {/* Mobile hamburger */}
      <button
        className="btn btn-ghost btn-sm md:hidden"
        onClick={onMobileMenuOpen}
        aria-label="Open navigation"
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Page title / breadcrumb */}
      <div className="flex-1 min-w-0">
        <h2
          className="text-base font-semibold truncate"
          style={{ color: 'var(--color-text-heading)' }}
        >
          {title}
        </h2>
      </div>

      {/* Research badge */}
      <div
        className="badge badge-lavender hidden sm:inline-flex"
        aria-label="Application type"
      >
        Research Platform
      </div>
    </header>
  )
}
