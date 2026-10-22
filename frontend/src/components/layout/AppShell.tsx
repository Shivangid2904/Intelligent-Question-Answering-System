import { useEffect, useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'
import { ENDPOINTS } from '../../lib/api'
import { cn } from '../../lib/utils'

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [backendReady, setBackendReady] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false

    async function checkHealth() {
      try {
        const res = await fetch(ENDPOINTS.health, { method: 'GET' })
        if (!cancelled) {
          setBackendReady(res.ok)
        }
      } catch {
        if (!cancelled) {
          setBackendReady(false)
        }
      }
    }

    checkHealth()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div
      className="flex min-h-screen"
      style={{ background: 'var(--color-bg-main)' }}
    >
      {/* Desktop Sidebar */}
      <div className="hidden md:flex">
        <Sidebar
          collapsed={collapsed}
          onToggle={() => setCollapsed(!collapsed)}
          backendReady={backendReady}
        />
      </div>

      {/* Mobile Drawer Backdrop */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-xs md:hidden"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Mobile Sidebar Drawer */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-50 transform md:hidden transition-transform duration-300 ease-in-out',
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <Sidebar
          collapsed={false}
          onToggle={() => setMobileMenuOpen(false)}
          backendReady={backendReady}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header onMobileMenuOpen={() => setMobileMenuOpen(true)} />

        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
