import { useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  FileText,
  MessageSquareText,
  ShieldCheck,
  Sparkles,
  Table2,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { ROUTES } from '../routes/routes'

export default function OverviewPage() {
  const navigate = useNavigate()

  return (
    <div className="page-container flex flex-col gap-8">
      {/* Hero Welcome Section */}
      <section
        className="rounded-3xl p-8 sm:p-10 relative overflow-hidden"
        style={{
          background: 'var(--gradient-hero)',
          border: '1px solid var(--color-border)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div className="max-w-2xl relative z-10 flex flex-col gap-3">
          <div className="inline-flex items-center gap-2">
            <Badge variant="pink">
              <Sparkles className="w-3.5 h-3.5 mr-1" /> Research Intelligence
            </Badge>
            <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Phase 2F Foundation
            </span>
          </div>

          <h1 className="page-title text-2xl sm:text-3xl lg:text-4xl">
            Your research, understood.
          </h1>

          <p className="text-sm sm:text-base leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
            IntelliAsk is a multi-document research assistant built for high-stakes scientific
            and clinical inquiries. It provides evidence-gated retrieval, fine-grained citation
            provenance, and structured literature synthesis with zero hallucinated claims.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => navigate(ROUTES.chat)}
              className="btn btn-primary"
            >
              Start Inquiring <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => navigate(ROUTES.library)}
              className="btn btn-secondary"
            >
              <BookOpen className="w-4 h-4 mr-1" /> View Library
            </button>
          </div>
        </div>
      </section>

      {/* Primary Workspaces Navigation */}
      <section className="flex flex-col gap-4">
        <div>
          <h2 className="section-heading">Research Workspaces</h2>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Navigate core capabilities designed for multi-document synthesis and grounded reasoning.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card
            interactive
            onClick={() => navigate(ROUTES.library)}
            className="flex flex-col justify-between"
          >
            <div>
              <CardHeader>
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-2"
                  style={{ background: 'var(--color-lavender-soft)' }}
                >
                  <BookOpen className="w-5 h-5" style={{ color: 'var(--color-primary)' }} />
                </div>
                <CardTitle>Research Library</CardTitle>
                <CardDescription>
                  Upload and manage academic PDFs, extract structured metadata, and maintain full-text chunk indices.
                </CardDescription>
              </CardHeader>
            </div>
            <div className="flex items-center justify-between pt-4 mt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
              <span className="text-xs font-medium" style={{ color: 'var(--color-primary)' }}>
                Open Library
              </span>
              <ArrowRight className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
            </div>
          </Card>

          <Card
            interactive
            onClick={() => navigate(ROUTES.chat)}
            className="flex flex-col justify-between"
          >
            <div>
              <CardHeader>
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-2"
                  style={{ background: 'var(--color-pink-light)' }}
                >
                  <MessageSquareText className="w-5 h-5" style={{ color: '#8B3A6A' }} />
                </div>
                <CardTitle>Evidence-Gated Chat</CardTitle>
                <CardDescription>
                  Query the literature with strict evidence-sufficiency checks, real-time SSE streaming, and grounded citations.
                </CardDescription>
              </CardHeader>
            </div>
            <div className="flex items-center justify-between pt-4 mt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
              <span className="text-xs font-medium" style={{ color: 'var(--color-primary)' }}>
                Start Conversation
              </span>
              <ArrowRight className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
            </div>
          </Card>

          <Card
            interactive
            onClick={() => navigate(ROUTES.matrix)}
            className="flex flex-col justify-between"
          >
            <div>
              <CardHeader>
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-2"
                  style={{ background: 'var(--color-bg-secondary)' }}
                >
                  <Table2 className="w-5 h-5" style={{ color: 'var(--color-primary-light)' }} />
                </div>
                <CardTitle>Literature Matrix</CardTitle>
                <CardDescription>
                  Compare methodologies, sample cohorts, and key findings side-by-side with direct provenance tracking and CSV export.
                </CardDescription>
              </CardHeader>
            </div>
            <div className="flex items-center justify-between pt-4 mt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
              <span className="text-xs font-medium" style={{ color: 'var(--color-primary)' }}>
                Explore Matrix
              </span>
              <ArrowRight className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
            </div>
          </Card>
        </div>
      </section>

      {/* System Status & Architecture Overview */}
      <section className="flex flex-col gap-4">
        <div>
          <h2 className="section-heading">Platform Architecture</h2>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            System indicators and core research guarantees (live metrics connect in Phase 2F.2).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="mb-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                  Backend Service
                </span>
                <span className="status-dot status-dot-muted" />
              </div>
              <CardTitle className="text-sm font-semibold">FastAPI & Uvicorn</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Endpoints: /api/documents, /api/query, /api/matrix, /api/history.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="mb-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                  Evidence Gate
                </span>
                <ShieldCheck className="w-4 h-4" style={{ color: 'var(--color-success)' }} />
              </div>
              <CardTitle className="text-sm font-semibold">Strict Verification</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Prevents unsupported speculation if retrieval score falls below threshold.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="mb-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                  Citation Engine
                </span>
                <FileText className="w-4 h-4" style={{ color: 'var(--color-primary-light)' }} />
              </div>
              <CardTitle className="text-sm font-semibold">Page-Level Tracing</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Every asserted finding links directly to source document chunk and page.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="mb-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                  Persistence
                </span>
                <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--color-success)' }} />
              </div>
              <CardTitle className="text-sm font-semibold">SQLite Storage</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Document metadata, extraction matrix, and query history safely persisted.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  )
}
