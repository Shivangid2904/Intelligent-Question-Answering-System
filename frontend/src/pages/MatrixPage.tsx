import { Download, Table2 } from 'lucide-react'
import { Card, CardContent } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Badge } from '../components/ui/Badge'

export default function MatrixPage() {
  return (
    <div className="page-container flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="page-title text-2xl">Literature Matrix</h1>
            <Badge variant="lavender">Comparative Synthesis</Badge>
          </div>
          <p className="page-subtitle mb-0">
            Side-by-side paper analysis of study methodology, sample size, metrics, and conclusions.
          </p>
        </div>

        <button className="btn btn-secondary self-start sm:self-auto" disabled>
          <Download className="w-4 h-4 mr-1" /> Export CSV (Phase 2F.2)
        </button>
      </div>

      <Card>
        <CardContent className="p-8">
          <EmptyState
            icon={Table2}
            title="Literature Matrix Grid Ready"
            description="The comparative matrix will consume GET /api/matrix, field-level provenance via /api/documents/{id}/provenance, and CSV export via /api/matrix/export."
            action={
              <Badge variant="muted">
                Endpoints: /api/matrix & /api/matrix/export
              </Badge>
            }
          />
        </CardContent>
      </Card>
    </div>
  )
}
