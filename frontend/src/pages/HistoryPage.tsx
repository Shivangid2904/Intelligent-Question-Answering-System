import { History, Trash2 } from 'lucide-react'
import { Card, CardContent } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Badge } from '../components/ui/Badge'

export default function HistoryPage() {
  return (
    <div className="page-container flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="page-title text-2xl">Query History</h1>
            <Badge variant="muted">Audit Trail</Badge>
          </div>
          <p className="page-subtitle mb-0">
            Chronological log of past scientific queries, sufficiency decisions, and execution times.
          </p>
        </div>

        <button className="btn btn-ghost self-start sm:self-auto text-xs" disabled>
          <Trash2 className="w-3.5 h-3.5 mr-1" /> Clear History (Phase 2F.2)
        </button>
      </div>

      <Card>
        <CardContent className="p-8">
          <EmptyState
            icon={History}
            title="Query History Ready for Integration"
            description="Historical research queries and sufficiency audit data will be fetched from GET /api/history with deletion supported via DELETE /api/history."
            action={
              <Badge variant="muted">
                Endpoint: /api/history
              </Badge>
            }
          />
        </CardContent>
      </Card>
    </div>
  )
}
