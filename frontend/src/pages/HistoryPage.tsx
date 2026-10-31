import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  History,
  Loader2,
  MessageSquare,
  RefreshCw,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import {
  APIError,
  type ClearHistoryResponse,
  type HistoryEntry,
  clearHistory,
  fetchHistory,
} from '../lib/api'
import { formatTimestamp } from '../lib/utils'
import { Badge } from '../components/ui/Badge'
import { Card, CardContent } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ROUTES } from '../routes/routes'

// ── Constants ─────────────────────────────────────────────────────────────────
const HISTORY_LIMIT = 100
const ANSWER_COLLAPSE_THRESHOLD = 220 // chars before collapsing answer text

// ── Main Component ────────────────────────────────────────────────────────────
export default function HistoryPage() {
  const navigate = useNavigate()

  // ── Data State ──────────────────────────────────────────────────────────────
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // ── Clear History State ──────────────────────────────────────────────────────
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const [clearSuccess, setClearSuccess] = useState<string | null>(null)
  const [clearError, setClearError] = useState<string | null>(null)

  // ── Expanded Answer State ────────────────────────────────────────────────────
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set())

  // ── Fetch ─────────────────────────────────────────────────────────────────────
  const loadHistory = useCallback(async (isRefresh = false) => {
    if (isRefresh) {
      setIsRefreshing(true)
    } else {
      setIsLoading(true)
    }
    setLoadError(null)

    try {
      const data = await fetchHistory(HISTORY_LIMIT)
      setEntries(data)
    } catch (err) {
      const msg =
        err instanceof APIError
          ? err.message
          : 'Unable to reach the backend service. Please check that FastAPI is running.'
      setLoadError(msg)
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function initialFetch() {
      try {
        const data = await fetchHistory(HISTORY_LIMIT)
        if (!cancelled) setEntries(data)
      } catch (err) {
        if (!cancelled) {
          const msg =
            err instanceof APIError
              ? err.message
              : 'Unable to reach the backend service. Please check that FastAPI is running.'
          setLoadError(msg)
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    void initialFetch()
    return () => {
      cancelled = true
    }
  }, [])

  // ── Clear History ─────────────────────────────────────────────────────────────
  const handleClearHistory = async () => {
    if (isClearing) return
    setIsClearing(true)
    setClearError(null)
    setClearSuccess(null)

    try {
      const result: ClearHistoryResponse = await clearHistory()
      setEntries([])
      setClearSuccess(result.message)
      setShowClearConfirm(false)
      setTimeout(() => setClearSuccess(null), 6000)
    } catch (err) {
      const msg =
        err instanceof APIError
          ? err.message
          : 'Failed to clear query history. Backend may be offline.'
      setClearError(msg)
      // Preserve existing records on failure — do not reset entries
    } finally {
      setIsClearing(false)
    }
  }

  // ── Answer Toggle ─────────────────────────────────────────────────────────────
  const toggleExpanded = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const recordCount = entries.length

  return (
    <div className="page-container flex flex-col gap-6">
      {/* ── Page Header ──────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="page-title text-2xl">Query History</h1>
            <Badge variant="muted">Audit Trail</Badge>
          </div>
          <p className="page-subtitle mb-0">
            Chronological log of past scientific queries, sufficiency decisions, and evidence citations.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto flex-wrap">
          <button
            type="button"
            onClick={() => void loadHistory(true)}
            disabled={isLoading || isRefreshing}
            className="btn btn-secondary btn-sm flex items-center gap-1.5"
            aria-label="Refresh query history"
            title="Reload latest query history from backend"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
            />
            <span>{isRefreshing ? 'Refreshing…' : 'Refresh'}</span>
          </button>

          {/* Clear History — only enabled when there is data to clear */}
          {!isLoading && !loadError && recordCount > 0 && (
            <button
              type="button"
              onClick={() => {
                setShowClearConfirm(true)
                setClearError(null)
              }}
              disabled={isClearing}
              className="btn btn-ghost btn-sm flex items-center gap-1.5 text-red-600 hover:bg-red-50/60 hover:text-red-700"
              aria-label="Clear all query history"
              title="Permanently delete all query history records"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </button>
          )}
        </div>
      </div>

      {/* ── Notification Banners ─────────────────────────────────────────────── */}
      {clearSuccess && (
        <div
          className="flex items-center justify-between p-3 rounded-xl border text-xs"
          style={{
            background: 'var(--color-success-bg)',
            borderColor: 'var(--color-success-border)',
            color: 'var(--color-success)',
          }}
          role="status"
          aria-live="polite"
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span className="font-semibold">{clearSuccess}</span>
          </div>
          <button
            type="button"
            onClick={() => setClearSuccess(null)}
            className="p-1 hover:opacity-70"
            aria-label="Dismiss message"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {clearError && (
        <div
          className="flex items-center justify-between p-3 rounded-xl border text-xs"
          style={{
            background: 'var(--color-error-bg)',
            borderColor: 'var(--color-error-border)',
            color: 'var(--color-error)',
          }}
          role="alert"
          aria-live="assertive"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>
              <strong className="font-semibold">Clear Failed: </strong>
              {clearError}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setClearError(null)}
            className="p-1 hover:opacity-70"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ── Clear History Confirmation Modal ─────────────────────────────────── */}
      {showClearConfirm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs"
          onClick={() => {
            if (!isClearing) setShowClearConfirm(false)
          }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="confirm-clear-title"
        >
          <div
            className="card max-w-sm w-full p-6 flex flex-col gap-4 shadow-xl"
            style={{
              background: 'var(--color-bg-card)',
              borderColor: 'var(--color-border-strong)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Icon + Title */}
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                style={{ background: 'var(--color-error-bg)', color: 'var(--color-error)' }}
              >
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h3
                  id="confirm-clear-title"
                  className="font-bold text-sm"
                  style={{ color: 'var(--color-text-heading)' }}
                >
                  Clear All Query History?
                </h3>
                <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                  This will permanently delete{' '}
                  <strong style={{ color: 'var(--color-error)' }}>
                    {recordCount} {recordCount === 1 ? 'record' : 'records'}
                  </strong>{' '}
                  from the audit log.
                </p>
              </div>
            </div>

            {/* Scope Guarantee */}
            <p
              className="text-xs p-3 rounded-lg border"
              style={{
                background: 'var(--color-warning-bg)',
                borderColor: 'var(--color-warning-border)',
                color: 'var(--color-warning)',
              }}
            >
              Research documents, extracted matrices, and uploaded PDFs are{' '}
              <strong>not affected</strong>. Only the query log is cleared.
            </p>

            {/* Actions */}
            <div className="flex items-center gap-2 justify-end">
              <button
                type="button"
                onClick={() => setShowClearConfirm(false)}
                disabled={isClearing}
                className="btn btn-secondary btn-sm"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleClearHistory()}
                disabled={isClearing}
                className="btn btn-sm flex items-center gap-1.5"
                style={{
                  background: 'var(--color-error)',
                  color: '#fff',
                  borderColor: 'var(--color-error)',
                }}
                aria-label="Confirm clear all history"
              >
                {isClearing ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Clearing…</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>
                      Clear {recordCount} {recordCount === 1 ? 'Record' : 'Records'}
                    </span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Summary Count ─────────────────────────────────────────────────────── */}
      {!isLoading && !loadError && recordCount > 0 && (
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <History className="w-3.5 h-3.5" />
          <span>
            {recordCount} {recordCount === 1 ? 'query' : 'queries'} in audit log
          </span>
        </div>
      )}

      {/* ── 1. Loading Skeleton ───────────────────────────────────────────────── */}
      {isLoading && (
        <div className="flex flex-col gap-3">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardContent className="p-5 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="h-4 bg-purple-100/70 rounded-md w-2/3 animate-pulse" />
                  <div className="h-4 bg-purple-100/50 rounded-md w-20 animate-pulse" />
                </div>
                <div className="h-3 bg-purple-50 rounded w-full animate-pulse" />
                <div className="h-3 bg-purple-50 rounded w-5/6 animate-pulse" />
                <div className="flex gap-2">
                  <div className="h-5 bg-purple-100/50 rounded-full w-16 animate-pulse" />
                  <div className="h-5 bg-purple-100/50 rounded-full w-20 animate-pulse" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* ── 2. Error State with Retry ─────────────────────────────────────────── */}
      {!isLoading && loadError && (
        <Card className="border-red-200 bg-red-50/20">
          <CardContent className="p-8">
            <EmptyState
              icon={AlertCircle}
              title="Unable to Load Query History"
              description={loadError}
              action={
                <button
                  type="button"
                  onClick={() => void loadHistory(false)}
                  className="btn btn-primary btn-sm flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry Connection</span>
                </button>
              }
            />
          </CardContent>
        </Card>
      )}

      {/* ── 3. Empty State ────────────────────────────────────────────────────── */}
      {!isLoading && !loadError && recordCount === 0 && (
        <Card>
          <CardContent className="p-8">
            <EmptyState
              icon={Search}
              title="No Query History Found"
              description="Queries submitted through the Research Chat will appear here with their evidence decisions and citation details."
              action={
                <button
                  type="button"
                  onClick={() => navigate(ROUTES.chat)}
                  className="btn btn-primary btn-sm flex items-center gap-1.5"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  <span>Go to Research Chat</span>
                </button>
              }
            />
          </CardContent>
        </Card>
      )}

      {/* ── 4. History Record List ────────────────────────────────────────────── */}
      {!isLoading && !loadError && recordCount > 0 && (
        <div className="flex flex-col gap-3" role="list" aria-label="Query history records">
          {entries.map((entry) => {
            const isExpanded = expandedIds.has(entry.history_id)
            const hasLongAnswer =
              typeof entry.answer === 'string' &&
              entry.answer.length > ANSWER_COLLAPSE_THRESHOLD
            const hasCitations =
              Array.isArray(entry.citations) && entry.citations.length > 0

            return (
              <HistoryCard
                key={entry.history_id}
                entry={entry}
                isExpanded={isExpanded}
                hasLongAnswer={hasLongAnswer}
                hasCitations={hasCitations}
                onToggleExpand={() => toggleExpanded(entry.history_id)}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── HistoryCard Sub-Component ─────────────────────────────────────────────────

interface HistoryCardProps {
  entry: HistoryEntry
  isExpanded: boolean
  hasLongAnswer: boolean
  hasCitations: boolean
  onToggleExpand: () => void
}

function HistoryCard({
  entry,
  isExpanded,
  hasLongAnswer,
  hasCitations,
  onToggleExpand,
}: HistoryCardProps) {
  const answerText = entry.answer ?? null
  const displayedAnswer =
    hasLongAnswer && !isExpanded && answerText
      ? answerText.slice(0, ANSWER_COLLAPSE_THRESHOLD).trimEnd() + '…'
      : answerText

  return (
    <div
      className="card p-4 sm:p-5 flex flex-col gap-3 transition-colors hover:bg-[var(--color-bg-hover)]"
      style={{
        background: 'var(--color-bg-card)',
        borderColor: 'var(--color-border)',
      }}
      role="listitem"
      aria-label={`Query: ${entry.query}`}
    >
      {/* Header: query text + timestamp */}
      <div className="flex items-start justify-between gap-3">
        <p
          className="font-semibold text-sm leading-snug flex-1 break-words"
          style={{ color: 'var(--color-text-heading)' }}
        >
          {entry.query}
        </p>
        <div
          className="flex items-center gap-1 shrink-0 text-[11px] font-mono"
          style={{ color: 'var(--color-text-muted)' }}
          title={entry.timestamp}
        >
          <Clock className="w-3 h-3" />
          <span>{formatTimestampSafe(entry.timestamp)}</span>
        </div>
      </div>

      {/* Badge Row: evidence + source + score + best_source */}
      <div className="flex flex-wrap items-center gap-1.5">
        <EvidenceBadge sufficient={entry.has_supporting_evidence} />
        <SourceBadge source={entry.answer_source} />
        {typeof entry.best_score === 'number' && (
          <Badge variant="muted" className="text-[10px] font-mono">
            Score: {entry.best_score.toFixed(3)}
          </Badge>
        )}
        {entry.best_source && (
          <Badge variant="muted" className="text-[10px]">
            {entry.best_source}
          </Badge>
        )}
      </div>

      {/* Answer Text */}
      {answerText !== null && (
        <div className="flex flex-col gap-1">
          <div
            className="text-xs leading-relaxed break-words whitespace-pre-wrap"
            style={{ color: 'var(--color-text-main)' }}
          >
            {displayedAnswer}
          </div>
          {hasLongAnswer && (
            <button
              type="button"
              onClick={onToggleExpand}
              className="inline-flex items-center gap-0.5 text-[10px] font-semibold text-purple-700 hover:text-purple-900 self-start mt-0.5"
              aria-label={isExpanded ? 'Collapse answer' : 'Expand full answer'}
            >
              {isExpanded ? (
                <>
                  <ChevronUp className="w-3 h-3" /> Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-3 h-3" /> Show full answer ({answerText.length} chars)
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Citations */}
      {hasCitations && (
        <div className="flex flex-wrap gap-1.5" aria-label="Evidence citations">
          {entry.citations.map((cite, idx) => (
            <span
              // biome-ignore lint/suspicious/noArrayIndexKey: citation list has no stable key
              key={idx}
              className="badge badge-lavender text-[10px] font-mono"
            >
              {cite}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Badge Helpers ─────────────────────────────────────────────────────────────

function EvidenceBadge({ sufficient }: { sufficient: boolean }) {
  return (
    <Badge variant={sufficient ? 'success' : 'warning'} className="text-[10px]">
      {sufficient ? '✓ Evidence Supported' : '⚠ Insufficient Evidence'}
    </Badge>
  )
}

function SourceBadge({ source }: { source: string }) {
  const variant =
    source === 'ollama' ? 'lavender' : source === 'abstention' ? 'warning' : 'muted'
  const label =
    source === 'ollama'
      ? 'Ollama LLM'
      : source === 'abstention'
      ? 'Abstained'
      : source || 'Unknown source'

  return (
    <Badge variant={variant} className="text-[10px]">
      {label}
    </Badge>
  )
}

// ── Timestamp Helper ──────────────────────────────────────────────────────────

function formatTimestampSafe(ts: unknown): string {
  if (!ts || typeof ts !== 'string') return '—'
  try {
    return formatTimestamp(ts)
  } catch {
    return String(ts)
  }
}
