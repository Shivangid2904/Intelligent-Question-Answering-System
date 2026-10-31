import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  BookOpen,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  Database,
  Download,
  Eye,
  FileSpreadsheet,
  FileText,
  Filter,
  Layers,
  Loader2,
  RefreshCw,
  Search,
  Sparkles,
  Table2,
  Users,
  X,
} from 'lucide-react'
import {
  APIError,
  CANONICAL_MATRIX_COLUMNS,
  exportMatrixCSV,
  fetchMatrix,
} from '../lib/api'
import type { MatrixResponse, MatrixRow } from '../lib/api'
import { formatTimestamp } from '../lib/utils'
import { Badge } from '../components/ui/Badge'
import { Card, CardContent } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ROUTES } from '../routes/routes'

export default function MatrixPage() {
  const navigate = useNavigate()

  // ── State ─────────────────────────────────────────────────────────────────
  const [matrixData, setMatrixData] = useState<MatrixResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Export State
  const [isExporting, setIsExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportSuccess, setExportSuccess] = useState<string | null>(null)

  // Inspection Modal & Expanded Cells State
  const [selectedRow, setSelectedRow] = useState<MatrixRow | null>(null)
  const [expandedCellKeys, setExpandedCellKeys] = useState<Set<string>>(new Set())

  // ── Fetch Matrix ──────────────────────────────────────────────────────────
  const loadMatrix = useCallback(async (isRefresh = false) => {
    if (isRefresh) {
      setIsRefreshing(true)
    } else {
      setIsLoading(true)
    }
    setLoadError(null)

    try {
      const data = await fetchMatrix()
      setMatrixData(data)
    } catch (err) {
      const msg =
        err instanceof APIError
          ? err.message
          : 'Unable to reach the backend service. Please check that FastAPI is running.'
      setLoadError(msg)
      setMatrixData(null)
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function initialFetch() {
      try {
        const data = await fetchMatrix()
        if (!cancelled) setMatrixData(data)
      } catch (err) {
        if (!cancelled) {
          const msg =
            err instanceof APIError
              ? err.message
              : 'Unable to reach the backend service. Please check that FastAPI is running.'
          setLoadError(msg)
          setMatrixData(null)
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

  // ── CSV Export ────────────────────────────────────────────────────────────
  const handleExportCSV = async () => {
    if (isExporting) return
    setIsExporting(true)
    setExportError(null)
    setExportSuccess(null)

    try {
      const { blob, filename } = await exportMatrixCSV()

      // Trigger browser download safely
      if (typeof window !== 'undefined' && typeof document !== 'undefined') {
        const downloadUrl = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.style.display = 'none'
        a.href = downloadUrl
        a.download = filename
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(downloadUrl)
        document.body.removeChild(a)
      }

      setExportSuccess(`Downloaded ${filename} successfully`)
      setTimeout(() => {
        setExportSuccess(null)
      }, 5000)
    } catch (err) {
      const msg =
        err instanceof APIError
          ? err.message
          : 'Failed to download matrix CSV from backend.'
      setExportError(msg)
    } finally {
      setIsExporting(false)
    }
  };

  // ── Cell Expansion Toggle ─────────────────────────────────────────────────
  const toggleCellExpansion = (key: string) => {
    setExpandedCellKeys((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  // ── Columns ───────────────────────────────────────────────────────────────
  // Prefer backend columns if provided, otherwise fallback to canonical columns
  const activeColumns = useMemo(() => {
    if (matrixData?.columns && matrixData.columns.length > 0) {
      return matrixData.columns
    }
    return Array.from(CANONICAL_MATRIX_COLUMNS)
  }, [matrixData?.columns])

  // ── Filtered Rows ─────────────────────────────────────────────────────────
  const filteredRows = useMemo(() => {
    if (!matrixData?.rows) return []
    const q = searchQuery.trim().toLowerCase()
    if (!q) return matrixData.rows

    return matrixData.rows.filter((row) => {
      return activeColumns.some((col) => {
        const val = row[col]
        if (val === null || val === undefined) return false
        return String(val).toLowerCase().includes(q)
      })
    })
  }, [matrixData, activeColumns, searchQuery])

  // Paper count: derive accurately from response
  const paperCount = matrixData?.total_rows ?? matrixData?.rows?.length ?? 0

  return (
    <div className="page-container flex flex-col gap-6">
      {/* ── Page Header ────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="page-title text-2xl">Literature Matrix</h1>
            <Badge variant="lavender">Comparative Synthesis</Badge>
          </div>
          <p className="page-subtitle mb-0">
            Structured 12-dimensional comparison of research methodology, datasets, hardware sensors, quantitative metrics, and limitations.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto flex-wrap">
          <button
            type="button"
            onClick={() => void loadMatrix(true)}
            disabled={isLoading || isRefreshing}
            className="btn btn-secondary btn-sm flex items-center gap-1.5"
            aria-label="Refresh literature matrix"
            title="Reload latest matrix data from backend"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
            />
            <span>{isRefreshing ? 'Refreshing…' : 'Refresh'}</span>
          </button>

          <button
            type="button"
            onClick={() => void handleExportCSV()}
            disabled={isLoading || isExporting || paperCount === 0}
            className="btn btn-primary btn-sm flex items-center gap-1.5"
            aria-label="Export matrix as CSV"
            title={
              paperCount === 0
                ? 'No matrix rows to export'
                : 'Download RFC 4180 CSV generated by FastAPI backend'
            }
          >
            {isExporting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Exporting CSV…</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Export CSV</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Notification Banners (Export Success / Error) ───────────────────── */}
      {exportSuccess && (
        <div
          className="flex items-center justify-between p-3 rounded-xl border text-xs"
          style={{
            background: 'var(--color-success-bg)',
            borderColor: 'var(--color-success-border)',
            color: 'var(--color-success)',
          }}
          role="status"
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span className="font-semibold">{exportSuccess}</span>
          </div>
          <button
            type="button"
            onClick={() => setExportSuccess(null)}
            className="p-1 hover:opacity-80"
            aria-label="Dismiss message"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {exportError && (
        <div
          className="flex items-center justify-between p-3 rounded-xl border text-xs"
          style={{
            background: 'var(--color-error-bg)',
            borderColor: 'var(--color-error-border)',
            color: 'var(--color-error)',
          }}
          role="alert"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>
              <strong className="font-semibold">Export Failed: </strong>
              {exportError}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setExportError(null)}
            className="p-1 hover:opacity-80"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ── Key Dimension Summary Metrics ──────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div
          className="card p-3.5 flex items-center gap-3"
          style={{ background: 'var(--color-bg-card)', borderColor: 'var(--color-border)' }}
        >
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: 'var(--color-lavender-soft)', color: 'var(--color-primary)' }}
          >
            <FileSpreadsheet className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Total Papers
            </div>
            <div className="text-base font-bold" style={{ color: 'var(--color-text-heading)' }}>
              {isLoading ? '…' : paperCount}
            </div>
          </div>
        </div>

        <div
          className="card p-3.5 flex items-center gap-3"
          style={{ background: 'var(--color-bg-card)', borderColor: 'var(--color-border)' }}
        >
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: 'var(--color-lavender-soft)', color: 'var(--color-primary)' }}
          >
            <Table2 className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Dimensions
            </div>
            <div className="text-base font-bold" style={{ color: 'var(--color-text-heading)' }}>
              {activeColumns.length} Columns
            </div>
          </div>
        </div>

        <div
          className="card p-3.5 flex items-center gap-3"
          style={{ background: 'var(--color-bg-card)', borderColor: 'var(--color-border)' }}
        >
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: 'var(--color-lavender-soft)', color: 'var(--color-primary)' }}
          >
            <CheckCircle2 className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Export Format
            </div>
            <div className="text-xs font-semibold" style={{ color: 'var(--color-text-heading)' }}>
              RFC 4180 CSV
            </div>
          </div>
        </div>

        <div
          className="card p-3.5 flex items-center gap-3"
          style={{ background: 'var(--color-bg-card)', borderColor: 'var(--color-border)' }}
        >
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: 'var(--color-lavender-soft)', color: 'var(--color-primary)' }}
          >
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Synthesis Mode
            </div>
            <div className="text-xs font-semibold" style={{ color: 'var(--color-text-heading)' }}>
              Cross-Study
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Matrix Section ────────────────────────────────────────────── */}
      <div className="flex flex-col gap-4">
        {/* Search & Filter Bar (Only shown when data is loaded and not empty) */}
        {!isLoading && !loadError && paperCount > 0 && (
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Filter by title, author, methodology, or metric…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input pl-9 pr-8 text-xs w-full"
                aria-label="Filter matrix table"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 p-0.5 text-gray-400 hover:text-gray-600"
                  aria-label="Clear filter"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              <Filter className="w-3.5 h-3.5" />
              <span>
                Showing {filteredRows.length} of {paperCount} {paperCount === 1 ? 'paper' : 'papers'}
              </span>
            </div>
          </div>
        )}

        {/* 1. Loading Skeleton State */}
        {isLoading && (
          <Card>
            <CardContent className="p-6 flex flex-col gap-4">
              <div className="flex items-center justify-between pb-2">
                <div className="h-5 bg-purple-100/70 rounded-md w-48 animate-pulse" />
                <div className="h-5 bg-purple-100/50 rounded-md w-28 animate-pulse" />
              </div>
              <div className="overflow-hidden border rounded-xl" style={{ borderColor: 'var(--color-border)' }}>
                <div className="h-10 bg-purple-100/60 animate-pulse" />
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="h-14 border-t flex items-center px-4 gap-4 animate-pulse"
                    style={{ borderColor: 'var(--color-border)', background: i % 2 === 0 ? 'var(--color-bg-main)' : 'var(--color-bg-card)' }}
                  >
                    <div className="h-4 bg-purple-100/60 rounded w-1/6" />
                    <div className="h-4 bg-purple-50 rounded w-1/4" />
                    <div className="h-4 bg-purple-50 rounded w-1/6" />
                    <div className="h-4 bg-purple-50 rounded w-1/12" />
                    <div className="h-4 bg-purple-50 rounded w-1/3" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* 2. Error State with Retry Button */}
        {!isLoading && loadError && (
          <Card className="border-red-200 bg-red-50/20">
            <CardContent className="p-8">
              <EmptyState
                icon={AlertCircle}
                title="Unable to Load Literature Matrix"
                description={loadError}
                action={
                  <button
                    type="button"
                    onClick={() => void loadMatrix(false)}
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

        {/* 3. Empty State (No extractions yet) */}
        {!isLoading && !loadError && paperCount === 0 && (
          <Card>
            <CardContent className="p-8">
              <EmptyState
                icon={Table2}
                title="No Literature Extractions Found"
                description="The literature matrix presents comparative extractions across 12 analytical dimensions. To populate the matrix, upload research papers in the Research Library."
                action={
                  <button
                    type="button"
                    onClick={() => navigate(ROUTES.library)}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                  >
                    <BookOpen className="w-3.5 h-3.5" />
                    <span>Go to Research Library</span>
                  </button>
                }
              />
            </CardContent>
          </Card>
        )}

        {/* 4. Filter Yielded Zero Matches */}
        {!isLoading && !loadError && paperCount > 0 && filteredRows.length === 0 && (
          <Card>
            <CardContent className="p-6">
              <EmptyState
                icon={Search}
                title="No Matching Papers Found"
                description={`No literature extractions match "${searchQuery}".`}
                action={
                  <button
                    type="button"
                    onClick={() => setSearchQuery('')}
                    className="btn btn-ghost btn-sm"
                  >
                    Clear search filter
                  </button>
                }
              />
            </CardContent>
          </Card>
        )}

        {/* 5. Literature Matrix Table */}
        {!isLoading && !loadError && filteredRows.length > 0 && (
          <div
            className="card rounded-2xl overflow-hidden border shadow-sm flex flex-col"
            style={{
              background: 'var(--color-bg-card)',
              borderColor: 'var(--color-border)',
            }}
          >
            {/* Horizontally scrollable responsive table container */}
            <div className="overflow-x-auto w-full max-h-[70vh] relative">
              <table className="w-full text-left text-xs border-collapse min-w-[1500px]">
                {/* Canonical Table Header */}
                <thead
                  className="sticky top-0 z-10 border-b select-none shadow-xs"
                  style={{
                    background: 'var(--color-bg-sidebar)',
                    borderColor: 'var(--color-border)',
                  }}
                >
                  <tr>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider sticky left-0 z-20 border-r"
                      style={{
                        background: 'var(--color-bg-sidebar)',
                        borderColor: 'var(--color-border)',
                        color: 'var(--color-text-heading)',
                        width: '180px',
                        minWidth: '180px',
                      }}
                    >
                      Document
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '220px', maxWidth: '280px' }}
                    >
                      Title
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '150px', maxWidth: '200px' }}
                    >
                      Authors
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-3 font-semibold text-[11px] uppercase tracking-wider text-center"
                      style={{ color: 'var(--color-text-heading)', width: '70px', minWidth: '70px' }}
                    >
                      Year
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '220px', maxWidth: '320px' }}
                    >
                      Research Objective
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '220px', maxWidth: '320px' }}
                    >
                      Methodology
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '160px', maxWidth: '220px' }}
                    >
                      Dataset
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '160px', maxWidth: '220px' }}
                    >
                      Hardware / Sensors
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '200px', maxWidth: '280px' }}
                    >
                      Key Metrics
                    </th>
                    <th
                      scope="col"
                      className="px-3.5 py-3 font-semibold text-[11px] uppercase tracking-wider"
                      style={{ color: 'var(--color-text-heading)', minWidth: '200px', maxWidth: '280px' }}
                    >
                      Limitations
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-3 font-semibold text-[11px] uppercase tracking-wider text-center"
                      style={{ color: 'var(--color-text-heading)', minWidth: '110px' }}
                    >
                      Pages Cited
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-3 font-semibold text-[11px] uppercase tracking-wider text-right"
                      style={{ color: 'var(--color-text-heading)', minWidth: '140px' }}
                    >
                      Extracted
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-3 font-semibold text-[11px] uppercase tracking-wider text-center"
                      style={{ color: 'var(--color-text-heading)', width: '60px' }}
                    >
                      Inspect
                    </th>
                  </tr>
                </thead>

                {/* Table Body */}
                <tbody className="divide-y divide-[var(--color-border)]">
                  {filteredRows.map((row, rowIdx) => {
                    const docName = String(row['Document'] || 'Untitled Document')
                    const yearVal = row['Year']
                    const pagesCited = row['Pages Cited']
                    const timestampVal = row['Extraction Timestamp']

                    return (
                      <tr
                        key={docName + rowIdx}
                        className="transition-colors hover:bg-[var(--color-bg-hover)] group"
                      >
                        {/* 1. Document (Sticky Column for Easy Context) */}
                        <td
                          className="px-3.5 py-3 align-top font-semibold sticky left-0 z-10 border-r"
                          style={{
                            background: 'var(--color-bg-card)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-primary)',
                          }}
                        >
                          <div className="flex items-start gap-1.5">
                            <FileText className="w-3.5 h-3.5 shrink-0 mt-0.5 opacity-70" />
                            <span className="break-words font-medium text-xs leading-snug">
                              {docName}
                            </span>
                          </div>
                        </td>

                        {/* 2. Title */}
                        <td className="px-3.5 py-3 align-top font-medium" style={{ color: 'var(--color-text-heading)' }}>
                          <div className="leading-snug max-w-xs break-words">
                            {formatCellValue(row['Title'])}
                          </div>
                        </td>

                        {/* 3. Authors */}
                        <td className="px-3.5 py-3 align-top" style={{ color: 'var(--color-text-muted)' }}>
                          <div className="leading-snug max-w-xs break-words">
                            {formatCellValue(row['Authors'])}
                          </div>
                        </td>

                        {/* 4. Year */}
                        <td className="px-3 py-3 align-top text-center">
                          {yearVal ? (
                            <span className="badge badge-muted font-mono text-[11px]">
                              {String(yearVal)}
                            </span>
                          ) : (
                            <span className="text-gray-400 italic">—</span>
                          )}
                        </td>

                        {/* 5. Research Objective (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `obj-${rowIdx}`,
                            row['Research Objective'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 6. Methodology (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `meth-${rowIdx}`,
                            row['Methodology'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 7. Dataset (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `data-${rowIdx}`,
                            row['Dataset'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 8. Hardware/Sensors (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `hw-${rowIdx}`,
                            row['Hardware/Sensors'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 9. Key Metrics (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `metrics-${rowIdx}`,
                            row['Key Metrics'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 10. Limitations (Expandable) */}
                        <td className="px-3.5 py-3 align-top">
                          {renderExpandableCell(
                            `limits-${rowIdx}`,
                            row['Limitations'],
                            expandedCellKeys,
                            toggleCellExpansion,
                          )}
                        </td>

                        {/* 11. Pages Cited */}
                        <td className="px-3 py-3 align-top text-center">
                          {pagesCited ? (
                            <Badge variant="lavender" className="text-[10px] font-mono">
                              p. {String(pagesCited)}
                            </Badge>
                          ) : (
                            <span className="text-gray-400 italic">—</span>
                          )}
                        </td>

                        {/* 12. Extraction Timestamp */}
                        <td
                          className="px-3 py-3 align-top text-right text-[11px] font-mono"
                          style={{ color: 'var(--color-text-muted)' }}
                        >
                          {timestampVal ? formatTimestamp(String(timestampVal)) : '—'}
                        </td>

                        {/* Inspect Action */}
                        <td className="px-3 py-3 align-top text-center">
                          <button
                            type="button"
                            onClick={() => setSelectedRow(row)}
                            className="p-1.5 rounded-lg text-purple-700 hover:bg-purple-100/60 transition-colors"
                            title="Inspect full paper extraction details"
                            aria-label={`Inspect extraction details for ${docName}`}
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Table Footer with Summary */}
            <div
              className="p-3.5 border-t flex flex-col sm:flex-row items-center justify-between gap-2 text-xs"
              style={{
                borderColor: 'var(--color-border)',
                background: 'var(--color-bg-secondary)',
                color: 'var(--color-text-muted)',
              }}
            >
              <div className="flex items-center gap-2">
                <span className="font-semibold text-gray-700">Literature Matrix Summary:</span>
                <span>
                  {paperCount} {paperCount === 1 ? 'study' : 'studies'} evaluated across 12 canonical fields.
                </span>
              </div>
              <div className="flex items-center gap-3 text-[11px]">
                <span className="flex items-center gap-1">
                  <span className="status-dot status-dot-success" /> Schema Version 1
                </span>
                <span>•</span>
                <span>RFC 4180 Validated Export</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Paper Details Modal (Inspect Full Extraction) ───────────────────── */}
      {selectedRow && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs animate-in fade-in duration-200"
          onClick={() => setSelectedRow(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="paper-details-title"
        >
          <div
            className="card max-w-2xl w-full max-h-[85vh] flex flex-col overflow-hidden shadow-xl"
            style={{
              background: 'var(--color-bg-card)',
              borderColor: 'var(--color-border-strong)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              className="p-4 sm:p-5 border-b flex items-start justify-between gap-3"
              style={{
                background: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-start gap-3 min-w-0">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 mt-0.5"
                  style={{ background: 'var(--color-lavender-soft)', color: 'var(--color-primary)' }}
                >
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <div className="min-w-0">
                  <h3
                    id="paper-details-title"
                    className="font-bold text-base leading-snug break-words"
                    style={{ color: 'var(--color-text-heading)' }}
                  >
                    {String(selectedRow['Document'] || 'Paper Extraction Details')}
                  </h3>
                  <div className="flex flex-wrap items-center gap-2 text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                    {selectedRow['Year'] && (
                      <span className="badge badge-muted font-mono text-[10px]">
                        Year {String(selectedRow['Year'])}
                      </span>
                    )}
                    {selectedRow['Pages Cited'] && (
                      <Badge variant="lavender" className="text-[10px]">
                        Pages Cited: {String(selectedRow['Pages Cited'])}
                      </Badge>
                    )}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setSelectedRow(null)}
                className="p-1 rounded-lg hover:bg-gray-200/50 text-gray-500 hover:text-gray-800"
                aria-label="Close modal"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body: Complete 12 Canonical Dimensions */}
            <div className="p-5 sm:p-6 overflow-y-auto flex flex-col gap-4 text-xs leading-relaxed">
              <DetailField
                icon={FileText}
                label="Full Title"
                value={selectedRow['Title']}
              />
              <DetailField
                icon={Users}
                label="Authors"
                value={selectedRow['Authors']}
              />
              <DetailField
                icon={Sparkles}
                label="Research Objective"
                value={selectedRow['Research Objective']}
              />
              <DetailField
                icon={Layers}
                label="Methodology & Algorithms"
                value={selectedRow['Methodology']}
              />
              <DetailField
                icon={Database}
                label="Datasets & Benchmarks"
                value={selectedRow['Dataset']}
              />
              <DetailField
                icon={Cpu}
                label="Hardware / Sensors"
                value={selectedRow['Hardware/Sensors']}
              />
              <DetailField
                icon={CheckCircle2}
                label="Key Findings & Quantitative Metrics"
                value={selectedRow['Key Metrics']}
              />
              <DetailField
                icon={AlertCircle}
                label="Reported Limitations"
                value={selectedRow['Limitations']}
              />
              <DetailField
                icon={Calendar}
                label="Publication Year"
                value={selectedRow['Year']}
              />
              <DetailField
                icon={Clock}
                label="Extraction Timestamp"
                value={selectedRow['Extraction Timestamp']}
              />
            </div>

            {/* Modal Footer */}
            <div
              className="p-3.5 border-t flex items-center justify-end gap-2"
              style={{
                borderColor: 'var(--color-border)',
                background: 'var(--color-bg-secondary)',
              }}
            >
              <button
                type="button"
                onClick={() => setSelectedRow(null)}
                className="btn btn-secondary btn-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Helper Sub-Components ───────────────────────────────────────────────────

function formatCellValue(val: unknown): string {
  if (val === null || val === undefined || val === '') {
    return '—'
  }
  return String(val)
}

function renderExpandableCell(
  key: string,
  val: unknown,
  expandedKeys: Set<string>,
  toggle: (key: string) => void,
) {
  if (val === null || val === undefined || val === '') {
    return <span className="text-gray-400 italic">—</span>
  }

  const str = String(val)
  const isExpanded = expandedKeys.has(key)
  const isLong = str.length > 90

  if (!isLong) {
    return (
      <div className="leading-snug max-w-xs break-words" style={{ color: 'var(--color-text-main)' }}>
        {str}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 max-w-xs">
      <div
        className={`leading-snug break-words transition-all duration-200 ${
          isExpanded ? '' : 'line-clamp-3'
        }`}
        style={{ color: 'var(--color-text-main)' }}
      >
        {str}
      </div>
      <button
        type="button"
        onClick={() => toggle(key)}
        className="inline-flex items-center gap-0.5 text-[10px] font-semibold text-purple-700 hover:text-purple-900 self-start mt-0.5"
        aria-label={isExpanded ? 'Collapse text' : 'Expand full text'}
      >
        {isExpanded ? (
          <>
            <ChevronUp className="w-3 h-3" /> Show less
          </>
        ) : (
          <>
            <ChevronDown className="w-3 h-3" /> Show full ({str.length} chars)
          </>
        )}
      </button>
    </div>
  )
}

interface DetailFieldProps {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: unknown
}

function DetailField({ icon: Icon, label, value }: DetailFieldProps) {
  const formatted = formatCellValue(value)
  const isEmpty = formatted === '—'

  return (
    <div
      className="p-3 rounded-xl border flex flex-col gap-1"
      style={{
        background: 'var(--color-bg-main)',
        borderColor: 'var(--color-border)',
      }}
    >
      <div className="flex items-center gap-1.5 font-semibold text-[11px]" style={{ color: 'var(--color-primary)' }}>
        <Icon className="w-3.5 h-3.5 opacity-80" />
        <span>{label}</span>
      </div>
      <div
        className={`text-xs whitespace-pre-wrap break-words leading-relaxed ${
          isEmpty ? 'text-gray-400 italic' : 'text-gray-800'
        }`}
      >
        {formatted}
      </div>
    </div>
  )
}
