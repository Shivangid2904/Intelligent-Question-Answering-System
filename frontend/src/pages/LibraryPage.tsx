import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Clock,
  Database,
  FileCheck2,
  FileText,
  FileUp,
  Hash,
  Layers,
  Loader2,
  RefreshCw,
  Search,
  UploadCloud,
  X,
} from 'lucide-react'
import {
  APIError,
  MAX_UPLOAD_SIZE_BYTES,
  fetchDocuments,
  uploadDocument,
} from '../lib/api'
import type {
  DocumentRecord,
  DocumentUploadResponse,
} from '../lib/api'
import { formatBytes, formatTimestamp, truncate } from '../lib/utils'
import { Badge } from '../components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'

export default function LibraryPage() {
  // ── Document List State ───────────────────────────────────────────────────
  const [documents, setDocuments] = useState<DocumentRecord[] | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [searchFilter, setSearchFilter] = useState('')

  // ── Upload State ──────────────────────────────────────────────────────────
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState<DocumentUploadResponse | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Fetch Documents ───────────────────────────────────────────────────────
  const refreshDocuments = useCallback(async () => {
    setIsRefreshing(true)
    setLoadError(null)

    try {
      const data = await fetchDocuments()
      setDocuments(data)
    } catch (err) {
      const msg =
        err instanceof APIError
          ? err.message
          : 'Unable to reach the backend service. Please check that FastAPI is running on port 8000.'
      setLoadError(msg)
      setDocuments(null)
    } finally {
      setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function initialFetch() {
      try {
        const data = await fetchDocuments()
        if (!cancelled) setDocuments(data)
      } catch (err) {
        if (!cancelled) {
          const msg =
            err instanceof APIError
              ? err.message
              : 'Unable to reach the backend service. Please check that FastAPI is running on port 8000.'
          setLoadError(msg)
          setDocuments(null)
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

  // ── File Selection & Validation ───────────────────────────────────────────
  const validateAndSelectFile = (file: File | undefined | null) => {
    setValidationError(null)
    setUploadError(null)
    setUploadSuccess(null)

    if (!file) {
      setSelectedFile(null)
      return
    }

    const isPdf =
      file.name.toLowerCase().endsWith('.pdf') ||
      file.type === 'application/pdf'

    if (!isPdf) {
      setValidationError('Invalid file format. Only PDF documents (.pdf) are supported.')
      setSelectedFile(null)
      return
    }

    if (file.size === 0) {
      setValidationError('Selected file is empty (0 bytes). Please select a valid document.')
      setSelectedFile(null)
      return
    }

    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1)
      setValidationError(
        `File is too large (${sizeMb} MB). Maximum supported file size is 50 MB.`,
      )
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    validateAndSelectFile(file)
  }

  // ── Drag & Drop Handlers ──────────────────────────────────────────────────
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (!isUploading) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (isUploading) return

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      validateAndSelectFile(files[0])
    }
  }

  // ── Upload Submission ─────────────────────────────────────────────────────
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile || isUploading) return

    setIsUploading(true)
    setUploadError(null)
    setUploadSuccess(null)

    try {
      const response = await uploadDocument(selectedFile)
      setUploadSuccess(response)
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      // Refresh library list with updated documents
      await refreshDocuments()
    } catch (err) {
      const message =
        err instanceof APIError
          ? err.message
          : 'An unexpected error occurred during document upload.'
      setUploadError(message)
    } finally {
      setIsUploading(false)
    }
  }

  const clearSelection = () => {
    setSelectedFile(null)
    setValidationError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Filtered documents by user search
  const filteredDocuments = documents?.filter((doc) =>
    doc.filename.toLowerCase().includes(searchFilter.trim().toLowerCase()),
  )

  return (
    <div className="page-container flex flex-col gap-8">
      {/* ── Page Header ───────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="page-title text-2xl sm:text-3xl">Research Library</h1>
            {documents !== null && (
              <Badge variant="lavender" id="doc-count-badge">
                <FileText className="w-3 h-3 mr-1" />
                {documents.length} {documents.length === 1 ? 'Paper' : 'Papers'}
              </Badge>
            )}
          </div>
          <p className="page-subtitle mb-0">
            Manage scientific research papers indexed for evidence-gated inquiry and literature synthesis.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={() => refreshDocuments()}
            disabled={isLoading || isRefreshing}
            className="btn btn-secondary btn-sm flex items-center gap-1.5"
            aria-label="Refresh library"
            title="Refresh documents list"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
            />
            <span>{isRefreshing ? 'Refreshing…' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* ── PDF Upload Card ───────────────────────────────────────────────── */}
      <Card
        className="overflow-hidden border-2 transition-all duration-200"
        style={{
          background: isDragging
            ? 'linear-gradient(135deg, #F0E8FF 0%, #FDF4FA 100%)'
            : 'linear-gradient(135deg, #FBF8FF 0%, #F8F1FD 100%)',
          borderColor: isDragging ? 'var(--color-primary)' : 'var(--color-border)',
        }}
      >
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <UploadCloud className="w-5 h-5" style={{ color: 'var(--color-primary)' }} />
              Add Papers to Your Research Library
            </CardTitle>
            <Badge variant="pink">Max 50 MB / PDF</Badge>
          </div>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleUpload} className="flex flex-col gap-4">
            {/* Drag & Drop Target Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !isUploading && fileInputRef.current?.click()}
              className={`
                flex flex-col items-center justify-center p-8 rounded-2xl
                border-2 border-dashed cursor-pointer transition-all duration-200
                ${
                  isDragging
                    ? 'border-purple-500 bg-purple-50/50 scale-[1.005]'
                    : 'border-purple-200 hover:border-purple-400 bg-white/70 hover:bg-white/90'
                }
              `}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  fileInputRef.current?.click()
                }
              }}
              aria-label="Select PDF file to upload"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileInputChange}
                className="hidden"
                disabled={isUploading}
                aria-hidden="true"
              />

              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3 shadow-xs"
                style={{
                  background: isDragging
                    ? 'var(--color-lavender)'
                    : 'var(--color-lavender-soft)',
                }}
              >
                <FileUp
                  className="w-7 h-7"
                  style={{ color: 'var(--color-primary)' }}
                />
              </div>

              <div className="text-center">
                <p
                  className="text-sm font-semibold mb-1"
                  style={{ color: 'var(--color-text-heading)' }}
                >
                  {isDragging
                    ? 'Drop the research PDF here'
                    : 'Click to browse or drag and drop your PDF'}
                </p>
                <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  Accepts scientific papers in PDF format. Text is chunked and indexed into the active QA engine.
                </p>
              </div>
            </div>

            {/* Selected File Card */}
            {selectedFile && (
              <div
                className="flex items-center justify-between p-3.5 rounded-xl border"
                style={{
                  background: 'var(--color-bg-card)',
                  borderColor: 'var(--color-border-strong)',
                }}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                    style={{ background: 'var(--color-lavender-soft)' }}
                  >
                    <FileText
                      className="w-5 h-5"
                      style={{ color: 'var(--color-primary)' }}
                    />
                  </div>
                  <div className="min-w-0">
                    <p
                      className="text-sm font-semibold truncate"
                      style={{ color: 'var(--color-text-heading)' }}
                    >
                      {selectedFile.name}
                    </p>
                    <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                      {formatBytes(selectedFile.size)} • PDF ready for upload
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={clearSelection}
                    disabled={isUploading}
                    className="btn btn-ghost btn-sm p-1.5"
                    aria-label="Remove selected file"
                    title="Deselect file"
                  >
                    <X className="w-4 h-4" />
                  </button>

                  <button
                    type="submit"
                    disabled={isUploading}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                    aria-label="Confirm upload"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Uploading & Indexing…</span>
                      </>
                    ) : (
                      <>
                        <FileUp className="w-3.5 h-3.5" />
                        <span>Upload & Index</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Validation Error Banner */}
            {validationError && (
              <div
                className="flex items-start gap-2.5 p-3 rounded-xl border text-xs"
                style={{
                  background: 'var(--color-warning-bg)',
                  borderColor: 'var(--color-warning-border)',
                  color: 'var(--color-warning)',
                }}
                role="alert"
              >
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold">Validation warning: </span>
                  {validationError}
                </div>
              </div>
            )}

            {/* Server Upload Error Banner */}
            {uploadError && (
              <div
                className="flex items-start gap-2.5 p-3 rounded-xl border text-xs"
                style={{
                  background: 'var(--color-error-bg)',
                  borderColor: 'var(--color-error-border)',
                  color: 'var(--color-error)',
                }}
                role="alert"
              >
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold">Upload failed: </span>
                  {uploadError}
                </div>
                <button
                  type="button"
                  onClick={() => setUploadError(null)}
                  className="p-0.5 text-current opacity-70 hover:opacity-100"
                  aria-label="Dismiss error"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Upload Success Banner */}
            {uploadSuccess && (
              <div
                className="flex items-start gap-3 p-4 rounded-xl border"
                style={{
                  background: 'var(--color-success-bg)',
                  borderColor: 'var(--color-success-border)',
                  color: 'var(--color-success)',
                }}
                role="status"
              >
                <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" />
                <div className="flex-1 flex flex-col gap-1 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm">
                      {uploadSuccess.filename} registered successfully
                    </span>
                    {uploadSuccess.is_duplicate && (
                      <Badge variant="muted">Duplicate Record</Badge>
                    )}
                    {uploadSuccess.is_searchable ? (
                      <Badge variant="success">Indexed & Searchable</Badge>
                    ) : (
                      <Badge variant="warning">
                        Index: {uploadSuccess.indexing_status}
                      </Badge>
                    )}
                  </div>
                  <p className="opacity-90">{uploadSuccess.message}</p>
                  <div className="flex items-center gap-3 pt-1 text-[11px] opacity-80">
                    <span>{uploadSuccess.page_count} pages</span>
                    <span>•</span>
                    <span>{uploadSuccess.chunk_count} text chunks</span>
                    <span>•</span>
                    <span>{uploadSuccess.char_count.toLocaleString()} characters</span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setUploadSuccess(null)}
                  className="p-1 text-current opacity-70 hover:opacity-100"
                  aria-label="Dismiss notification"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
          </form>
        </CardContent>
      </Card>

      {/* ── Document List Section ─────────────────────────────────────────── */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="section-heading mb-0.5">Indexed Research Papers</h2>
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              Scientific literature available for vector similarity and evidence-gated inquiry.
            </p>
          </div>

          {/* Search filter for documents */}
          {documents && documents.length > 0 && (
            <div className="relative w-full sm:w-64">
              <Search
                className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2"
                style={{ color: 'var(--color-text-placeholder)' }}
              />
              <input
                type="text"
                placeholder="Filter by filename…"
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="input pl-9 text-xs"
                aria-label="Filter documents by filename"
              />
              {searchFilter && (
                <button
                  onClick={() => setSearchFilter('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 p-0.5 text-gray-400 hover:text-gray-600"
                  aria-label="Clear filter"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* 1. Loading Skeleton */}
        {isLoading && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="card p-5 animate-pulse flex flex-col gap-3"
                style={{ background: 'var(--color-bg-card)' }}
              >
                <div className="h-5 bg-purple-100/60 rounded-md w-3/4" />
                <div className="flex gap-2">
                  <div className="h-4 bg-purple-50 rounded-md w-16" />
                  <div className="h-4 bg-purple-50 rounded-md w-20" />
                  <div className="h-4 bg-purple-50 rounded-md w-24" />
                </div>
                <div className="h-3 bg-purple-50 rounded-md w-1/2 mt-2" />
              </div>
            ))}
          </div>
        )}

        {/* 2. Connection / Fetch Error State */}
        {!isLoading && loadError && (
          <Card className="border-red-200 bg-red-50/20">
            <CardContent className="p-8">
              <EmptyState
                icon={AlertCircle}
                title="Unable to Load Research Library"
                description={loadError}
                action={
                  <button
                    onClick={() => refreshDocuments()}
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

        {/* 3. Empty Library State */}
        {!isLoading && !loadError && documents && documents.length === 0 && (
          <Card>
            <CardContent className="p-8">
              <EmptyState
                icon={BookOpen}
                title="Your Research Library is Empty"
                description="Upload an academic PDF above to start extracting text chunks and grounding queries in verified evidence."
                action={
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                  >
                    <FileUp className="w-3.5 h-3.5" />
                    <span>Upload Your First PDF</span>
                  </button>
                }
              />
            </CardContent>
          </Card>
        )}

        {/* 4. Filter Yielded No Results */}
        {!isLoading &&
          !loadError &&
          documents &&
          documents.length > 0 &&
          filteredDocuments?.length === 0 && (
            <Card>
              <CardContent className="p-6">
                <EmptyState
                  icon={Search}
                  title="No Matching Papers Found"
                  description={`No registered documents match "${searchFilter}".`}
                  action={
                    <button
                      onClick={() => setSearchFilter('')}
                      className="btn btn-ghost btn-sm"
                    >
                      Clear search filter
                    </button>
                  }
                />
              </CardContent>
            </Card>
          )}

        {/* 5. Document Cards Grid */}
        {!isLoading &&
          !loadError &&
          filteredDocuments &&
          filteredDocuments.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredDocuments.map((doc) => (
                <div
                  key={doc.document_id}
                  className="card p-5 flex flex-col justify-between gap-4 transition-all duration-200 hover:shadow-md"
                  style={{
                    background: 'var(--color-bg-card)',
                    borderColor: 'var(--color-border)',
                  }}
                >
                  <div className="flex flex-col gap-2.5">
                    {/* Header: Document Name and Status Badges */}
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-2.5 min-w-0">
                        <div
                          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                          style={{ background: 'var(--color-lavender-soft)' }}
                        >
                          <FileText
                            className="w-4 h-4"
                            style={{ color: 'var(--color-primary)' }}
                          />
                        </div>
                        <div className="min-w-0">
                          <h3
                            className="text-sm font-semibold leading-snug break-words"
                            style={{ color: 'var(--color-text-heading)' }}
                            title={doc.filename}
                          >
                            {truncate(doc.filename, 70)}
                          </h3>
                          <div className="flex items-center gap-1 text-[11px] font-mono mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                            <Hash className="w-3 h-3 opacity-60" />
                            <span>{doc.document_id.slice(0, 12)}…</span>
                          </div>
                        </div>
                      </div>

                      <div className="shrink-0 flex flex-col items-end gap-1">
                        {doc.is_searchable ? (
                          <Badge variant="success" className="text-[11px]">
                            <FileCheck2 className="w-3 h-3 mr-0.5" /> Searchable
                          </Badge>
                        ) : (
                          <Badge variant="warning" className="text-[11px]">
                            Unindexed
                          </Badge>
                        )}
                        {doc.has_extraction ? (
                          <Badge variant="lavender" className="text-[10px]">
                            Matrix Ready
                          </Badge>
                        ) : null}
                      </div>
                    </div>

                    {/* Metadata Badges */}
                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <span className="badge badge-muted text-[11px]">
                        <Layers className="w-3 h-3 mr-1 opacity-70" />
                        {doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}
                      </span>
                      <span className="badge badge-muted text-[11px]">
                        <Database className="w-3 h-3 mr-1 opacity-70" />
                        {doc.chunk_count} chunks
                      </span>
                      <span className="badge badge-muted text-[11px]">
                        {doc.char_count.toLocaleString()} chars
                      </span>
                    </div>
                  </div>

                  {/* Footer: Ingestion Timestamp */}
                  <div
                    className="flex items-center justify-between pt-2.5 border-t text-[11px]"
                    style={{
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text-muted)',
                    }}
                  >
                    <div className="flex items-center gap-1.5">
                      <Clock className="w-3 h-3 opacity-70" />
                      <span>Ingested {formatTimestamp(doc.ingestion_timestamp)}</span>
                    </div>
                    <span className="font-mono text-[10px] opacity-60">PDF</span>
                  </div>
                </div>
              ))}
            </div>
          )}
      </section>
    </div>
  )
}
