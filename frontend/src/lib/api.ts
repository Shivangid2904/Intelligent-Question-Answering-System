// Centralized API configuration and client helpers
// The base URL defaults to '' so Vite's proxy forwards /api/* to FastAPI (localhost:8000).
// Override VITE_API_BASE_URL in a .env file for production deploys.

import { readSSEStream } from './sse'
export { SSEParser, readSSEStream } from './sse'
export type { SSEEvent, SSEEventCallback } from './sse'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export const ENDPOINTS = {
  health: `${API_BASE_URL}/api/health`,
  readiness: `${API_BASE_URL}/api/readiness`,
  query: `${API_BASE_URL}/api/query`,
  queryStream: `${API_BASE_URL}/api/query/stream`,
  documents: `${API_BASE_URL}/api/documents`,
  documentsUpload: `${API_BASE_URL}/api/documents/upload`,
  matrix: `${API_BASE_URL}/api/matrix`,
  matrixExport: `${API_BASE_URL}/api/matrix/export`,
  history: `${API_BASE_URL}/api/history`,
} as const

/** Maximum allowed upload size in bytes (50 MB default aligned with backend APIConfig) */
export const MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

// ── Document Schemas ─────────────────────────────────────────────────────────

export interface DocumentRecord {
  document_id: string
  filename: string
  page_count: number
  chunk_count: number
  char_count: number
  ingestion_timestamp: string
  is_searchable: boolean
  has_extraction: boolean
}

export interface DocumentUploadResponse {
  document_id: string
  filename: string
  page_count: number
  chunk_count: number
  char_count: number
  processing_status: 'processed' | 'empty' | 'failed' | string
  indexing_status: 'indexed' | 'failed' | 'skipped' | string
  is_searchable: boolean
  is_duplicate: boolean
  message: string
}

// ── Streaming & Query Schemas ───────────────────────────────────────────────

export interface EvidenceVerdict {
  sufficient: boolean
  should_abstain: boolean
  confidence: number
  reason: string
  top_score: number
  chunk_count: number
  scoped_chunk_count: number
  missing_entities: string[]
  unscored_chunk_count: number
  missing_sources: string[]
  citations: string[]
}

export interface PassageSchema {
  text: string
  citation: string
  score: number
  source_filename: string
  page_number: number | null
  qualifies: boolean
}

export interface StreamMetadataPayload {
  query_id: string
  query: string
  model: string
  has_supporting_evidence: boolean
  verdict: EvidenceVerdict
  best_source: string
  best_score: number
  citations: string[]
}

export interface StreamTokenPayload {
  token: string
}

export interface StreamFinalPayload {
  query_id: string
  query: string
  answer: string
  answer_source: string
  has_supporting_evidence: boolean
  best_source: string
  best_score: number
  citations: string[]
  supporting_passages: PassageSchema[]
  all_retrieved: PassageSchema[]
  verdict: EvidenceVerdict
  completion_status: 'complete' | 'abstention' | string
}

export interface StreamErrorPayload {
  error: string
  detail: string
}

export interface StreamDonePayload {
  done: boolean
}

// ── Query History Schemas ───────────────────────────────────────────────────

export interface HistoryEntry {
  history_id: number
  query: string
  answer: string | null
  answer_source: string
  has_supporting_evidence: boolean
  best_source: string | null
  best_score: number | null
  citations: string[]
  timestamp: string
}

export interface APIErrorDetail {
  error?: string
  detail?: string
  status_code?: number
}

export class APIError extends Error {
  statusCode: number
  detail?: string

  constructor(message: string, statusCode = 500, detail?: string) {
    super(message)
    this.name = 'APIError'
    this.statusCode = statusCode
    this.detail = detail
  }
}

// ── Document Methods ────────────────────────────────────────────────────────

/**
 * Fetch list of registered documents from GET /api/documents.
 */
export async function fetchDocuments(): Promise<DocumentRecord[]> {
  try {
    const res = await fetch(ENDPOINTS.documents)
    if (!res.ok) {
      let detailMsg = `Server returned status ${res.status}`
      try {
        const errorJson = (await res.json()) as APIErrorDetail
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }
    return (await res.json()) as DocumentRecord[]
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Unable to connect to the backend server. Please verify the API is running.',
      0,
      String(err),
    )
  }
}

/**
 * Upload a research PDF file via POST /api/documents/upload.
 */
export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  if (!file) {
    throw new APIError('No file selected.', 400)
  }
  if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
    throw new APIError('Invalid file type. Only PDF documents (.pdf) are supported.', 400)
  }
  if (file.size === 0) {
    throw new APIError('Selected file is empty (0 bytes).', 400)
  }
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1)
    throw new APIError(
      `File size (${mb} MB) exceeds maximum allowed limit of 50 MB.`,
      413,
    )
  }

  const formData = new FormData()
  formData.append('file', file)

  try {
    const res = await fetch(ENDPOINTS.documentsUpload, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      let detailMsg = `Upload failed with status ${res.status}`
      try {
        const errorJson = (await res.json()) as APIErrorDetail
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }

    return (await res.json()) as DocumentUploadResponse
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Network error while uploading document. Please verify the API is running.',
      0,
      String(err),
    )
  }
}

// ── Streaming Query Methods ─────────────────────────────────────────────────

export interface StreamQueryCallbacks {
  onMetadata?: (meta: StreamMetadataPayload) => void
  onToken?: (token: string) => void
  onFinal?: (final: StreamFinalPayload) => void
  onError?: (err: StreamErrorPayload) => void
  onDone?: () => void
}

export interface StreamQueryOptions {
  model?: string
  signal?: AbortSignal
}

/**
 * Execute streaming question-answering query via POST /api/query/stream.
 */
export async function streamQuery(
  query: string,
  callbacks: StreamQueryCallbacks,
  options: StreamQueryOptions = {},
): Promise<void> {
  const trimmed = query.trim()
  if (!trimmed) {
    throw new APIError('Query must not be blank or whitespace-only.', 422)
  }
  if (trimmed.length > 2000) {
    throw new APIError('Query exceeds maximum allowed length of 2000 characters.', 422)
  }

  let res: Response
  try {
    res = await fetch(ENDPOINTS.queryStream, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: trimmed,
        model: options.model || 'llama3.2:1b',
      }),
      signal: options.signal,
    })
  } catch (err: unknown) {
    if (options.signal?.aborted) {
      throw new APIError('Query cancelled by user.', 0, 'AbortError')
    }
    throw new APIError(
      'Unable to connect to the research API. Please verify the backend is running.',
      0,
      String(err),
    )
  }

  if (!res.ok) {
    let detailMsg = `Request failed with status ${res.status}`
    try {
      const errJson = (await res.json()) as APIErrorDetail
      detailMsg = errJson.detail || errJson.error || detailMsg
    } catch {
      // fallback
    }
    throw new APIError(detailMsg, res.status, detailMsg)
  }

  const contentType = res.headers.get('content-type') || ''
  if (!contentType.includes('text/event-stream')) {
    throw new APIError(
      `Unexpected content-type received from streaming endpoint: ${contentType}`,
      502,
    )
  }

  if (!res.body) {
    throw new APIError('No response body received from stream endpoint.', 502)
  }

  const reader = res.body.getReader()
  let hasReceivedFinal = false
  let hasReceivedError = false
  let hasReceivedDone = false

  await readSSEStream(
    reader,
    (evt) => {
      try {
        if (evt.event === 'metadata') {
          const payload = JSON.parse(evt.data) as StreamMetadataPayload
          callbacks.onMetadata?.(payload)
        } else if (evt.event === 'token') {
          const payload = JSON.parse(evt.data) as StreamTokenPayload
          if (payload.token) {
            callbacks.onToken?.(payload.token)
          }
        } else if (evt.event === 'final') {
          hasReceivedFinal = true
          const payload = JSON.parse(evt.data) as StreamFinalPayload
          callbacks.onFinal?.(payload)
        } else if (evt.event === 'error') {
          hasReceivedError = true
          const payload = JSON.parse(evt.data) as StreamErrorPayload
          callbacks.onError?.(payload)
        } else if (evt.event === 'done') {
          hasReceivedDone = true
          callbacks.onDone?.()
        }
      } catch (parseErr) {
        callbacks.onError?.({
          error: 'JSONParseError',
          detail: `Failed to parse SSE payload for event '${evt.event}': ${String(parseErr)}`,
        })
      }
    },
    options.signal,
  )

  // Verify stream completion semantics:
  if (!options.signal?.aborted && !hasReceivedDone && !hasReceivedError && !hasReceivedFinal) {
    callbacks.onError?.({
      error: 'UnexpectedEOF',
      detail: 'Stream connection closed unexpectedly before receiving final answer.',
    })
  }
}

// ── Query History Methods ───────────────────────────────────────────────────

/**
 * Fetch recent research query history from GET /api/history.
 */
export async function fetchHistory(limit = 50): Promise<HistoryEntry[]> {
  try {
    const res = await fetch(`${ENDPOINTS.history}?limit=${limit}`)
    if (!res.ok) {
      let detailMsg = `Failed to fetch query history: status ${res.status}`
      try {
        const errorJson = (await res.json()) as APIErrorDetail
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }
    return (await res.json()) as HistoryEntry[]
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Unable to load query history. Backend may be offline.',
      0,
      String(err),
    )
  }
}
