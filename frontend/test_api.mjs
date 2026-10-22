import test from 'node:test'
import assert from 'node:assert/strict'

// ── Shared API & Protocol Definitions ───────────────────────────────────────

const MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

class APIError extends Error {
  constructor(message, statusCode = 500, detail) {
    super(message)
    this.name = 'APIError'
    this.statusCode = statusCode
    this.detail = detail
  }
}

// ── SSE Parser Implementation ───────────────────────────────────────────────

class SSEParser {
  constructor(onEvent) {
    this.buffer = ''
    this.currentEvent = 'message'
    this.currentData = []
    this.onEvent = onEvent
  }

  feed(chunk) {
    this.buffer += chunk
    let lineStart = 0
    let i = 0

    while (i < this.buffer.length) {
      const char = this.buffer[i]

      if (char === '\r') {
        const isCRLF = i + 1 < this.buffer.length && this.buffer[i + 1] === '\n'
        if (i + 1 === this.buffer.length) {
          break
        }
        const line = this.buffer.slice(lineStart, i)
        this.processLine(line)
        i = isCRLF ? i + 2 : i + 1
        lineStart = i
      } else if (char === '\n') {
        const line = this.buffer.slice(lineStart, i)
        this.processLine(line)
        i += 1
        lineStart = i
      } else {
        i += 1
      }
    }

    this.buffer = this.buffer.slice(lineStart)
  }

  processLine(line) {
    if (line === '') {
      this.dispatchEvent()
      return
    }

    if (line.startsWith(':')) {
      return
    }

    const colonIdx = line.indexOf(':')
    let field
    let value = ''

    if (colonIdx === -1) {
      field = line
      value = ''
    } else {
      field = line.slice(0, colonIdx)
      value = line.slice(colonIdx + 1)
      if (value.startsWith(' ')) {
        value = value.slice(1)
      }
    }

    if (field === 'event') {
      this.currentEvent = value
    } else if (field === 'data') {
      this.currentData.push(value)
    }
  }

  dispatchEvent() {
    if (this.currentData.length > 0 || this.currentEvent !== 'message') {
      const dataStr = this.currentData.join('\n')
      this.onEvent({
        event: this.currentEvent,
        data: dataStr,
      })
    }
    this.currentEvent = 'message'
    this.currentData = []
  }

  flush() {
    if (this.buffer.length > 0) {
      this.processLine(this.buffer)
      this.buffer = ''
    }
    this.dispatchEvent()
  }
}

async function readSSEStream(reader, onEvent, signal) {
  const decoder = new TextDecoder('utf-8')
  const parser = new SSEParser(onEvent)

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel('Aborted by user')
        break
      }

      const { value, done } = await reader.read()

      if (done) {
        const rest = decoder.decode()
        if (rest) {
          parser.feed(rest)
        }
        parser.flush()
        break
      }

      if (value) {
        const text = decoder.decode(value, { stream: true })
        parser.feed(text)
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // ignore
    }
  }
}

// ── Streaming Query Client ──────────────────────────────────────────────────

async function streamQuery(query, callbacks, options = {}, fetchFn = globalThis.fetch) {
  const trimmed = query.trim()
  if (!trimmed) {
    throw new APIError('Query must not be blank or whitespace-only.', 422)
  }
  if (trimmed.length > 2000) {
    throw new APIError('Query exceeds maximum allowed length of 2000 characters.', 422)
  }

  let res
  try {
    res = await fetchFn('/api/query/stream', {
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
  } catch (err) {
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
      const errJson = await res.json()
      detailMsg = errJson.detail || errJson.error || detailMsg
    } catch {
      // fallback
    }
    throw new APIError(detailMsg, res.status, detailMsg)
  }

  const contentType = res.headers?.get?.('content-type') || res.headers?.['content-type'] || ''
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
          callbacks.onMetadata?.(JSON.parse(evt.data))
        } else if (evt.event === 'token') {
          const payload = JSON.parse(evt.data)
          if (payload.token) callbacks.onToken?.(payload.token)
        } else if (evt.event === 'final') {
          hasReceivedFinal = true
          callbacks.onFinal?.(JSON.parse(evt.data))
        } else if (evt.event === 'error') {
          hasReceivedError = true
          callbacks.onError?.(JSON.parse(evt.data))
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

  if (!options.signal?.aborted && !hasReceivedDone && !hasReceivedError && !hasReceivedFinal) {
    callbacks.onError?.({
      error: 'UnexpectedEOF',
      detail: 'Stream connection closed unexpectedly before receiving final answer.',
    })
  }
}

// ── Query History Client ────────────────────────────────────────────────────

async function fetchHistory(limit = 50, fetchFn = globalThis.fetch) {
  try {
    const res = await fetchFn(`/api/history?limit=${limit}`)
    if (!res.ok) {
      let detailMsg = `Failed to fetch query history: status ${res.status}`
      try {
        const errorJson = await res.json()
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Unable to load query history. Backend may be offline.',
      0,
      String(err),
    )
  }
}

// ── Document Methods (From Phase 2F.2) ──────────────────────────────────────

async function fetchDocuments(fetchFn, endpoint = '/api/documents') {
  try {
    const res = await fetchFn(endpoint)
    if (!res.ok) {
      let detailMsg = `Server returned status ${res.status}`
      try {
        const errorJson = await res.json()
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Unable to connect to the backend server. Please verify the API is running.',
      0,
      String(err),
    )
  }
}

async function uploadDocument(file, fetchFn, endpoint = '/api/documents/upload') {
  if (!file) throw new APIError('No file selected.', 400)
  if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
    throw new APIError('Invalid file type. Only PDF documents (.pdf) are supported.', 400)
  }
  if (file.size === 0) throw new APIError('Selected file is empty (0 bytes).', 400)
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1)
    throw new APIError(`File size (${mb} MB) exceeds maximum allowed limit of 50 MB.`, 413)
  }

  const formData = new FormData()
  formData.append('file', file)

  try {
    const res = await fetchFn(endpoint, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      let detailMsg = `Upload failed with status ${res.status}`
      try {
        const errorJson = await res.json()
        detailMsg = errorJson.detail || errorJson.error || detailMsg
      } catch {
        // fallback
      }
      throw new APIError(detailMsg, res.status, detailMsg)
    }

    return await res.json()
  } catch (err) {
    if (err instanceof APIError) throw err
    throw new APIError(
      'Network error while uploading document. Please verify the API is running.',
      0,
      String(err),
    )
  }
}

// ── Mock Stream Helper ──────────────────────────────────────────────────────

function createMockReadableStream(chunks) {
  let index = 0
  const encoder = new TextEncoder()
  return {
    getReader() {
      return {
        async read() {
          if (index < chunks.length) {
            const chunk = chunks[index++]
            return { value: encoder.encode(chunk), done: false }
          }
          return { value: undefined, done: true }
        },
        async cancel() {},
        releaseLock() {},
      }
    },
  }
}

// ── Phase 2F.3 Tests: SSE Parser ────────────────────────────────────────────

test('SSEParser: handles events split across multiple network chunks', () => {
  const events = []
  const parser = new SSEParser((e) => events.push(e))

  parser.feed('event: to')
  assert.equal(events.length, 0)

  parser.feed('ken\ndata: {"tok')
  assert.equal(events.length, 0)

  parser.feed('en":"hello"}\n\n')
  assert.equal(events.length, 1)
  assert.equal(events[0].event, 'token')
  assert.equal(events[0].data, '{"token":"hello"}')
})

test('SSEParser: handles multiple events in a single chunk', () => {
  const events = []
  const parser = new SSEParser((e) => events.push(e))

  parser.feed(
    'event: token\ndata: {"token":"A"}\n\nevent: token\ndata: {"token":"B"}\n\nevent: done\ndata: {"done":true}\n\n',
  )
  assert.equal(events.length, 3)
  assert.equal(events[0].event, 'token')
  assert.equal(events[0].data, '{"token":"A"}')
  assert.equal(events[1].event, 'token')
  assert.equal(events[1].data, '{"token":"B"}')
  assert.equal(events[2].event, 'done')
})

test('SSEParser: handles CRLF (\\r\\n) and LF (\\n) framing', () => {
  const events = []
  const parser = new SSEParser((e) => events.push(e))

  parser.feed('event: metadata\r\ndata: {"query_id":"q1"}\r\n\r\n')
  parser.feed('event: done\ndata: {"done":true}\n\n')

  assert.equal(events.length, 2)
  assert.equal(events[0].event, 'metadata')
  assert.equal(events[1].event, 'done')
})

test('SSEParser: handles multiline data fields according to SSE spec', () => {
  const events = []
  const parser = new SSEParser((e) => events.push(e))

  parser.feed('event: final\ndata: Line 1\ndata: Line 2\ndata: Line 3\n\n')
  assert.equal(events.length, 1)
  assert.equal(events[0].data, 'Line 1\nLine 2\nLine 3')
})

test('SSEParser: ignores comment and heartbeat lines starting with colon', () => {
  const events = []
  const parser = new SSEParser((e) => events.push(e))

  parser.feed(': keepalive\n: heartbeat ping\nevent: done\ndata: {"done":true}\n\n')
  assert.equal(events.length, 1)
  assert.equal(events[0].event, 'done')
})

// ── Phase 2F.3 Tests: Streaming Query Client ────────────────────────────────

test('streamQuery: rejects empty and whitespace-only queries (client validation)', async () => {
  await assert.rejects(
    async () => streamQuery('   ', {}),
    (err) => {
      assert.equal(err.statusCode, 422)
      assert.match(err.message, /must not be blank/)
      return true
    },
  )
})

test('streamQuery: rejects queries exceeding 2000 characters (client validation)', async () => {
  const longQuery = 'x'.repeat(2001)
  await assert.rejects(
    async () => streamQuery(longQuery, {}),
    (err) => {
      assert.equal(err.statusCode, 422)
      assert.match(err.message, /exceeds maximum allowed length/)
      return true
    },
  )
})

test('streamQuery: successful streaming, progressive token accumulation, and reconciliation', async () => {
  const chunks = [
    'event: metadata\ndata: {"query_id":"q100","query":"What is LiDAR?","model":"llama3.2:1b","has_supporting_evidence":true,"verdict":{"sufficient":true,"should_abstain":false,"confidence":0.85,"reason":"Strong evidence","top_score":0.72,"chunk_count":3,"scoped_chunk_count":3,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":["[lidar.pdf, p. 2]"]},"best_source":"lidar.pdf","best_score":0.72,"citations":["[lidar.pdf, p. 2]"]}\n\n',
    'event: token\ndata: {"token":"LiDAR "}\n\n',
    'event: token\ndata: {"token":"measures "}\n\n',
    'event: token\ndata: {"token":"distance."}\n\n',
    'event: final\ndata: {"query_id":"q100","query":"What is LiDAR?","answer":"LiDAR measures distance.","answer_source":"ollama","has_supporting_evidence":true,"best_source":"lidar.pdf","best_score":0.72,"citations":["[lidar.pdf, p. 2]"],"supporting_passages":[{"text":"LiDAR sensors use laser light to calculate distance","citation":"[lidar.pdf, p. 2]","score":0.72,"source_filename":"lidar.pdf","page_number":2,"qualifies":true}],"all_retrieved":[],"verdict":{"sufficient":true,"should_abstain":false,"confidence":0.85,"reason":"Strong evidence","top_score":0.72,"chunk_count":3,"scoped_chunk_count":3,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":["[lidar.pdf, p. 2]"]},"completion_status":"complete"}\n\n',
    'event: done\ndata: {"done":true}\n\n',
  ]

  const mockResponse = {
    ok: true,
    status: 200,
    headers: {
      get: (h) => (h === 'content-type' ? 'text/event-stream; charset=utf-8' : null),
    },
    body: createMockReadableStream(chunks),
  }

  let capturedMeta = null
  const tokens = []
  let capturedFinal = null
  let doneCalled = false

  await streamQuery(
    'What is LiDAR?',
    {
      onMetadata: (m) => (capturedMeta = m),
      onToken: (t) => tokens.push(t),
      onFinal: (f) => (capturedFinal = f),
      onDone: () => (doneCalled = true),
    },
    {},
    async () => mockResponse,
  )

  assert.ok(capturedMeta)
  assert.equal(capturedMeta.query_id, 'q100')
  assert.equal(capturedMeta.has_supporting_evidence, true)
  assert.equal(tokens.join(''), 'LiDAR measures distance.')
  assert.ok(capturedFinal)
  assert.equal(capturedFinal.answer, 'LiDAR measures distance.')
  assert.equal(capturedFinal.citations.length, 1)
  assert.equal(capturedFinal.supporting_passages.length, 1)
  assert.equal(capturedFinal.completion_status, 'complete')
  assert.equal(doneCalled, true)
})

test('streamQuery: evidence-gated abstention without token events', async () => {
  const chunks = [
    'event: metadata\ndata: {"query_id":"q200","query":"Unsupported query","model":"llama3.2:1b","has_supporting_evidence":false,"verdict":{"sufficient":false,"should_abstain":true,"confidence":0.1,"reason":"Below threshold 0.38","top_score":0.21,"chunk_count":1,"scoped_chunk_count":1,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":[]},"best_source":"","best_score":0.0,"citations":[]}\n\n',
    'event: final\ndata: {"query_id":"q200","query":"Unsupported query","answer":"I do not have sufficient evidence in the indexed literature to answer this question.","answer_source":"abstention","has_supporting_evidence":false,"best_source":"","best_score":0.0,"citations":[],"supporting_passages":[],"all_retrieved":[{"text":"Unrelated text snippet","citation":"[other.pdf, p. 1]","score":0.21,"source_filename":"other.pdf","page_number":1,"qualifies":false}],"verdict":{"sufficient":false,"should_abstain":true,"confidence":0.1,"reason":"Below threshold 0.38","top_score":0.21,"chunk_count":1,"scoped_chunk_count":1,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":[]},"completion_status":"abstention"}\n\n',
    'event: done\ndata: {"done":true}\n\n',
  ]

  const mockResponse = {
    ok: true,
    status: 200,
    headers: {
      get: (h) => (h === 'content-type' ? 'text/event-stream' : null),
    },
    body: createMockReadableStream(chunks),
  }

  const tokens = []
  let capturedFinal = null
  let doneCalled = false

  await streamQuery(
    'Unsupported query',
    {
      onToken: (t) => tokens.push(t),
      onFinal: (f) => (capturedFinal = f),
      onDone: () => (doneCalled = true),
    },
    {},
    async () => mockResponse,
  )

  // Zero tokens must be emitted on abstention path
  assert.equal(tokens.length, 0)
  assert.ok(capturedFinal)
  assert.equal(capturedFinal.has_supporting_evidence, false)
  assert.equal(capturedFinal.completion_status, 'abstention')
  assert.equal(capturedFinal.answer_source, 'abstention')
  assert.match(capturedFinal.answer, /not have sufficient evidence/)
  assert.equal(doneCalled, true)
})

test('streamQuery: handles generation failure mid-stream', async () => {
  const chunks = [
    'event: metadata\ndata: {"query_id":"q300","query":"Test","model":"llama3.2:1b","has_supporting_evidence":true,"verdict":{"sufficient":true,"should_abstain":false,"confidence":0.8,"reason":"OK","top_score":0.6,"chunk_count":1,"scoped_chunk_count":1,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":[]},"best_source":"","best_score":0.0,"citations":[]}\n\n',
    'event: token\ndata: {"token":"Partial text "}\n\n',
    'event: error\ndata: {"error":"GenerationError","detail":"Ollama server connection lost"}\n\n',
    'event: done\ndata: {"done":true}\n\n',
  ]

  const mockResponse = {
    ok: true,
    status: 200,
    headers: {
      get: () => 'text/event-stream',
    },
    body: createMockReadableStream(chunks),
  }

  const tokens = []
  let capturedError = null
  let capturedFinal = null

  await streamQuery(
    'Test query',
    {
      onToken: (t) => tokens.push(t),
      onError: (err) => (capturedError = err),
      onFinal: (f) => (capturedFinal = f),
    },
    {},
    async () => mockResponse,
  )

  assert.equal(tokens.join(''), 'Partial text ')
  assert.ok(capturedError)
  assert.equal(capturedError.error, 'GenerationError')
  assert.match(capturedError.detail, /connection lost/)
  assert.equal(capturedFinal, null) // No final answer on generation failure
})

test('streamQuery: handles unexpected EOF before final event', async () => {
  const chunks = [
    'event: metadata\ndata: {"query_id":"q400","query":"Test","model":"llama3.2:1b","has_supporting_evidence":true,"verdict":{"sufficient":true,"should_abstain":false,"confidence":0.8,"reason":"OK","top_score":0.6,"chunk_count":1,"scoped_chunk_count":1,"missing_entities":[],"unscored_chunk_count":0,"missing_sources":[],"citations":[]},"best_source":"","best_score":0.0,"citations":[]}\n\n',
    'event: token\ndata: {"token":"Abrupt end"}\n\n',
    // Stream terminates without final or done event
  ]

  const mockResponse = {
    ok: true,
    status: 200,
    headers: {
      get: () => 'text/event-stream',
    },
    body: createMockReadableStream(chunks),
  }

  let capturedError = null
  await streamQuery(
    'Test',
    {
      onError: (err) => (capturedError = err),
    },
    {},
    async () => mockResponse,
  )

  assert.ok(capturedError)
  assert.equal(capturedError.error, 'UnexpectedEOF')
})

test('streamQuery: handles HTTP error status (503 Service Unavailable)', async () => {
  const mockResponse = {
    ok: false,
    status: 503,
    json: async () => ({
      detail: 'No document index is available. Please upload a PDF first.',
    }),
  }

  await assert.rejects(
    async () => streamQuery('Test', {}, {}, async () => mockResponse),
    (err) => {
      assert.equal(err.statusCode, 503)
      assert.match(err.message, /No document index is available/)
      return true
    },
  )
})

test('streamQuery: supports client-side stop / AbortController', async () => {
  const controller = new AbortController()

  // Abort immediately
  controller.abort()

  const fakeFetch = async () => {
    throw new DOMException('The user aborted a request.', 'AbortError')
  }

  await assert.rejects(
    async () => streamQuery('Test', {}, { signal: controller.signal }, fakeFetch),
    (err) => {
      assert.match(err.message, /cancelled by user/)
      return true
    },
  )
})

// ── Phase 2F.3 Tests: Query History Client ──────────────────────────────────

test('fetchHistory: successfully loads query history items', async () => {
  const mockHistory = [
    {
      history_id: 1,
      query: 'What sensor was evaluated?',
      answer: 'The Velodyne VLP-16 was evaluated.',
      answer_source: 'ollama',
      has_supporting_evidence: true,
      best_source: 'paper1.pdf',
      best_score: 0.65,
      citations: ['[paper1.pdf, p. 3]'],
      timestamp: '2026-10-22T14:30:00Z',
    },
  ]

  const fakeFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => mockHistory,
  })

  const history = await fetchHistory(50, fakeFetch)
  assert.equal(history.length, 1)
  assert.equal(history[0].query, 'What sensor was evaluated?')
  assert.equal(history[0].has_supporting_evidence, true)
})

test('fetchHistory: handles empty history and network errors', async () => {
  const fakeEmptyFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => [],
  })
  const empty = await fetchHistory(50, fakeEmptyFetch)
  assert.deepEqual(empty, [])

  const fakeErrorFetch = async () => {
    throw new Error('ECONNREFUSED')
  }
  await assert.rejects(
    async () => fetchHistory(50, fakeErrorFetch),
    (err) => {
      assert.match(err.message, /Unable to load query history/)
      return true
    },
  )
})

// ── Phase 2F.2 Document Tests (Preserved) ───────────────────────────────────

test('fetchDocuments: successfully returns document list', async () => {
  const mockDocs = [
    {
      document_id: 'doc-1',
      filename: 'sample.pdf',
      page_count: 5,
      chunk_count: 12,
      char_count: 3400,
      ingestion_timestamp: '2026-10-22T14:00:00Z',
      is_searchable: true,
      has_extraction: true,
    },
  ]

  const fakeFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => mockDocs,
  })

  const result = await fetchDocuments(fakeFetch)
  assert.equal(result.length, 1)
  assert.equal(result[0].document_id, 'doc-1')
  assert.equal(result[0].is_searchable, true)
})

test('fetchDocuments: handles empty library state', async () => {
  const fakeFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => [],
  })

  const result = await fetchDocuments(fakeFetch)
  assert.deepEqual(result, [])
})

test('uploadDocument: sends multipart FormData with field "file"', async () => {
  const validFile = new File(['%PDF-1.4 sample content'], 'paper.pdf', {
    type: 'application/pdf',
  })

  let capturedInit = null
  const fakeFetch = async (url, init) => {
    capturedInit = init
    return {
      ok: true,
      status: 200,
      json: async () => ({
        document_id: 'doc-pdf-1',
        filename: 'paper.pdf',
        page_count: 2,
        chunk_count: 6,
        char_count: 1200,
        processing_status: 'processed',
        indexing_status: 'indexed',
        is_searchable: true,
        is_duplicate: false,
        message: 'Document registered and indexed successfully',
      }),
    }
  }

  const res = await uploadDocument(validFile, fakeFetch)
  assert.equal(capturedInit.method, 'POST')
  assert.ok(capturedInit.body instanceof FormData)
  assert.equal(capturedInit.body.get('file').name, 'paper.pdf')
  assert.equal(res.document_id, 'doc-pdf-1')
  assert.equal(res.is_searchable, true)
})
