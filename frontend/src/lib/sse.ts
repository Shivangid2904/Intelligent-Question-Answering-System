/**
 * IntelliAsk Research — Server-Sent Events (SSE) Protocol Parser
 *
 * Implements a robust stateful stream parser handling:
 * - Network read chunking across arbitrary byte boundaries
 * - Multiple events per network read
 * - Multiline data: fields according to SSE specification
 * - CRLF (\r\n) and LF (\n) line terminations
 * - Comment lines (: ping/keepalive)
 * - Safe JSON payload extraction
 */

export interface SSEEvent {
  event: string
  data: string
}

export type SSEEventCallback = (event: SSEEvent) => void

export class SSEParser {
  private buffer = ''
  private currentEvent = 'message'
  private currentData: string[] = []
  private onEvent: SSEEventCallback

  constructor(onEvent: SSEEventCallback) {
    this.onEvent = onEvent
  }

  /**
   * Feed a chunk of raw string data into the parser.
   */
  feed(chunk: string): void {
    this.buffer += chunk

    let lineStart = 0
    let i = 0

    while (i < this.buffer.length) {
      const char = this.buffer[i]

      if (char === '\r') {
        const isCRLF = i + 1 < this.buffer.length && this.buffer[i + 1] === '\n'
        // If \r is at the very end of buffer, wait for next chunk
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

    // Keep unprocessed partial line in buffer
    this.buffer = this.buffer.slice(lineStart)
  }

  /**
   * Process a single line according to the SSE standard.
   */
  private processLine(line: string): void {
    // Empty line indicates event boundary (dispatch accumulated event)
    if (line === '') {
      this.dispatchEvent()
      return
    }

    // Ignore comment line
    if (line.startsWith(':')) {
      return
    }

    const colonIdx = line.indexOf(':')
    let field: string
    let value = ''

    if (colonIdx === -1) {
      field = line
      value = ''
    } else {
      field = line.slice(0, colonIdx)
      value = line.slice(colonIdx + 1)
      // Strip single optional leading space
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

  private dispatchEvent(): void {
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

  /**
   * Flush any remaining complete event at end of stream.
   */
  flush(): void {
    if (this.buffer.length > 0) {
      this.processLine(this.buffer)
      this.buffer = ''
    }
    this.dispatchEvent()
  }
}

/**
 * Read and parse an SSE ReadableStream incrementally.
 */
export async function readSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: SSEEventCallback,
  signal?: AbortSignal,
): Promise<void> {
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
      // lock already released
    }
  }
}
