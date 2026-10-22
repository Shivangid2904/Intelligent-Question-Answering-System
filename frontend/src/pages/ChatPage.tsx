import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Copy,
  FileText,
  History,
  MessageSquareText,
  RefreshCw,
  Send,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
  X,
} from 'lucide-react'
import {
  APIError,
  fetchHistory,
  streamQuery,
} from '../lib/api'
import type {
  HistoryEntry,
  PassageSchema,
  StreamErrorPayload,
  StreamFinalPayload,
  StreamMetadataPayload,
} from '../lib/api'
import { formatTimestamp, truncate } from '../lib/utils'
import { Badge } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'

// ── Types ───────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  status?: 'streaming' | 'complete' | 'abstention' | 'error' | 'cancelled'
  metadata?: StreamMetadataPayload
  finalData?: StreamFinalPayload
  errorMessage?: string
}

export default function ChatPage() {
  // ── Conversation State ────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputQuery, setInputQuery] = useState('')
  const [selectedModel, setSelectedModel] = useState('llama3.2:1b')
  const [isStreaming, setIsStreaming] = useState(false)
  const [inputError, setInputError] = useState<string | null>(null)

  // ── Active Stream & Cancellation Ref ──────────────────────────────────────
  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // ── History Drawer State ──────────────────────────────────────────────────
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [historyItems, setHistoryItems] = useState<HistoryEntry[] | null>(null)
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryEntry | null>(null)

  // ── Auto Scroll ───────────────────────────────────────────────────────────
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, isStreaming, scrollToBottom])

  // ── Load Query History ────────────────────────────────────────────────────
  const loadHistory = useCallback(async () => {
    setIsHistoryLoading(true)
    setHistoryError(null)
    try {
      const data = await fetchHistory(30)
      setHistoryItems(data)
    } catch (err) {
      setHistoryError(
        err instanceof APIError
          ? err.message
          : 'Failed to load past research queries from history.',
      )
      setHistoryItems(null)
    } finally {
      setIsHistoryLoading(false)
    }
  }, [])

  const handleToggleHistory = () => {
    const next = !isHistoryOpen
    setIsHistoryOpen(next)
    if (next && historyItems === null) {
      void loadHistory()
    }
  }

  // Cleanup active controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  // ── Submit Query ──────────────────────────────────────────────────────────
  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (isStreaming) return

    const trimmed = inputQuery.trim()
    if (!trimmed) {
      setInputError('Please enter a research question.')
      return
    }
    if (trimmed.length > 2000) {
      setInputError('Question exceeds maximum limit of 2,000 characters.')
      return
    }

    setInputError(null)
    const userMsgId = `u_${Date.now()}`
    const assistantMsgId = `a_${Date.now()}`

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    }

    const initialAssistantMessage: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      status: 'streaming',
    }

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage])
    setInputQuery('')
    setIsStreaming(true)

    const controller = new AbortController()
    abortControllerRef.current = controller

    let streamedTokens = ''

    try {
      await streamQuery(
        trimmed,
        {
          onMetadata: (meta: StreamMetadataPayload) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      metadata: meta,
                    }
                  : msg,
              ),
            )
          },
          onToken: (token: string) => {
            streamedTokens += token
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      content: streamedTokens,
                    }
                  : msg,
              ),
            )
          },
          onFinal: (final: StreamFinalPayload) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      // Reconcile complete answer from final payload
                      content: final.answer || streamedTokens,
                      finalData: final,
                      status:
                        final.completion_status === 'abstention' ||
                        !final.has_supporting_evidence
                          ? 'abstention'
                          : 'complete',
                    }
                  : msg,
              ),
            )
          },
          onError: (err: StreamErrorPayload) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      status: 'error',
                      errorMessage: `${err.error}: ${err.detail}`,
                    }
                  : msg,
              ),
            )
          },
          onDone: () => {
            setIsStreaming(false)
            abortControllerRef.current = null
            // Refresh history quietly in background
            void loadHistory()
          },
        },
        {
          model: selectedModel,
          signal: controller.signal,
        },
      )
    } catch (err: unknown) {
      if (controller.signal.aborted) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  status: 'cancelled',
                  errorMessage: 'Generation stopped by user.',
                }
              : msg,
          ),
        )
      } else {
        const errorMsg =
          err instanceof APIError
            ? err.message
            : 'Unable to stream response. Please verify the backend is running.'
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  status: 'error',
                  errorMessage: errorMsg,
                }
              : msg,
          ),
        )
      }
    } finally {
      setIsStreaming(false)
      abortControllerRef.current = null
    }
  }

  // ── Stop Generation ───────────────────────────────────────────────────────
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsStreaming(false)
  }

  // ── Keyboard Navigation ───────────────────────────────────────────────────
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSubmit()
    }
  }

  return (
    <div className="flex h-[calc(100vh-var(--space-header-height))] overflow-hidden relative">
      {/* ── Main Chat Area ────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#FBF8FF]">
        {/* Chat Sub-header with Evidence Gate Notice & History Trigger */}
        <div
          className="flex items-center justify-between px-6 py-3 border-b shrink-0 bg-white/80 backdrop-blur-xs"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: 'var(--color-lavender-soft)' }}
            >
              <MessageSquareText className="w-4 h-4 text-purple-700" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold leading-none text-purple-950">
                  Research Chat
                </h1>
                <Badge variant="pink" className="text-[10px] py-0 px-1.5">
                  <ShieldCheck className="w-2.5 h-2.5 mr-1 text-purple-800" /> Evidence-Gated
                </Badge>
              </div>
              <p className="text-[11px] text-gray-500 mt-0.5 leading-none">
                Grounded in indexed corpus • Threshold: dense score ≥ 0.38
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Model select */}
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isStreaming}
              className="text-xs bg-purple-50/60 border border-purple-200 rounded-lg px-2.5 py-1 text-purple-900 focus:outline-none focus:border-purple-400"
              aria-label="Select generator model"
            >
              <option value="llama3.2:1b">llama3.2:1b (Fast)</option>
              <option value="qwen2.5:7b-instruct">qwen2.5:7b-instruct (Deep)</option>
            </select>

            {/* Toggle History Drawer */}
            <button
              onClick={handleToggleHistory}
              className="btn btn-secondary btn-sm flex items-center gap-1.5"
              aria-label="Toggle query history"
              title="View query history"
            >
              <History className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">History</span>
            </button>
          </div>
        </div>

        {/* Messages Scroll Area */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 flex flex-col gap-6">
          {messages.length === 0 ? (
            <div className="my-auto max-w-xl mx-auto w-full">
              <EmptyState
                icon={Sparkles}
                title="Grounded Scientific Question Answering"
                description="Inquire against indexed literature. If sufficient evidence exists across retrieved chunks, an evidence-grounded answer with verifiable citations is generated; otherwise, the system safely abstains."
                action={
                  <div className="flex flex-col gap-2 w-full pt-2">
                    <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider text-center">
                      Suggested Research Questions
                    </span>
                    <div className="flex flex-col gap-2">
                      {[
                        'What sensor configuration is used for autonomous ground vehicle navigation?',
                        'What are the reported accuracy benchmarks and sample sizes across the papers?',
                        'Compare the methodology and experimental setup of the included studies.',
                      ].map((suggestion, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            setInputQuery(suggestion)
                            textareaRef.current?.focus()
                          }}
                          className="text-left text-xs p-3 rounded-xl border border-purple-100 bg-white hover:bg-purple-50/50 text-purple-900 transition-colors flex items-center justify-between group shadow-2xs"
                        >
                          <span>{suggestion}</span>
                          <ArrowRight className="w-3.5 h-3.5 text-purple-400 group-hover:text-purple-700 transition-transform group-hover:translate-x-0.5" />
                        </button>
                      ))}
                    </div>
                  </div>
                }
              />
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isStreaming={isStreaming && message.status === 'streaming'}
                onStop={handleStop}
              />
            ))
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Composer */}
        <div
          className="p-4 sm:px-8 border-t bg-white shrink-0"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <form
            onSubmit={(e) => void handleSubmit(e)}
            className="max-w-4xl mx-auto flex flex-col gap-2"
          >
            {inputError && (
              <div
                className="flex items-center gap-2 text-xs text-amber-800 bg-amber-50 p-2.5 rounded-lg border border-amber-200"
                role="alert"
              >
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                <span>{inputError}</span>
              </div>
            )}

            <div className="relative flex items-end rounded-2xl border border-purple-200 bg-purple-50/20 focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-200 transition-all shadow-xs">
              <textarea
                ref={textareaRef}
                value={inputQuery}
                onChange={(e) => {
                  setInputQuery(e.target.value)
                  if (inputError) setInputError(null)
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask a scientific question grounded in your research corpus (Shift+Enter for newline)…"
                rows={2}
                disabled={isStreaming}
                className="w-full p-3.5 pr-24 bg-transparent text-xs sm:text-sm text-purple-950 placeholder-gray-400 resize-none focus:outline-none"
                aria-label="Research inquiry input"
              />

              <div className="absolute right-2.5 bottom-2.5 flex items-center gap-1.5">
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="btn btn-sm bg-red-100 hover:bg-red-200 text-red-800 border border-red-300 flex items-center gap-1 shadow-xs"
                    aria-label="Stop generation"
                  >
                    <Square className="w-3 h-3 fill-current" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!inputQuery.trim()}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                    aria-label="Submit query"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Ask</span>
                  </button>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] text-gray-400 px-1">
              <span>Enter to send • Shift+Enter for newline</span>
              <span>{inputQuery.length} / 2,000</span>
            </div>
          </form>
        </div>
      </div>

      {/* ── Query History Side Drawer ──────────────────────────────────────── */}
      {isHistoryOpen && (
        <aside
          className="w-80 sm:w-96 border-l flex flex-col h-full bg-white shrink-0 z-20 shadow-lg animate-in slide-in-from-right duration-200"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div className="flex items-center justify-between px-4 py-3.5 border-b shrink-0 bg-purple-50/40">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-purple-700" />
              <h2 className="text-sm font-bold text-purple-950">Query History</h2>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => void loadHistory()}
                disabled={isHistoryLoading}
                className="p-1.5 text-gray-500 hover:text-purple-700 rounded-lg hover:bg-purple-100/50"
                aria-label="Refresh history"
                title="Refresh query history"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isHistoryLoading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setIsHistoryOpen(false)}
                className="p-1.5 text-gray-500 hover:text-purple-700 rounded-lg hover:bg-purple-100/50"
                aria-label="Close history drawer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
            {isHistoryLoading && !historyItems && (
              <div className="flex flex-col gap-2 p-3">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-16 rounded-xl bg-purple-50/60 animate-pulse" />
                ))}
              </div>
            )}

            {historyError && (
              <div className="p-4 text-xs text-red-700 bg-red-50 rounded-xl border border-red-200 flex flex-col gap-2">
                <div className="flex items-center gap-1.5 font-semibold">
                  <AlertCircle className="w-4 h-4" />
                  <span>Unable to load history</span>
                </div>
                <p className="text-[11px] opacity-90">{historyError}</p>
                <button
                  onClick={() => void loadHistory()}
                  className="btn btn-secondary btn-sm self-start mt-1"
                >
                  Retry
                </button>
              </div>
            )}

            {!isHistoryLoading && historyItems && historyItems.length === 0 && (
              <div className="text-center py-12 px-4 text-gray-400 flex flex-col items-center gap-2">
                <History className="w-8 h-8 opacity-40 text-purple-300" />
                <p className="text-xs font-semibold text-gray-600">No past queries logged</p>
                <p className="text-[11px] text-gray-400">
                  Queries executed against the corpus will automatically appear in this audit log.
                </p>
              </div>
            )}

            {historyItems &&
              historyItems.map((item) => (
                <div
                  key={item.history_id}
                  onClick={() => setSelectedHistoryItem(item)}
                  className={`
                    p-3 rounded-xl border text-left cursor-pointer transition-all flex flex-col gap-1.5
                    ${
                      selectedHistoryItem?.history_id === item.history_id
                        ? 'border-purple-400 bg-purple-50/80 ring-1 ring-purple-300'
                        : 'border-purple-100 bg-white hover:bg-purple-50/40 hover:border-purple-200'
                    }
                  `}
                >
                  <div className="flex items-center justify-between gap-2">
                    {item.has_supporting_evidence ? (
                      <Badge variant="success" className="text-[9px] py-0 px-1.5">
                        <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" /> Evidence
                      </Badge>
                    ) : (
                      <Badge variant="warning" className="text-[9px] py-0 px-1.5">
                        <ShieldAlert className="w-2.5 h-2.5 mr-0.5" /> Abstained
                      </Badge>
                    )}
                    <span className="text-[10px] text-gray-400 flex items-center gap-1">
                      <Clock className="w-2.5 h-2.5" />
                      {formatTimestamp(item.timestamp)}
                    </span>
                  </div>

                  <p className="text-xs font-medium text-purple-950 line-clamp-2">
                    {item.query}
                  </p>

                  {item.best_source && (
                    <div className="text-[10px] text-gray-500 truncate flex items-center gap-1">
                      <FileText className="w-2.5 h-2.5 text-purple-400 shrink-0" />
                      <span className="truncate">{item.best_source}</span>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-1 border-t border-purple-50">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        setInputQuery(item.query)
                        textareaRef.current?.focus()
                      }}
                      className="text-[10px] font-medium text-purple-700 hover:text-purple-900 flex items-center gap-1"
                    >
                      <Copy className="w-2.5 h-2.5" /> Copy question
                    </button>
                    <span className="text-[10px] text-gray-400 capitalize">
                      {item.answer_source}
                    </span>
                  </div>
                </div>
              ))}
          </div>

          {/* Selected History Item Modal / Detail View */}
          {selectedHistoryItem && (
            <div className="p-3 border-t bg-purple-50/30 flex flex-col gap-2 max-h-60 overflow-y-auto">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-purple-900">Historical Answer</span>
                <button
                  onClick={() => setSelectedHistoryItem(null)}
                  className="text-gray-400 hover:text-gray-600 text-[10px]"
                >
                  Close
                </button>
              </div>
              <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed">
                {selectedHistoryItem.answer || 'No answer recorded.'}
              </p>
            </div>
          )}
        </aside>
      )}
    </div>
  )
}

// ── Message Bubble Component ────────────────────────────────────────────────

interface MessageBubbleProps {
  message: ChatMessage
  isStreaming: boolean
  onStop: () => void
}

function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const [showPassages, setShowPassages] = useState(false)
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div
          className="max-w-2xl rounded-2xl rounded-tr-xs px-5 py-3.5 shadow-2xs text-xs sm:text-sm"
          style={{
            background: 'linear-gradient(135deg, #F0E8FF 0%, #FBEBF4 100%)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text-heading)',
          }}
        >
          <p className="whitespace-pre-wrap leading-relaxed font-medium">{message.content}</p>
          <div className="text-[10px] text-purple-400 mt-1.5 text-right">
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    )
  }

  // Assistant Message
  const hasMetadata = Boolean(message.metadata)
  const final = message.finalData
  const isAbstained =
    message.status === 'abstention' ||
    (final && !final.has_supporting_evidence)
  const isError = message.status === 'error'
  const isCancelled = message.status === 'cancelled'

  const citations = final?.citations || message.metadata?.citations || []
  const supportingPassages = final?.supporting_passages || []
  const allRetrieved = final?.all_retrieved || []
  const passagesToDisplay =
    supportingPassages.length > 0 ? supportingPassages : allRetrieved

  return (
    <div className="flex justify-start">
      <Card
        className="max-w-3xl w-full p-5 sm:p-6 flex flex-col gap-4 border"
        style={{
          background: 'var(--color-bg-card)',
          borderColor: isAbstained
            ? 'var(--color-warning-border)'
            : isError
              ? 'var(--color-error-border)'
              : 'var(--color-border)',
        }}
      >
        {/* Header: Assistant Identity, Verification Status, and Model */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2">
            <div
              className="w-6 h-6 rounded-md flex items-center justify-center shrink-0"
              style={{ background: 'var(--gradient-btn)' }}
            >
              <Sparkles className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="text-xs font-bold text-purple-950">IntelliAsk Research</span>

            {/* Evidence Gate Status */}
            {isAbstained ? (
              <Badge variant="warning" className="text-[10px] py-0 px-2">
                <ShieldAlert className="w-3 h-3 mr-1" /> Abstained — Insufficient Evidence
              </Badge>
            ) : message.status === 'complete' ? (
              <Badge variant="success" className="text-[10px] py-0 px-2">
                <ShieldCheck className="w-3 h-3 mr-1" /> Evidence Verified (Score ≥ 0.38)
              </Badge>
            ) : isStreaming ? (
              <Badge variant="lavender" className="text-[10px] py-0 px-2 animate-pulse">
                <Sparkles className="w-3 h-3 mr-1" /> Streaming Answer…
              </Badge>
            ) : null}
          </div>

          <div className="flex items-center gap-2 text-[11px] text-gray-400">
            {hasMetadata && (
              <span className="font-mono text-[10px]">
                {message.metadata?.model}
              </span>
            )}
            <span>{formatTimestamp(message.timestamp)}</span>
          </div>
        </div>

        {/* Abstention Banner */}
        {isAbstained && (
          <div
            className="flex items-start gap-2.5 p-3 rounded-xl border text-xs text-amber-900 bg-amber-50/70 border-amber-200"
            role="alert"
          >
            <ShieldAlert className="w-4 h-4 shrink-0 text-amber-700 mt-0.5" />
            <div className="flex flex-col gap-0.5">
              <span className="font-bold">Evidence Sufficiency Gate Refusal</span>
              <p className="text-[11px] leading-relaxed opacity-90">
                The retrieved corpus chunks scored below the strict evidence threshold (0.38) or lacked required query entity overlap. To prevent speculative hallucinations, answer generation was abstained.
              </p>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {isError && (
          <div
            className="flex items-start gap-2.5 p-3 rounded-xl border text-xs text-red-900 bg-red-50 border-red-200"
            role="alert"
          >
            <AlertCircle className="w-4 h-4 shrink-0 text-red-700 mt-0.5" />
            <div className="flex flex-col gap-0.5">
              <span className="font-bold">Query Error</span>
              <p className="text-[11px] leading-relaxed opacity-90">
                {message.errorMessage || 'An unexpected error interrupted the generation stream.'}
              </p>
            </div>
          </div>
        )}

        {/* User Cancellation Notice */}
        {isCancelled && (
          <div className="flex items-center gap-2 p-2.5 rounded-lg bg-gray-50 border border-gray-200 text-xs text-gray-600">
            <Square className="w-3 h-3 fill-gray-400" />
            <span>Generation stopped by user. Partial output is preserved below.</span>
          </div>
        )}

        {/* Answer Content Area */}
        <div
          className="text-xs sm:text-sm text-gray-800 leading-relaxed whitespace-pre-wrap font-sans"
          aria-live="polite"
        >
          {message.content ? (
            message.content
          ) : isStreaming ? (
            <span className="text-gray-400 italic">Evaluating corpus evidence and initializing synthesis…</span>
          ) : isAbstained ? (
            <span className="italic text-gray-500">No grounded answer available.</span>
          ) : null}
        </div>

        {/* Citations Section */}
        {citations.length > 0 && (
          <div className="flex flex-col gap-1.5 pt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
            <span className="text-[11px] font-bold uppercase tracking-wider text-purple-900 flex items-center gap-1">
              <FileText className="w-3 h-3 text-purple-600" /> Grounded Citations
            </span>
            <div className="flex flex-wrap gap-2">
              {citations.map((cite, i) => (
                <div
                  key={i}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-purple-50 border border-purple-200 text-[11px] font-medium text-purple-900 shadow-2xs"
                >
                  <BookOpen className="w-3 h-3 text-purple-600 shrink-0" />
                  <span>{cite}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Supporting Passages Section (Collapsible Accordion) */}
        {passagesToDisplay.length > 0 && (
          <div className="pt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
            <button
              type="button"
              onClick={() => setShowPassages(!showPassages)}
              className="flex items-center justify-between w-full py-1 text-left text-xs font-semibold text-purple-950 hover:text-purple-700 transition-colors"
              aria-expanded={showPassages}
            >
              <div className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-purple-600" />
                <span>
                  {isAbstained ? 'Retrieved Corpus Passages (Unqualified)' : 'Supporting Evidence Passages'}
                </span>
                <Badge variant="muted" className="text-[10px] py-0 px-1.5">
                  {passagesToDisplay.length}
                </Badge>
              </div>
              {showPassages ? (
                <ChevronDown className="w-4 h-4 text-purple-600" />
              ) : (
                <ChevronRight className="w-4 h-4 text-purple-600" />
              )}
            </button>

            {showPassages && (
              <div className="flex flex-col gap-2.5 mt-3 animate-in fade-in duration-150">
                {passagesToDisplay.map((p: PassageSchema, idx: number) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl border bg-purple-50/20 text-xs flex flex-col gap-1.5"
                    style={{ borderColor: 'var(--color-border)' }}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5 text-purple-900 font-semibold truncate">
                        <FileText className="w-3 h-3 text-purple-500 shrink-0" />
                        <span className="truncate">{p.source_filename}</span>
                        {p.page_number && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 text-purple-800">
                            p. {p.page_number}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-mono text-gray-500">
                          Score: {typeof p.score === 'number' ? p.score.toFixed(3) : p.score}
                        </span>
                        {p.qualifies ? (
                          <Badge variant="success" className="text-[9px] py-0 px-1">
                            Qualifies
                          </Badge>
                        ) : (
                          <Badge variant="muted" className="text-[9px] py-0 px-1">
                            Below Threshold
                          </Badge>
                        )}
                      </div>
                    </div>
                    <p className="text-[11px] text-gray-600 leading-relaxed font-serif bg-white p-2 rounded-lg border border-purple-100/60 italic">
                      "{truncate(p.text, 280)}"
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}
