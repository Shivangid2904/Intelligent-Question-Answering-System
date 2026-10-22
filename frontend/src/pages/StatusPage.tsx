import { Cpu, Database, HardDrive, ShieldCheck } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'

export default function StatusPage() {
  return (
    <div className="page-container flex flex-col gap-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <h1 className="page-title text-2xl">System Status</h1>
          <Badge variant="lavender">Diagnostics</Badge>
        </div>
        <p className="page-subtitle mb-0">
          Component health, vector store statistics, and LLM readiness metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Database className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                SQLite & Vector Store
              </CardTitle>
              <Badge variant="muted">Live in 2F.2</Badge>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Database Engine:</span>
              <span className="font-mono">SQLite (WAL Mode)</span>
            </div>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Chunk Store:</span>
              <span className="font-mono">FAISS / Persistent Store</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Health Endpoint:</span>
              <span className="font-mono">/api/health</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Cpu className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                LLM & Embeddings
              </CardTitle>
              <Badge variant="muted">Live in 2F.2</Badge>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Generator Model:</span>
              <span className="font-mono">Ollama (qwen2.5:7b-instruct)</span>
            </div>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Embedding Model:</span>
              <span className="font-mono">nomic-embed-text</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Streaming Interface:</span>
              <span className="font-mono">Server-Sent Events</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <ShieldCheck className="w-4 h-4" style={{ color: 'var(--color-success)' }} />
                Sufficiency Guardrails
              </CardTitle>
              <Badge variant="success">Active</Badge>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Strict Mode:</span>
              <span className="font-medium" style={{ color: 'var(--color-success)' }}>Enforced</span>
            </div>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Threshold Filtering:</span>
              <span className="font-mono">Cosine / BM25 Combined</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Fallback Strategy:</span>
              <span>Explicit Refusal on Insufficient Evidence</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <HardDrive className="w-4 h-4" style={{ color: 'var(--color-primary-light)' }} />
                API Server Configuration
              </CardTitle>
              <Badge variant="lavender">FastAPI</Badge>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Default Host:</span>
              <span className="font-mono">127.0.0.1:8000</span>
            </div>
            <div className="flex justify-between py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>
              <span>Frontend Dev Port:</span>
              <span className="font-mono">5173 (Vite Proxy)</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Streamlit Fallback:</span>
              <span className="font-mono">Preserved (app.py)</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
