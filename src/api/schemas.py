"""
IntelliAsk Research — API Pydantic Schemas

Defines typed response and error models for the FastAPI application layer.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Probe schemas ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(default="ok", description="Liveness status of the API service")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    models_loaded: bool = Field(
        default=False,
        description="Whether heavy ML models are currently loaded in memory",
    )


class ComponentStatus(BaseModel):
    """Individual service/component status model for readiness probe."""

    status: str = Field(
        ...,
        description="Component status (e.g. 'connected', 'available', 'unavailable', 'uninitialized', 'lazy')",
    )
    message: Optional[str] = Field(default=None, description="Descriptive status message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional component metadata"
    )


class ReadinessResponse(BaseModel):
    """Readiness probe response model distinguishing core API from optional backend services."""

    status: str = Field(
        ...,
        description="Overall readiness state: 'ready', 'degraded', or 'unready'",
    )
    version: str = Field(..., description="API version")
    components: Dict[str, ComponentStatus] = Field(
        ..., description="Status breakdown of subsystems"
    )
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class ErrorResponse(BaseModel):
    """Standardized API error response model."""

    error: str = Field(..., description="Short error name or description")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


# ── Query endpoint schemas ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /api/query request body."""

    query: str = Field(
        ...,
        description="The research question to answer against the indexed corpus.",
        min_length=1,
        max_length=2000,
    )
    model: str = Field(
        default="llama3.2:1b",
        description="Ollama model name to use for answer generation.",
        max_length=128,
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return v


class VerdictSchema(BaseModel):
    """Serialisable form of EvidenceVerdict."""

    sufficient: bool
    should_abstain: bool
    confidence: float
    reason: str
    top_score: float
    chunk_count: int
    scoped_chunk_count: int
    missing_entities: List[str]
    unscored_chunk_count: int
    missing_sources: List[str]
    citations: List[str]


class PassageSchema(BaseModel):
    """A single retrieved passage with provenance metadata."""

    text: str
    citation: str
    score: float
    source_filename: str
    page_number: Optional[int]
    qualifies: bool = Field(
        description="True iff dense_score >= evidence threshold (0.38)"
    )


class QueryResponse(BaseModel):
    """POST /api/query response body."""

    query: str
    answer: str = Field(
        description=(
            "Generated answer text, or abstention message when evidence is insufficient. "
            "Check has_supporting_evidence to distinguish confident answers from abstentions."
        )
    )
    answer_source: str = Field(
        description="'ollama' | 'abstention' | 'error'"
    )
    has_supporting_evidence: bool = Field(
        description=(
            "True only when evidence passes the sufficiency gate AND at least one chunk "
            "has dense_score >= 0.38.  False means the answer is an abstention."
        )
    )
    verdict: VerdictSchema
    best_source: str = Field(default="")
    best_score: float = Field(default=0.0)
    citations: List[str] = Field(default_factory=list)
    supporting_passages: List[PassageSchema] = Field(default_factory=list)
    all_retrieved: List[PassageSchema] = Field(default_factory=list)


# ── Document endpoint schemas ─────────────────────────────────────────────────

class DocumentRecord(BaseModel):
    """Registered document metadata returned by GET /api/documents."""

    document_id: str = Field(..., description="Stable deterministic document identifier")
    filename: str = Field(..., description="Original filename of the document")
    page_count: int = Field(..., description="Total physical pages in the PDF")
    chunk_count: int = Field(..., description="Number of text chunks extracted")
    char_count: int = Field(..., description="Total characters across all chunks")
    ingestion_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of ingestion")
    is_searchable: bool = Field(
        ...,
        description="Whether this document's chunks are currently indexed in the active in-memory search engine",
    )
    has_extraction: bool = Field(
        default=False,
        description="Whether structured extraction data exists in SQLite for this document",
    )


class DocumentUploadResponse(BaseModel):
    """POST /api/documents/upload response model."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    processing_status: str = Field(
        ...,
        description="Status of document parsing ('processed', 'empty', 'failed')",
    )
    indexing_status: str = Field(
        ...,
        description="Status of search index construction ('indexed', 'failed', 'skipped')",
    )
    is_searchable: bool = Field(
        ...,
        description="True if document was successfully added to active QAEngine and can be queried",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True if document was already registered in SQLite",
    )
    message: str = Field(..., description="Human-readable summary of upload outcome")


# ── Streaming event schemas ───────────────────────────────────────────────────

class StreamMetadataPayload(BaseModel):
    """Payload for 'metadata' SSE event."""

    query_id: str
    query: str
    model: str
    has_supporting_evidence: bool
    verdict: VerdictSchema
    best_source: str = ""
    best_score: float = 0.0
    citations: List[str] = Field(default_factory=list)


class StreamTokenPayload(BaseModel):
    """Payload for 'token' SSE event."""

    token: str


class StreamFinalPayload(BaseModel):
    """Payload for 'final' SSE event."""

    query_id: str
    query: str
    answer: str
    answer_source: str
    has_supporting_evidence: bool
    best_source: str = ""
    best_score: float = 0.0
    citations: List[str] = Field(default_factory=list)
    supporting_passages: List[PassageSchema] = Field(default_factory=list)
    all_retrieved: List[PassageSchema] = Field(default_factory=list)
    verdict: VerdictSchema
    completion_status: str  # "complete" | "abstention"


class StreamErrorPayload(BaseModel):
    """Payload for 'error' SSE event."""

    error: str
    detail: str


class StreamDonePayload(BaseModel):
    """Payload for 'done' SSE event."""

    done: bool = True


# ── Phase 2E: Matrix endpoint schemas ─────────────────────────────────────────

class MatrixResponse(BaseModel):
    """GET /api/matrix response model."""

    columns: List[str] = Field(
        ...,
        description="Deterministic list of column names defining the literature matrix",
    )
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="Matrix rows where each row dictionary has keys matching columns",
    )
    total_rows: int = Field(
        ...,
        description="Total number of extraction rows in the matrix",
    )


# ── Phase 2E: Field Provenance schemas ───────────────────────────────────────

class FieldProvenanceSpan(BaseModel):
    """Single provenance span for an extracted field."""

    field_name: str = Field(..., description="Field name (e.g. 'title', 'methodology')")
    page_number: int = Field(..., description="1-indexed physical page number")
    verbatim_quote: Optional[str] = Field(
        default=None,
        description="Exact verbatim quote from the source PDF supporting this field",
    )


class DocumentProvenanceResponse(BaseModel):
    """GET /api/documents/{document_id}/provenance response model."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    provenance_records: List[FieldProvenanceSpan] = Field(
        default_factory=list,
        description="List of stored field-level provenance spans",
    )
    total_records: int = Field(..., description="Count of stored provenance spans")
    provenance_status: str = Field(
        ...,
        description="'available' if spans exist, 'no_spans_stored' if extraction exists without spans, 'unextracted' if document is registered but not yet extracted",
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Honest statement of any provenance limitations or extraction state",
    )


# ── Phase 2E: Query History schemas ──────────────────────────────────────────

class HistoryEntry(BaseModel):
    """A single record in the query history."""

    history_id: int
    query: str
    answer: Optional[str] = None
    answer_source: str
    has_supporting_evidence: bool
    best_source: Optional[str] = None
    best_score: Optional[float] = None
    citations: List[str] = Field(default_factory=list)
    timestamp: str


class ClearHistoryResponse(BaseModel):
    """DELETE /api/history response model."""

    deleted_count: int = Field(..., description="Number of query history records deleted")
    message: str = Field(..., description="Confirmation of history deletion scope")
    scope: str = Field(
        default="query_history_only",
        description="Explicit guarantee that only query logs were cleared; documents and extractions untouched",
    )




