"""
IntelliAsk Research — FastAPI Application Entry Point

Provides lightweight, robust REST endpoints for health, readiness, and
configuration probes without triggering model loading or PDF processing.
Phase 2B adds POST /api/query with explicit engine dependency injection.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, List, Optional

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.config import APIConfig
from src.api.schemas import (
    ClearHistoryResponse,
    ComponentStatus,
    DocumentProvenanceResponse,
    DocumentRecord,
    DocumentUploadResponse,
    ErrorResponse,
    FieldProvenanceSpan,
    HealthResponse,
    HistoryEntry,
    MatrixResponse,
    PassageSchema,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    VerdictSchema,
)
from src.services.document_service import DocumentService, QAEngineManager
from src.services.matrix_service import MatrixService
from src.services.qa_service import QAResult, run_qa_query, stream_qa_query
from src.storage.db import SQLiteManager


# Benign fallback for Starlette HTTP_422 deprecation
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)



def _check_database(db_path: str) -> ComponentStatus:
    """
    Safely inspect database accessibility without running migrations or mutations.
    """
    try:
        path = Path(db_path)
        if not path.exists():
            # Check if directory is or can be made writable
            parent = path.parent
            if parent.exists() and not os_access_writable(parent):
                return ComponentStatus(
                    status="unavailable",
                    message=f"Database directory {parent} is not writable",
                    details={"db_path": db_path, "exists": False},
                )
            return ComponentStatus(
                status="uninitialized",
                message=f"Database file not yet created at {db_path}; will be initialized on first ingestion",
                details={"db_path": db_path, "exists": False},
            )

        # File exists; attempt a read-only query
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            return ComponentStatus(
                status="connected",
                message=f"Database accessible with {table_count} table(s)",
                details={"db_path": db_path, "table_count": table_count, "exists": True},
            )
        finally:
            conn.close()
    except Exception as exc:
        return ComponentStatus(
            status="unavailable",
            message=f"Database check failed: {type(exc).__name__}: {str(exc)}",
            details={"db_path": db_path, "error": str(exc)},
        )


def os_access_writable(path: Path) -> bool:
    """Helper to verify directory accessibility."""
    import os
    return os.access(str(path), os.W_OK)


async def _check_ollama(ollama_url: str, target_model: str) -> ComponentStatus:
    """
    Probe Ollama server availability with a strict 1-second timeout.
    """
    endpoint = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(endpoint)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                model_found = any(target_model in m for m in models)
                return ComponentStatus(
                    status="connected",
                    message="Ollama instance is online and reachable",
                    details={
                        "url": ollama_url,
                        "target_model": target_model,
                        "model_present": model_found,
                        "available_models": models,
                    },
                )
            return ComponentStatus(
                status="unavailable",
                message=f"Ollama returned HTTP status {resp.status_code}",
                details={"url": ollama_url, "status_code": resp.status_code},
            )
    except Exception as exc:
        return ComponentStatus(
            status="unavailable",
            message=f"Cannot reach Ollama at {ollama_url}: {type(exc).__name__}",
            details={"url": ollama_url, "error": str(exc)},
        )


def _check_embedding_model() -> ComponentStatus:
    """
    Report embedding model configuration without triggering lazy loading.
    """
    return ComponentStatus(
        status="lazy",
        message="Embedding model configured for on-demand lazy initialization (not loaded during startup)",
        details={"model_name": "all-MiniLM-L6-v2", "loaded": False},
    )


# ── Engine dependency ─────────────────────────────────────────────────────────

class _EngineHolder:
    """
    Thread-safe, test-injectable holder for the active QAEngine instance.

    The holder starts empty (engine=None) because the API foundation does not
    load models or build FAISS indices at startup.  The POST /api/query
    endpoint checks the holder and returns HTTP 503 when no engine is present,
    instead of silently constructing an empty engine.

    In tests, replace the singleton holder's engine with a fake QAEngine:

        from src.api.main import _engine_holder
        _engine_holder.engine = my_fake_engine
    """
    def __init__(self) -> None:
        self.engine: Optional[Any] = None

    def set(self, engine: Any) -> None:
        self.engine = engine

    def clear(self) -> None:
        self.engine = None


# Module-level singleton — replaced in tests by direct attribute assignment
_engine_holder = _EngineHolder()

# Module-level QAEngineManager singleton managing multi-document search state
_qa_engine_manager = QAEngineManager()


def get_engine_manager() -> QAEngineManager:
    """Dependency that returns the active QAEngineManager."""
    return _qa_engine_manager


def get_db_manager(request: Request) -> SQLiteManager:
    """Dependency that returns a SQLiteManager configured with application db_path."""
    cfg: APIConfig = getattr(request.app.state, "config", APIConfig())
    return SQLiteManager(cfg.db_path)


def get_document_service(
    request: Request,
    db: SQLiteManager = Depends(get_db_manager),
    engine_mgr: QAEngineManager = Depends(get_engine_manager),
) -> DocumentService:
    """Dependency providing a DocumentService instance configured for the request."""
    cfg: APIConfig = getattr(request.app.state, "config", APIConfig())
    return DocumentService(
        db_manager=db,
        engine_manager=engine_mgr,
        max_upload_size_bytes=cfg.max_upload_size_bytes,
    )


def get_matrix_service(
    db: SQLiteManager = Depends(get_db_manager),
) -> MatrixService:
    """Dependency providing a MatrixService instance configured for the request."""
    return MatrixService(db_manager=db)



def engine_provider() -> Any:
    """
    FastAPI dependency that returns the active QAEngine, or raises HTTP 503
    when no engine has been initialised (i.e. no PDF has been indexed yet).
    Checks QAEngineManager first, then falls back to _engine_holder (used in tests).
    """
    engine = _qa_engine_manager.active_engine or _engine_holder.engine
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No document index is available. "
                "Please upload a PDF via POST /api/documents/upload first."
            ),
        )
    return engine



# ── Result conversion ─────────────────────────────────────────────────────────

def _qa_result_to_response(result: QAResult) -> QueryResponse:
    """Convert a QAResult dataclass into a serialisable QueryResponse."""
    verdict = VerdictSchema(
        sufficient=result.verdict.sufficient,
        should_abstain=result.verdict.should_abstain,
        confidence=result.verdict.confidence,
        reason=result.verdict.reason,
        top_score=result.verdict.top_score,
        chunk_count=result.verdict.chunk_count,
        scoped_chunk_count=result.verdict.scoped_chunk_count,
        missing_entities=result.verdict.missing_entities,
        unscored_chunk_count=result.verdict.unscored_chunk_count,
        missing_sources=result.verdict.missing_sources,
        citations=result.verdict.citations,
    )
    supporting = [
        PassageSchema(
            text=p.text,
            citation=p.citation,
            score=p.score,
            source_filename=p.source_filename,
            page_number=p.page_number,
            qualifies=p.qualifies,
        )
        for p in result.supporting_passages
    ]
    all_retrieved = [
        PassageSchema(
            text=p.text,
            citation=p.citation,
            score=p.score,
            source_filename=p.source_filename,
            page_number=p.page_number,
            qualifies=p.qualifies,
        )
        for p in result.all_retrieved
    ]
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        answer_source=result.answer_source,
        has_supporting_evidence=result.has_supporting_evidence,
        verdict=verdict,
        best_source=str(result.best_source or ""),
        best_score=float(result.best_score or 0.0),

        citations=result.citations,
        supporting_passages=supporting,
        all_retrieved=all_retrieved,
    )


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    FastAPI application factory.

    Configures CORS, custom exception handlers, foundational probe endpoints,
    and the Phase 2B POST /api/query endpoint.
    """
    cfg = config or APIConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Phase 2A: Explicitly lightweight startup — no ML models or PDFs loaded
        app.state.config = cfg
        yield
        # Explicit clean teardown

    app = FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        description=cfg.api_description,
        lifespan=lifespan,
    )

    app.state.config = cfg

    # Configure CORS for local development (Vite frontend on port 5173 / 3000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Standardized Error Handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                detail=detail,
                status_code=exc.status_code,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=HTTP_422,
            content=ErrorResponse(
                error="ValidationError",
                detail=str(exc),
                status_code=HTTP_422,
            ).model_dump(),
        )


    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                detail="An unexpected internal server error occurred",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ).model_dump(),
        )

    # Foundational Endpoints
    @app.get(
        "/api/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Liveness Probe",
        description="Non-blocking liveness check. Always returns status 'ok' if API server is running.",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=cfg.api_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            models_loaded=False,
        )

    @app.get(
        "/api/readiness",
        response_model=ReadinessResponse,
        tags=["System"],
        summary="Readiness Probe",
        description="Subsystem readiness check distinguishing API availability from optional Ollama and DB services.",
    )
    async def readiness(response: Response) -> ReadinessResponse:
        db_status = _check_database(cfg.db_path)
        ollama_status = await _check_ollama(cfg.ollama_url, cfg.ollama_model)
        embed_status = _check_embedding_model()

        components = {
            "api": ComponentStatus(
                status="ready",
                message="FastAPI application foundation operational",
                details={"version": cfg.api_version},
            ),
            "database": db_status,
            "ollama": ollama_status,
            "embedding_model": embed_status,
        }

        # Determine overall readiness
        if db_status.status == "unavailable":
            overall_status = "unready"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif ollama_status.status == "connected":
            overall_status = "ready"
            response.status_code = status.HTTP_200_OK
        else:
            # API and DB are accessible, but Ollama is offline/unreachable
            overall_status = "degraded"
            response.status_code = status.HTTP_200_OK

        return ReadinessResponse(
            status=overall_status,
            version=cfg.api_version,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── Phase 2B: Query endpoint ───────────────────────────────────────────────
    @app.post(
        "/api/query",
        response_model=QueryResponse,
        tags=["Query"],
        summary="Non-streaming QA Query",
        description=(
            "Run a hybrid-retrieval + evidence-gated question-answering query against "
            "the currently indexed corpus. Returns a fully-structured JSON response. "
            "When evidence is insufficient the endpoint still returns HTTP 200 with "
            "has_supporting_evidence=false and answer_source='abstention' — the "
            "abstention is a first-class result, not an error."
        ),
        responses={
            503: {"model": ErrorResponse, "description": "No document index available"},
            422: {"model": ErrorResponse, "description": "Invalid request body"},
        },
    )
    async def query_endpoint(
        body: QueryRequest,
        engine: Any = Depends(engine_provider),
        db: SQLiteManager = Depends(get_db_manager),
    ) -> QueryResponse:
        result = run_qa_query(
            query=body.query,
            engine=engine,
            ollama_model=body.model,
        )
        try:
            db.record_query_history(
                query=result.query,
                answer=result.answer,
                answer_source=result.answer_source,
                has_supporting_evidence=result.has_supporting_evidence,
                best_source=str(result.best_source or "") if result.best_source else None,
                best_score=float(result.best_score or 0.0) if result.best_score is not None else None,
                citations=result.citations,
            )
        except Exception:
            pass
        return _qa_result_to_response(result)

    # ── Phase 2D: Streaming Query endpoint (SSE) ──────────────────────────────
    @app.post(
        "/api/query/stream",
        tags=["Query"],
        summary="Server-Sent Events (SSE) Streaming QA Query",
        description=(
            "Execute a hybrid-retrieval + evidence-gated query with real-time SSE token streaming. "
            "Streams 'metadata', 'token', 'final', and 'done' events over text/event-stream. "
            "When evidence is insufficient, emits an abstention 'final' event without invoking the LLM."
        ),
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "Progressive event stream (metadata, token, final, done)",
            },
            422: {"model": ErrorResponse, "description": "Invalid query payload"},
            503: {"model": ErrorResponse, "description": "No document index available"},
        },
    )
    async def query_stream_endpoint(
        body: QueryRequest,
        engine: Any = Depends(engine_provider),
        db: SQLiteManager = Depends(get_db_manager),
    ) -> StreamingResponse:
        def event_generator():
            try:
                for event_type, payload in stream_qa_query(
                    query=body.query,
                    engine=engine,
                    ollama_model=body.model,
                ):
                    if event_type == "final":
                        try:
                            db.record_query_history(
                                query=payload.get("query", body.query),
                                answer=payload.get("answer"),
                                answer_source=payload.get("answer_source", "ollama"),
                                has_supporting_evidence=payload.get("has_supporting_evidence", False),
                                best_source=payload.get("best_source") or None,
                                best_score=payload.get("best_score"),
                                citations=payload.get("citations"),
                            )
                        except Exception:
                            pass
                    yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
            except Exception as exc:
                err_data = {
                    "error": "StreamError",
                    "detail": f"Stream generation failed: {type(exc).__name__}",
                }
                yield f"event: error\ndata: {json.dumps(err_data)}\n\n"
                yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"


        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


    # ── Phase 2C: Document endpoints ───────────────────────────────────────────
    @app.post(
        "/api/documents/upload",
        response_model=DocumentUploadResponse,
        tags=["Documents"],
        summary="Upload and Index PDF Document",
        description=(
            "Accepts a multipart PDF upload, validates file type and signature, "
            "extracts page-bounded chunks, registers document metadata in SQLite, "
            "and updates the active in-memory search index."
        ),
        responses={
            400: {"model": ErrorResponse, "description": "Invalid file type, empty file, or bad signature"},
            413: {"model": ErrorResponse, "description": "File exceeds maximum upload size"},
            422: {"model": ErrorResponse, "description": "Malformed or unreadable PDF"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
    )
    async def upload_document(
        file: UploadFile = File(...),
        doc_service: DocumentService = Depends(get_document_service),
    ) -> DocumentUploadResponse:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only PDF documents (.pdf) are supported.",
            )

        try:
            content = await file.read()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to read uploaded file payload.",
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        max_size = doc_service.max_upload_size_bytes
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(content)} bytes) exceeds maximum allowed limit of {max_size} bytes.",
            )

        if b"%PDF-" not in content[:1024]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF signature. File content does not match the PDF specification.",
            )

        try:
            result = doc_service.process_and_register(filename, content)
            if result.is_searchable and doc_service.engine_manager.active_engine:
                _engine_holder.set(doc_service.engine_manager.active_engine)
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except RuntimeError:
            raise HTTPException(
                status_code=HTTP_422,
                detail="Failed to parse PDF document: malformed or unreadable file.",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while processing the document.",
            )

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            char_count=result.char_count,
            processing_status=result.processing_status,
            indexing_status=result.indexing_status,
            is_searchable=result.is_searchable,
            is_duplicate=result.is_duplicate,
            message=result.message,
        )

    @app.get(
        "/api/documents",
        response_model=List[DocumentRecord],
        tags=["Documents"],
        summary="List Registered Documents",
        description="Returns all registered documents from SQLite with their active searchability status.",
    )
    async def list_documents(
        doc_service: DocumentService = Depends(get_document_service),
    ) -> List[DocumentRecord]:
        docs = doc_service.list_documents()
        return [
            DocumentRecord(
                document_id=d.document_id,
                filename=d.filename,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                char_count=d.char_count,
                ingestion_timestamp=d.ingestion_timestamp,
                is_searchable=d.is_searchable,
                has_extraction=d.has_extraction,
            )
            for d in docs
        ]

    # ── Phase 2E: Matrix endpoints ─────────────────────────────────────────────
    @app.get(
        "/api/matrix",
        response_model=MatrixResponse,
        tags=["Matrix"],
        summary="Retrieve Literature Matrix",
        description="Retrieves the structured literature review matrix with 12 canonical columns and all saved paper extractions.",
    )
    async def get_matrix(
        matrix_service: MatrixService = Depends(get_matrix_service),
    ) -> MatrixResponse:
        data = matrix_service.get_matrix()
        return MatrixResponse(
            columns=data["columns"],
            rows=data["rows"],
            total_rows=data["total_rows"],
        )

    @app.get(
        "/api/matrix/export",
        tags=["Matrix"],
        summary="Export Literature Matrix as CSV",
        description="Exports the literature review matrix as a downloadable RFC 4180-compliant CSV file.",
        responses={
            200: {
                "content": {"text/csv": {}},
                "description": "Downloadable CSV representation of the literature matrix",
            }
        },
    )
    async def export_matrix_csv(
        matrix_service: MatrixService = Depends(get_matrix_service),
    ) -> Response:
        csv_content = matrix_service.export_csv()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"literature_matrix_{timestamp}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "text/csv; charset=utf-8",
            },
        )

    # ── Phase 2E: Field Provenance endpoint ───────────────────────────────────
    @app.get(
        "/api/documents/{document_id}/provenance",
        response_model=DocumentProvenanceResponse,
        tags=["Documents"],
        summary="Retrieve Field Provenance for Document",
        description="Retrieves stored field-level provenance spans (page number and verbatim quote) for a registered document.",
        responses={
            404: {"model": ErrorResponse, "description": "Document not found"},
        },
    )
    async def get_document_provenance(
        document_id: str,
        db: SQLiteManager = Depends(get_db_manager),
    ) -> DocumentProvenanceResponse:
        doc = db.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        has_ext = db.extraction_exists(document_id)
        raw_spans = db.get_all_provenance_for_document(document_id)

        spans = [
            FieldProvenanceSpan(
                field_name=s["field_name"],
                page_number=s["page_number"],
                verbatim_quote=s.get("verbatim_quote"),
            )
            for s in raw_spans
        ]

        if spans:
            status_str = "available"
            limitations = None
        elif has_ext:
            status_str = "no_spans_stored"
            limitations = "Paper extraction exists but no field-level provenance quotes were preserved for this document."
        else:
            status_str = "unextracted"
            limitations = "Document is registered but structured paper extraction has not been executed yet."

        return DocumentProvenanceResponse(
            document_id=document_id,
            filename=doc["filename"],
            provenance_records=spans,
            total_records=len(spans),
            provenance_status=status_str,
            limitations=limitations,
        )

    # ── Phase 2E: Query History endpoints ─────────────────────────────────────
    @app.get(
        "/api/history",
        response_model=List[HistoryEntry],
        tags=["History"],
        summary="Retrieve Query History",
        description="Retrieves recent research query history ordered by timestamp descending.",
    )
    async def get_history(
        limit: int = 100,
        db: SQLiteManager = Depends(get_db_manager),
    ) -> List[HistoryEntry]:
        records = db.get_query_history(limit=limit)
        return [
            HistoryEntry(
                history_id=r["history_id"],
                query=r["query"],
                answer=r.get("answer"),
                answer_source=r["answer_source"],
                has_supporting_evidence=r["has_supporting_evidence"],
                best_source=r.get("best_source"),
                best_score=r.get("best_score"),
                citations=r.get("citations", []),
                timestamp=r["timestamp"],
            )
            for r in records
        ]

    @app.delete(
        "/api/history",
        response_model=ClearHistoryResponse,
        tags=["History"],
        summary="Clear Query History",
        description="Clears query logs from the query_history table. Scope is strictly restricted to query history; documents, extractions, provenance, and corpus files are never deleted.",
    )
    async def clear_history(
        db: SQLiteManager = Depends(get_db_manager),
    ) -> ClearHistoryResponse:
        deleted = db.clear_query_history()
        return ClearHistoryResponse(
            deleted_count=deleted,
            message=f"Successfully cleared {deleted} query history record(s).",
            scope="query_history_only",
        )

    return app



# Default application instance for Uvicorn ASGI server
app = create_app()
