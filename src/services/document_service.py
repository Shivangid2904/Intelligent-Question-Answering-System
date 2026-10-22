"""
IntelliAsk Research — Document Service & QA Engine Manager

Provides:
- QAEngineManager: Explicit lifecycle manager for indexed document chunks
  and active QAEngine instances. Injectable and mockable for tests.
- DocumentService: Validates PDFs, extracts page-bounded chunks via DocumentProcessor,
  registers documents in SQLite via SQLiteManager, and updates QAEngineManager.
- Structured dataclasses: DocumentUploadResult, DocumentInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional

from src.document_processor import Chunk, DocumentProcessor
from src.storage.db import SQLiteManager


# ── Structured Data Models ───────────────────────────────────────────────────

@dataclass
class DocumentUploadResult:
    """Outcome of document validation, extraction, database storage, and indexing."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    processing_status: str  # "processed" | "empty" | "failed"
    indexing_status: str    # "indexed" | "skipped" | "failed"
    is_searchable: bool
    is_duplicate: bool
    message: str


@dataclass
class DocumentInfo:
    """Representation of a registered document with active searchability status."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    ingestion_timestamp: str
    is_searchable: bool
    has_extraction: bool = False


# ── QA Engine Manager ────────────────────────────────────────────────────────

class QAEngineManager:
    """
    Manages active in-memory QAEngine state and indexed document chunks.
    Injectable and decoupled from global state or FastAPI routers.

    Supports:
    - Lazy loading of heavy ML models (SentenceTransformer, spaCy)
    - Full mockability in unit tests without model downloads
    - Multi-document chunk aggregation without duplicate chunks
    - Clear distinction between registered documents and searchable documents
    """

    def __init__(
        self,
        embedding_model: Any = None,
        nlp: Any = None,
        embedding_model_loader: Optional[Callable[[], Any]] = None,
        nlp_loader: Optional[Callable[[], Any]] = None,
        engine_builder: Optional[Callable[[List[Any], Any], Any]] = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._nlp = nlp
        self._embedding_model_loader = embedding_model_loader
        self._nlp_loader = nlp_loader
        self._engine_builder = engine_builder
        self._chunks_by_document: Dict[str, List[Any]] = {}
        self._active_engine: Optional[Any] = None

    def get_nlp(self) -> Any:
        """Obtain spaCy NLP pipeline lazily."""
        if self._nlp is None:
            if self._nlp_loader:
                self._nlp = self._nlp_loader()
            else:
                from src.document_processor import get_nlp
                self._nlp = get_nlp()
        return self._nlp

    def get_embedding_model(self) -> Any:
        """Obtain embedding model lazily."""
        if self._embedding_model is None:
            if self._embedding_model_loader:
                self._embedding_model = self._embedding_model_loader()
            else:
                from src.qa_engine import get_embedding_model
                self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def active_engine(self) -> Optional[Any]:
        """Return the currently constructed QAEngine, or None if no documents indexed."""
        return self._active_engine

    def is_document_indexed(self, document_id: str) -> bool:
        """Return True strictly if document chunks are present in active search engine."""
        return (
            document_id in self._chunks_by_document
            and self._active_engine is not None
            and len(self._chunks_by_document[document_id]) > 0
        )

    def get_indexed_document_ids(self) -> List[str]:
        """Return list of document IDs currently indexed in the active engine."""
        return list(self._chunks_by_document.keys())

    def get_chunks_for_document(self, document_id: str) -> List[Any]:
        """Return all chunks registered for a specific document."""
        return list(self._chunks_by_document.get(document_id, []))

    def add_document_chunks(self, document_id: str, chunks: List[Any]) -> bool:
        """
        Register chunks for document_id and rebuild active QAEngine over all indexed documents.
        Returns True if engine rebuilt successfully, False if indexing failed.
        """
        if not chunks:
            return False

        old_chunks = self._chunks_by_document.get(document_id)
        # Store chunks for this document (replacing if document was previously indexed)
        self._chunks_by_document[document_id] = list(chunks)

        # Aggregate all chunks from all loaded documents
        all_chunks: List[Any] = []
        for doc_chunks in self._chunks_by_document.values():
            all_chunks.extend(doc_chunks)

        try:
            model = self.get_embedding_model()
            if self._engine_builder:
                new_engine = self._engine_builder(all_chunks, model)
            else:
                from src.qa_engine import QAEngine
                new_engine = QAEngine(all_chunks, model)
            self._active_engine = new_engine
            return True
        except Exception:
            # Rollback chunk registration on engine build failure
            if old_chunks is not None:
                self._chunks_by_document[document_id] = old_chunks
            else:
                self._chunks_by_document.pop(document_id, None)
            return False

    def remove_document(self, document_id: str) -> bool:
        """Remove document chunks and rebuild engine. Returns True if found."""
        if document_id not in self._chunks_by_document:
            return False
        self._chunks_by_document.pop(document_id)
        all_chunks: List[Any] = []
        for doc_chunks in self._chunks_by_document.values():
            all_chunks.extend(doc_chunks)

        if all_chunks:
            try:
                model = self.get_embedding_model()
                if self._engine_builder:
                    self._active_engine = self._engine_builder(all_chunks, model)
                else:
                    from src.qa_engine import QAEngine
                    self._active_engine = QAEngine(all_chunks, model)
            except Exception:
                self._active_engine = None
        else:
            self._active_engine = None
        return True

    def clear(self) -> None:
        """Reset all in-memory chunk storage and clear the active engine."""
        self._chunks_by_document.clear()
        self._active_engine = None


# ── Document Service ─────────────────────────────────────────────────────────

class DocumentService:
    """
    Coordinates PDF validation, chunking, SQLite persistence, and QAEngine indexing.
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        engine_manager: QAEngineManager,
        processor_factory: Optional[Callable[[], DocumentProcessor]] = None,
        max_upload_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.db_manager = db_manager
        self.engine_manager = engine_manager
        self.processor_factory = processor_factory or DocumentProcessor
        self.max_upload_size_bytes = max_upload_size_bytes

    def validate_pdf_input(self, filename: str, content: bytes) -> None:
        """
        Validate file extension, size, and PDF signature.
        Raises ValueError or specific exception on violation.
        """
        if not filename or not filename.strip():
            raise ValueError("Filename must not be empty.")

        clean_ext = Path(filename).suffix.lower()
        if clean_ext != ".pdf":
            raise ValueError(
                f"Invalid file extension '{clean_ext}'. Only .pdf documents are supported."
            )

        if len(content) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        if len(content) > self.max_upload_size_bytes:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds maximum limit "
                f"of {self.max_upload_size_bytes} bytes."
            )

        # PDF magic bytes: '%PDF-' must appear within first 1024 bytes per PDF spec
        header_sample = content[:1024]
        if b"%PDF-" not in header_sample:
            raise ValueError(
                "Invalid PDF signature. File content does not match the PDF specification."
            )

    def process_and_register(
        self,
        filename: str,
        content: bytes,
    ) -> DocumentUploadResult:
        """
        Validate, extract chunks, register in SQLite, and index in active QAEngine.
        Safely manages temporary files and cleans them up upon completion or failure.
        """
        # 1. Validate input bytes and format
        self.validate_pdf_input(filename, content)

        # 2. Extract chunks using DocumentProcessor in a guaranteed-clean temporary directory
        processor = self.processor_factory()
        nlp = self.engine_manager.get_nlp()

        safe_filename = Path(filename).name
        safe_base = re.sub(r"[^\w\-.]", "_", safe_filename)
        if not safe_base.lower().endswith(".pdf"):
            safe_base += ".pdf"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file_path = Path(tmpdir) / safe_base
            temp_file_path.write_bytes(content)

            try:
                chunks = processor.process_file(str(temp_file_path), nlp)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse PDF document: malformed or unreadable file."
                ) from exc

        # Extract processor metadata
        doc_id = processor.document_id
        page_count = processor.page_count
        chunk_count = len(chunks)
        char_count = sum(len(c.text) for c in chunks)

        # 3. Register in SQLite database
        self.db_manager.initialize_schema()
        # register_document returns True if new insert, False if updated existing
        is_new_insert = self.db_manager.register_document(
            document_id=doc_id,
            filename=safe_filename,
            page_count=page_count,
            chunk_count=chunk_count,
            char_count=char_count,
        )
        is_duplicate = not is_new_insert

        # 4. Index in QAEngineManager
        if chunk_count > 0:
            indexing_success = self.engine_manager.add_document_chunks(doc_id, chunks)
            if indexing_success:
                indexing_status = "indexed"
                is_searchable = True
                if is_duplicate:
                    message = (
                        f"Document '{safe_filename}' re-indexed successfully (updated existing record; "
                        f"{page_count} pages, {chunk_count} chunks)."
                    )
                else:
                    message = (
                        f"Document '{safe_filename}' successfully processed, stored, and indexed "
                        f"({page_count} pages, {chunk_count} chunks)."
                    )
            else:
                indexing_status = "failed"
                is_searchable = False
                message = (
                    f"Document '{safe_filename}' processed and stored, but search index "
                    f"construction failed."
                )
        else:
            indexing_status = "skipped"
            is_searchable = False
            message = (
                f"Document '{safe_filename}' processed ({page_count} pages), but contained "
                f"no extractable text chunks. Search indexing skipped."
            )

        return DocumentUploadResult(
            document_id=doc_id,
            filename=safe_filename,
            page_count=page_count,
            chunk_count=chunk_count,
            char_count=char_count,
            processing_status="processed" if chunk_count > 0 else "empty",
            indexing_status=indexing_status,
            is_searchable=is_searchable,
            is_duplicate=is_duplicate,
            message=message,
        )

    def list_documents(self) -> List[DocumentInfo]:
        """
        List all registered documents from SQLite with active searchability status.
        Does not mutate the database or create tables if DB file does not exist.
        """
        if not self.db_manager.db_path.exists():
            return []

        try:
            records = self.db_manager.get_all_documents()
        except Exception:
            return []

        results: List[DocumentInfo] = []
        for rec in records:
            d_id = rec["document_id"]
            is_searchable = self.engine_manager.is_document_indexed(d_id)
            has_extraction = self.db_manager.extraction_exists(d_id)
            results.append(
                DocumentInfo(
                    document_id=d_id,
                    filename=rec["filename"],
                    page_count=rec["page_count"],
                    chunk_count=rec["chunk_count"],
                    char_count=rec["char_count"],
                    ingestion_timestamp=rec["ingestion_timestamp"],
                    is_searchable=is_searchable,
                    has_extraction=has_extraction,
                )
            )
        return results

    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get single registered document with active searchability status."""
        if not self.db_manager.db_path.exists():
            return None
        rec = self.db_manager.get_document(document_id)
        if not rec:
            return None
        d_id = rec["document_id"]
        return DocumentInfo(
            document_id=d_id,
            filename=rec["filename"],
            page_count=rec["page_count"],
            chunk_count=rec["chunk_count"],
            char_count=rec["char_count"],
            ingestion_timestamp=rec["ingestion_timestamp"],
            is_searchable=self.engine_manager.is_document_indexed(d_id),
            has_extraction=self.db_manager.extraction_exists(d_id),
        )
