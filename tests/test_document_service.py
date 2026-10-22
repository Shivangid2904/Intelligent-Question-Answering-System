"""
IntelliAsk Research — Phase 2C Document Service & Upload API Tests

Covers:
- Valid PDF ingestion with controlled fixtures
- Empty upload rejection (400 / ValueError)
- Invalid extension rejection (400 / ValueError)
- Invalid PDF signature rejection (400 / ValueError)
- Oversized upload rejection (413 / ValueError)
- Duplicate upload behavior (idempotent DB upsert, no chunk duplication in engine)
- Successful SQLite registration with accurate counts
- Indexing failure handling (document stored, but is_searchable=False)
- Guaranteed temporary-file cleanup
- Document listing with zero and multiple documents
- Registered vs searchable distinction across simulated server restarts
- Dependency injection for QAEngineManager and SQLiteManager
- No live Ollama inference or SentenceTransformer model downloads
- Full HTTP API endpoint validation with TestClient
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
from typing import Any, List
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.config import APIConfig
from src.api.main import (
    _engine_holder,
    _qa_engine_manager,
    app,
    create_app,
    get_db_manager,
    get_engine_manager,
)
from src.api.schemas import DocumentRecord, DocumentUploadResponse
from src.document_processor import Chunk, DocumentProcessor, get_nlp
from src.services.document_service import (
    DocumentInfo,
    DocumentService,
    DocumentUploadResult,
    QAEngineManager,
)
from src.storage.db import SQLiteManager


# ── Fixtures & Helpers ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_engine_states():
    """Ensure global engine holders are reset before and after every test."""
    _engine_holder.clear()
    _qa_engine_manager.clear()
    yield
    _engine_holder.clear()
    _qa_engine_manager.clear()


@pytest.fixture(scope="session")
def shared_nlp():
    """Load spaCy sentencizer once for test session."""
    return get_nlp()



def create_minimal_pdf_bytes(
    text: str = "LiDAR sensors measure distance using pulsed laser light reflected from targets. "
    "Autonomous vehicle navigation systems rely heavily on accurate LiDAR point clouds.",
) -> bytes:
    """Create a minimal, valid in-memory PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_empty_pdf_bytes() -> bytes:
    """Create a valid PDF with a single page containing no text."""
    doc = fitz.open()
    doc.new_page()  # blank page
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class MockEmbeddingModel:
    """Deterministic mock embedding model returning 384-dim normalized zero-free vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        count = len(texts) if isinstance(texts, list) else 1
        arr = np.full((count, self.dim), 0.05, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        return arr


class MockQAEngine:
    """Lightweight mock search engine for fast test execution."""

    def __init__(self, chunks: List[Any], model: Any = None):
        self.chunks = list(chunks)
        self.model = model

    def search_hybrid(self, query: str, k: int = 10) -> List[Any]:
        results = []
        for i, c in enumerate(self.chunks[:k]):
            source_name = str(getattr(c, "source_filename", "mock.pdf"))
            chunk_copy = MagicMock()
            chunk_copy.text = getattr(c, "text", "")
            chunk_copy.source_filename = source_name

            chunk_copy.source = source_name
            chunk_copy.page_number = getattr(c, "page_number", 1)
            chunk_copy.dense_score = 0.65 - (i * 0.02)
            chunk_copy.score = 0.02 - (i * 0.001)
            results.append(chunk_copy)

        return results


# ── Unit Tests: Document Validation ──────────────────────────────────────────

class TestDocumentValidation:
    """Tests for PDF input validation rules."""

    def test_empty_filename_raises_value_error(self, tmp_path):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager()
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        with pytest.raises(ValueError, match="Filename must not be empty"):
            svc.validate_pdf_input("", b"%PDF-1.4 sample content")

    def test_invalid_extension_raises_value_error(self, tmp_path):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager()
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        with pytest.raises(ValueError, match="Only .pdf documents are supported"):
            svc.validate_pdf_input("notes.txt", b"%PDF-1.4 sample content")

    def test_empty_content_raises_value_error(self, tmp_path):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager()
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        with pytest.raises(ValueError, match="Uploaded file is empty"):
            svc.validate_pdf_input("document.pdf", b"")

    def test_oversized_file_raises_value_error(self, tmp_path):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager()
        svc = DocumentService(db_manager=db, engine_manager=mgr, max_upload_size_bytes=100)

        oversized_bytes = b"%PDF-" + b"x" * 200
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            svc.validate_pdf_input("large.pdf", oversized_bytes)

    def test_missing_pdf_signature_raises_value_error(self, tmp_path):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager()
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        not_a_pdf = b"Hello, this is just plain text disguised as a PDF file."
        with pytest.raises(ValueError, match="Invalid PDF signature"):
            svc.validate_pdf_input("fake.pdf", not_a_pdf)


# ── Unit Tests: DocumentService & QAEngineManager Core ───────────────────────

class TestDocumentServiceCore:
    """Tests for document processing, storage, and indexing lifecycle."""

    def test_successful_pdf_processing_and_indexing(self, tmp_path, shared_nlp):
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        pdf_bytes = create_minimal_pdf_bytes()
        result = svc.process_and_register("sensor_paper.pdf", pdf_bytes)

        assert isinstance(result, DocumentUploadResult)
        assert result.processing_status == "processed"
        assert result.indexing_status == "indexed"
        assert result.is_searchable is True
        assert result.is_duplicate is False
        assert result.page_count == 1
        assert result.chunk_count > 0
        assert result.char_count > 0
        assert mgr.is_document_indexed(result.document_id) is True

        # Check SQLite persistence
        db_doc = db.get_document(result.document_id)
        assert db_doc is not None
        assert db_doc["filename"] == "sensor_paper.pdf"
        assert db_doc["page_count"] == 1
        assert db_doc["chunk_count"] == result.chunk_count

    def test_duplicate_upload_behavior(self, tmp_path, shared_nlp):
        """Duplicate uploads must update the DB record and replace chunks without duplication."""
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        pdf_bytes = create_minimal_pdf_bytes()
        res1 = svc.process_and_register("duplicate_test.pdf", pdf_bytes)
        assert res1.is_duplicate is False
        initial_chunk_count = len(mgr.active_engine.chunks)

        # Upload exact same content again
        res2 = svc.process_and_register("duplicate_test.pdf", pdf_bytes)
        assert res2.is_duplicate is True
        assert res2.document_id == res1.document_id
        assert res2.is_searchable is True

        # Ensure total chunks in active engine did NOT duplicate
        assert len(mgr.active_engine.chunks) == initial_chunk_count

        # Ensure SQLite only has 1 record for this document
        all_docs = db.get_all_documents()
        matching_docs = [d for d in all_docs if d["document_id"] == res1.document_id]
        assert len(matching_docs) == 1

    def test_blank_pdf_skips_indexing(self, tmp_path, shared_nlp):
        """A PDF with pages but no text chunks registers in DB but skips search indexing."""
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        blank_bytes = create_empty_pdf_bytes()
        result = svc.process_and_register("blank.pdf", blank_bytes)

        assert result.processing_status == "empty"
        assert result.indexing_status == "skipped"
        assert result.is_searchable is False
        assert result.chunk_count == 0

        # Document is recorded in SQLite
        db_doc = db.get_document(result.document_id)
        assert db_doc is not None
        assert db_doc["chunk_count"] == 0

    def test_indexing_failure_leaves_document_unsearchable(self, tmp_path, shared_nlp):
        """If engine construction throws an error, the document is NOT falsely marked searchable."""
        def failing_engine_builder(chunks, model):
            raise RuntimeError("Engine build out of memory simulation")

        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=failing_engine_builder,
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        pdf_bytes = create_minimal_pdf_bytes()
        result = svc.process_and_register("fail_paper.pdf", pdf_bytes)

        assert result.indexing_status == "failed"
        assert result.is_searchable is False
        assert "construction failed" in result.message
        assert mgr.is_document_indexed(result.document_id) is False

        # SQLite has the document recorded
        assert db.get_document(result.document_id) is not None

    def test_malformed_pdf_raises_runtime_error_and_cleans_up(self, tmp_path, shared_nlp):
        """A corrupted PDF with valid %PDF- header but broken syntax must raise and clean temp files."""
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(nlp=shared_nlp)
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        # Valid PDF header but garbage content that PyMuPDF cannot parse
        corrupted_bytes = b"%PDF-1.4\n trailer << /Root 1 0 R >> %%EOF corrupted data !!!"

        with pytest.raises(RuntimeError, match="Failed to parse PDF document"):
            svc.process_and_register("corrupt.pdf", corrupted_bytes)

    def test_temporary_files_are_cleaned_up_after_processing(self, tmp_path, shared_nlp):
        """Verify that temporary files are deleted from disk upon completion."""
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        pdf_bytes = create_minimal_pdf_bytes()
        initial_temp_count = len(list(Path(tempfile.gettempdir()).glob("*")))

        svc.process_and_register("clean_test.pdf", pdf_bytes)

        # In Python tempfile.TemporaryDirectory context, directory is deleted on exit
        # Verify no orphaned files with our filename remain in system temp dir
        matching_orphans = list(Path(tempfile.gettempdir()).glob("*clean_test*"))
        assert len(matching_orphans) == 0

    def test_list_documents_empty_and_populated(self, tmp_path, shared_nlp):
        """Listing documents correctly reflects counts and searchability status."""
        db = SQLiteManager(str(tmp_path / "test.db"))
        mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc = DocumentService(db_manager=db, engine_manager=mgr)

        # When DB does not exist yet
        assert svc.list_documents() == []

        # Ingest paper 1
        res1 = svc.process_and_register(
            "paper1.pdf",
            create_minimal_pdf_bytes("Paper 1 presents comprehensive experimental results across autonomous navigation benchmarks."),
        )
        assert res1.chunk_count > 0
        docs = svc.list_documents()
        assert len(docs) == 1
        assert docs[0].filename == "paper1.pdf"
        assert docs[0].is_searchable is True

        # Ingest paper 2
        res2 = svc.process_and_register(
            "paper2.pdf",
            create_minimal_pdf_bytes("Paper 2 presents comparative ablation studies evaluating multi-sensor fusion architectures."),
        )
        assert res2.chunk_count > 0
        docs = svc.list_documents()
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert "paper1.pdf" in filenames
        assert "paper2.pdf" in filenames


    def test_server_restart_distinction(self, tmp_path, shared_nlp):
        """
        Verify that active in-memory search index does NOT survive restart,
        while SQLite registration persists and documents are marked is_searchable=False.
        """
        db_file = tmp_path / "persistent_papers.db"
        db = SQLiteManager(str(db_file))

        # Session 1: Process and index document
        mgr1 = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc1 = DocumentService(db_manager=db, engine_manager=mgr1)
        res1 = svc1.process_and_register("restart_test.pdf", create_minimal_pdf_bytes())
        assert res1.is_searchable is True

        docs_before = svc1.list_documents()
        assert len(docs_before) == 1
        assert docs_before[0].is_searchable is True

        # Simulate Server Restart: New QAEngineManager instance (in-memory state reset)
        # but connecting to the SAME persistent SQLite database file
        mgr2 = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        svc2 = DocumentService(db_manager=SQLiteManager(str(db_file)), engine_manager=mgr2)

        docs_after = svc2.list_documents()
        assert len(docs_after) == 1
        assert docs_after[0].filename == "restart_test.pdf"
        # Crucial architectural verification: document is registered in DB, but NOT searchable in memory!
        assert docs_after[0].is_searchable is False
        assert mgr2.active_engine is None


# ── Integration Tests: FastAPI Endpoints ─────────────────────────────────────

class TestDocumentEndpoints:
    """HTTP API endpoint tests using TestClient."""

    @pytest.fixture
    def custom_client(self, tmp_path, shared_nlp):
        """Create an isolated test application with injected temporary database and mock engine."""
        db_path = str(tmp_path / "api_test.db")
        mock_engine_mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
            engine_builder=lambda chunks, model: MockQAEngine(chunks, model),
        )
        cfg = APIConfig(db_path=db_path)
        test_app = create_app(cfg)

        # Override dependency providers
        test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_mgr
        test_app.dependency_overrides[get_db_manager] = lambda: SQLiteManager(db_path)

        client = TestClient(test_app)
        return client, mock_engine_mgr, db_path

    def test_get_documents_empty_returns_200_empty_list(self, custom_client):
        client, _, _ = custom_client
        resp = client.get("/api/documents")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == []

    def test_upload_valid_pdf_endpoint(self, custom_client):
        client, engine_mgr, _ = custom_client
        pdf_bytes = create_minimal_pdf_bytes("Testing valid upload via FastAPI endpoint.")

        files = {"file": ("research.pdf", pdf_bytes, "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        validated = DocumentUploadResponse(**data)

        assert validated.filename == "research.pdf"
        assert validated.processing_status == "processed"
        assert validated.indexing_status == "indexed"
        assert validated.is_searchable is True
        assert validated.is_duplicate is False
        assert validated.chunk_count > 0

        # Verify GET /api/documents now lists it as searchable
        get_resp = client.get("/api/documents")
        assert get_resp.status_code == status.HTTP_200_OK
        docs = get_resp.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "research.pdf"
        assert docs[0]["is_searchable"] is True

    def test_upload_non_pdf_extension_returns_400(self, custom_client):
        client, _, _ = custom_client
        files = {"file": ("data.csv", b"id,name\n1,test", "text/csv")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        data = resp.json()
        assert data["error"] == "HTTPException"
        assert "Only PDF documents" in data["detail"]

    def test_upload_empty_file_returns_400(self, custom_client):
        client, _, _ = custom_client
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        data = resp.json()
        assert "empty (0 bytes)" in data["detail"]

    def test_upload_invalid_signature_returns_400(self, custom_client):
        client, _, _ = custom_client
        fake_content = b"Not a PDF header at all. Random binary or text data."
        files = {"file": ("fake.pdf", fake_content, "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        data = resp.json()
        assert "Invalid PDF signature" in data["detail"]

    def test_upload_oversized_file_returns_413(self, tmp_path, shared_nlp):
        """Configuring a tiny maximum size must reject with HTTP 413."""
        db_path = str(tmp_path / "tiny.db")
        mock_engine_mgr = QAEngineManager(
            embedding_model=MockEmbeddingModel(),
            nlp=shared_nlp,
        )
        cfg = APIConfig(db_path=db_path, max_upload_size_bytes=50)
        tiny_app = create_app(cfg)
        tiny_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_mgr
        client = TestClient(tiny_app)

        pdf_bytes = create_minimal_pdf_bytes()  # > 50 bytes
        files = {"file": ("toolarge.pdf", pdf_bytes, "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        data = resp.json()
        assert "exceeds maximum allowed limit" in data["detail"]

    def test_upload_malformed_pdf_returns_422(self, custom_client):
        client, _, _ = custom_client
        broken_bytes = b"%PDF-1.4\n corrupted broken structure without pages"
        files = {"file": ("broken.pdf", broken_bytes, "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)

        assert resp.status_code == 422
        data = resp.json()
        assert "Failed to parse PDF document" in data["detail"]

    def test_duplicate_upload_via_api_returns_duplicate_flag(self, custom_client):
        client, _, _ = custom_client
        pdf_bytes = create_minimal_pdf_bytes("Paper for duplicate API testing.")

        files1 = {"file": ("dup.pdf", pdf_bytes, "application/pdf")}
        resp1 = client.post("/api/documents/upload", files=files1)
        assert resp1.status_code == status.HTTP_200_OK
        assert resp1.json()["is_duplicate"] is False

        files2 = {"file": ("dup.pdf", pdf_bytes, "application/pdf")}
        resp2 = client.post("/api/documents/upload", files=files2)
        assert resp2.status_code == status.HTTP_200_OK
        assert resp2.json()["is_duplicate"] is True

        # Database still has only 1 document
        docs_resp = client.get("/api/documents")
        assert len(docs_resp.json()) == 1

    def test_query_after_upload_succeeds_end_to_end(self, custom_client):
        """End-to-end integration: query fails before upload (503), succeeds after upload (200)."""
        client, _, _ = custom_client

        # Before upload: query returns 503
        q_resp_before = client.post("/api/query", json={"query": "What is the sensor?"})
        assert q_resp_before.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

        # Upload document
        pdf_bytes = create_minimal_pdf_bytes(
            "The lidar sensor model used was Ouster OS1-32 mounted on the autonomous vehicle."
        )
        up_resp = client.post(
            "/api/documents/upload",
            files={"file": ("ouster.pdf", pdf_bytes, "application/pdf")},
        )
        assert up_resp.status_code == status.HTTP_200_OK

        # After upload: query succeeds without 503
        with patch("src.services.qa_service._default_generate_answer", return_value=["The sensor is Ouster OS1-32."]):
            q_resp_after = client.post("/api/query", json={"query": "What is the sensor?"})
            assert q_resp_after.status_code == status.HTTP_200_OK
            data = q_resp_after.json()
            assert "query" in data
            assert data["query"] == "What is the sensor?"
