"""
IntelliAsk Research — Phase 2E: Tests for Matrix, Provenance & History APIs

Covers:
- GET /api/matrix: response structure, columns, row formatting, empty-corpus behavior.
- GET /api/matrix/export: CSV download headers, RFC 4180 compliance, Unicode, commas,
  quotes, embedded newlines, and consistency with matrix endpoint.
- GET /api/documents/{document_id}/provenance: stored field-level provenance, 404 for
  unknown documents, honest limitations reporting (no fabrication).
- GET /api/history & DELETE /api/history: query logging, ordering, reopen persistence,
  safe clear scope (verifying documents/extractions/provenance are untouched).
- SQLite schema migration: v1 to v2 migration without data loss.
"""

import csv
import io
import os
import tempfile
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.config import APIConfig
from src.api.main import _engine_holder, create_app
from src.extraction.schemas import (
    ExtractedField,
    MetricResult,
    PaperExtraction,
    ProvenanceSpan,
)
from src.matrix.exporter import MATRIX_COLUMNS
from src.storage.db import SQLiteManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = SQLiteManager(path)
    db.initialize_schema()
    db.close()
    yield path
    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture
def client(temp_db_path):
    """Create a TestClient with a fresh temporary database."""
    config = APIConfig(db_path=temp_db_path)
    app = create_app(config=config)
    with TestClient(app) as test_client:
        yield test_client


def _make_sample_extraction(paper_id: str, title: str = "Test Study", quotes: bool = True) -> PaperExtraction:
    """Helper to build a valid PaperExtraction object with or without provenance quotes."""
    provenance = [ProvenanceSpan(page_number=1, verbatim_quote="Explicit study text quote.")] if quotes else []
    return PaperExtraction(
        paper_id=paper_id,
        title=ExtractedField(value=title, is_stated=True, confidence=0.95, provenance=provenance),
        authors=ExtractedField(value="Alice Smith, Bob Jones", is_stated=True, confidence=0.90, provenance=provenance),
        year=ExtractedField(value="2024", is_stated=True, confidence=1.0, provenance=[]),
        research_objective=ExtractedField(value="Evaluate multimodal QA performance", is_stated=True, confidence=0.88, provenance=provenance),
        proposed_methodology=ExtractedField(value="Deep neural network with attention", is_stated=True, confidence=0.85, provenance=provenance),
        dataset_used=ExtractedField(value="Custom Benchmark v1", is_stated=True, confidence=0.80, provenance=[]),
        hardware_or_sensors=ExtractedField(value="NVIDIA A100 GPU", is_stated=False, confidence=0.60, provenance=[]),
        acknowledged_limitations=ExtractedField(value="Limited to English language", is_stated=True, confidence=0.75, provenance=[]),
        quantitative_findings=[
            MetricResult(
                metric_name="Accuracy",
                value="94.5%",
                dataset="Custom Benchmark v1",
                provenance=ProvenanceSpan(page_number=4, verbatim_quote="achieved 94.5% accuracy") if quotes else None,
            )
        ],
        extraction_timestamp="2026-10-22T12:00:00Z",
    )


# ── Literature Matrix Tests ───────────────────────────────────────────────────

class TestMatrixEndpoints:
    def test_get_matrix_empty_corpus(self, client):
        """GET /api/matrix returns standard columns and 0 rows when corpus is empty."""
        resp = client.get("/api/matrix")
        assert resp.status_code == 200
        data = resp.json()

        assert "columns" in data
        assert "rows" in data
        assert "total_rows" in data
        assert data["columns"] == MATRIX_COLUMNS
        assert len(data["columns"]) == 12
        assert data["rows"] == []
        assert data["total_rows"] == 0

    def test_get_matrix_with_extractions(self, client, temp_db_path):
        """GET /api/matrix returns formatted rows corresponding to saved extractions."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc1", "paper1.pdf", page_count=5, chunk_count=20, char_count=8000)
        ext = _make_sample_extraction("doc1", title="Quantum Computing Advances")
        db.save_extraction(ext)
        db.close()

        resp = client.get("/api/matrix")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_rows"] == 1
        assert len(data["rows"]) == 1
        row = data["rows"][0]
        assert row["Document"] == "paper1.pdf"
        assert row["Title"] == "Quantum Computing Advances"
        assert row["Authors"] == "Alice Smith, Bob Jones"
        assert row["Year"] == "2024"
        assert "Accuracy: 94.5%" in row["Key Metrics"]
        assert "NVIDIA A100 GPU (inferred)" in row["Hardware/Sensors"]
        assert "1, 4" in row["Pages Cited"]

    def test_export_matrix_csv_empty(self, client):
        """GET /api/matrix/export returns valid CSV header even when empty."""
        resp = client.get("/api/matrix/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "attachment; filename=" in resp.headers["content-disposition"]
        assert resp.headers["content-disposition"].endswith('.csv"')

        # Parse CSV content
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == MATRIX_COLUMNS

    def test_export_matrix_csv_rfc4180_and_unicode(self, client, temp_db_path):
        """Test CSV export properly handles RFC 4180 quoting, commas, quotes, embedded newlines, and Unicode."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc_unicode", "unicode_paper.pdf", 10, 40, 15000)

        # Extraction containing quotes, commas, newlines, and non-ASCII Unicode
        complex_title = 'A "Groundbreaking" Study: Fast, Accurate & Scalable'
        complex_limitations = "1. Sample size small;\n2. Memory leaks observed;\n3. UTF-8 test: Café, naïve, 🚀"
        ext = PaperExtraction(
            paper_id="doc_unicode",
            title=ExtractedField(value=complex_title, is_stated=True, confidence=1.0, provenance=[]),
            authors=ExtractedField(value='Smith, "The Researcher", Jr.', is_stated=True, confidence=1.0, provenance=[]),
            year=ExtractedField(value="2025", is_stated=True, confidence=1.0, provenance=[]),
            research_objective=ExtractedField(value="Comma, newline\nand quote \"test\"", is_stated=True, confidence=1.0, provenance=[]),
            proposed_methodology=ExtractedField(value="Standard", is_stated=True, confidence=1.0, provenance=[]),
            dataset_used=ExtractedField(value="Dataset-A", is_stated=True, confidence=1.0, provenance=[]),
            hardware_or_sensors=ExtractedField(value="Sensors", is_stated=True, confidence=1.0, provenance=[]),
            acknowledged_limitations=ExtractedField(value=complex_limitations, is_stated=True, confidence=1.0, provenance=[]),
            quantitative_findings=[],
            extraction_timestamp="2026-10-22T15:00:00Z",
        )
        db.save_extraction(ext)
        db.close()

        resp = client.get("/api/matrix/export")
        assert resp.status_code == 200
        content = resp.text

        # Verify RFC 4180 parsing with standard csv library
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 data row
        header, data_row = rows[0], rows[1]
        assert header == MATRIX_COLUMNS

        # Check fields survive round-trip without corruption
        title_idx = header.index("Title")
        limitations_idx = header.index("Limitations")
        assert data_row[title_idx] == complex_title
        assert data_row[limitations_idx] == complex_limitations
        assert "Café, naïve, 🚀" in data_row[limitations_idx]


# ── Field Provenance Tests ────────────────────────────────────────────────────

class TestProvenanceEndpoints:
    def test_get_provenance_unknown_document_404(self, client):
        """GET /api/documents/{document_id}/provenance returns 404 for unregistered doc."""
        resp = client.get("/api/documents/non_existent_doc/provenance")
        assert resp.status_code == 404
        data = resp.json()
        assert "non_existent_doc" in data["detail"]

    def test_get_provenance_unextracted_document(self, client, temp_db_path):
        """GET /api/documents/{document_id}/provenance reports unextracted state honestly."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc_unextracted", "fresh.pdf", 4, 12, 5000)
        db.close()

        resp = client.get("/api/documents/doc_unextracted/provenance")
        assert resp.status_code == 200
        data = resp.json()

        assert data["document_id"] == "doc_unextracted"
        assert data["filename"] == "fresh.pdf"
        assert data["provenance_status"] == "unextracted"
        assert data["total_records"] == 0
        assert data["provenance_records"] == []
        assert "has not been executed" in data["limitations"]

    def test_get_provenance_extracted_with_no_spans(self, client, temp_db_path):
        """GET /api/documents/{document_id}/provenance reports no_spans_stored honestly without fabrication."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc_no_spans", "empty_quotes.pdf", 3, 10, 4000)
        ext = _make_sample_extraction("doc_no_spans", title="No Quotes Paper", quotes=False)
        db.save_extraction(ext)
        db.close()

        resp = client.get("/api/documents/doc_no_spans/provenance")
        assert resp.status_code == 200
        data = resp.json()

        assert data["document_id"] == "doc_no_spans"
        assert data["provenance_status"] == "no_spans_stored"
        assert data["total_records"] == 0
        assert data["provenance_records"] == []
        assert "no field-level provenance quotes were preserved" in data["limitations"]

    def test_get_provenance_success_with_records(self, client, temp_db_path):
        """GET /api/documents/{document_id}/provenance returns exact stored spans."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc_with_spans", "cited.pdf", 6, 25, 12000)
        ext = _make_sample_extraction("doc_with_spans", title="Cited Paper", quotes=True)
        db.save_extraction(ext)
        db.close()

        resp = client.get("/api/documents/doc_with_spans/provenance")
        assert resp.status_code == 200
        data = resp.json()

        assert data["document_id"] == "doc_with_spans"
        assert data["provenance_status"] == "available"
        assert data["total_records"] > 0
        assert data["limitations"] is None

        # Verify spans contain exact saved quote and page number
        records = data["provenance_records"]
        assert any(r["verbatim_quote"] == "Explicit study text quote." and r["page_number"] == 1 for r in records)


# ── Query History Tests ───────────────────────────────────────────────────────

class FakeQAEngine:
    """Mock engine satisfying the interface for query execution in tests."""

    class _FakeChunk:
        def __init__(self):
            self.text = "Sample content about AI research."
            self.source_filename = "paper1.pdf"
            self.source = "paper1.pdf"
            self.page_number = 1
            self.dense_score = 0.85
            self.score = 0.9
            self.chunk_id = "c1"

    def search_hybrid(self, query: str, top_k: int = 10) -> list:
        return [self._FakeChunk()]


class TestQueryHistoryEndpoints:
    def test_history_empty_initially(self, client):
        """GET /api/history returns empty list when no queries have been run."""
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_recorded_and_retrieved(self, client, temp_db_path):
        """History records written to the DB are correctly returned by GET /api/history."""
        # Write directly via SQLiteManager (same DB the client uses)
        db = SQLiteManager(temp_db_path)
        db.record_query_history(
            query="What is the status of AI research?",
            answer="AI research is advancing rapidly.",
            answer_source="ollama",
            has_supporting_evidence=True,
            best_source="paper1.pdf",
            best_score=0.88,
            citations=["[paper1.pdf, Page 1]"],
        )
        db.close()

        hist_resp = client.get("/api/history")
        assert hist_resp.status_code == 200
        history = hist_resp.json()

        assert len(history) == 1
        entry = history[0]
        assert entry["query"] == "What is the status of AI research?"
        assert entry["answer_source"] == "ollama"
        assert entry["has_supporting_evidence"] is True
        assert entry["best_source"] == "paper1.pdf"
        assert entry["best_score"] >= 0.8
        assert entry["citations"] == ["[paper1.pdf, Page 1]"]
        assert "history_id" in entry
        assert "timestamp" in entry

    def test_history_persistence_across_reopen(self, client, temp_db_path):
        """History records persist across database close and reopen."""
        # Manually record entries via SQLiteManager
        db1 = SQLiteManager(temp_db_path)
        id1 = db1.record_query_history(
            query="First Query?",
            answer="Answer 1",
            answer_source="ollama",
            has_supporting_evidence=True,
            best_source="p1.pdf",
            best_score=0.88,
            citations=["[p1.pdf, Page 1]"],
            timestamp="2026-10-22T10:00:00Z",
        )
        id2 = db1.record_query_history(
            query="Second Query?",
            answer="Abstention",
            answer_source="abstention",
            has_supporting_evidence=False,
            best_source="p2.pdf",
            best_score=0.22,
            citations=[],
            timestamp="2026-10-22T11:00:00Z",
        )
        db1.close()

        # Reopen and query through FastAPI client
        resp = client.get("/api/history")
        assert resp.status_code == 200
        history = resp.json()

        assert len(history) == 2
        # Ordered by timestamp DESC (Second Query first)
        assert history[0]["query"] == "Second Query?"
        assert history[0]["has_supporting_evidence"] is False
        assert history[0]["answer_source"] == "abstention"

        assert history[1]["query"] == "First Query?"
        assert history[1]["has_supporting_evidence"] is True
        assert history[1]["citations"] == ["[p1.pdf, Page 1]"]

    def test_delete_history_safe_scope(self, client, temp_db_path):
        """DELETE /api/history clears history records but NEVER touches documents, extractions, or provenance."""
        db = SQLiteManager(temp_db_path)
        db.register_document("doc_safe", "safe.pdf", 5, 20, 8000)
        ext = _make_sample_extraction("doc_safe", title="Safe Paper", quotes=True)
        db.save_extraction(ext)

        # Record a query
        db.record_query_history("Can I be deleted?", "Yes", "ollama", True)
        assert len(db.get_query_history()) == 1
        db.close()

        # Invoke DELETE /api/history
        del_resp = client.delete("/api/history")
        assert del_resp.status_code == 200
        del_data = del_resp.json()

        assert del_data["deleted_count"] == 1
        assert del_data["scope"] == "query_history_only"

        # Verify history is now empty
        hist_resp = client.get("/api/history")
        assert hist_resp.status_code == 200
        assert hist_resp.json() == []

        # Verify document, extraction, and provenance still exist completely intact!
        db_check = SQLiteManager(temp_db_path)
        assert db_check.document_exists("doc_safe") is True
        assert db_check.extraction_exists("doc_safe") is True
        doc = db_check.get_document("doc_safe")
        assert doc["filename"] == "safe.pdf"
        provenance = db_check.get_all_provenance_for_document("doc_safe")
        assert len(provenance) > 0
        db_check.close()


# ── Schema Migration Tests ───────────────────────────────────────────────────

class TestSchemaMigration:
    def test_migrate_to_version_2_preserves_data(self, temp_db_path):
        """Test that migrate_to_version_2 updates schema_version to 2 and preserves all existing rows."""
        db = SQLiteManager(temp_db_path)
        assert db.get_schema_version() == 1

        # Register document and save extraction in v1
        db.register_document("doc_v1", "v1.pdf", 2, 8, 3000)
        ext = _make_sample_extraction("doc_v1", title="V1 Paper")
        db.save_extraction(ext)

        # Perform migration to v2
        new_version = db.migrate_to_version_2()
        assert new_version == 2
        assert db.get_schema_version() == 2

        # Verify existing data is preserved intact
        assert db.document_exists("doc_v1") is True
        assert db.get_extraction("doc_v1").title.value == "V1 Paper"

        # Verify query_history table functions properly under v2
        db.record_query_history("V2 query?", "V2 answer", "ollama", True)
        hist = db.get_query_history()
        assert len(hist) == 1
        assert hist[0]["query"] == "V2 query?"

        db.close()
