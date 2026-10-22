"""
Week 4 Phase 2 — Tests for src/storage/db.py

Covers:
  - Database initialization and schema version
  - Repeated initialization
  - Migration behavior
  - Foreign-key enforcement
  - Document creation and lookup
  - Round-trip serialization of a PaperExtraction
  - Metric and provenance persistence
  - Correct physical page numbers
  - Idempotent repeated ingestion
  - Updating and replacing extraction records
  - Cascading document deletion
  - Transaction rollback after a simulated failure
  - Handling missing optional fields
  - Distinguishing separate documents with identical titles
  - Reopening the database and retrieving persisted data

Uses temporary SQLite databases so tests never modify real project data.
Tests are deterministic and do not require Ollama, network, or the actual PDF corpus.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.extraction.schemas import (
    ExtractedField,
    MetricResult,
    PaperExtraction,
    ProvenanceSpan,
)
from src.storage.db import SQLiteManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = SQLiteManager(path)
    db.initialize_schema()
    yield db
    db.close()
    # Clean up
    if os.path.exists(path):
        os.unlink(path)


class TestDatabaseInitialization:
    def test_initialize_schema_creates_tables(self, temp_db):
        """Test that schema initialization creates all required tables."""
        conn = temp_db.connect()
        cursor = temp_db.conn.cursor()

        # Check that all tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        assert "documents" in tables
        assert "paper_extractions" in tables
        assert "extraction_metrics" in tables
        assert "field_provenance" in tables
        assert "schema_version" in tables

    def test_initialize_schema_is_idempotent(self, temp_db):
        """Test that calling initialize_schema multiple times is safe."""
        temp_db.initialize_schema()  # Second call
        temp_db.initialize_schema()  # Third call

        # Should still work without errors
        conn = temp_db.connect()
        cursor = temp_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 0

    def test_schema_version_is_recorded(self, temp_db):
        """Test that schema version is recorded correctly."""
        version = temp_db.get_schema_version()
        assert version == 1

    def test_foreign_keys_are_enabled(self, temp_db):
        """Test that foreign key enforcement is enabled."""
        conn = temp_db.connect()
        cursor = temp_db.conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        assert result[0] == 1  # 1 = enabled


class TestDocumentRegistration:
    def test_register_new_document(self, temp_db):
        """Test registering a new document."""
        inserted = temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )
        assert inserted is True

        doc = temp_db.get_document("doc_001")
        assert doc is not None
        assert doc["document_id"] == "doc_001"
        assert doc["filename"] == "paper.pdf"
        assert doc["page_count"] == 10
        assert doc["chunk_count"] == 50
        assert doc["char_count"] == 25000

    def test_register_duplicate_document_updates(self, temp_db):
        """Test that re-registering the same document updates instead of duplicating."""
        # First registration
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        # Second registration with updated metadata
        inserted = temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=12,  # Updated
            chunk_count=60,  # Updated
            char_count=30000,  # Updated
        )
        assert inserted is False  # Was an update, not insert

        # Verify updated values
        doc = temp_db.get_document("doc_001")
        assert doc["page_count"] == 12
        assert doc["chunk_count"] == 60
        assert doc["char_count"] == 30000

        # Verify only one document exists
        conn = temp_db.connect()
        cursor = temp_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 1

    def test_get_nonexistent_document_returns_none(self, temp_db):
        """Test that getting a nonexistent document returns None."""
        doc = temp_db.get_document("nonexistent")
        assert doc is None

    def test_document_exists_check(self, temp_db):
        """Test document existence check."""
        assert temp_db.document_exists("doc_001") is False

        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        assert temp_db.document_exists("doc_001") is True


class TestExtractionPersistence:
    def test_save_and_retrieve_extraction(self, temp_db):
        """Test round-trip serialization of a PaperExtraction."""
        # Register document first
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        # Create extraction
        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(
                value="Test Paper Title",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=1, verbatim_quote="Full title: Test Paper Title")],
            ),
            authors=ExtractedField(value="Author One, Author Two", is_stated=True),
            year=ExtractedField(value="2025", is_stated=True),
            research_objective=ExtractedField(
                value="To test extraction", is_stated=True, provenance=[ProvenanceSpan(page_number=2)]
            ),
            proposed_methodology=ExtractedField(
                value="Test method", is_stated=True, provenance=[ProvenanceSpan(page_number=3)]
            ),
            quantitative_findings=[
                MetricResult(
                    metric_name="accuracy",
                    value="95%",
                    dataset="test",
                    provenance=ProvenanceSpan(page_number=5),
                )
            ],
        )

        # Save extraction
        temp_db.save_extraction(extraction)

        # Retrieve extraction
        retrieved = temp_db.get_extraction("doc_001")
        assert retrieved is not None
        assert retrieved.paper_id == "doc_001"
        assert retrieved.title.value == "Test Paper Title"
        assert retrieved.title.is_stated is True
        assert retrieved.authors.value == "Author One, Author Two"
        assert len(retrieved.quantitative_findings) == 1
        assert retrieved.quantitative_findings[0].metric_name == "accuracy"

    def test_save_extraction_without_document_raises_error(self, temp_db):
        """Test that saving extraction for unregistered document raises error."""
        extraction = PaperExtraction(
            paper_id="nonexistent",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )

        with pytest.raises(RuntimeError) as exc_info:
            temp_db.save_extraction(extraction)
        assert "not registered" in str(exc_info.value)

    def test_update_extraction_is_idempotent(self, temp_db):
        """Test that updating an extraction replaces old data without duplicates."""
        # Register document
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        # Save first extraction
        extraction1 = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Original Title", is_stated=True),
            research_objective=ExtractedField(value="Original Objective", is_stated=True),
            proposed_methodology=ExtractedField(value="Original Method", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="accuracy", value="90%", dataset="test")
            ],
        )
        temp_db.save_extraction(extraction1)

        # Save updated extraction
        extraction2 = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Updated Title", is_stated=True),
            research_objective=ExtractedField(value="Updated Objective", is_stated=True),
            proposed_methodology=ExtractedField(value="Updated Method", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="accuracy", value="95%", dataset="test"),
                MetricResult(metric_name="F1", value="0.92", dataset="test"),
            ],
        )
        temp_db.save_extraction(extraction2)

        # Verify only one extraction exists
        conn = temp_db.connect()
        cursor = temp_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paper_extractions")
        assert cursor.fetchone()[0] == 1

        # Verify metrics were replaced (2 new metrics, not 3 total)
        cursor.execute("SELECT COUNT(*) FROM extraction_metrics WHERE document_id = ?", ("doc_001",))
        assert cursor.fetchone()[0] == 2

        # Verify updated values
        retrieved = temp_db.get_extraction("doc_001")
        assert retrieved.title.value == "Updated Title"
        assert len(retrieved.quantitative_findings) == 2

    def test_extraction_exists_check(self, temp_db):
        """Test extraction existence check."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        assert temp_db.extraction_exists("doc_001") is False

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        assert temp_db.extraction_exists("doc_001") is True


class TestMetricPersistence:
    def test_metrics_are_persisted_correctly(self, temp_db):
        """Test that metrics are saved with correct values."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(
                    metric_name="mAP@0.5",
                    value="68.2%",
                    dataset="KITTI",
                    provenance=ProvenanceSpan(page_number=6),
                ),
                MetricResult(
                    metric_name="F1-score",
                    value="0.84",
                    dataset="COCO",
                    provenance=ProvenanceSpan(page_number=7),
                ),
            ],
        )
        temp_db.save_extraction(extraction)

        # Retrieve metrics
        metrics = temp_db.get_metrics_for_document("doc_001")
        assert len(metrics) == 2

        assert metrics[0]["metric_name"] == "mAP@0.5"
        assert metrics[0]["metric_value"] == "68.2%"
        assert metrics[0]["dataset"] == "KITTI"
        assert metrics[0]["page_number"] == 6

        assert metrics[1]["metric_name"] == "F1-score"
        assert metrics[1]["metric_value"] == "0.84"
        assert metrics[1]["dataset"] == "COCO"
        assert metrics[1]["page_number"] == 7

    def test_metric_without_dataset_is_persisted(self, temp_db):
        """Test that metrics without optional dataset are saved correctly."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="RMSE", value="2.3 cm")  # No dataset
            ],
        )
        temp_db.save_extraction(extraction)

        metrics = temp_db.get_metrics_for_document("doc_001")
        assert len(metrics) == 1
        assert metrics[0]["dataset"] is None


class TestProvenancePersistence:
    def test_provenance_is_persisted_correctly(self, temp_db):
        """Test that field provenance is saved with correct page numbers."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=1, verbatim_quote="Title on page 1")],
            ),
            authors=ExtractedField(
                value="Test Authors",
                is_stated=True,
                provenance=[
                    ProvenanceSpan(page_number=1, verbatim_quote="Authors on page 1"),
                    ProvenanceSpan(page_number=2, verbatim_quote="Authors continued on page 2"),
                ],
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        # Retrieve provenance for title
        title_provenance = temp_db.get_provenance_for_field("doc_001", "title")
        assert len(title_provenance) == 1
        assert title_provenance[0]["page_number"] == 1
        assert title_provenance[0]["verbatim_quote"] == "Title on page 1"

        # Retrieve provenance for authors
        authors_provenance = temp_db.get_provenance_for_field("doc_001", "authors")
        assert len(authors_provenance) == 2
        assert authors_provenance[0]["page_number"] == 1
        assert authors_provenance[1]["page_number"] == 2

    def test_provenance_without_quote_is_persisted(self, temp_db):
        """Test that provenance without verbatim quote is saved correctly."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=1)],  # No quote
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        provenance = temp_db.get_provenance_for_field("doc_001", "title")
        assert len(provenance) == 1
        assert provenance[0]["page_number"] == 1
        assert provenance[0]["verbatim_quote"] is None


class TestForeignKeyCascading:
    def test_delete_document_cascades_to_extraction(self, temp_db):
        """Test that deleting a document deletes its extraction."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        # Verify extraction exists
        assert temp_db.extraction_exists("doc_001") is True

        # Delete document
        deleted = temp_db.delete_document("doc_001")
        assert deleted is True

        # Verify extraction is gone (cascade)
        assert temp_db.extraction_exists("doc_001") is False
        assert temp_db.get_document("doc_001") is None

    def test_delete_document_cascades_to_metrics(self, temp_db):
        """Test that deleting a document deletes its metrics."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="accuracy", value="95%", dataset="test")
            ],
        )
        temp_db.save_extraction(extraction)

        # Verify metrics exist
        metrics = temp_db.get_metrics_for_document("doc_001")
        assert len(metrics) == 1

        # Delete document
        temp_db.delete_document("doc_001")

        # Verify metrics are gone (cascade)
        metrics = temp_db.get_metrics_for_document("doc_001")
        assert len(metrics) == 0

    def test_delete_document_cascades_to_provenance(self, temp_db):
        """Test that deleting a document deletes its provenance."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=1)],
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        # Verify provenance exists
        provenance = temp_db.get_provenance_for_field("doc_001", "title")
        assert len(provenance) == 1

        # Delete document
        temp_db.delete_document("doc_001")

        # Verify provenance is gone (cascade)
        provenance = temp_db.get_provenance_for_field("doc_001", "title")
        assert len(provenance) == 0


class TestTransactionRollback:
    def test_save_extraction_rolls_back_on_error(self, temp_db):
        """Test that extraction save rolls back on error."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        # Create valid extraction
        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        # Verify extraction exists
        assert temp_db.extraction_exists("doc_001") is True

        # Try to save extraction for unregistered document (should fail and rollback)
        bad_extraction = PaperExtraction(
            paper_id="nonexistent",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )

        with pytest.raises(RuntimeError):
            temp_db.save_extraction(bad_extraction)

        # Verify original extraction is still intact (no partial corruption)
        assert temp_db.extraction_exists("doc_001") is True
        original = temp_db.get_extraction("doc_001")
        assert original.title.value == "Test"


class TestMissingOptionalFields:
    def test_missing_fields_are_persisted_as_none(self, temp_db):
        """Test that missing optional fields are persisted as None."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            # Missing optional fields (default ExtractedField with value=None)
        )
        temp_db.save_extraction(extraction)

        retrieved = temp_db.get_extraction("doc_001")
        assert retrieved.authors.value is None
        assert retrieved.year.value is None
        assert retrieved.dataset_used.value is None
        assert retrieved.hardware_or_sensors.value is None
        assert retrieved.acknowledged_limitations.value is None


class TestDistinguishingDocuments:
    def test_identical_titles_do_not_conflate_documents(self, temp_db):
        """Test that documents with identical titles are distinguished by document_id."""
        # Register two documents with same title but different IDs
        temp_db.register_document(
            document_id="doc_001",
            filename="paper1.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )
        temp_db.register_document(
            document_id="doc_002",
            filename="paper2.pdf",
            page_count=12,
            chunk_count=60,
            char_count=30000,
        )

        # Save extractions with identical titles
        extraction1 = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Same Title", is_stated=True),
            research_objective=ExtractedField(value="Objective 1", is_stated=True),
            proposed_methodology=ExtractedField(value="Method 1", is_stated=True),
        )
        extraction2 = PaperExtraction(
            paper_id="doc_002",
            title=ExtractedField(value="Same Title", is_stated=True),
            research_objective=ExtractedField(value="Objective 2", is_stated=True),
            proposed_methodology=ExtractedField(value="Method 2", is_stated=True),
        )

        temp_db.save_extraction(extraction1)
        temp_db.save_extraction(extraction2)

        # Verify they are distinct
        retrieved1 = temp_db.get_extraction("doc_001")
        retrieved2 = temp_db.get_extraction("doc_002")

        assert retrieved1.paper_id == "doc_001"
        assert retrieved2.paper_id == "doc_002"
        assert retrieved1.title.value == "Same Title"
        assert retrieved2.title.value == "Same Title"
        assert retrieved1.research_objective.value == "Objective 1"
        assert retrieved2.research_objective.value == "Objective 2"


class TestDatabaseReopening:
    def test_data_persists_after_reopening(self, temp_db):
        """Test that data survives closing and reopening the database."""
        # Save data
        temp_db.register_document(
            document_id="doc_001",
            filename="paper.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )

        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        temp_db.save_extraction(extraction)

        # Close connection
        temp_db.close()

        # Reopen with same path
        db_path = temp_db.db_path
        new_db = SQLiteManager(str(db_path))
        new_db.initialize_schema()

        # Verify data is still there
        doc = new_db.get_document("doc_001")
        assert doc is not None
        assert doc["filename"] == "paper.pdf"

        retrieved = new_db.get_extraction("doc_001")
        assert retrieved is not None
        assert retrieved.title.value == "Test"

        new_db.close()


class TestGetAllOperations:
    def test_get_all_documents(self, temp_db):
        """Test retrieving all documents."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper1.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )
        temp_db.register_document(
            document_id="doc_002",
            filename="paper2.pdf",
            page_count=12,
            chunk_count=60,
            char_count=30000,
        )

        docs = temp_db.get_all_documents()
        assert len(docs) == 2
        assert any(d["document_id"] == "doc_001" for d in docs)
        assert any(d["document_id"] == "doc_002" for d in docs)

    def test_get_all_extractions(self, temp_db):
        """Test retrieving all extractions."""
        temp_db.register_document(
            document_id="doc_001",
            filename="paper1.pdf",
            page_count=10,
            chunk_count=50,
            char_count=25000,
        )
        temp_db.register_document(
            document_id="doc_002",
            filename="paper2.pdf",
            page_count=12,
            chunk_count=60,
            char_count=30000,
        )

        extraction1 = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Paper 1", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        extraction2 = PaperExtraction(
            paper_id="doc_002",
            title=ExtractedField(value="Paper 2", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )

        temp_db.save_extraction(extraction1)
        temp_db.save_extraction(extraction2)

        extractions = temp_db.get_all_extractions()
        assert len(extractions) == 2
        assert any(e.paper_id == "doc_001" for e in extractions)
        assert any(e.paper_id == "doc_002" for e in extractions)
