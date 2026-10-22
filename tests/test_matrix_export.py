"""
Week 4 Phase 3 — Tests for src/matrix/exporter.py

Covers:
  - Multiple papers with different optional fields
  - Missing and unreported values
  - Multiple metrics per paper
  - Unicode text
  - Commas, double quotes, and embedded newlines
  - Stable header ordering
  - Round-trip parsing with csv.reader
  - Empty matrix
  - Missing provenance
  - Multiple provenance spans and physical page numbers
  - Retrieval from SQLite without an Ollama dependency
  - Correct handling of papers without stored extractions
"""

import csv
import io
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
from src.matrix.exporter import (
    MATRIX_COLUMNS,
    build_matrix_from_db,
    build_matrix_from_extractions,
    export_db_to_csv,
    export_matrix_to_csv,
    extraction_to_matrix_row,
    _format_field_value,
    _format_metrics,
    _format_pages_cited,
)
from src.storage.db import SQLiteManager


class TestFormatFieldValues:
    def test_explicitly_stated_value(self):
        """Test that explicitly stated values are formatted correctly."""
        result = _format_field_value("YOLOv8", is_stated=True)
        assert result == "YOLOv8"

    def test_inferred_value(self):
        """Test that inferred values are marked as such."""
        result = _format_field_value("2025", is_stated=False)
        assert result == "2025 (inferred)"

    def test_none_value(self):
        """Test that None values are formatted as 'Not reported'."""
        result = _format_field_value(None, is_stated=True)
        assert result == "Not reported"


class TestFormatMetrics:
    def test_single_metric(self):
        """Test formatting a single metric."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="accuracy", value="95%", dataset="KITTI")
            ],
        )
        result = _format_metrics(extraction)
        assert "accuracy: 95%" in result
        assert "(KITTI)" in result

    def test_multiple_metrics(self):
        """Test formatting multiple metrics."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="mAP@0.5", value="68.2%", dataset="KITTI"),
                MetricResult(metric_name="F1", value="0.84", dataset="COCO"),
            ],
        )
        result = _format_metrics(extraction)
        assert "mAP@0.5: 68.2% (KITTI)" in result
        assert "F1: 0.84 (COCO)" in result
        assert ";" in result  # Semicolon separator

    def test_no_metrics(self):
        """Test formatting when no metrics are present."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[],
        )
        result = _format_metrics(extraction)
        assert result == "Not reported"


class TestFormatPagesCited:
    def test_single_page(self):
        """Test formatting a single page."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=3)]
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        result = _format_pages_cited(extraction)
        assert result == "3"

    def test_multiple_pages(self):
        """Test formatting multiple pages."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[ProvenanceSpan(page_number=1), ProvenanceSpan(page_number=5)]
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        result = _format_pages_cited(extraction)
        assert result == "1, 5"

    def test_no_pages(self):
        """Test formatting when no provenance is present."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        result = _format_pages_cited(extraction)
        assert result == "Not reported"


class TestExtractionToMatrixRow:
    def test_complete_extraction(self):
        """Test transforming a complete extraction."""
        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test Paper", is_stated=True),
            authors=ExtractedField(value="Author One", is_stated=True),
            year=ExtractedField(value="2025", is_stated=True),
            research_objective=ExtractedField(value="Test objective", is_stated=True),
            proposed_methodology=ExtractedField(value="Test method", is_stated=True),
            dataset_used=ExtractedField(value="Test dataset", is_stated=True),
            hardware_or_sensors=ExtractedField(value="GPU", is_stated=True),
            acknowledged_limitations=ExtractedField(value="Small dataset", is_stated=True),
        )
        row = extraction_to_matrix_row(extraction, filename="paper.pdf")
        
        assert row["Document"] == "paper.pdf"
        assert row["Title"] == "Test Paper"
        assert row["Authors"] == "Author One"
        assert row["Year"] == "2025"
        assert row["Research Objective"] == "Test objective"
        assert row["Methodology"] == "Test method"
        assert row["Dataset"] == "Test dataset"
        assert row["Hardware/Sensors"] == "GPU"
        assert row["Limitations"] == "Small dataset"

    def test_missing_optional_fields(self):
        """Test transformation with missing optional fields."""
        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            # Other fields have default ExtractedField with value=None
        )
        row = extraction_to_matrix_row(extraction)
        
        assert row["Authors"] == "Not reported"
        assert row["Year"] == "Not reported"
        assert row["Dataset"] == "Not reported"
        assert row["Hardware/Sensors"] == "Not reported"
        assert row["Limitations"] == "Not reported"

    def test_uses_paper_id_when_no_filename(self):
        """Test that paper_id is used when filename is not provided."""
        extraction = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        row = extraction_to_matrix_row(extraction, filename=None)
        assert row["Document"] == "doc_001"


class TestBuildMatrixFromExtractions:
    def test_multiple_extractions(self):
        """Test building matrix from multiple extractions."""
        extraction1 = PaperExtraction(
            paper_id="doc_001",
            title=ExtractedField(value="Paper 1", is_stated=True),
            research_objective=ExtractedField(value="Obj 1", is_stated=True),
            proposed_methodology=ExtractedField(value="Method 1", is_stated=True),
        )
        extraction2 = PaperExtraction(
            paper_id="doc_002",
            title=ExtractedField(value="Paper 2", is_stated=True),
            research_objective=ExtractedField(value="Obj 2", is_stated=True),
            proposed_methodology=ExtractedField(value="Method 2", is_stated=True),
        )
        
        matrix = build_matrix_from_extractions([extraction1, extraction2])
        assert len(matrix) == 2
        assert matrix[0]["Title"] == "Paper 1"
        assert matrix[1]["Title"] == "Paper 2"

    def test_with_db_lookup(self):
        """Test building matrix with database filename lookup."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = None
        try:
            db = SQLiteManager(db_path)
            db.initialize_schema()

            # Register documents
            db.register_document("doc_001", "paper1.pdf", 10, 50, 25000)
            db.register_document("doc_002", "paper2.pdf", 12, 60, 30000)

            # Create extractions
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

            matrix = build_matrix_from_extractions([extraction1, extraction2], db)

            assert matrix[0]["Document"] == "paper1.pdf"
            assert matrix[1]["Document"] == "paper2.pdf"
        finally:
            if db:
                db.close()
            try:
                os.unlink(db_path)
            except OSError:
                pass


class TestExportMatrixToCsv:
    def test_empty_matrix(self):
        """Test CSV export of empty matrix."""
        csv_data = export_matrix_to_csv([])
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)
        assert len(rows) == 1  # Header row only
        assert rows[0] == MATRIX_COLUMNS

    def test_single_row_matrix(self):
        """Test CSV export of single row."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": "Test Paper",
            "Authors": "Author One",
            "Year": "2025",
            "Research Objective": "Test objective",
            "Methodology": "Test method",
            "Dataset": "Test dataset",
            "Hardware/Sensors": "GPU",
            "Key Metrics": "accuracy: 95%",
            "Limitations": "Small dataset",
            "Pages Cited": "1, 3, 5",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)

        # Parse with csv.reader to verify RFC 4180 compliance
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert len(rows) == 2  # Header + 1 data row
        assert rows[0] == MATRIX_COLUMNS
        assert rows[1][0] == "paper.pdf"
        assert rows[1][1] == "Test Paper"

    def test_comma_escaping(self):
        """Test that commas in values are properly escaped."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": "Test Paper, with comma",
            "Authors": "Not reported",
            "Year": "Not reported",
            "Research Objective": "Test",
            "Methodology": "Test",
            "Dataset": "Not reported",
            "Hardware/Sensors": "Not reported",
            "Key Metrics": "Not reported",
            "Limitations": "Not reported",
            "Pages Cited": "Not reported",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert rows[1][1] == "Test Paper, with comma"

    def test_quote_escaping(self):
        """Test that quotes in values are properly escaped."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": 'Test "quoted" paper',
            "Authors": "Not reported",
            "Year": "Not reported",
            "Research Objective": "Test",
            "Methodology": "Test",
            "Dataset": "Not reported",
            "Hardware/Sensors": "Not reported",
            "Key Metrics": "Not reported",
            "Limitations": "Not reported",
            "Pages Cited": "Not reported",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert rows[1][1] == 'Test "quoted" paper'

    def test_newline_escaping(self):
        """Test that newlines in values are properly escaped."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": "Test\nPaper",
            "Authors": "Not reported",
            "Year": "Not reported",
            "Research Objective": "Test",
            "Methodology": "Test",
            "Dataset": "Not reported",
            "Hardware/Sensors": "Not reported",
            "Key Metrics": "Not reported",
            "Limitations": "Not reported",
            "Pages Cited": "Not reported",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert rows[1][1] == "Test\nPaper"

    def test_unicode_text(self):
        """Test that Unicode characters are preserved."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": "Test αβγ 日本語",
            "Authors": "Not reported",
            "Year": "Not reported",
            "Research Objective": "Test",
            "Methodology": "Test",
            "Dataset": "Not reported",
            "Hardware/Sensors": "Not reported",
            "Key Metrics": "Not reported",
            "Limitations": "Not reported",
            "Pages Cited": "Not reported",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert rows[1][1] == "Test αβγ 日本語"

    def test_multiple_metrics_formatting(self):
        """Test that multiple metrics are formatted correctly in CSV."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(value="Test", is_stated=True),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
            quantitative_findings=[
                MetricResult(metric_name="mAP@0.5", value="68.2%", dataset="KITTI"),
                MetricResult(metric_name="F1", value="0.84", dataset="COCO"),
            ],
        )
        matrix = [extraction_to_matrix_row(extraction)]
        csv_data = export_matrix_to_csv(matrix)

        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        assert "mAP@0.5: 68.2% (KITTI)" in rows[1][8]
        assert "F1: 0.84 (COCO)" in rows[1][8]

    def test_stable_column_order(self):
        """Test that column order is deterministic."""
        matrix = [{
            "Document": "paper.pdf",
            "Title": "Test",
            "Authors": "Not reported",
            "Year": "Not reported",
            "Research Objective": "Test",
            "Methodology": "Test",
            "Dataset": "Not reported",
            "Hardware/Sensors": "Not reported",
            "Key Metrics": "Not reported",
            "Limitations": "Not reported",
            "Pages Cited": "Not reported",
            "Extraction Timestamp": "2025-01-01T00:00:00+00:00",
        }]

        csv_data = export_matrix_to_csv(matrix)
        reader = csv.reader(io.StringIO(csv_data))
        header = list(reader)[0]

        assert header == MATRIX_COLUMNS


class TestBuildMatrixFromDb:
    def test_empty_database(self):
        """Test building matrix from empty database."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = None
        try:
            db = SQLiteManager(db_path)
            db.initialize_schema()

            matrix = build_matrix_from_db(db)
            assert matrix == []
        finally:
            if db:
                db.close()
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_database_with_extractions(self):
        """Test building matrix from database with extractions."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = None
        try:
            db = SQLiteManager(db_path)
            db.initialize_schema()

            # Register document and save extraction
            db.register_document("doc_001", "paper.pdf", 10, 50, 25000)

            extraction = PaperExtraction(
                paper_id="doc_001",
                title=ExtractedField(value="Test Paper", is_stated=True),
                research_objective=ExtractedField(value="Test objective", is_stated=True),
                proposed_methodology=ExtractedField(value="Test method", is_stated=True),
            )
            db.save_extraction(extraction)

            matrix = build_matrix_from_db(db)
            assert len(matrix) == 1
            assert matrix[0]["Document"] == "paper.pdf"
            assert matrix[0]["Title"] == "Test Paper"
        finally:
            if db:
                db.close()
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_export_db_to_csv(self):
        """Test one-step CSV export from database."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = None
        try:
            db = SQLiteManager(db_path)
            db.initialize_schema()

            db.register_document("doc_001", "paper.pdf", 10, 50, 25000)

            extraction = PaperExtraction(
                paper_id="doc_001",
                title=ExtractedField(value="Test Paper", is_stated=True),
                research_objective=ExtractedField(value="Test objective", is_stated=True),
                proposed_methodology=ExtractedField(value="Test method", is_stated=True),
            )
            db.save_extraction(extraction)

            csv_data = export_db_to_csv(db)

            reader = csv.reader(io.StringIO(csv_data))
            rows = list(reader)

            assert len(rows) == 2  # Header + 1 data row
            assert rows[1][0] == "paper.pdf"
        finally:
            if db:
                db.close()
            try:
                os.unlink(db_path)
            except OSError:
                pass


class TestMultipleProvenanceSpans:
    def test_multiple_provenance_in_matrix(self):
        """Test that multiple provenance spans are reflected in pages cited."""
        extraction = PaperExtraction(
            paper_id="test",
            title=ExtractedField(
                value="Test",
                is_stated=True,
                provenance=[
                    ProvenanceSpan(page_number=1),
                    ProvenanceSpan(page_number=3),
                    ProvenanceSpan(page_number=7),
                ]
            ),
            research_objective=ExtractedField(value="Test", is_stated=True),
            proposed_methodology=ExtractedField(value="Test", is_stated=True),
        )
        row = extraction_to_matrix_row(extraction)
        pages = row["Pages Cited"]
        
        assert "1" in pages
        assert "3" in pages
        assert "7" in pages
