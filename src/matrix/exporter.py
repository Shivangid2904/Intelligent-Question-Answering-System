"""
IntelliAsk Research — Literature Matrix Export

Provides transformation of PaperExtraction objects into tabular literature review
matrices and RFC 4180-compliant CSV export.
"""

import csv
import io
from datetime import datetime, timezone
from typing import List, Optional

from src.extraction.schemas import PaperExtraction
from src.storage.db import SQLiteManager


# Matrix column definitions (deterministic order)
MATRIX_COLUMNS = [
    "Document",
    "Title",
    "Authors",
    "Year",
    "Research Objective",
    "Methodology",
    "Dataset",
    "Hardware/Sensors",
    "Key Metrics",
    "Limitations",
    "Pages Cited",
    "Extraction Timestamp",
]


def _format_field_value(value: Optional[str], is_stated: bool = True) -> str:
    """
    Format an ExtractedField value for display in the matrix.

    Args:
        value: The field value (may be None).
        is_stated: Whether the value was explicitly stated.

    Returns:
        Formatted string for display.
    """
    if value is None:
        return "Not reported"
    if not is_stated:
        return f"{value} (inferred)"
    return value


def _format_metrics(extraction: PaperExtraction) -> str:
    """
    Format quantitative findings into a semicolon-separated string.

    Args:
        extraction: PaperExtraction object.

    Returns:
        Semicolon-separated metrics string, e.g., "mAP@0.5: 68.2%; F1: 0.84"
    """
    if not extraction.quantitative_findings:
        return "Not reported"

    metric_strings = []
    for metric in extraction.quantitative_findings:
        parts = [f"{metric.metric_name}: {metric.value}"]
        if metric.dataset:
            parts.append(f"({metric.dataset})")
        metric_strings.append(" ".join(parts))

    return "; ".join(metric_strings)


def _format_pages_cited(extraction: PaperExtraction) -> str:
    """
    Extract unique page numbers from all provenance spans.

    Args:
        extraction: PaperExtraction object.

    Returns:
        Comma-separated page numbers, e.g., "1, 3, 5, 7"
    """
    pages = set()
    
    # Collect pages from all fields
    for field in [
        extraction.title,
        extraction.authors,
        extraction.year,
        extraction.research_objective,
        extraction.proposed_methodology,
        extraction.dataset_used,
        extraction.hardware_or_sensors,
        extraction.acknowledged_limitations,
    ]:
        for span in field.provenance:
            pages.add(span.page_number)
    
    # Collect pages from metrics
    for metric in extraction.quantitative_findings:
        if metric.provenance:
            pages.add(metric.provenance.page_number)
    
    if not pages:
        return "Not reported"

    return ", ".join(str(p) for p in sorted(pages))


def extraction_to_matrix_row(
    extraction: PaperExtraction,
    filename: Optional[str] = None,
) -> dict:
    """
    Transform a PaperExtraction into a matrix row dict.

    Args:
        extraction: PaperExtraction object.
        filename: Optional source filename (if available from document metadata).

    Returns:
        Dict with keys matching MATRIX_COLUMNS.
    """
    return {
        "Document": filename or extraction.paper_id,
        "Title": _format_field_value(extraction.title.value, extraction.title.is_stated),
        "Authors": _format_field_value(extraction.authors.value, extraction.authors.is_stated),
        "Year": _format_field_value(extraction.year.value, extraction.year.is_stated),
        "Research Objective": _format_field_value(
            extraction.research_objective.value, extraction.research_objective.is_stated
        ),
        "Methodology": _format_field_value(
            extraction.proposed_methodology.value, extraction.proposed_methodology.is_stated
        ),
        "Dataset": _format_field_value(
            extraction.dataset_used.value, extraction.dataset_used.is_stated
        ),
        "Hardware/Sensors": _format_field_value(
            extraction.hardware_or_sensors.value, extraction.hardware_or_sensors.is_stated
        ),
        "Key Metrics": _format_metrics(extraction),
        "Limitations": _format_field_value(
            extraction.acknowledged_limitations.value, extraction.acknowledged_limitations.is_stated
        ),
        "Pages Cited": _format_pages_cited(extraction),
        "Extraction Timestamp": extraction.extraction_timestamp,
    }


def build_matrix_from_extractions(
    extractions: List[PaperExtraction],
    db: Optional[SQLiteManager] = None,
) -> List[dict]:
    """
    Build a complete literature matrix from a list of extractions.

    If a SQLiteManager is provided, it will be used to look up filenames
    for each document. Otherwise, paper_id is used as the document identifier.

    Args:
        extractions: List of PaperExtraction objects.
        db: Optional SQLiteManager for filename lookup.

    Returns:
        List of matrix row dicts.
    """
    matrix = []
    
    for extraction in extractions:
        filename = None
        if db:
            doc = db.get_document(extraction.paper_id)
            if doc:
                filename = doc["filename"]
        
        row = extraction_to_matrix_row(extraction, filename)
        matrix.append(row)
    
    return matrix


def export_matrix_to_csv(matrix: List[dict]) -> str:
    """
    Export a literature matrix to RFC 4180-compliant CSV string.

    Handles:
    - UTF-8 encoding
    - Commas, quotes, and newlines in values
    - Proper escaping according to RFC 4180
    - Consistent newline handling (\r\n)

    Args:
        matrix: List of matrix row dicts with keys matching MATRIX_COLUMNS.

    Returns:
        CSV string ready for download or file writing.
    """
    if not matrix:
        # Return header row only for empty matrix
        output = io.StringIO()
        writer = csv.writer(output, dialect="excel", lineterminator="\r\n")
        writer.writerow(MATRIX_COLUMNS)
        return output.getvalue()

    output = io.StringIO()
    writer = csv.writer(output, dialect="excel", lineterminator="\r\n")
    
    # Write header
    writer.writerow(MATRIX_COLUMNS)
    
    # Write data rows
    for row in matrix:
        # Ensure all columns are present in correct order
        writer.writerow([row.get(col, "") for col in MATRIX_COLUMNS])
    
    return output.getvalue()


def build_matrix_from_db(db: SQLiteManager) -> List[dict]:
    """
    Build a literature matrix directly from the SQLite database.

    This is a convenience function that retrieves all extractions from the
    database and builds the matrix in one step.

    Args:
        db: SQLiteManager instance.

    Returns:
        List of matrix row dicts.
    """
    extractions = db.get_all_extractions()
    return build_matrix_from_extractions(extractions, db)


def export_db_to_csv(db: SQLiteManager) -> str:
    """
    Export all extractions from the database to CSV in one step.

    Args:
        db: SQLiteManager instance.

    Returns:
        CSV string ready for download or file writing.
    """
    matrix = build_matrix_from_db(db)
    return export_matrix_to_csv(matrix)
