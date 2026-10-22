"""
IntelliAsk Research — Literature Matrix Module

Provides transformation of PaperExtraction objects into tabular literature review
matrices and RFC 4180-compliant CSV export.
"""

from src.matrix.exporter import (
    build_matrix_from_db,
    build_matrix_from_extractions,
    export_db_to_csv,
    export_matrix_to_csv,
    extraction_to_matrix_row,
    MATRIX_COLUMNS,
)

__all__ = [
    "MATRIX_COLUMNS",
    "extraction_to_matrix_row",
    "build_matrix_from_extractions",
    "build_matrix_from_db",
    "export_matrix_to_csv",
    "export_db_to_csv",
]
