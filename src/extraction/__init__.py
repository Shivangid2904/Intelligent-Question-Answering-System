"""
IntelliAsk Research — Structured Extraction Module

Provides schemas and extraction logic for structured paper metadata extraction.
"""

from src.extraction.schemas import (
    ExtractedField,
    MetricResult,
    PaperExtraction,
    ProvenanceSpan,
)

__all__ = [
    "PaperExtraction",
    "ExtractedField",
    "ProvenanceSpan",
    "MetricResult",
]
