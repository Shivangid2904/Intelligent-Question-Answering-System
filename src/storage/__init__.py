"""
IntelliAsk Research — Storage Module

Provides SQLite persistence for documents, structured paper extractions,
quantitative metrics, and field-level provenance.
"""

from src.storage.db import SQLiteManager

__all__ = ["SQLiteManager"]
