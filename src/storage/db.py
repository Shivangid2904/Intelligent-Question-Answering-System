"""
IntelliAsk Research — SQLite Persistence Manager

Provides canonical SQLite storage for documents, structured paper extractions,
quantitative metrics, and field-level provenance with foreign-key cascades
and idempotent ingestion support.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.extraction.schemas import (
    ExtractedField,
    MetricResult,
    PaperExtraction,
    ProvenanceSpan,
)

# Database schema version for migration tracking
SCHEMA_VERSION = 1


class SQLiteManager:
    """
    Manages SQLite database connection, schema initialization, and CRUD operations
    for document ingestion and structured paper extraction persistence.
    """

    def __init__(self, db_path: str = "data/papers.db"):
        """
        Initialize the SQLite manager.

        Args:
            db_path: Path to the SQLite database file. Directory will be created if needed.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """
        Establish a database connection with foreign-key enforcement enabled.

        Returns:
            SQLite connection object.
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.row_factory = sqlite3.Row  # Enable dict-like row access
        return self.conn

    def close(self):
        """Close the database connection if open."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic close."""
        self.close()

    def initialize_schema(self) -> None:
        """
        Create all database tables and indexes if they don't exist.
        This is idempotent and safe to call multiple times.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # 1. Documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                ingestion_timestamp TEXT NOT NULL
            )
        """
        )

        # 2. Paper extractions table (enhanced to preserve is_stated and confidence)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_extractions (
                extraction_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE REFERENCES documents(document_id) ON DELETE CASCADE,
                title TEXT,
                title_is_stated INTEGER DEFAULT 1,
                title_confidence REAL DEFAULT 1.0,
                authors TEXT,
                authors_is_stated INTEGER DEFAULT 1,
                authors_confidence REAL DEFAULT 1.0,
                year TEXT,
                year_is_stated INTEGER DEFAULT 1,
                year_confidence REAL DEFAULT 1.0,
                research_objective TEXT,
                research_objective_is_stated INTEGER DEFAULT 1,
                research_objective_confidence REAL DEFAULT 1.0,
                methodology TEXT,
                methodology_is_stated INTEGER DEFAULT 1,
                methodology_confidence REAL DEFAULT 1.0,
                dataset TEXT,
                dataset_is_stated INTEGER DEFAULT 1,
                dataset_confidence REAL DEFAULT 1.0,
                hardware_sensors TEXT,
                hardware_sensors_is_stated INTEGER DEFAULT 1,
                hardware_sensors_confidence REAL DEFAULT 1.0,
                limitations TEXT,
                limitations_is_stated INTEGER DEFAULT 1,
                limitations_confidence REAL DEFAULT 1.0,
                raw_json TEXT NOT NULL,
                extracted_at TEXT NOT NULL
            )
        """
        )

        # 3. Extraction metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                dataset TEXT,
                page_number INTEGER
            )
        """
        )

        # 5. Field provenance table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS field_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                field_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                verbatim_quote TEXT
            )
        """
        )

        # 6. Query history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT,
                answer_source TEXT NOT NULL,
                has_supporting_evidence INTEGER NOT NULL,
                best_source TEXT,
                best_score REAL,
                citations_json TEXT,
                timestamp TEXT NOT NULL
            )
        """
        )

        # 7. Indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_provenance ON field_provenance(document_id, field_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_metrics ON extraction_metrics(document_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON query_history(timestamp DESC)"
        )

        # 8. Schema version tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """
        )

        # Record current schema version if not present
        cursor.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
        )

        conn.commit()

    def get_schema_version(self) -> int:
        """
        Get the current schema version from the database.

        Returns:
            Schema version integer, or 0 if not initialized.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        return row["version"] if row else 0

    def register_document(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        char_count: int,
    ) -> bool:
        """
        Register a document in the database with idempotent upsert behavior.

        If the document_id already exists, the record is updated with new metadata.
        This supports re-ingestion without creating duplicates.

        Args:
            document_id: Stable document identifier (from DocumentProcessor).
            filename: Original PDF filename.
            page_count: Number of physical pages in the PDF.
            chunk_count: Number of chunks generated.
            char_count: Total character count.

        Returns:
            True if document was inserted, False if it was updated (already existed).
        """
        conn = self.connect()
        cursor = conn.cursor()

        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if document exists
        cursor.execute("SELECT document_id FROM documents WHERE document_id = ?", (document_id,))
        exists = cursor.fetchone() is not None

        if exists:
            # Update existing document
            cursor.execute(
                """
                UPDATE documents
                SET filename = ?, page_count = ?, chunk_count = ?, char_count = ?, ingestion_timestamp = ?
                WHERE document_id = ?
                """,
                (filename, page_count, chunk_count, char_count, timestamp, document_id),
            )
            conn.commit()
            return False
        else:
            # Insert new document
            cursor.execute(
                """
                INSERT INTO documents (document_id, filename, page_count, chunk_count, char_count, ingestion_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, filename, page_count, chunk_count, char_count, timestamp),
            )
            conn.commit()
            return True

    def get_document(self, document_id: str) -> Optional[dict]:
        """
        Retrieve document metadata by ID.

        Args:
            document_id: Document identifier.

        Returns:
            Document metadata dict or None if not found.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def save_extraction(self, extraction: PaperExtraction) -> None:
        """
        Save a PaperExtraction to the database with idempotent upsert behavior.

        This operation:
        1. Saves the main extraction record (upserts by document_id)
        2. Replaces all associated metrics (delete + insert)
        3. Replaces all associated provenance (delete + insert)
        4. Stores the full JSON for round-trip preservation

        All operations run in a single transaction. On failure, everything rolls back.

        Args:
            extraction: Validated PaperExtraction object.

        Raises:
            RuntimeError: If document_id is not registered in documents table.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Verify document exists
        doc = self.get_document(extraction.paper_id)
        if not doc:
            raise RuntimeError(
                f"Cannot save extraction: document '{extraction.paper_id}' is not registered. "
                f"Call register_document() first."
            )

        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")

            # Helper to convert bool to int for SQLite
            def bool_to_int(b: bool) -> int:
                return 1 if b else 0

            # 1. Upsert main extraction record
            extraction_id = f"{extraction.paper_id}_extraction"
            cursor.execute(
                """
                INSERT INTO paper_extractions (
                    extraction_id, document_id,
                    title, title_is_stated, title_confidence,
                    authors, authors_is_stated, authors_confidence,
                    year, year_is_stated, year_confidence,
                    research_objective, research_objective_is_stated, research_objective_confidence,
                    methodology, methodology_is_stated, methodology_confidence,
                    dataset, dataset_is_stated, dataset_confidence,
                    hardware_sensors, hardware_sensors_is_stated, hardware_sensors_confidence,
                    limitations, limitations_is_stated, limitations_confidence,
                    raw_json, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    title_is_stated = excluded.title_is_stated,
                    title_confidence = excluded.title_confidence,
                    authors = excluded.authors,
                    authors_is_stated = excluded.authors_is_stated,
                    authors_confidence = excluded.authors_confidence,
                    year = excluded.year,
                    year_is_stated = excluded.year_is_stated,
                    year_confidence = excluded.year_confidence,
                    research_objective = excluded.research_objective,
                    research_objective_is_stated = excluded.research_objective_is_stated,
                    research_objective_confidence = excluded.research_objective_confidence,
                    methodology = excluded.methodology,
                    methodology_is_stated = excluded.methodology_is_stated,
                    methodology_confidence = excluded.methodology_confidence,
                    dataset = excluded.dataset,
                    dataset_is_stated = excluded.dataset_is_stated,
                    dataset_confidence = excluded.dataset_confidence,
                    hardware_sensors = excluded.hardware_sensors,
                    hardware_sensors_is_stated = excluded.hardware_sensors_is_stated,
                    hardware_sensors_confidence = excluded.hardware_sensors_confidence,
                    limitations = excluded.limitations,
                    limitations_is_stated = excluded.limitations_is_stated,
                    limitations_confidence = excluded.limitations_confidence,
                    raw_json = excluded.raw_json,
                    extracted_at = excluded.extracted_at
                """,
                (
                    extraction_id,
                    extraction.paper_id,
                    extraction.title.value,
                    bool_to_int(extraction.title.is_stated),
                    extraction.title.confidence,
                    extraction.authors.value,
                    bool_to_int(extraction.authors.is_stated),
                    extraction.authors.confidence,
                    extraction.year.value,
                    bool_to_int(extraction.year.is_stated),
                    extraction.year.confidence,
                    extraction.research_objective.value,
                    bool_to_int(extraction.research_objective.is_stated),
                    extraction.research_objective.confidence,
                    extraction.proposed_methodology.value,
                    bool_to_int(extraction.proposed_methodology.is_stated),
                    extraction.proposed_methodology.confidence,
                    extraction.dataset_used.value,
                    bool_to_int(extraction.dataset_used.is_stated),
                    extraction.dataset_used.confidence,
                    extraction.hardware_or_sensors.value,
                    bool_to_int(extraction.hardware_or_sensors.is_stated),
                    extraction.hardware_or_sensors.confidence,
                    extraction.acknowledged_limitations.value,
                    bool_to_int(extraction.acknowledged_limitations.is_stated),
                    extraction.acknowledged_limitations.confidence,
                    extraction.model_dump_json(),
                    extraction.extraction_timestamp,
                ),
            )

            # 2. Delete existing metrics for this document
            cursor.execute(
                "DELETE FROM extraction_metrics WHERE document_id = ?", (extraction.paper_id,)
            )

            # 3. Insert new metrics
            for metric in extraction.quantitative_findings:
                page_num = metric.provenance.page_number if metric.provenance else None
                cursor.execute(
                    """
                    INSERT INTO extraction_metrics (document_id, metric_name, metric_value, dataset, page_number)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (extraction.paper_id, metric.metric_name, metric.value, metric.dataset, page_num),
                )

            # 4. Delete existing provenance for this document
            cursor.execute(
                "DELETE FROM field_provenance WHERE document_id = ?", (extraction.paper_id,)
            )

            # 5. Insert new provenance for each field
            field_mappings = {
                "title": extraction.title,
                "authors": extraction.authors,
                "year": extraction.year,
                "research_objective": extraction.research_objective,
                "methodology": extraction.proposed_methodology,
                "dataset": extraction.dataset_used,
                "hardware_sensors": extraction.hardware_or_sensors,
                "limitations": extraction.acknowledged_limitations,
            }

            for field_name, field in field_mappings.items():
                for span in field.provenance:
                    cursor.execute(
                        """
                        INSERT INTO field_provenance (document_id, field_name, page_number, verbatim_quote)
                        VALUES (?, ?, ?, ?)
                        """,
                        (extraction.paper_id, field_name, span.page_number, span.verbatim_quote),
                    )

            # Commit transaction
            conn.commit()

        except Exception as e:
            # Rollback on any error
            conn.rollback()
            raise RuntimeError(f"Failed to save extraction: {str(e)}") from e

    def get_extraction(self, document_id: str) -> Optional[PaperExtraction]:
        """
        Retrieve a PaperExtraction by document ID.

        Args:
            document_id: Document identifier.

        Returns:
            PaperExtraction object or None if not found.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Get main extraction record
        cursor.execute(
            """
            SELECT raw_json FROM paper_extractions WHERE document_id = ?
            """,
            (document_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize from JSON
        try:
            extraction_dict = json.loads(row["raw_json"])
            return PaperExtraction.model_validate(extraction_dict)
        except (json.JSONDecodeError, Exception) as e:
            raise RuntimeError(f"Failed to deserialize extraction: {str(e)}") from e

    def get_all_documents(self) -> List[dict]:
        """
        Retrieve all registered documents.

        Returns:
            List of document metadata dicts.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY ingestion_timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_all_extractions(self) -> List[PaperExtraction]:
        """
        Retrieve all saved extractions.

        Returns:
            List of PaperExtraction objects.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT raw_json FROM paper_extractions ORDER BY extracted_at DESC")

        extractions = []
        for row in cursor.fetchall():
            try:
                extraction_dict = json.loads(row["raw_json"])
                extractions.append(PaperExtraction.model_validate(extraction_dict))
            except (json.JSONDecodeError, Exception):
                # Skip corrupted records
                continue

        return extractions

    def get_metrics_for_document(self, document_id: str) -> List[dict]:
        """
        Retrieve all metrics for a document.

        Args:
            document_id: Document identifier.

        Returns:
            List of metric dicts with metric_name, value, dataset, page_number.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT metric_name, metric_value, dataset, page_number
            FROM extraction_metrics
            WHERE document_id = ?
            ORDER BY metric_id
            """,
            (document_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_provenance_for_field(self, document_id: str, field_name: str) -> List[dict]:
        """
        Retrieve provenance spans for a specific field.

        Args:
            document_id: Document identifier.
            field_name: Field name (e.g., 'title', 'methodology').

        Returns:
            List of provenance dicts with page_number and verbatim_quote.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT page_number, verbatim_quote
            FROM field_provenance
            WHERE document_id = ? AND field_name = ?
            ORDER BY id
            """,
            (document_id, field_name),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated records (extraction, metrics, provenance).

        This leverages foreign-key ON DELETE CASCADE for automatic cleanup.

        Args:
            document_id: Document identifier.

        Returns:
            True if document was deleted, False if not found.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Check if document exists
        cursor.execute("SELECT document_id FROM documents WHERE document_id = ?", (document_id,))
        if not cursor.fetchone():
            return False

        # Delete document (cascade will handle extraction, metrics, provenance)
        cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.commit()
        return True

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document is registered.

        Args:
            document_id: Document identifier.

        Returns:
            True if document exists, False otherwise.
        """
        return self.get_document(document_id) is not None

    def extraction_exists(self, document_id: str) -> bool:
        """
        Check if an extraction exists for a document.

        Args:
            document_id: Document identifier.

        Returns:
            True if extraction exists, False otherwise.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT extraction_id FROM paper_extractions WHERE document_id = ?", (document_id,)
        )
        return cursor.fetchone() is not None

    def get_all_provenance_for_document(self, document_id: str) -> List[dict]:
        """
        Retrieve all field provenance records for a document.

        Args:
            document_id: Document identifier.

        Returns:
            List of dicts with field_name, page_number, and verbatim_quote.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT field_name, page_number, verbatim_quote
            FROM field_provenance
            WHERE document_id = ?
            ORDER BY field_name, id
            """,
            (document_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def record_query_history(
        self,
        query: str,
        answer: Optional[str],
        answer_source: str,
        has_supporting_evidence: bool,
        best_source: Optional[str] = None,
        best_score: Optional[float] = None,
        citations: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """
        Record a query and answer outcome in query history.

        Args:
            query: The user research query string.
            answer: Answer text or abstention message.
            answer_source: 'ollama' | 'abstention' | 'error'.
            has_supporting_evidence: True if evidence sufficiency gate passed.
            best_source: Top retrieval document source.
            best_score: Dense retrieval score.
            citations: List of citation strings.
            timestamp: ISO 8601 UTC timestamp (defaults to current UTC time).

        Returns:
            The newly inserted history_id integer.
        """
        conn = self.connect()
        cursor = conn.cursor()
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations) if citations is not None else None
        has_ev = 1 if has_supporting_evidence else 0

        cursor.execute(
            """
            INSERT INTO query_history (
                query, answer, answer_source, has_supporting_evidence,
                best_source, best_score, citations_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (query, answer, answer_source, has_ev, best_source, best_score, citations_json, ts),
        )
        conn.commit()
        return cursor.lastrowid

    def get_query_history(self, limit: int = 100) -> List[dict]:
        """
        Retrieve query history ordered by timestamp descending.

        Args:
            limit: Maximum number of history entries to return.

        Returns:
            List of history dicts.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT history_id, query, answer, answer_source, has_supporting_evidence,
                   best_source, best_score, citations_json, timestamp
            FROM query_history
            ORDER BY timestamp DESC, history_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        records = []
        for row in cursor.fetchall():
            d = dict(row)
            d["has_supporting_evidence"] = bool(d["has_supporting_evidence"])
            if d.get("citations_json"):
                try:
                    d["citations"] = json.loads(d["citations_json"])
                except Exception:
                    d["citations"] = []
            else:
                d["citations"] = []
            records.append(d)
        return records

    def clear_query_history(self) -> int:
        """
        Safely delete all records in the query_history table.
        Does NOT delete any documents, extractions, provenance, or metrics.

        Returns:
            The number of history records deleted.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM query_history")
        count = cursor.fetchone()["cnt"]
        cursor.execute("DELETE FROM query_history")
        conn.commit()
        return count

    def migrate_to_version_2(self) -> int:
        """
        Migrate database schema to version 2.
        Ensures query_history table and index exist, and records version 2 in schema_version.
        Preserves all existing data.

        Returns:
            Current schema version (2).
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT,
                answer_source TEXT NOT NULL,
                has_supporting_evidence INTEGER NOT NULL,
                best_source TEXT,
                best_score REAL,
                citations_json TEXT,
                timestamp TEXT NOT NULL
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON query_history(timestamp DESC)"
        )
        cursor.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (2, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return 2

