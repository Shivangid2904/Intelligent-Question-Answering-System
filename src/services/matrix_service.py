"""
IntelliAsk Research — Matrix Service

Encapsulates literature matrix retrieval and RFC 4180 CSV export
leveraging SQLite-persisted extractions and the matrix exporter module.
"""

from typing import Any, Dict, List

from src.matrix.exporter import MATRIX_COLUMNS, build_matrix_from_db, export_db_to_csv
from src.storage.db import SQLiteManager


class MatrixService:
    """
    Service layer providing literature matrix generation and CSV export
    from registered documents and extractions.
    """

    def __init__(self, db_manager: SQLiteManager):
        """
        Initialize the MatrixService with a SQLiteManager instance.

        Args:
            db_manager: Initialized SQLiteManager connected to the database.
        """
        self.db = db_manager

    def get_columns(self) -> List[str]:
        """
        Get the 12 canonical matrix column names in deterministic order.

        Returns:
            List of column name strings.
        """
        return list(MATRIX_COLUMNS)

    def get_matrix(self) -> Dict[str, Any]:
        """
        Retrieve the literature review matrix rows and columns.

        Returns:
            Dict containing 'columns', 'rows', and 'total_rows'.
        """
        rows = build_matrix_from_db(self.db)
        return {
            "columns": list(MATRIX_COLUMNS),
            "rows": rows,
            "total_rows": len(rows),
        }

    def export_csv(self) -> str:
        """
        Export the literature matrix as an RFC 4180-compliant CSV string.

        Returns:
            CSV string with Windows CRLF (\r\n) line endings and header.
        """
        return export_db_to_csv(self.db)
