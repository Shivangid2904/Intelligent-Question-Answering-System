"""
IntelliAsk Research — Services Package
"""

from src.services.qa_service import (
    PassageInfo,
    PreparedQAContext,
    QAResult,
    prepare_qa_context,
    run_qa_query,
    stream_qa_query,
)
from src.services.document_service import (
    DocumentInfo,
    DocumentService,
    DocumentUploadResult,
    QAEngineManager,
)
from src.services.matrix_service import MatrixService

__all__ = [
    "PassageInfo",
    "PreparedQAContext",
    "QAResult",
    "prepare_qa_context",
    "run_qa_query",
    "stream_qa_query",
    "DocumentInfo",
    "DocumentService",
    "DocumentUploadResult",
    "QAEngineManager",
    "MatrixService",
]


