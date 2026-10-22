"""
IntelliAsk Research — API Configuration
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class APIConfig:
    """Configuration settings for the IntelliAsk FastAPI backend."""

    api_title: str = "IntelliAsk Research API"
    api_version: str = "0.1.0"
    api_description: str = (
        "REST and streaming API for scientific literature question answering, "
        "evidence sufficiency gating, and structured paper extraction."
    )
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    db_path: str = os.getenv("PAPERS_DB_PATH", "data/papers.db")
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))

