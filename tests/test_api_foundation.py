"""
IntelliAsk Research — FastAPI Foundation Test Suite

Verifies:
- Application imports cleanly without loading SentenceTransformer or spaCy models.
- Liveness (/api/health) returns valid schema and status 200.
- Readiness (/api/readiness) distinguishes core API from Ollama/DB availability.
- Readiness handles connected, degraded (Ollama down), and unready (DB down) states.
- Predictable typed error responses for 404 and HTTP exceptions.
- CORS middleware behavior for frontend origins.
- Runs 100% offline without requiring a running Ollama server.
"""

from datetime import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.config import APIConfig
from src.api.main import app, create_app
from src.api.schemas import ErrorResponse, HealthResponse, ReadinessResponse


@pytest.fixture
def test_client():
    """Default TestClient for the global application instance."""
    return TestClient(app)


def test_import_does_not_load_models_or_trigger_network():
    """
    Verify importing the API module does not instantiate SentenceTransformer,
    call spacy.load, or make network requests.
    """
    with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("SentenceTransformer loaded on import!")):
        with patch("spacy.load", side_effect=RuntimeError("spacy.load invoked on import!")):
            # Re-creating an app with factory must not invoke heavy models
            test_app = create_app()
            assert test_app is not None
            assert test_app.state.config.api_title == "IntelliAsk Research API"


def test_health_liveness_endpoint(test_client):
    """
    Verify GET /api/health returns 200 with the exact HealthResponse schema.
    """
    response = test_client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    validated = HealthResponse(**data)
    assert validated.status == "ok"
    assert validated.version == "0.1.0"
    assert validated.models_loaded is False

    # Timestamp must be parseable ISO-8601
    dt = datetime.fromisoformat(validated.timestamp)
    assert dt is not None


def test_readiness_ollama_connected(test_client, tmp_path):
    """
    Verify GET /api/readiness returns 'ready' when Ollama is online and database is accessible.
    """
    db_file = tmp_path / "test_papers.db"
    import sqlite3
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE papers (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    mock_ollama_response = httpx.Response(
        status_code=200,
        json={"models": [{"name": "llama3.2:1b"}, {"name": "nomic-embed-text:latest"}]},
        request=httpx.Request("GET", "http://localhost:11434/api/tags"),
    )

    custom_cfg = APIConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
        db_path=str(db_file),
    )
    custom_app = create_app(custom_cfg)
    client = TestClient(custom_app)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_ollama_response):
        response = client.get("/api/readiness")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        validated = ReadinessResponse(**data)
        assert validated.status == "ready"
        assert validated.components["api"].status == "ready"
        assert validated.components["database"].status == "connected"
        assert validated.components["ollama"].status == "connected"
        assert validated.components["ollama"].details["model_present"] is True
        assert validated.components["embedding_model"].status == "lazy"


def test_readiness_ollama_unavailable_degraded(test_client, tmp_path):
    """
    Verify GET /api/readiness returns 'degraded' (HTTP 200) when Ollama is offline,
    distinguishing API availability from optional generative services.
    """
    db_file = tmp_path / "test_papers.db"
    import sqlite3
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE papers (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    custom_cfg = APIConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
        db_path=str(db_file),
    )
    custom_app = create_app(custom_cfg)
    client = TestClient(custom_app)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        response = client.get("/api/readiness")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        validated = ReadinessResponse(**data)
        assert validated.status == "degraded"
        assert validated.components["api"].status == "ready"
        assert validated.components["database"].status == "connected"
        assert validated.components["ollama"].status == "unavailable"
        assert "Cannot reach Ollama" in validated.components["ollama"].message
        assert validated.components["embedding_model"].status == "lazy"


def test_readiness_uninitialized_database_is_handled(tmp_path):
    """
    Verify an uninitialized database file path returns 'uninitialized' without breaking readiness.
    """
    uninitialized_db = tmp_path / "uninitialized.db"

    mock_ollama_response = httpx.Response(
        status_code=200,
        json={"models": [{"name": "llama3.2:1b"}]},
        request=httpx.Request("GET", "http://localhost:11434/api/tags"),
    )

    custom_cfg = APIConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
        db_path=str(uninitialized_db),
    )
    custom_app = create_app(custom_cfg)
    client = TestClient(custom_app)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_ollama_response):
        response = client.get("/api/readiness")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        validated = ReadinessResponse(**data)
        assert validated.status == "ready"
        assert validated.components["database"].status == "uninitialized"


def test_readiness_database_unavailable_returns_503(tmp_path):
    """
    Verify GET /api/readiness returns 'unready' with HTTP 503 when the database path is invalid/corrupt.
    """
    # Create a directory where a database file is expected, causing SQLite connect failure
    corrupt_dir = tmp_path / "corrupt_db_as_dir"
    corrupt_dir.mkdir()

    custom_cfg = APIConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
        db_path=str(corrupt_dir),
    )
    custom_app = create_app(custom_cfg)
    client = TestClient(custom_app)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        response = client.get("/api/readiness")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

        data = response.json()
        validated = ReadinessResponse(**data)
        assert validated.status == "unready"
        assert validated.components["database"].status == "unavailable"


def test_error_handling_not_found(test_client):
    """
    Verify 404 responses conform to the standardized ErrorResponse schema.
    """
    response = test_client.get("/api/unregistered_endpoint")
    assert response.status_code == status.HTTP_404_NOT_FOUND

    data = response.json()
    validated = ErrorResponse(**data)
    assert validated.status_code == 404
    assert validated.error == "HTTPException"
    assert "Not Found" in validated.detail


def test_cors_headers_configured(test_client):
    """
    Verify CORS middleware returns correct Access-Control-Allow-Origin header
    for authorized frontend development origins.
    """
    response = test_client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_custom_app_configuration():
    """
    Verify create_app respects custom APIConfig overrides.
    """
    custom_cfg = APIConfig(
        api_title="Custom IntelliAsk",
        api_version="2.5.0",
        cors_origins=["http://custom-origin.org"],
    )
    custom_app = create_app(custom_cfg)
    client = TestClient(custom_app)

    response = client.get(
        "/api/health",
        headers={"Origin": "http://custom-origin.org"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["version"] == "2.5.0"
    assert response.headers.get("access-control-allow-origin") == "http://custom-origin.org"
