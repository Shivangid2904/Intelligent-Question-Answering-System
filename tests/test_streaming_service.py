"""
IntelliAsk Research — Phase 2D Server-Sent Events (SSE) Streaming Query Tests

Covers:
- Valid SSE content type ('text/event-stream') and wire framing ('event: ...\\ndata: ...\\n\\n')
- Progressive token emission with multiple incremental chunks
- Metadata event carrying query_id, verdict, citations, and evidence sufficiency
- Abstention path emitting metadata, final abstention, and done with ZERO token events and ZERO LLM calls
- Generation failure emitting a safe 'error' event and 'done' event without tracebacks
- Input validation consistency with non-streaming query (422 on empty, blank, or >2000 chars)
- Missing engine returns HTTP 503
- Non-streaming endpoint (POST /api/query) continues functioning identically
- 100% offline execution with mock generators and no live Ollama dependency
"""

from __future__ import annotations

import json
from typing import Any, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import (
    _engine_holder,
    _qa_engine_manager,
    app,
    create_app,
)
from src.api.schemas import (
    QueryRequest,
    StreamDonePayload,
    StreamErrorPayload,
    StreamFinalPayload,
    StreamMetadataPayload,
    StreamTokenPayload,
)
from src.services.qa_service import (
    PassageInfo,
    PreparedQAContext,
    prepare_qa_context,
    run_qa_query,
    stream_qa_query,
)


# ── Helpers & Fixtures ───────────────────────────────────────────────────────

class _MockChunk:
    """Mock Chunk instance carrying required dense retrieval attributes."""

    def __init__(
        self,
        text: str,
        source_filename: str = "lidar_paper.pdf",
        page_number: int = 3,
        dense_score: Optional[float] = 0.65,
        score: float = 0.02,
        chunk_id: str = "chunk_001",
    ):
        self.text = text
        self.source_filename = source_filename
        self.source = source_filename
        self.page_number = page_number
        self.dense_score = dense_score
        self.score = score
        self.chunk_id = chunk_id


def _make_engine(chunks: Optional[List[Any]] = None) -> MagicMock:
    """Return a mock QAEngine whose search_hybrid returns the given chunks."""
    engine = MagicMock()
    engine.search_hybrid.return_value = chunks or []
    return engine


def parse_sse_events(raw_text: str) -> List[tuple[str, dict]]:
    """
    Parse a raw SSE string into a list of (event_type, payload_dict) tuples.
    Frames are separated by double newlines ('\\n\\n').
    """
    events = []
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    for block in blocks:
        event_name = None
        data_str = None
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
        if event_name and data_str:
            try:
                payload = json.loads(data_str)
            except Exception:
                payload = {"raw": data_str}
            events.append((event_name, payload))
    return events


@pytest.fixture(autouse=True)
def reset_engine_states():
    """Ensure engine holders are completely clean before and after every test."""
    _engine_holder.clear()
    _qa_engine_manager.clear()
    yield
    _engine_holder.clear()
    _qa_engine_manager.clear()


@pytest.fixture
def api_client():
    return TestClient(app)


# ── Service-Level Unit Tests: stream_qa_query ────────────────────────────────

class TestStreamQAServiceUnit:
    """Unit tests for stream_qa_query generator logic."""

    def test_stream_sufficient_evidence_emits_expected_sequence(self):
        """
        When evidence is sufficient:
        Sequence must be: metadata -> tokens (in order) -> final -> done.
        """
        chunk = _MockChunk(
            text="The Ouster OS1-32 lidar sensor is used for autonomous ground vehicle navigation.",
            dense_score=0.72,
            page_number=4,
            source_filename="ouster.pdf",
        )
        engine = _make_engine([chunk])

        stub_tokens = ["The", " sensor", " is", " Ouster", " OS1-32."]

        def _stub_gen(q, p, model):
            yield from stub_tokens

        events = list(
            stream_qa_query(
                query="What sensor is used for autonomous navigation?",
                engine=engine,
                answer_generator=_stub_gen,
                query_id="q_unit_001",
            )
        )

        event_names = [e[0] for e in events]
        assert event_names[0] == "metadata"
        assert event_names[1:6] == ["token"] * len(stub_tokens)
        assert event_names[6] == "final"
        assert event_names[7] == "done"

        # Verify metadata payload
        meta = events[0][1]
        assert meta["query_id"] == "q_unit_001"
        assert meta["has_supporting_evidence"] is True
        assert meta["verdict"]["should_abstain"] is False
        assert meta["best_source"] == "ouster.pdf"

        # Verify token payloads
        emitted_tokens = [e[1]["token"] for e in events[1:6]]
        assert emitted_tokens == stub_tokens

        # Verify final payload
        final = events[6][1]
        assert final["answer"] == "".join(stub_tokens)
        assert final["answer_source"] == "ollama"
        assert final["has_supporting_evidence"] is True
        assert final["completion_status"] == "complete"
        assert len(final["supporting_passages"]) >= 1
        assert any("ouster.pdf" in c for c in final["citations"])

        # Verify done payload
        done = events[7][1]
        assert done["done"] is True

    def test_stream_abstention_no_tokens_no_llm_call(self):
        """
        When evidence is insufficient:
        Sequence must be: metadata -> final (abstention) -> done.
        NO 'token' events and NO answer_generator invocations are allowed.
        """
        chunk = _MockChunk(
            text="Unrelated cooking recipe for tomato soup.",
            dense_score=0.10,
        )
        engine = _make_engine([chunk])

        llm_called = []

        def _bad_gen(q, p, model):
            llm_called.append(True)
            yield "Should not be called!"

        events = list(
            stream_qa_query(
                query="What is the lidar sensor model?",
                engine=engine,
                answer_generator=_bad_gen,
            )
        )

        assert len(llm_called) == 0, "LLM was called during abstention!"

        event_names = [e[0] for e in events]
        assert "token" not in event_names
        assert event_names == ["metadata", "final", "done"]

        meta = events[0][1]
        assert meta["has_supporting_evidence"] is False
        assert meta["verdict"]["should_abstain"] is True

        final = events[1][1]
        assert final["has_supporting_evidence"] is False
        assert final["answer_source"] == "abstention"
        assert final["completion_status"] == "abstention"
        assert "Insufficient evidence" in final["answer"]

        done = events[2][1]
        assert done["done"] is True

    def test_stream_llm_generation_error_emits_error_and_done(self):
        """
        If answer_generator fails midway through iteration:
        Emits: metadata -> tokens prior to error -> error -> done.
        """
        chunk = _MockChunk(
            text="Laser rangefinder specifications and performance metrics.",
            dense_score=0.68,
        )
        engine = _make_engine([chunk])

        def _failing_gen(q, p, model):
            yield "First token."
            raise ConnectionError("Ollama network connection reset by peer")

        events = list(
            stream_qa_query(
                query="What are the laser specifications?",
                engine=engine,
                answer_generator=_failing_gen,
            )
        )

        event_names = [e[0] for e in events]
        assert event_names == ["metadata", "token", "error", "done"]

        err = events[2][1]
        assert err["error"] == "GenerationError"
        assert "ConnectionError" in err["detail"]

        done = events[3][1]
        assert done["done"] is True

    def test_stream_retrieval_failure_emits_error_and_done(self):
        """If engine.search_hybrid raises an exception, emits error and done safely."""
        engine = MagicMock()
        engine.search_hybrid.side_effect = RuntimeError("FAISS search internal failure")

        events = list(
            stream_qa_query(
                query="Any question?",
                engine=engine,
            )
        )

        event_names = [e[0] for e in events]
        assert event_names == ["error", "done"]
        assert events[0][1]["error"] == "RetrievalError"
        assert events[1][1]["done"] is True


# ── HTTP API Integration Tests: POST /api/query/stream ───────────────────────

class TestStreamingAPIEndpoint:
    """Integration tests for the HTTP SSE endpoint using TestClient."""

    def test_stream_missing_engine_returns_503(self, api_client):
        """When no engine is indexed/loaded, endpoint returns HTTP 503."""
        resp = api_client.post("/api/query/stream", json={"query": "What sensor?"})
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = resp.json()
        assert data["error"] == "HTTPException"
        assert "No document index is available" in data["detail"]

    def test_stream_empty_query_returns_422(self, api_client):
        """Validation rejection: empty string returns 422."""
        _engine_holder.set(_make_engine())
        resp = api_client.post("/api/query/stream", json={"query": ""})
        assert resp.status_code == 422

    def test_stream_whitespace_only_query_returns_422(self, api_client):
        """Validation rejection: whitespace query returns 422."""
        _engine_holder.set(_make_engine())
        resp = api_client.post("/api/query/stream", json={"query": "    \n   "})
        assert resp.status_code == 422

    def test_stream_oversized_query_returns_422(self, api_client):
        """Validation rejection: query exceeding 2000 characters returns 422."""
        _engine_holder.set(_make_engine())
        resp = api_client.post("/api/query/stream", json={"query": "Q" * 2001})
        assert resp.status_code == 422

    def test_stream_valid_sse_framing_and_tokens(self, api_client):
        """
        Verify valid SSE wire framing, Content-Type, progressive tokens,
        and final metadata payloads via HTTP.
        """
        chunk = _MockChunk(
            text="Autonomous navigation with Ouster OS1-32 lidar produces highly accurate point clouds.",
            dense_score=0.75,
            page_number=5,
            source_filename="nav_paper.pdf",
        )
        _engine_holder.set(_make_engine([chunk]))

        stub_tokens = ["Ouster", " OS1-32", " is", " used."]

        with patch("src.services.qa_service._default_generate_answer", return_value=iter(stub_tokens)):
            resp = api_client.post(
                "/api/query/stream",
                json={"query": "What lidar is used for navigation?", "model": "llama3.2:1b"},
            )

        assert resp.status_code == status.HTTP_200_OK
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers.get("cache-control") == "no-cache"

        # Parse SSE events from response text
        events = parse_sse_events(resp.text)
        event_names = [e[0] for e in events]

        assert "metadata" in event_names
        assert "token" in event_names
        assert "final" in event_names
        assert "done" in event_names

        # Validate typed payloads
        meta_event = next(e[1] for e in events if e[0] == "metadata")
        StreamMetadataPayload(**meta_event)  # validates schema
        assert meta_event["has_supporting_evidence"] is True

        tokens_received = [e[1]["token"] for e in events if e[0] == "token"]
        assert tokens_received == stub_tokens

        final_event = next(e[1] for e in events if e[0] == "final")
        StreamFinalPayload(**final_event)  # validates schema
        assert final_event["answer"] == "".join(stub_tokens)
        assert final_event["completion_status"] == "complete"
        assert final_event["best_source"] == "nav_paper.pdf"

        done_event = next(e[1] for e in events if e[0] == "done")
        StreamDonePayload(**done_event)
        assert done_event["done"] is True

    def test_stream_abstention_via_sse(self, api_client):
        """
        Verify that an abstention query streams metadata, final abstention,
        and done without any token events.
        """
        chunk = _MockChunk(
            text="Unrelated biology notes on cellular division.",
            dense_score=0.15,
        )
        _engine_holder.set(_make_engine([chunk]))

        llm_called = []

        def _throw_if_called(*args, **kwargs):
            llm_called.append(True)
            return iter(["Should not happen!"])

        with patch("src.services.qa_service._default_generate_answer", side_effect=_throw_if_called):
            resp = api_client.post(
                "/api/query/stream",
                json={"query": "What is the lidar wavelength?", "model": "llama3.2:1b"},
            )

        assert resp.status_code == status.HTTP_200_OK
        assert len(llm_called) == 0

        events = parse_sse_events(resp.text)
        event_names = [e[0] for e in events]

        assert "token" not in event_names
        assert event_names == ["metadata", "final", "done"]

        final_event = next(e[1] for e in events if e[0] == "final")
        assert final_event["has_supporting_evidence"] is False
        assert final_event["completion_status"] == "abstention"
        assert final_event["answer_source"] == "abstention"

    def test_stream_generation_error_via_sse(self, api_client):
        """Verify that mid-stream generator exceptions emit an SSE error event and done event."""
        chunk = _MockChunk(
            text="Autonomous vehicle localization using lidar sensors.",
            dense_score=0.70,
        )
        _engine_holder.set(_make_engine([chunk]))

        def _broken_stream(*args, **kwargs):
            yield "Token 1"
            raise RuntimeError("Ollama connection terminated unexpectedly")

        with patch("src.services.qa_service._default_generate_answer", side_effect=_broken_stream):
            resp = api_client.post(
                "/api/query/stream",
                json={"query": "How does vehicle localization work?"},
            )

        assert resp.status_code == status.HTTP_200_OK
        events = parse_sse_events(resp.text)
        event_names = [e[0] for e in events]

        assert "error" in event_names
        assert "done" in event_names

        error_event = next(e[1] for e in events if e[0] == "error")
        StreamErrorPayload(**error_event)
        assert error_event["error"] == "GenerationError"
        assert "RuntimeError" in error_event["detail"]

    def test_non_streaming_endpoint_still_functions_correctly(self, api_client):
        """Confirm that POST /api/query remains 100% operational alongside POST /api/query/stream."""
        chunk = _MockChunk(
            text="The sensor mounted on top was an Ouster OS1-32.",
            dense_score=0.75,
            source_filename="ouster.pdf",
        )
        _engine_holder.set(_make_engine([chunk]))

        with patch("src.services.qa_service._default_generate_answer", return_value=iter(["OS1-32 sensor."])):
            resp = api_client.post("/api/query", json={"query": "What sensor was mounted on top?"})

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["answer"] == "OS1-32 sensor."
        assert data["answer_source"] == "ollama"
        assert data["has_supporting_evidence"] is True
        assert data["best_source"] == "ouster.pdf"
