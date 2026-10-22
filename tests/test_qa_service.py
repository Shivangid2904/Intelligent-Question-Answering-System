"""
IntelliAsk Research — Phase 2B QA Service & Query Endpoint Tests

Covers:
- QA service with sufficient evidence → supported answer
- QA service with insufficient evidence → abstention (no LLM call)
- Citations and passages carry physical page numbers and source filenames
- Hybrid retrieval depth and dense-score gating match app.py semantics
- Empty and excessively-long queries are rejected by schema validation
- Unavailable engine → HTTP 503
- Internal exceptions do not leak stack traces or paths
- Runs 100% offline (no live Ollama server)
"""

from __future__ import annotations

import math
from typing import Any, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import _engine_holder, app, create_app
from src.api.schemas import QueryRequest
from src.services.qa_service import (
    SCORE_THRESHOLD,
    PassageInfo,
    QAResult,
    _extract_dense_score,
    _get_display_score,
    _is_qualifying_chunk,
    run_qa_query,
)


# ─── Helpers / Fixtures ────────────────────────────────────────────────────────

class _FakeChunk:
    """Minimal Chunk look-alike for testing without importing DocumentProcessor."""

    def __init__(
        self,
        text: str,
        source_filename: str = "paper.pdf",
        page_number: int = 3,
        dense_score: Optional[float] = 0.55,
        score: float = 0.016,
        chunk_id: str = "chunk_001",
    ):
        self.text = text
        self.source_filename = source_filename
        self.source = source_filename  # alias used by app.py
        self.page_number = page_number
        self.dense_score = dense_score
        self.score = score
        self.chunk_id = chunk_id


def _make_engine(
    chunks: Optional[List[_FakeChunk]] = None,
    search_results: Optional[List[_FakeChunk]] = None,
) -> MagicMock:
    """Return a mock QAEngine that returns fixed search results."""
    engine = MagicMock()
    results = search_results if search_results is not None else (chunks or [])
    engine.search_hybrid.return_value = results
    return engine


def _stub_generator(text: str):
    """Return a stub answer generator that yields a fixed string."""
    def _gen(question: str, passages: list, model: str = "llama3.2:1b"):
        yield text
    return _gen


@pytest.fixture(autouse=True)
def clear_engine_holder():
    """Ensure _engine_holder is reset between tests."""
    _engine_holder.clear()
    yield
    _engine_holder.clear()


@pytest.fixture
def api_client():
    return TestClient(app)


# ─── Score helper unit tests ───────────────────────────────────────────────────

class TestScoreHelpers:
    """Verify _extract_dense_score, _is_qualifying_chunk, _get_display_score."""

    def test_valid_score_returned(self):
        chunk = _FakeChunk("text", dense_score=0.6)
        assert _extract_dense_score(chunk) == pytest.approx(0.6)

    def test_none_score_returns_none(self):
        chunk = _FakeChunk("text", dense_score=None)
        assert _extract_dense_score(chunk) is None

    def test_nan_score_returns_none(self):
        chunk = _FakeChunk("text", dense_score=float("nan"))
        assert _extract_dense_score(chunk) is None

    def test_positive_infinity_returns_none(self):
        chunk = _FakeChunk("text", dense_score=float("inf"))
        assert _extract_dense_score(chunk) is None

    def test_negative_infinity_returns_none(self):
        chunk = _FakeChunk("text", dense_score=float("-inf"))
        assert _extract_dense_score(chunk) is None

    def test_sentinel_minus_one_returns_none(self):
        chunk = _FakeChunk("text", dense_score=-1.0)
        assert _extract_dense_score(chunk) is None

    def test_sentinel_below_minus_one_returns_none(self):
        chunk = _FakeChunk("text", dense_score=-1.5)
        assert _extract_dense_score(chunk) is None

    def test_dict_chunk_score_extracted(self):
        chunk = {"dense_score": 0.42, "text": "hello"}
        assert _extract_dense_score(chunk) == pytest.approx(0.42)

    def test_qualifies_above_threshold(self):
        chunk = _FakeChunk("text", dense_score=0.40)
        assert _is_qualifying_chunk(chunk, threshold=0.38) is True

    def test_does_not_qualify_below_threshold(self):
        chunk = _FakeChunk("text", dense_score=0.30)
        assert _is_qualifying_chunk(chunk, threshold=0.38) is False

    def test_does_not_qualify_missing_score(self):
        chunk = _FakeChunk("text", dense_score=None)
        assert _is_qualifying_chunk(chunk, threshold=0.38) is False

    def test_get_display_score_valid(self):
        chunk = _FakeChunk("text", dense_score=0.51)
        assert _get_display_score(chunk) == pytest.approx(0.51)

    def test_get_display_score_missing_returns_zero(self):
        chunk = _FakeChunk("text", dense_score=None)
        assert _get_display_score(chunk) == 0.0


# ─── QA Service — sufficient evidence ─────────────────────────────────────────

class TestQAServiceSufficientEvidence:

    def test_supported_answer_returned(self):
        chunk = _FakeChunk(
            "LiDAR sensor OS1-32 was used in the experiment.",
            dense_score=0.65,
            page_number=5,
            source_filename="karukayil.pdf",
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("The Ouster OS1-32 LiDAR sensor was used.")

        result = run_qa_query("Which LiDAR sensor was used?", engine, answer_generator=gen)

        assert result.answer_source == "ollama"
        assert result.has_supporting_evidence is True
        assert "OS1-32" in result.answer or "Ouster" in result.answer
        assert result.verdict.should_abstain is False

    def test_retrieval_uses_k10(self):
        chunk = _FakeChunk("Some text.", dense_score=0.55)
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("Answer.")

        run_qa_query("What is X?", engine, answer_generator=gen)

        engine.search_hybrid.assert_called_once_with("What is X?", k=10)

    def test_citations_contain_source_and_page(self):
        chunk = _FakeChunk(
            "The sensor is the Velodyne HDL-64E.",
            dense_score=0.72,
            page_number=12,
            source_filename="sensor_paper.pdf",
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("The Velodyne sensor.")

        result = run_qa_query("What sensor?", engine, answer_generator=gen)

        assert result.has_supporting_evidence is True
        assert len(result.citations) >= 1
        assert any("sensor_paper.pdf" in c for c in result.citations)
        assert any("p. 12" in c for c in result.citations)

    def test_supporting_passages_carry_physical_page_number(self):
        # Chunk text must contain a content word from the query for evidence sufficiency
        chunk = _FakeChunk(
            "Some methodology happened on page seven of the experiment.",
            dense_score=0.6,
            page_number=7,
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("Answer here.")

        result = run_qa_query("What methodology happened?", engine, answer_generator=gen)

        assert len(result.supporting_passages) >= 1
        assert result.supporting_passages[0].page_number == 7
        assert result.supporting_passages[0].qualifies is True

    def test_best_source_populated_from_qualifying_chunk(self):
        chunk = _FakeChunk(
            "Results from the ablation study.",
            dense_score=0.62,
            source_filename="ablation.pdf",
            page_number=9,
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("Results here.")

        result = run_qa_query("Ablation results?", engine, answer_generator=gen)

        assert "ablation.pdf" in result.best_source
        assert result.best_score > 0.0

    def test_all_retrieved_includes_non_qualifying(self):
        high_chunk = _FakeChunk("Good passage.", dense_score=0.70, chunk_id="c1")
        low_chunk = _FakeChunk("Irrelevant text.", dense_score=0.10, chunk_id="c2")
        engine = _make_engine(search_results=[high_chunk, low_chunk])
        gen = _stub_generator("Good answer.")

        result = run_qa_query("What?", engine, answer_generator=gen)

        qualifies_flags = [p.qualifies for p in result.all_retrieved]
        assert True in qualifies_flags
        assert False in qualifies_flags


# ─── QA Service — abstention ───────────────────────────────────────────────────

class TestQAServiceAbstention:

    def test_insufficient_evidence_abstains(self):
        chunk = _FakeChunk("Unrelated passage about potatoes.", dense_score=0.15)
        engine = _make_engine(search_results=[chunk])
        llm_called = []

        def _gen(q, p, model="llama3.2:1b"):
            llm_called.append(True)
            yield "LLM should not be called"

        result = run_qa_query("What LiDAR sensor?", engine, answer_generator=_gen)

        assert result.answer_source == "abstention"
        assert result.has_supporting_evidence is False
        assert result.verdict.should_abstain is True
        assert len(llm_called) == 0, "LLM was called despite insufficient evidence"

    def test_no_results_abstains(self):
        engine = _make_engine(search_results=[])
        llm_called = []

        def _gen(q, p, model="llama3.2:1b"):
            llm_called.append(True)
            yield "SHOULD NOT REACH"

        result = run_qa_query("Anything?", engine, answer_generator=_gen)

        assert result.answer_source == "abstention"
        assert result.has_supporting_evidence is False
        assert len(llm_called) == 0

    def test_abstention_message_not_empty(self):
        engine = _make_engine(search_results=[])
        result = run_qa_query("What is the sensor?", engine)
        assert len(result.answer.strip()) > 10

    def test_abstention_supporting_passages_empty(self):
        chunk = _FakeChunk("Low relevance.", dense_score=0.10)
        engine = _make_engine(search_results=[chunk])
        result = run_qa_query("What is the methodology?", engine)
        assert result.has_supporting_evidence is False
        assert result.supporting_passages == []


# ─── QA Service — score gating (dense_score strictly controls qualifying) ──────

class TestScoreGating:

    def test_chunk_exactly_at_threshold_qualifies(self):
        # Text must contain a content word (>3 chars) from the query
        chunk = _FakeChunk(
            "This methodology study is at border threshold.",
            dense_score=0.38,
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("Answer.")

        result = run_qa_query("What methodology study?", engine, answer_generator=gen)

        # Only qualifies if dense_score >= 0.38
        assert result.has_supporting_evidence is True
        assert result.supporting_passages[0].qualifies is True

    def test_chunk_just_below_threshold_does_not_qualify(self):
        # dense_score 0.379 is below 0.38
        chunk = _FakeChunk("Almost qualifying.", dense_score=0.379)
        engine = _make_engine(search_results=[chunk])

        result = run_qa_query("Query?", engine)

        assert result.has_supporting_evidence is False

    def test_high_rrf_score_with_no_dense_score_does_not_qualify(self):
        # score (RRF) is high but dense_score is None: must not qualify
        chunk = _FakeChunk("Good BM25 passage.", dense_score=None, score=0.9)
        engine = _make_engine(search_results=[chunk])

        result = run_qa_query("Something?", engine)

        assert result.has_supporting_evidence is False

    def test_rrf_score_not_used_for_gating(self):
        """RRF score (chunk.score) is preserved but never used for evidence gating."""
        # Dense score >= threshold; text must contain a query content word
        chunk = _FakeChunk(
            "Relevant methodology passage from paper.",
            dense_score=0.55,
            score=0.002,
        )
        engine = _make_engine(search_results=[chunk])
        gen = _stub_generator("Answer.")

        result = run_qa_query("What methodology?", engine, answer_generator=gen)

        # Low RRF score must not cause abstention if dense_score >= threshold
        assert result.has_supporting_evidence is True


# ─── Internal exception safety ────────────────────────────────────────────────

class TestExceptionSafety:

    def test_engine_exception_returns_error_result(self):
        engine = MagicMock()
        engine.search_hybrid.side_effect = RuntimeError("FAISS internal crash")

        result = run_qa_query("What?", engine)

        assert result.answer_source == "error"
        # No raw exception trace in the answer
        assert "Traceback" not in result.answer
        assert "RuntimeError" in result.answer or "error" in result.answer.lower()

    def test_generator_exception_returns_error_result(self):
        chunk = _FakeChunk(
            "Great methodology passage with detailed results.",
            dense_score=0.70,
        )
        engine = _make_engine(search_results=[chunk])

        def _bad_gen(q, p, model="llama3.2:1b"):
            # Raise inside the iteration loop (not at call time)
            yield from []  # ensure it's a generator
            raise ConnectionError("Ollama unreachable")

        # Patch the default so the service uses our bad generator
        with patch("src.services.qa_service._default_generate_answer", side_effect=_bad_gen):
            result = run_qa_query(
                "What methodology results?",
                engine,
                # Do NOT pass answer_generator kwarg — let default be patched
            )

        assert result.answer_source in ("error", "abstention")
        assert "Traceback" not in result.answer

    def test_error_result_does_not_contain_filesystem_paths(self):
        engine = MagicMock()
        engine.search_hybrid.side_effect = FileNotFoundError("/data/papers.db")

        result = run_qa_query("Test?", engine)

        assert result.answer_source == "error"
        # Absolute paths must not be surfaced
        assert "/data/papers.db" not in result.answer


# ─── POST /api/query endpoint tests ──────────────────────────────────────────

class TestQueryEndpoint:

    def _inject_engine(self, engine):
        _engine_holder.set(engine)

    def test_engine_unavailable_returns_503(self, api_client):
        # engine holder is empty (cleared by fixture)
        response = api_client.post("/api/query", json={"query": "What sensor?"})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "No document index" in data.get("detail", "")

    def test_empty_query_returns_422(self, api_client):
        chunk = _FakeChunk("Good passage.", dense_score=0.65)
        self._inject_engine(_make_engine(search_results=[chunk]))

        response = api_client.post("/api/query", json={"query": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_whitespace_only_query_returns_422(self, api_client):
        chunk = _FakeChunk("Good passage.", dense_score=0.65)
        self._inject_engine(_make_engine(search_results=[chunk]))

        response = api_client.post("/api/query", json={"query": "   "})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_exceeding_max_length_returns_422(self, api_client):
        chunk = _FakeChunk("Good passage.", dense_score=0.65)
        self._inject_engine(_make_engine(search_results=[chunk]))

        response = api_client.post("/api/query", json={"query": "A" * 2001})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_sufficient_evidence_returns_200_with_supported_answer(self, api_client):
        chunk = _FakeChunk(
            "The Ouster OS1-32 sensor is used in SLAM.",
            dense_score=0.70,
            page_number=4,
            source_filename="slam_paper.pdf",
        )
        engine = _make_engine(search_results=[chunk])
        self._inject_engine(engine)

        with patch(
            "src.services.qa_service._default_generate_answer",
            side_effect=lambda q, p, model: iter(["OS1-32 is the sensor used."]),
        ):
            response = api_client.post(
                "/api/query",
                json={"query": "What sensor is used in SLAM?", "model": "llama3.2:1b"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer_source"] == "ollama"
        assert data["has_supporting_evidence"] is True
        assert data["verdict"]["should_abstain"] is False
        assert len(data["supporting_passages"]) >= 1
        assert data["supporting_passages"][0]["page_number"] == 4
        assert data["supporting_passages"][0]["source_filename"] == "slam_paper.pdf"

    def test_insufficient_evidence_returns_200_with_abstention(self, api_client):
        chunk = _FakeChunk("Unrelated text.", dense_score=0.10)
        engine = _make_engine(search_results=[chunk])
        self._inject_engine(engine)

        response = api_client.post(
            "/api/query",
            json={"query": "What sensor was used?"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer_source"] == "abstention"
        assert data["has_supporting_evidence"] is False
        assert data["verdict"]["should_abstain"] is True
        # Abstention must carry a non-empty message
        assert len(data["answer"].strip()) > 0

    def test_response_schema_fully_valid(self, api_client):
        from src.api.schemas import QueryResponse, VerdictSchema, PassageSchema

        chunk = _FakeChunk("Evidence passage.", dense_score=0.65, page_number=2)
        engine = _make_engine(search_results=[chunk])
        self._inject_engine(engine)

        with patch(
            "src.services.qa_service._default_generate_answer",
            side_effect=lambda q, p, model: iter(["Generated answer."]),
        ):
            response = api_client.post("/api/query", json={"query": "Test query?"})

        assert response.status_code == status.HTTP_200_OK
        validated = QueryResponse(**response.json())
        assert isinstance(validated.verdict, VerdictSchema)
        for p in validated.supporting_passages:
            assert isinstance(p, PassageSchema)

    def test_no_stack_trace_in_error_response(self, api_client):
        engine = MagicMock()
        engine.search_hybrid.side_effect = RuntimeError("FAISS crash")
        self._inject_engine(engine)

        response = api_client.post("/api/query", json={"query": "What happened?"})

        # The global error handler + QA service must swallow the exception
        # and return either a 200 with answer_source="error" or a structured 500
        assert response.status_code in (200, 500)
        body = response.text
        assert "Traceback" not in body

    def test_missing_query_field_returns_422(self, api_client):
        chunk = _FakeChunk("Passage.", dense_score=0.65)
        self._inject_engine(_make_engine(search_results=[chunk]))

        response = api_client.post("/api/query", json={"model": "llama3.2:1b"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_custom_model_forwarded_to_generator(self, api_client):
        # Use a query and chunk text with overlapping content words
        chunk = _FakeChunk(
            "Evidence from the methodology study on sensors.",
            dense_score=0.65,
        )
        engine = _make_engine(search_results=[chunk])
        self._inject_engine(engine)

        received_models = []

        def _capture_gen(q, passages, model="llama3.2:1b"):
            received_models.append(model)
            yield "Answer."

        with patch("src.services.qa_service._default_generate_answer", side_effect=_capture_gen):
            response = api_client.post(
                "/api/query",
                json={"query": "What methodology study?", "model": "mistral:7b"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert received_models == ["mistral:7b"]

    def test_all_retrieved_rank_and_qualify_flags_present(self, api_client):
        high = _FakeChunk("Good.", dense_score=0.70, chunk_id="c1")
        low = _FakeChunk("Bad.", dense_score=0.05, chunk_id="c2")
        engine = _make_engine(search_results=[high, low])
        self._inject_engine(engine)

        with patch(
            "src.services.qa_service._default_generate_answer",
            side_effect=lambda q, p, model: iter(["Answer."]),
        ):
            response = api_client.post("/api/query", json={"query": "Something?"})

        assert response.status_code == status.HTTP_200_OK
        all_r = response.json()["all_retrieved"]
        assert len(all_r) == 2
        assert any(p["qualifies"] is True for p in all_r)
        assert any(p["qualifies"] is False for p in all_r)
