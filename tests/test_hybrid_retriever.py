"""
test_hybrid_retriever.py
========================
Unit and integration tests for hybrid retrieval using Reciprocal Rank Fusion (RRF).

Coverage:
  1. Correct RRF score calculation and rank ordering
  2. A result appearing in both ranked lists
  3. Empty dense or BM25 results
  4. Deterministic tie handling
  5. Duplicate prevention and provenance preservation
  6. Dense-only behavior remaining available
  7. Integration with the existing QA engine
"""

from __future__ import annotations

import os
import sys
from typing import Any, List
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.document_processor import Chunk
from src.qa_engine import QAEngine, reciprocal_rank_fusion
from tests.test_qa_engine import MockEmbeddingModel


def make_chunk(
    text: str,
    source_filename: str = "doc1.pdf",
    page_number: int = 1,
    document_id: str = "doc1_hash",
    chunk_index: int = 0,
    chunk_id: str | None = None,
) -> Chunk:
    """Helper to create a Chunk with explicit provenance."""
    return Chunk(
        text=text,
        source_filename=source_filename,
        page_number=page_number,
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
    )


# ---------------------------------------------------------------------------
# 1. RRF Score Calculation and Ordering
# ---------------------------------------------------------------------------


class TestRRFCalculationAndOrdering:
    def test_single_list_rrf_formula(self):
        """Single list items get 1 / (60 + rank)."""
        c1 = make_chunk("Chunk 1", chunk_id="c1")
        c2 = make_chunk("Chunk 2", chunk_id="c2")
        fused = reciprocal_rank_fusion([[c1, c2]], k=2, rrf_k=60)

        assert len(fused) == 2
        # Rank 1: 1 / 61
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0
        # Rank 2: 1 / 62
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 62.0
        assert fused[0].chunk_id == "c1"
        assert fused[1].chunk_id == "c2"

    def test_custom_rrf_k_parameter(self):
        """Custom rrf_k (e.g. 20) correctly scales denominator."""
        c1 = make_chunk("Chunk 1", chunk_id="c1")
        fused = reciprocal_rank_fusion([[c1]], k=1, rrf_k=20)
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 21.0

    def test_two_lists_sum_reciprocal_ranks(self):
        """Item in both lists has score = 1/(60+r1) + 1/(60+r2)."""
        target = make_chunk("Target Chunk", chunk_id="target")
        other_dense = make_chunk("Other Dense", chunk_id="other_d")
        other_lex = make_chunk("Other Lexical", chunk_id="other_l")

        # target is rank 1 in dense, rank 2 in lexical
        dense_list = [target, other_dense]
        lexical_list = [other_lex, target]

        fused = reciprocal_rank_fusion([dense_list, lexical_list], k=3, rrf_k=60)

        assert len(fused) == 3
        # target appeared at rank 1 in list 1 and rank 2 in list 2
        expected_target_score = (1.0 / 61.0) + (1.0 / 62.0)
        assert fused[0].chunk_id == "target"
        assert pytest.approx(fused[0].score, rel=1e-6) == expected_target_score

        # other_lex was rank 1 in list 2 only: 1/61
        assert fused[1].chunk_id == "other_l"
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 61.0

        # other_dense was rank 2 in list 1 only: 1/62
        assert fused[2].chunk_id == "other_d"
        assert pytest.approx(fused[2].score, rel=1e-6) == 1.0 / 62.0


# ---------------------------------------------------------------------------
# 2. Overlap and Deduplication
# ---------------------------------------------------------------------------


class TestOverlapAndDeduplication:
    def test_result_appearing_in_both_lists_is_deduplicated(self):
        """Chunk in both lists must appear exactly once in output."""
        shared = make_chunk("Shared content", chunk_id="shared_chunk")
        dense = [shared]
        lexical = [shared]

        fused = reciprocal_rank_fusion([dense, lexical], k=10)
        assert len(fused) == 1
        assert fused[0].chunk_id == "shared_chunk"
        # Score must be 2 / 61
        assert pytest.approx(fused[0].score, rel=1e-6) == 2.0 / 61.0

    def test_multi_list_overlap_three_lists(self):
        """RRF correctly fuses across >2 lists."""
        shared = make_chunk("Shared across 3", chunk_id="all_three")
        l1 = [shared]
        l2 = [shared]
        l3 = [shared]

        fused = reciprocal_rank_fusion([l1, l2, l3], k=5)
        assert len(fused) == 1
        assert pytest.approx(fused[0].score, rel=1e-6) == 3.0 / 61.0

    def test_intra_list_duplicate_chunk_contributes_only_first_rank(self):
        """Duplicate occurrences of a chunk within the same ranked list contribute only once."""
        dup = make_chunk("Duplicate chunk", chunk_id="dup_chunk")
        other = make_chunk("Other chunk", chunk_id="other_chunk")

        # dup appears at rank 1 and rank 3 within the SAME list
        single_list_with_dups = [dup, other, dup]

        fused = reciprocal_rank_fusion([single_list_with_dups], k=5, rrf_k=60)
        assert len(fused) == 2
        assert fused[0].chunk_id == "dup_chunk"
        # Score must be 1 / 61 (rank 1 only; rank 3 intra-list duplicate ignored)
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0
        assert fused[1].chunk_id == "other_chunk"
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 62.0


# ---------------------------------------------------------------------------
# 3. Empty and Asymmetric Results Handling
# ---------------------------------------------------------------------------


class TestEmptyAndAsymmetricLists:
    def test_all_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([], k=5) == []
        assert reciprocal_rank_fusion([[], []], k=5) == []

    def test_empty_dense_list_returns_lexical_results(self):
        """When dense retriever returns [], BM25 results are preserved."""
        c1 = make_chunk("Lexical hit 1", chunk_id="lex1")
        c2 = make_chunk("Lexical hit 2", chunk_id="lex2")

        fused = reciprocal_rank_fusion([[], [c1, c2]], k=5)
        assert len(fused) == 2
        assert fused[0].chunk_id == "lex1"
        assert fused[1].chunk_id == "lex2"
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 62.0

    def test_empty_lexical_list_returns_dense_results(self):
        """When BM25 returns [] (e.g. no lexical overlap), dense results are preserved."""
        c1 = make_chunk("Dense hit 1", chunk_id="dense1")
        c2 = make_chunk("Dense hit 2", chunk_id="dense2")

        fused = reciprocal_rank_fusion([[c1, c2], []], k=5)
        assert len(fused) == 2
        assert fused[0].chunk_id == "dense1"
        assert fused[1].chunk_id == "dense2"
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0

    def test_k_clamping(self):
        """k larger than total unique candidates returns all candidates."""
        c1 = make_chunk("Hit 1", chunk_id="c1")
        fused = reciprocal_rank_fusion([[c1]], k=100)
        assert len(fused) == 1


# ---------------------------------------------------------------------------
# 4. Deterministic Tie Handling
# ---------------------------------------------------------------------------


class TestDeterministicTies:
    def test_tied_scores_break_deterministically(self):
        """
        When two chunks have identical RRF scores (e.g. rank 1 in list 1 vs rank 1 in list 2),
        the tie is resolved deterministically by earliest discovery list order.
        """
        chunk_dense_only = make_chunk("Dense top", chunk_id="d_top")
        chunk_lex_only = make_chunk("Lexical top", chunk_id="l_top")

        list_a = [chunk_dense_only]
        list_b = [chunk_lex_only]

        fused_1 = reciprocal_rank_fusion([list_a, list_b], k=2)
        fused_2 = reciprocal_rank_fusion([list_a, list_b], k=2)

        # Both have score 1/61
        assert pytest.approx(fused_1[0].score, rel=1e-6) == 1.0 / 61.0
        assert pytest.approx(fused_1[1].score, rel=1e-6) == 1.0 / 61.0

        # Run 1 and Run 2 must be identical
        assert [c.chunk_id for c in fused_1] == [c.chunk_id for c in fused_2]
        # First-seen order: d_top was in list_a (index 0), l_top was in list_b (index 1)
        assert fused_1[0].chunk_id == "d_top"
        assert fused_1[1].chunk_id == "l_top"


# ---------------------------------------------------------------------------
# 5. Provenance Preservation and Immutability
# ---------------------------------------------------------------------------


class TestProvenancePreservationAndImmutability:
    def test_original_chunks_not_mutated(self):
        """Input chunks must keep original score (0.0); fused output is a copy."""
        orig = make_chunk(
            "Original content with metadata",
            source_filename="karukayil.pdf",
            page_number=5,
            document_id="doc_karukayil",
            chunk_index=42,
            chunk_id="orig_c42",
        )
        orig.score = 0.0

        fused = reciprocal_rank_fusion([[orig]], k=1)
        assert len(fused) == 1
        assert fused[0].score > 0.0
        # Original chunk object must still have score 0.0
        assert orig.score == 0.0

    def test_provenance_metadata_preserved_completely(self):
        """Physical page, source filename, document id survive fusion intact."""
        orig = make_chunk(
            "LiDAR pothole detection text",
            source_filename="sensor_paper.pdf",
            page_number=7,
            document_id="sensor_doc_123",
            chunk_index=15,
            chunk_id="sensor_p7_c15",
        )
        fused = reciprocal_rank_fusion([[orig]], k=1)
        out = fused[0]

        assert out.page_number == 7
        assert out.source_filename == "sensor_paper.pdf"
        assert out.document_id == "sensor_doc_123"
        assert out.chunk_index == 15
        assert out.chunk_id == "sensor_p7_c15"
        assert out.text == "LiDAR pothole detection text"

    def test_fallback_chunk_id_is_deterministic(self):
        """Chunks without chunk_id get a deterministic fallback ID via MD5 on full text."""
        from src.qa_engine import _get_chunk_id

        chunk_dict = {
            "source_filename": "paper.pdf",
            "page_number": 4,
            "text": "This is a sample sentence without explicit chunk_id.",
        }

        id_1 = _get_chunk_id(chunk_dict)
        id_2 = _get_chunk_id(chunk_dict)

        assert id_1 == id_2
        assert id_1.startswith("paper.pdf_p4_")
        digest = id_1.split("_")[-1]
        assert len(digest) == 8
        assert all(c in "0123456789abcdef" for c in digest)

        # Different full texts sharing the same first 30 characters must produce different fallback IDs
        prefix_30 = "123456789012345678901234567890"  # exactly 30 characters
        chunk_a = {
            "source_filename": "paper.pdf",
            "page_number": 4,
            "text": prefix_30 + " - Variant Alpha suffix",
        }
        chunk_b = {
            "source_filename": "paper.pdf",
            "page_number": 4,
            "text": prefix_30 + " - Variant Beta suffix",
        }
        id_a = _get_chunk_id(chunk_a)
        id_b = _get_chunk_id(chunk_b)
        assert id_a != id_b
        assert id_a.startswith("paper.pdf_p4_")
        assert id_b.startswith("paper.pdf_p4_")


# ---------------------------------------------------------------------------
# 6. QAEngine Hybrid Integration and Modes
# ---------------------------------------------------------------------------


class TestQAEngineHybridIntegration:
    @pytest.fixture
    def mock_model(self):
        return MockEmbeddingModel(dim=8)

    @pytest.fixture
    def sample_chunks(self):
        return [
            make_chunk(
                "The Ouster OS1-32 LiDAR sensor provides 32-beam point clouds.",
                source_filename="karukayil.pdf",
                page_number=5,
                document_id="doc_karukayil",
                chunk_index=0,
                chunk_id="k_p5_c0",
            ),
            make_chunk(
                "Deep learning models evaluate pavement distress from RGB images.",
                source_filename="survey.pdf",
                page_number=2,
                document_id="doc_survey",
                chunk_index=1,
                chunk_id="s_p2_c1",
            ),
            make_chunk(
                "iPhone 14 Pro equipped with LiDAR was tested in Lahore.",
                source_filename="butt.pdf",
                page_number=3,
                document_id="doc_butt",
                chunk_index=2,
                chunk_id="b_p3_c2",
            ),
        ]

    def test_dense_only_behavior_default(self, sample_chunks, mock_model):
        """Default search() calls dense FAISS retrieval (cosine similarity scores)."""
        engine = QAEngine(sample_chunks, mock_model)
        assert engine.default_mode == "dense"
        results = engine.search("Ouster OS1-32", k=2)

        assert len(results) == 2
        # Dense scores are cosine similarities (-1.0 to 1.0, typically > 0.0)
        assert results[0].score > 0.0

    def test_search_dense_explicit(self, sample_chunks, mock_model):
        """search_dense() executes dense FAISS retrieval directly."""
        engine = QAEngine(sample_chunks, mock_model)
        results = engine.search_dense("Ouster OS1-32", k=2)
        assert len(results) == 2
        assert isinstance(results[0].score, float)

    def test_search_lexical_explicit(self, sample_chunks, mock_model):
        """search_lexical() executes BM25 retrieval directly."""
        engine = QAEngine(sample_chunks, mock_model)
        results = engine.search_lexical("Ouster OS1-32", k=2)
        assert len(results) >= 1
        # BM25 finds the exact match chunk from page 5 at rank 1
        assert results[0].chunk_id == "k_p5_c0"
        assert results[0].page_number == 5

    def test_search_hybrid_explicit(self, sample_chunks, mock_model):
        """search_hybrid() fuses dense and BM25 candidates with RRF scores."""
        engine = QAEngine(sample_chunks, mock_model)
        results = engine.search_hybrid("Ouster OS1-32", k=2)

        assert len(results) >= 1
        # Exact match chunk surfaced by BM25 and dense should be rank 1
        assert results[0].chunk_id == "k_p5_c0"
        assert results[0].page_number == 5
        # Score is an RRF score (bounded by 2 * (1/61) ~ 0.033)
        assert 0.0 < results[0].score < 0.1

    def test_search_with_mode_parameter(self, sample_chunks, mock_model):
        """search() respects mode='hybrid', mode='dense', mode='lexical'."""
        engine = QAEngine(sample_chunks, mock_model)

        res_dense = engine.search("Ouster OS1-32", k=1, mode="dense")
        res_lex = engine.search("Ouster OS1-32", k=1, mode="lexical")
        res_hyb = engine.search("Ouster OS1-32", k=1, mode="hybrid")

        assert len(res_dense) == 1
        assert len(res_lex) == 1
        assert len(res_hyb) == 1

        # RRF score is distinct from cosine similarity and BM25 score
        assert res_hyb[0].score != res_dense[0].score

    def test_unknown_mode_raises_value_error(self, sample_chunks, mock_model):
        engine = QAEngine(sample_chunks, mock_model)
        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            engine.search("test", k=1, mode="invalid_mode")

    def test_empty_engine_hybrid_search(self, mock_model):
        """Empty QAEngine handles hybrid search gracefully without crashing."""
        engine = QAEngine([], mock_model)
        assert engine.search_hybrid("test query", k=5) == []
        assert engine.search("test query", k=5, mode="hybrid") == []

    def test_search_hybrid_when_lexical_retriever_is_none(self, sample_chunks, mock_model):
        """When lexical_retriever is None, search_hybrid() falls back gracefully to dense candidates."""
        engine = QAEngine(sample_chunks, mock_model)
        engine.lexical_retriever = None

        assert engine.search_lexical("Ouster OS1-32", k=2) == []

        dense_expected = engine.search_dense("Ouster OS1-32", k=2)
        assert len(dense_expected) == 2

        results = engine.search_hybrid("Ouster OS1-32", k=2)
        assert len(results) == 2

        # Order of returned candidates matches dense ranking exactly
        assert [c.chunk_id for c in results] == [c.chunk_id for c in dense_expected]

        # Dense candidates are scored via RRF (single list contribution: 1 / (60 + rank))
        assert pytest.approx(results[0].score, rel=1e-6) == 1.0 / 61.0
        assert pytest.approx(results[1].score, rel=1e-6) == 1.0 / 62.0
