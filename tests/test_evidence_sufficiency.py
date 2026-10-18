"""
tests/test_evidence_sufficiency.py
==================================
Tests for Week 3 Step 3 Evidence-Sufficiency Safeguards:
  1. Document Scoping (preventing cross-paper contamination)
  2. Critical-Entity Validation (token-aware matching, avoiding substring false positives)
  3. Dual-Score Support (explicit None for uncomputable dense scores, no fallback to RRF/BM25)
  4. QAEngine Score Preservation & Mapping Hardening (1D/2D vectors, dimension checks, ambiguous IDs)
"""

import pytest
from unittest.mock import MagicMock
import numpy as np

from src.citation import (
    check_evidence_sufficiency,
    EvidenceVerdict,
    _chunk_source_filename,
    _chunk_semantic_score,
    _normalize_entity,
    _tokenize_entity,
    _entity_present_in_text,
)
from src.document_processor import Chunk
from src.qa_engine import QAEngine
from tests.test_qa_engine import MockEmbeddingModel


def _make_chunk(
    text: str,
    source_filename: str,
    page_number: int,
    score: float = 0.45,
    dense_score: float | None = None,
    bm25_score: float | None = None,
    chunk_id: str | None = None,
):
    """Create a minimal Chunk-like mock object with required attributes."""
    c = MagicMock()
    c.text = text
    c.source_filename = source_filename
    c.source = source_filename
    c.page_number = page_number
    c.score = score
    c.document_id = f"doc_{source_filename[:8]}"
    c.chunk_id = chunk_id or f"{source_filename}_p{page_number}"
    c.dense_score = dense_score
    c.bm25_score = bm25_score
    return c


# ==============================================================================
# 1. Document Scoping Tests
# ==============================================================================

class TestDocumentScoping:
    def test_scoping_filters_unrelated_documents(self):
        """Chunks from un-scoped documents must be excluded from evidence."""
        c1 = _make_chunk(
            text="Ponnusamy paper discusses ground tripod setup at 0.8-1.2 m.",
            source_filename="ponnusamy_2026.pdf",
            page_number=3,
            score=0.45,
            dense_score=0.45,
        )
        c2 = _make_chunk(
            text="Unrelated paper discusses pitch angle from 15 to 75 degrees.",
            source_filename="other_paper.pdf",
            page_number=7,
            score=0.55,
            dense_score=0.55,
        )
        chunks = [c2, c1]

        # Scope only to ponnusamy_2026.pdf
        verdict = check_evidence_sufficiency(
            chunks,
            query="What pitch angle was used in Ponnusamy?",
            score_threshold=0.38,
            scope_filenames=["ponnusamy_2026.pdf"],
        )

        assert verdict.scoped_chunk_count == 1
        assert verdict.chunk_count == 2
        # Only c1 should be cited
        assert len(verdict.citations) == 1
        assert "ponnusamy_2026.pdf" in verdict.citations[0]
        assert "other_paper.pdf" not in verdict.citations[0]

    def test_scoping_no_matching_documents_abstains(self):
        """When scope produces 0 matching chunks, system must abstain with clear reason."""
        c1 = _make_chunk("Irrelevant content.", "doc_a.pdf", page_number=1, score=0.60, dense_score=0.60)
        chunks = [c1]

        verdict = check_evidence_sufficiency(
            chunks,
            query="What sensor is used in doc_b?",
            scope_filenames=["doc_b.pdf"],
        )

        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert verdict.scoped_chunk_count == 0
        assert "Document scoping is active" in verdict.reason
        assert "doc_b.pdf" in verdict.reason

    def test_multi_document_scoping(self):
        """Questions comparing multiple papers retain chunks from all specified papers."""
        c1 = _make_chunk("Paper A uses LiDAR sensor.", "paper_a.pdf", page_number=2, score=0.5, dense_score=0.5)
        c2 = _make_chunk("Paper B uses stereo camera.", "paper_b.pdf", page_number=4, score=0.52, dense_score=0.52)
        c3 = _make_chunk("Paper C is unrelated.", "paper_c.pdf", page_number=1, score=0.6, dense_score=0.6)
        chunks = [c1, c2, c3]

        verdict = check_evidence_sufficiency(
            chunks,
            query="Compare sensors between Paper A and Paper B",
            scope_filenames=["paper_a.pdf", "paper_b.pdf"],
        )

        assert verdict.scoped_chunk_count == 2
        assert len(verdict.citations) == 2
        assert any("paper_a.pdf" in c for c in verdict.citations)
        assert any("paper_b.pdf" in c for c in verdict.citations)
        assert not any("paper_c.pdf" in c for c in verdict.citations)

    def test_scoping_none_preserves_unscoped_behavior(self):
        """Default scope_filenames=None evaluates all chunks (backward compatibility)."""
        c1 = _make_chunk("Relevant text on attention models.", "doc1.pdf", page_number=1, score=0.45)
        c2 = _make_chunk("More text on transformers.", "doc2.pdf", page_number=2, score=0.48)
        verdict = check_evidence_sufficiency([c1, c2], query="attention models", scope_filenames=None)
        assert verdict.scoped_chunk_count == 2
        assert len(verdict.citations) == 2


# ==============================================================================
# 2. Critical-Entity Validation & Token-Aware Matching Tests
# ==============================================================================

class TestCriticalEntityValidation:
    def test_missing_critical_entity_abstains(self):
        """If a critical entity is absent from the evidence text, check must abstain."""
        chunk = _make_chunk(
            text="The vehicle used camera and GPS sensors for navigation across highway scenes.",
            source_filename="autonomous.pdf",
            page_number=5,
            score=0.62,
            dense_score=0.62,
        )
        verdict = check_evidence_sufficiency(
            [chunk],
            query="What model of Ouster OS1-32 LiDAR sensor was used?",
            score_threshold=0.38,
            critical_entities=["Ouster OS1-32"],
        )

        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert "Ouster OS1-32" in verdict.missing_entities
        assert "Required technical entity/entities not found" in verdict.reason

    def test_present_critical_entity_passes(self):
        """When critical entity is present, sufficiency verdict is True."""
        chunk = _make_chunk(
            text="The setup integrates an Ouster OS1-32 LiDAR with a high-resolution RGB camera.",
            source_filename="karukayil_2024.pdf",
            page_number=5,
            score=0.48,
            dense_score=0.48,
        )
        verdict = check_evidence_sufficiency(
            [chunk],
            query="What LiDAR sensor is used in Karukayil?",
            score_threshold=0.38,
            critical_entities=["Ouster OS1-32"],
        )

        assert verdict.sufficient is True
        assert verdict.should_abstain is False
        assert len(verdict.missing_entities) == 0

    def test_token_aware_avoids_substring_false_positive_zed_in_analyzed(self):
        """'ZED' must NOT match inside 'analyzed' or 'categorized'."""
        text = "We analyzed the point cloud and categorized the road distress."
        assert _entity_present_in_text("ZED", text) is False
        assert _entity_present_in_text("ZED 2i", text) is False

        # Positive match when ZED is an actual standalone identifier
        text_with_sensor = "Equipped with a ZED 2i stereo camera."
        assert _entity_present_in_text("ZED", text_with_sensor) is True
        assert _entity_present_in_text("ZED 2i", text_with_sensor) is True

    def test_token_aware_avoids_substring_false_positive_os1_in_os10(self):
        """'OS1' must NOT match inside 'OS10' or 'OS128'."""
        text = "The autonomous system ran on an OS10 operating system platform."
        assert _entity_present_in_text("OS1", text) is False

        # Positive match when OS1 is tokenized
        text_real = "The Ouster OS1-32 was mounted on the roof."
        assert _entity_present_in_text("OS1", text_real) is True
        assert _entity_present_in_text("OS1-32", text_real) is True

    def test_token_aware_avoids_substring_false_positive_dpt_in_adaptation(self):
        """'DPT' must NOT match inside 'adaptation'."""
        text = "We studied domain adaptation algorithms for pavement detection."
        assert _entity_present_in_text("DPT", text) is False

        # Positive match for DPT model
        text_dpt = "The MiDaS-DPT architecture was deployed for monocular depth."
        assert _entity_present_in_text("DPT", text_dpt) is True
        assert _entity_present_in_text("MiDaS-DPT", text_dpt) is True

    def test_token_aware_avoids_substring_false_positive_mae_in_remaining(self):
        """'MAE' must NOT match inside 'remaining'."""
        text = "The remaining samples were discarded from the training set."
        assert _entity_present_in_text("MAE", text) is False

        # Positive match for metric MAE
        text_mae = "Reported metrics include MAE = 0.089 and RMSE = 0.301."
        assert _entity_present_in_text("MAE", text_mae) is True

    def test_entity_normalization_punctuation_and_case(self):
        """Token-aware matching handles hyphens, underscores, spaces, and case transparently."""
        assert _tokenize_entity("OS1-32") == ["os1", "32"]
        assert _tokenize_entity("os1_32") == ["os1", "32"]
        assert _tokenize_entity("  DepthSense ") == ["depthsense"]
        assert _normalize_entity("OS1-32") == "os1 32"

        # Matching across hyphen / space / underscore variations
        assert _entity_present_in_text("OS1-32", "Ouster OS1 32 LiDAR") is True
        assert _entity_present_in_text("OS1-32", "Ouster OS1_32 LiDAR") is True
        assert _entity_present_in_text("OS1-32", "Ouster OS1-32 LiDAR") is True
        assert _entity_present_in_text("ZED-2i", "ZED 2i binocular camera") is True

    def test_combined_evidence_checks_multiple_entities(self):
        """Entities can be distributed across different scoped chunks."""
        c1 = _make_chunk("We evaluate on the KITTI dataset benchmark.", "eval.pdf", page_number=4, score=0.5, dense_score=0.5)
        c2 = _make_chunk("A ResNet-50 backbone is used for feature extraction.", "eval.pdf", page_number=5, score=0.52, dense_score=0.52)
        verdict = check_evidence_sufficiency(
            [c1, c2],
            query="What accuracy does ResNet-50 achieve on KITTI?",
            critical_entities=["ResNet-50", "KITTI"],
        )
        assert verdict.sufficient is True
        assert len(verdict.missing_entities) == 0


# ==============================================================================
# 3. Dual-Score Support & Fallback Hardening Tests
# ==============================================================================

class TestDualScoreSupport:
    def test_hybrid_uses_dense_score_for_threshold(self):
        """When use_dense_score=True, gating relies strictly on dense_score, not RRF score."""
        chunk = _make_chunk(
            text="Text mentioning Ouster OS1-32 and road pothole mapping.",
            source_filename="paper.pdf",
            page_number=5,
            score=0.032,        # Typical RRF score (1/61 + 1/62 ~ 0.0325), < 0.38
            dense_score=0.48,   # True cosine similarity, >= 0.38
        )

        # In dense mode (use_dense_score=False), score=0.032 < 0.38 -> abstains
        verdict_dense_mode = check_evidence_sufficiency(
            [chunk],
            query="What sensor is used for pothole mapping?",
            score_threshold=0.38,
            use_dense_score=False,
        )
        assert verdict_dense_mode.sufficient is False

        # In hybrid mode (use_dense_score=True), dense_score=0.48 >= 0.38 -> passes
        verdict_hybrid_mode = check_evidence_sufficiency(
            [chunk],
            query="What sensor is used for pothole mapping?",
            score_threshold=0.38,
            use_dense_score=True,
        )
        assert verdict_hybrid_mode.sufficient is True
        assert verdict_hybrid_mode.top_score == 0.48

    def test_hybrid_missing_dense_score_does_not_pass_on_large_bm25_score(self):
        """
        When use_dense_score=True, a chunk with chunk.score=14.5 (large BM25)
        and dense_score=None must NEVER fall back to chunk.score and falsely pass.
        """
        chunk = _make_chunk(
            text="High lexical match chunk about transformers and attention.",
            source_filename="paper.pdf",
            page_number=1,
            score=14.5,          # Large BM25 score >> 0.38
            dense_score=None,    # Missing dense score
        )
        score_val = _chunk_semantic_score(chunk, use_dense_score=True)
        assert score_val is None  # Must explicitly be None, not 14.5

        verdict = check_evidence_sufficiency(
            [chunk],
            query="attention transformers",
            score_threshold=0.38,
            use_dense_score=True,
        )
        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert verdict.unscored_chunk_count == 1
        assert "lacked valid semantic scores" in verdict.reason

    def test_hybrid_sentinel_dense_score_returns_none(self):
        """Sentinel dense_score <= -1.0 returns None and does not fall back to score."""
        chunk = _make_chunk(
            text="Chunk with uncomputable embedding.",
            source_filename="paper.pdf",
            page_number=1,
            score=0.033,         # RRF score
            dense_score=-1.0,    # Uncomputable sentinel
        )
        assert _chunk_semantic_score(chunk, use_dense_score=True) is None

    def test_hybrid_invalid_dense_score_returns_none(self):
        """Invalid dense_score types return None and do not crash."""
        chunk = _make_chunk(
            text="Chunk with malformed score.",
            source_filename="paper.pdf",
            page_number=1,
            score=0.55,
            dense_score="not_a_float",
        )
        assert _chunk_semantic_score(chunk, use_dense_score=True) is None

    def test_nan_dense_score_returns_none(self):
        """NaN dense_score is rejected by math.isfinite() and returns None."""
        import math as _math
        chunk = _make_chunk(
            text="Some relevant passage about depth estimation.",
            source_filename="paper.pdf",
            page_number=2,
            score=0.55,
            dense_score=float("nan"),
        )
        score_val = _chunk_semantic_score(chunk, use_dense_score=True)
        assert score_val is None

        # Verify full pipeline treats NaN chunk as unscored and abstains
        verdict = check_evidence_sufficiency(
            [chunk],
            query="depth estimation sensor",
            score_threshold=0.38,
            use_dense_score=True,
        )
        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert verdict.unscored_chunk_count == 1

    def test_positive_infinity_dense_score_returns_none(self):
        """Positive infinity dense_score is rejected by math.isfinite() and returns None."""
        chunk = _make_chunk(
            text="Some relevant passage about LiDAR calibration.",
            source_filename="paper.pdf",
            page_number=3,
            score=0.99,
            dense_score=float("inf"),
        )
        score_val = _chunk_semantic_score(chunk, use_dense_score=True)
        assert score_val is None

        verdict = check_evidence_sufficiency(
            [chunk],
            query="LiDAR calibration sensor",
            score_threshold=0.38,
            use_dense_score=True,
        )
        assert verdict.sufficient is False
        assert verdict.unscored_chunk_count == 1

    def test_negative_infinity_dense_score_returns_none(self):
        """Negative infinity dense_score is rejected by math.isfinite() and returns None."""
        chunk = _make_chunk(
            text="Passage about pothole depth measurement using stereo vision.",
            source_filename="paper.pdf",
            page_number=4,
            score=0.70,
            dense_score=float("-inf"),
        )
        score_val = _chunk_semantic_score(chunk, use_dense_score=True)
        assert score_val is None

        verdict = check_evidence_sufficiency(
            [chunk],
            query="pothole depth stereo vision",
            score_threshold=0.38,
            use_dense_score=True,
        )
        assert verdict.sufficient is False
        assert verdict.unscored_chunk_count == 1

    def test_valid_cosine_score_passes_isfinite_check(self):
        """Regression: a valid cosine score in (0.0, 1.0) is not rejected by math.isfinite()."""
        for valid_score in [0.0, 0.38, 0.50, 0.99, 1.0, -0.5]:
            chunk = _make_chunk(
                text="Passage mentioning ZED 2i stereo camera for depth.",
                source_filename="paper.pdf",
                page_number=1,
                score=0.01,
                dense_score=valid_score,
            )
            result = _chunk_semantic_score(chunk, use_dense_score=True)
            assert result == valid_score, (
                f"Expected {valid_score} to pass isfinite check, got {result}"
            )
    def test_dense_only_retrieval_uses_score(self):
        """When use_dense_score=False, chunk.score is used directly (backward compat)."""
        chunk = _make_chunk(
            text="Chunk in dense-only retrieval.",
            source_filename="paper.pdf",
            page_number=1,
            score=0.55,
            dense_score=None,
        )
        assert _chunk_semantic_score(chunk, use_dense_score=False) == 0.55


# ==============================================================================
# 4. QAEngine Score Preservation & Mapping Hardening Tests
# ==============================================================================

class TestQAEngineScorePreservation:
    def test_hybrid_search_preserves_dense_bm25_and_rrf_scores(self):
        """QAEngine.search_hybrid() outputs chunks with score (RRF), dense_score, and bm25_score."""
        mock_model = MockEmbeddingModel(dim=8)
        chunks = [
            Chunk(
                text="The Ouster OS1-32 LiDAR sensor produces high-accuracy point clouds.",
                source_filename="karukayil.pdf",
                page_number=5,
                document_id="k2024",
                chunk_index=0,
                chunk_id="k_c0",
            ),
            Chunk(
                text="Autonomous navigation uses visual SLAM algorithms.",
                source_filename="nav.pdf",
                page_number=2,
                document_id="nav2023",
                chunk_index=1,
                chunk_id="nav_c1",
            ),
            Chunk(
                text="Deep learning models evaluate pavement distress from aerial images.",
                source_filename="survey.pdf",
                page_number=1,
                document_id="survey2022",
                chunk_index=2,
                chunk_id="surv_c2",
            ),
        ]

        engine = QAEngine(chunks, mock_model)
        results = engine.search_hybrid("Ouster OS1-32 LiDAR", k=2)

        assert len(results) >= 1
        top = results[0]
        # 1. RRF score in top.score
        assert hasattr(top, "score")
        assert 0.0 < top.score < 0.1
        assert hasattr(top, "rrf_score")
        assert top.rrf_score == top.score

        # 2. Dense score preserved and is a float
        assert hasattr(top, "dense_score")
        assert isinstance(top.dense_score, float)

        # 3. BM25 score preserved and is a float
        assert hasattr(top, "bm25_score")
        assert isinstance(top.bm25_score, float)

    def test_lexical_only_candidate_computes_dense_score(self):
        """Lexical candidates absent from dense top-k have dense_score computed via embedding dot product."""
        mock_model = MockEmbeddingModel(dim=8)
        c1 = Chunk(
            text="Common word alpha beta gamma delta epsilon.",
            source_filename="p1.pdf",
            page_number=1,
            document_id="p1",
            chunk_index=0,
            chunk_id="c1",
        )
        c2 = Chunk(
            text="Specific rare keyword XYZ999 unique identifier.",
            source_filename="p2.pdf",
            page_number=2,
            document_id="p2",
            chunk_index=1,
            chunk_id="c2",
        )

        engine = QAEngine([c1, c2], mock_model)

        # Force dense_k=1 so c2 is NOT in the dense top-1
        results = engine.search_hybrid("XYZ999", k=2, dense_k=1, lexical_k=2)

        # Find c2 in results
        c2_result = next((r for r in results if r.chunk_id == "c2"), None)
        assert c2_result is not None
        assert hasattr(c2_result, "dense_score")
        assert c2_result.dense_score is not None
        assert isinstance(c2_result.dense_score, float)

    def test_1d_and_2d_query_vector_support(self):
        """_compute_chunk_dense_similarity supports both 1D and 2D query vectors."""
        mock_model = MockEmbeddingModel(dim=8)
        c = Chunk("Sample passage for vector shape testing.", "test.pdf", 1, "d1", 0, "c_test")
        engine = QAEngine([c], mock_model)

        # 2D query vector (1, 8)
        vec_2d = np.ones((1, 8), dtype=np.float32) / np.sqrt(8.0)
        sim_2d = engine._compute_chunk_dense_similarity("c_test", vec_2d)
        assert sim_2d is not None
        assert isinstance(sim_2d, float)

        # 1D query vector (8,)
        vec_1d = np.ones((8,), dtype=np.float32) / np.sqrt(8.0)
        sim_1d = engine._compute_chunk_dense_similarity("c_test", vec_1d)
        assert sim_1d is not None
        assert isinstance(sim_1d, float)

        # Results must be identical
        assert pytest.approx(sim_1d, rel=1e-6) == sim_2d

    def test_dimension_mismatch_returns_none(self):
        """Query vector with wrong dimensionality returns None."""
        mock_model = MockEmbeddingModel(dim=8)
        c = Chunk("Sample passage for dimension testing.", "test.pdf", 1, "d1", 0, "c_dim")
        engine = QAEngine([c], mock_model)

        wrong_dim_vec = np.ones((4,), dtype=np.float32)
        sim = engine._compute_chunk_dense_similarity("c_dim", wrong_dim_vec)
        assert sim is None

    def test_ambiguous_chunk_id_with_different_text_returns_none(self):
        """Duplicate chunk IDs with differing text are marked ambiguous and return None."""
        mock_model = MockEmbeddingModel(dim=8)
        c1 = Chunk("First text for colliding ID.", "test.pdf", 1, "d1", 0, "collide_id")
        c2 = Chunk("Completely different second text.", "test.pdf", 2, "d1", 1, "collide_id")

        engine = QAEngine([c1, c2], mock_model)
        assert "collide_id" in engine._ambiguous_chunk_ids

        vec = np.ones((8,), dtype=np.float32) / np.sqrt(8.0)
        sim = engine._compute_chunk_dense_similarity("collide_id", vec)
        assert sim is None

    def test_duplicate_chunk_id_with_identical_text_maps_cleanly(self):
        """Duplicate chunk IDs with identical text are not flagged as ambiguous."""
        mock_model = MockEmbeddingModel(dim=8)
        c1 = Chunk("Identical duplicate text.", "test.pdf", 1, "d1", 0, "dup_id")
        c2 = Chunk("Identical duplicate text.", "test.pdf", 1, "d1", 0, "dup_id")

        engine = QAEngine([c1, c2], mock_model)
        assert "dup_id" not in engine._ambiguous_chunk_ids

        vec = np.ones((8,), dtype=np.float32) / np.sqrt(8.0)
        sim = engine._compute_chunk_dense_similarity("dup_id", vec)
        assert sim is not None
        assert isinstance(sim, float)

    def test_unmapped_chunk_id_returns_none_dense_score(self):
        """An invalid/unmapped chunk ID returns None rather than silently using 0.0."""
        mock_model = MockEmbeddingModel(dim=8)
        engine = QAEngine([], mock_model)
        sim = engine._compute_chunk_dense_similarity("non_existent_chunk", query_embedding=None)
        assert sim is None


# ==============================================================================
# 5. Multi-Document Required Source Coverage Tests (Phase 1.4)
# ==============================================================================

class TestRequiredSourceCoverage:
    def test_missing_required_source_causes_abstention(self):
        """When a required source document has 0 above-threshold chunks, sufficiency fails."""
        c1 = _make_chunk("Evidence for paper 1 with high relevance score.", "paper1.pdf", 1, dense_score=0.55)
        c2 = _make_chunk("Another chunk for paper 1.", "paper1.pdf", 2, dense_score=0.50)

        # paper2.pdf is required but missing from retrieved evidence
        verdict = check_evidence_sufficiency(
            chunks=[c1, c2],
            query="Compare paper1 and paper2 methodology.",
            score_threshold=0.38,
            use_dense_score=True,
            required_source_filenames=["paper1.pdf", "paper2.pdf"],
        )
        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert verdict.missing_sources == ["paper2.pdf"]
        assert "Required source document(s) missing" in verdict.reason
        assert "paper2.pdf" in verdict.reason

    def test_all_required_sources_represented_succeeds(self):
        """When all required source documents have above-threshold chunks, sufficiency succeeds."""
        c1 = _make_chunk("Sensing modality details for Butt paper.", "butt.pdf", 1, dense_score=0.52)
        c2 = _make_chunk("Stereo vision sensor details for Wang paper.", "wang.pdf", 2, dense_score=0.48)
        c3 = _make_chunk("Time-of-flight sensor details for Ponnusamy paper.", "ponnusamy.pdf", 3, dense_score=0.49)

        verdict = check_evidence_sufficiency(
            chunks=[c1, c2, c3],
            query="Compare depth sensing across Butt, Wang, and Ponnusamy.",
            score_threshold=0.38,
            use_dense_score=True,
            required_source_filenames=["butt.pdf", "wang.pdf", "ponnusamy.pdf"],
        )
        assert verdict.sufficient is True
        assert verdict.should_abstain is False
        assert verdict.missing_sources == []
        assert "All 3 required source(s) represented" in verdict.reason

    def test_required_sources_backwards_compatible_when_none(self):
        """When required_source_filenames is None, missing_sources is empty and check passes."""
        c1 = _make_chunk("Single paper evidence passage.", "single.pdf", 1, dense_score=0.50)
        verdict = check_evidence_sufficiency(
            chunks=[c1],
            query="Single paper question.",
            score_threshold=0.38,
            use_dense_score=True,
            required_source_filenames=None,
        )
        assert verdict.sufficient is True
        assert verdict.missing_sources == []


# ==============================================================================
# 6. Cross-Chunk Entity Validation Tests (Phase 1.4 / Q11)
# ==============================================================================

class TestCrossChunkEntityValidation:
    def test_q11_style_distributed_evidence_passes(self):
        """
        When numerical metrics are in one chunk (c1) and equipment identifier is in another (c2)
        from the same document, entity validation succeeds as long as c1 exceeds threshold.
        """
        # c1 has the numerical results and high score, but not the camera model
        c1 = _make_chunk(
            text="Relative errors of 12.53% depth and 18.19% surface area at 40 km/h, reducing cost by 90%.",
            source_filename="wang_isprs.pdf",
            page_number=1,
            dense_score=0.49,
        )
        # c2 has the equipment name (ZED 2i) from the same paper, score below threshold
        c2 = _make_chunk(
            text="A ZED 2i binocular camera was used to build the vehicle-mounted acquisition system.",
            source_filename="wang_isprs.pdf",
            page_number=2,
            dense_score=0.32,
        )

        verdict = check_evidence_sufficiency(
            chunks=[c1, c2],
            query="In Wang et al., what relative errors are achieved at 40 km/h with cost reduction?",
            score_threshold=0.38,
            use_dense_score=True,
            scope_filenames=["wang_isprs.pdf"],
            critical_entities=["ZED 2i"],
        )
        assert verdict.sufficient is True
        assert verdict.should_abstain is False
        assert verdict.missing_entities == []
        assert "All 1 required entity/entities confirmed present" in verdict.reason

    def test_entity_in_unrelated_paper_cannot_satisfy_requirement(self):
        """An entity appearing in an unrelated paper cannot satisfy the target paper's requirement."""
        # Target paper has high score but lacks the entity
        c1 = _make_chunk(
            text="Ponnusamy paper reports ground-truth depth measurements using a tripod.",
            source_filename="ponnusamy.pdf",
            page_number=3,
            dense_score=0.55,
        )
        # Unrelated paper has the entity 'UAV'
        c2 = _make_chunk(
            text="Rauf Butt paper mentions UAV aerial survey in future work.",
            source_filename="butt.pdf",
            page_number=7,
            dense_score=0.45,
        )

        # Scoping is active for ponnusamy.pdf; butt.pdf is excluded
        verdict = check_evidence_sufficiency(
            chunks=[c1, c2],
            query="What flight altitude was used for UAV aerial collection in Ponnusamy?",
            score_threshold=0.38,
            use_dense_score=True,
            scope_filenames=["ponnusamy.pdf"],
            critical_entities=["UAV"],
        )
        assert verdict.sufficient is False
        assert verdict.should_abstain is True
        assert verdict.missing_entities == ["UAV"]

    def test_unanswerable_questions_still_abstain_under_cross_chunk(self):
        """Q16, Q17, Q18 must still abstain under cross-chunk entity checking."""
        # Q16 mock: Ponnusamy passages with no UAV/drone
        p_c1 = _make_chunk("Basler ToF camera on tripod.", "ponnusamy.pdf", 3, dense_score=0.46)
        p_c2 = _make_chunk("Depth error analysis ±3 mm.", "ponnusamy.pdf", 4, dense_score=0.42)
        v16 = check_evidence_sufficiency(
            chunks=[p_c1, p_c2],
            query="What flight altitude was used for UAV drone in Ponnusamy?",
            score_threshold=0.38,
            use_dense_score=True,
            scope_filenames=["ponnusamy.pdf"],
            critical_entities=["UAV", "drone"],
        )
        assert v16.should_abstain is True
        assert "UAV" in v16.missing_entities

        # Q18 mock: Dehao Wang passages with high score on YOLOv8, but no YOLOv10-X or COCO 2017
        w_c1 = _make_chunk("ACSH-YOLOv8 pothole area estimation.", "wang2025.pdf", 3, dense_score=0.75)
        w_c2 = _make_chunk("Clear Road and Dark Road evaluation.", "wang2025.pdf", 8, dense_score=0.68)
        v18 = check_evidence_sufficiency(
            chunks=[w_c1, w_c2],
            query="How does ACSH-YOLOv8 compare against YOLOv10-X on COCO 2017?",
            score_threshold=0.38,
            use_dense_score=True,
            scope_filenames=["wang2025.pdf"],
            critical_entities=["YOLOv10-X", "COCO 2017"],
        )
        assert v18.should_abstain is True
        assert "YOLOv10-X" in v18.missing_entities
        assert "COCO 2017" in v18.missing_entities


# ==============================================================================
# 7. QAEngine Source-Aware Search Tests (Phase 1.4)
# ==============================================================================

class TestQAEngineSourceAwareSearch:
    def test_search_source_aware_allocates_per_source(self):
        """search_source_aware retrieves balanced candidates across specified papers."""
        mock_model = MockEmbeddingModel(dim=8)
        # Create 10 chunks for paper A (dominant), and 2 chunks for paper B (short)
        chunks_a = [
            Chunk(f"Paper A long passage {i}.", "paperA.pdf", i, "dA", i, f"cA_{i}")
            for i in range(10)
        ]
        chunks_b = [
            Chunk(f"Paper B critical passage {i}.", "paperB.pdf", i, "dB", i, f"cB_{i}")
            for i in range(2)
        ]

        engine = QAEngine(chunks_a + chunks_b, mock_model)

        results = engine.search_source_aware(
            query="test query",
            sources=["paperA.pdf", "paperB.pdf"],
            k_per_source=2,
            mode="dense",
        )
        assert len(results) == 4
        sources_found = [c.source_filename for c in results]
        assert sources_found.count("paperA.pdf") == 2
        assert sources_found.count("paperB.pdf") == 2
