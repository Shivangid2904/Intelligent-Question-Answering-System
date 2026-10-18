"""
Tests for Phase 2.1A — Production Evidence-Gate Integration

Verifies:
  V1: App calls search_hybrid() rather than search() in its query path
  V2: check_evidence_sufficiency() receives use_dense_score=True
  V3: Only qualifying chunks (dense_score >= 0.38) reach generate_answer()
  V4: A failed evidence verdict results in zero LLM calls
  V5: Missing, None, NaN, +/-inf, non-numeric, and sentinel scores are rejected
  V6: Displayed and persisted relevance scores use dense_score, not RRF score
  V7: Original engine.search() interface remains compatible
  R1: No qualifying chunks results in abstention without fallback (no or results[:3])
  R2: Infinities and non-numeric scores in hybrid candidates do not qualify or crash
  R3: Unqualified chunk with high keyword overlap cannot become best supporting chunk
  R4: Abstention verdict suppresses supporting UI and history even if chunks pass threshold
  R5: LLM prompt in src/llm_engine.py contains approved recency and grounding instructions
"""

import ast
import inspect
import math
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from app import (
    _extract_dense_score,
    _get_display_score,
    _is_qualifying_chunk,
)
from src.citation import check_evidence_sufficiency, format_citation, format_citations
from src.document_processor import Chunk
from src.qa_engine import QAEngine
from tests.test_qa_engine import MockEmbeddingModel


def _make_dummy_chunk(
    text: str,
    source: str = "paper.pdf",
    page: int = 1,
    chunk_id: str = "c1",
    score: float = 0.02,
    dense_score: Any = 0.5,
    bm25_score: Any = 5.0,
) -> Chunk:
    """Helper to build a Chunk instance with explicit score attributes."""
    c = Chunk(
        text=text,
        source_filename=source,
        page_number=page,
        document_id=source.replace(".pdf", ""),
        chunk_index=0,
        chunk_id=chunk_id,
        pos_tags=[],
    )
    c.score = score
    if dense_score is not None:
        c.dense_score = dense_score
    if bm25_score is not None:
        c.bm25_score = bm25_score
    return c


def run_app_query_pipeline(
    engine: Any,
    query: str,
    ollama_model: str = "llama3.2:1b",
    generate_answer_fn: Any = None,
    check_evidence_sufficiency_fn: Any = None,
) -> dict:
    """
    Execution harness exercising the exact production query logic from app.py.

    Emulates:
      - search_hybrid(query, k=10)
      - check_evidence_sufficiency(..., use_dense_score=True)
      - qualifying_chunks filtering
      - has_supporting_evidence gating
      - best chunk selection
      - conditional LLM invocation
      - meta pills & supporting passages generation
      - history entry construction
    """
    results = engine.search_hybrid(query, k=10)

    if not results:
        return {
            "status": "no_results",
            "llm_called": False,
            "has_supporting_evidence": False,
            "passages_for_llm": [],
            "history_entry": None,
        }

    gate_fn = check_evidence_sufficiency_fn or check_evidence_sufficiency
    verdict = gate_fn(
        results,
        query=query,
        score_threshold=0.38,
        min_content_word_overlap=1,
        min_sufficient_chunks=1,
        use_dense_score=True,
    )

    qualifying_chunks = [r for r in results if _is_qualifying_chunk(r, 0.38)]
    has_supporting_evidence = not verdict.should_abstain and bool(qualifying_chunks)

    import re as _re
    _query_words = [_re.sub(r'[^\w]', '', w) for w in query.lower().split()
                    if _re.sub(r'[^\w]', '', w)]
    _content_words = [w for w in _query_words if len(w) > 3] or _query_words

    if has_supporting_evidence:
        best = qualifying_chunks[0]
        best_combined = -1.0
        for r in qualifying_chunks:
            mc = sum(word in r.text.lower() for word in _content_words)
            cs = _get_display_score(r) + 0.15 * mc
            if cs > best_combined:
                best_combined = cs
                best = r
    else:
        best = None

    llm_called = False
    llm_answer = ""
    answer_source = ""
    passages_for_llm: List[Any] = []

    if not has_supporting_evidence:
        from src.citation import build_abstention_message
        abstention_msg = build_abstention_message(verdict, query)
        llm_answer = abstention_msg
        answer_source = "abstention"
    else:
        passages_for_llm = qualifying_chunks
        llm_called = True
        answer_source = "ollama"
        if generate_answer_fn:
            llm_answer = "".join(list(generate_answer_fn(query, passages_for_llm, model=ollama_model)))
        else:
            llm_answer = "Mock answer"

    meta_pills = None
    if has_supporting_evidence and best is not None:
        meta_pills = {
            "score": round(_get_display_score(best), 3),
            "source": best.source,
            "citations": format_citations(qualifying_chunks[:3]),
        }

    supporting_passages = []
    if has_supporting_evidence:
        supporting_passages = [
            {
                "citation": format_citation(r),
                "score": round(_get_display_score(r), 3),
                "text": r.text,
            }
            for r in qualifying_chunks
        ]

    if has_supporting_evidence and best is not None:
        filtered_passages = [
            {"text": r.text, "score": round(_get_display_score(r), 3)}
            for r in qualifying_chunks
        ]
        source_val = best.source
        score_val = float(_get_display_score(best))
        citations_val = verdict.citations
    else:
        filtered_passages = []
        source_val = ""
        score_val = 0.0
        citations_val = []

    history_entry = {
        "question": query,
        "answer": llm_answer if isinstance(llm_answer, str) else "",
        "answer_source": answer_source,
        "source": source_val,
        "score": score_val,
        "citations": citations_val,
        "passages": filtered_passages,
    }

    return {
        "status": "success",
        "verdict": verdict,
        "qualifying_chunks": qualifying_chunks,
        "has_supporting_evidence": has_supporting_evidence,
        "best": best,
        "llm_called": llm_called,
        "passages_for_llm": passages_for_llm,
        "meta_pills": meta_pills,
        "supporting_passages": supporting_passages,
        "history_entry": history_entry,
    }


# ==============================================================================
# V1: Application calls search_hybrid() rather than search()
# ==============================================================================
def test_v1_app_calls_search_hybrid_in_query_path():
    """Verify that app.py invokes engine.search_hybrid() and passes k=10 in the query path."""
    import app
    app_source = inspect.getsource(app)
    tree = ast.parse(app_source)

    # Walk AST to find calls to engine methods in the query handling block
    hybrid_calls = []
    search_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "search_hybrid":
                hybrid_calls.append(node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "search":
                # Check if called on engine
                if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "engine":
                    search_calls.append(node)

    assert len(hybrid_calls) >= 1, "Expected app.py to call search_hybrid() on engine"
    assert len(search_calls) == 0, f"app.py should NOT call engine.search() in production query path, found: {search_calls}"

    # Also test via execution harness with mock engine
    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = []
    run_app_query_pipeline(mock_engine, "What is attention?")
    mock_engine.search_hybrid.assert_called_once_with("What is attention?", k=10)
    mock_engine.search.assert_not_called()


# ==============================================================================
# V2: check_evidence_sufficiency() receives use_dense_score=True
# ==============================================================================
def test_v2_evidence_sufficiency_receives_dense_score_flag():
    """Verify that app.py passes use_dense_score=True to check_evidence_sufficiency()."""
    import app
    app_source = inspect.getsource(app)
    assert "use_dense_score=True" in app_source, "app.py must pass use_dense_score=True to check_evidence_sufficiency"

    # AST verification of app.py to confirm check_evidence_sufficiency call syntax
    tree = ast.parse(app_source)
    found_ast_flag = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "check_evidence_sufficiency":
            for kw in node.keywords:
                if kw.arg == "use_dense_score" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    found_ast_flag = True
    assert found_ast_flag, "app.py AST must contain check_evidence_sufficiency(..., use_dense_score=True)"

    # Execution harness verification
    mock_engine = MagicMock()
    chunk = _make_dummy_chunk("Transformer models use self-attention mechanism.", dense_score=0.6)
    mock_engine.search_hybrid.return_value = [chunk]

    spy_check = MagicMock(wraps=check_evidence_sufficiency)
    res = run_app_query_pipeline(mock_engine, "self-attention mechanism", check_evidence_sufficiency_fn=spy_check)
    spy_check.assert_called_once()
    _, kwargs = spy_check.call_args
    assert kwargs.get("use_dense_score") is True
    assert kwargs.get("score_threshold") == 0.38


# ==============================================================================
# V3: Only qualifying chunks (dense_score >= 0.38) reach generate_answer()
# ==============================================================================
def test_v3_only_qualifying_chunks_reach_llm():
    """Verify that chunks below 0.38 are excluded from passages passed to LLM."""
    c1 = _make_dummy_chunk("Attention mechanism in transformers allows routing.", dense_score=0.65)
    c2 = _make_dummy_chunk("Self-attention computes key query value dot products.", dense_score=0.42)
    c3 = _make_dummy_chunk("Weakly relevant passage discussing natural language processing.", dense_score=0.35)
    c4 = _make_dummy_chunk("Unrelated chunk discussing biology and cells.", dense_score=0.12)

    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c1, c2, c3, c4]

    mock_gen = MagicMock(return_value=iter(["Answer token"]))
    res = run_app_query_pipeline(mock_engine, "attention mechanism in transformers", generate_answer_fn=mock_gen)

    assert res["llm_called"] is True
    assert len(res["passages_for_llm"]) == 2
    assert c1 in res["passages_for_llm"]
    assert c2 in res["passages_for_llm"]
    assert c3 not in res["passages_for_llm"]
    assert c4 not in res["passages_for_llm"]


# ==============================================================================
# V4: A failed evidence verdict results in zero LLM calls
# ==============================================================================
def test_v4_failed_verdict_results_in_zero_llm_calls():
    """When evidence gate fails (should_abstain=True), zero LLM calls must occur."""
    # Chunk with low dense score
    c = _make_dummy_chunk("Completely unrelated text about cooking recipes.", dense_score=0.20)
    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c]

    mock_gen = MagicMock()
    res = run_app_query_pipeline(mock_engine, "What is the LiDAR frame rate?", generate_answer_fn=mock_gen)

    assert res["llm_called"] is False
    mock_gen.assert_not_called()
    assert res["has_supporting_evidence"] is False
    assert res["history_entry"]["answer_source"] == "abstention"
    ans_lower = res["history_entry"]["answer"].lower()
    assert "cannot answer" in ans_lower or "unavailable" in ans_lower or "insufficient" in ans_lower


# ==============================================================================
# V5: Missing, None, NaN, +/-inf, non-numeric, and sentinel scores are rejected
# ==============================================================================
def test_v5_non_finite_and_invalid_dense_scores_rejected():
    """Verify that _extract_dense_score and _is_qualifying_chunk reject all non-finite/invalid values."""
    # Test helper directly
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=None)) is None
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=float("nan"))) is None
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=float("inf"))) is None
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=float("-inf"))) is None
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score="invalid_string")) is None
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=-2.0)) is None  # Sentinel
    assert _extract_dense_score(_make_dummy_chunk("test", dense_score=-1.0)) is None  # Sentinel boundary
    assert _extract_dense_score({"other_key": 0.5}) is None  # Missing attribute/key

    # None of these should qualify
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=None))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=float("nan")))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=float("inf")))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=float("-inf")))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score="invalid_string"))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=-2.0))

    # Valid finite score >= 0.38 qualifies
    assert _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=0.38))
    assert _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=0.75))
    assert not _is_qualifying_chunk(_make_dummy_chunk("test", dense_score=0.379))


# ==============================================================================
# V6: Displayed and persisted relevance scores use dense_score, not RRF score
# ==============================================================================
def test_v6_displayed_and_persisted_scores_use_dense_score():
    """Verify that UI badges and history use dense_score (cosine), not RRF rank score."""
    c = _make_dummy_chunk(
        "Transformers employ multi-head attention blocks.",
        score=0.0185,       # RRF fused score (small float)
        dense_score=0.642,  # Cosine similarity
    )
    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c]

    res = run_app_query_pipeline(mock_engine, "multi-head attention blocks")

    assert res["has_supporting_evidence"] is True
    # Meta pills score
    assert res["meta_pills"]["score"] == 0.642
    # Supporting passages score
    assert res["supporting_passages"][0]["score"] == 0.642
    # History entry score
    assert res["history_entry"]["score"] == 0.642
    assert res["history_entry"]["passages"][0]["score"] == 0.642
    # Must NOT equal the RRF score
    assert res["history_entry"]["score"] != 0.0185


# ==============================================================================
# V7: Original engine.search() interface remains compatible
# ==============================================================================
def test_v7_legacy_engine_search_interface_compatibility():
    """Verify that engine.search() still works across dense, lexical, and hybrid modes."""
    chunks = [
        _make_dummy_chunk("The Ouster OS1-32 LiDAR sensor provides 32-beam point clouds.", chunk_id="c1", source="karukayil.pdf"),
        _make_dummy_chunk("Deep learning models evaluate pavement distress from RGB images.", chunk_id="c2", source="survey.pdf"),
        _make_dummy_chunk("iPhone 14 Pro equipped with LiDAR was tested in urban environments.", chunk_id="c3", source="butt.pdf"),
    ]
    model = MockEmbeddingModel(dim=8)
    engine = QAEngine(chunks, model)

    # Legacy default (dense)
    res_dense = engine.search("Ouster LiDAR", k=2)
    assert len(res_dense) > 0
    assert hasattr(res_dense[0], "score")

    # Explicit dense
    res_dense_exp = engine.search("Ouster LiDAR", k=2, mode="dense")
    assert len(res_dense_exp) > 0

    # Explicit lexical
    res_lex = engine.search("Ouster OS1-32", k=2, mode="lexical")
    assert len(res_lex) > 0

    # Explicit hybrid
    res_hyb = engine.search("Ouster OS1-32", k=2, mode="hybrid")
    assert len(res_hyb) > 0
    assert hasattr(res_hyb[0], "dense_score")


# ==============================================================================
# R1: No qualifying chunks results in abstention without fallback
# ==============================================================================
def test_r1_no_qualifying_chunks_abstains_without_fallback():
    """Verify that when all chunks have dense_score < 0.38, no fallback occurs and LLM is not called."""
    chunks = [
        _make_dummy_chunk("Marginally related text alpha.", dense_score=0.30),
        _make_dummy_chunk("Marginally related text beta.", dense_score=0.25),
        _make_dummy_chunk("Marginally related text gamma.", dense_score=0.15),
    ]
    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = chunks

    mock_gen = MagicMock()
    res = run_app_query_pipeline(mock_engine, "quantum physics in semiconductors", generate_answer_fn=mock_gen)

    # Unsafe fallback or results[:3] MUST NOT trigger
    assert res["has_supporting_evidence"] is False
    assert res["llm_called"] is False
    mock_gen.assert_not_called()
    assert res["passages_for_llm"] == []
    assert res["meta_pills"] is None
    assert res["supporting_passages"] == []
    assert res["history_entry"]["source"] == ""
    assert res["history_entry"]["score"] == 0.0
    assert res["history_entry"]["passages"] == []
    assert res["history_entry"]["citations"] == []


# ==============================================================================
# R2: Infinities and non-numeric scores in hybrid candidates do not qualify or crash
# ==============================================================================
def test_r2_infinities_and_non_numeric_scores_in_hybrid_results():
    """Verify that chunks with +inf, -inf, or uncomputable sentinels do not cause crashes or qualify."""
    c_pos_inf = _make_dummy_chunk("Malicious positive infinity chunk.", dense_score=float("inf"))
    c_neg_inf = _make_dummy_chunk("Malicious negative infinity chunk.", dense_score=float("-inf"))
    c_sentinel = _make_dummy_chunk("Lexical-only uncomputable chunk.", dense_score=-2.0)
    c_valid = _make_dummy_chunk("Legitimate qualifying chunk with high similarity.", dense_score=0.55)

    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c_pos_inf, c_neg_inf, c_sentinel, c_valid]

    res = run_app_query_pipeline(mock_engine, "legitimate qualifying chunk")

    assert res["has_supporting_evidence"] is True
    # Only c_valid qualifies
    assert res["qualifying_chunks"] == [c_valid]
    assert res["passages_for_llm"] == [c_valid]
    assert res["best"] == c_valid


# ==============================================================================
# R3: Unqualified chunk with high keyword overlap cannot become best supporting chunk
# ==============================================================================
def test_r3_unqualified_chunk_with_highest_keyword_overlap_cannot_become_best():
    """
    An unqualified chunk with high keyword matches must NOT be chosen as best.
    best must be selected strictly from qualifying_chunks.
    """
    # c_unqual matches all words in query ("camera", "lidar", "sensor", "calibration")
    # but its dense similarity is below threshold (0.28 < 0.38)
    c_unqual = _make_dummy_chunk(
        "camera lidar sensor calibration setup description without results",
        source="unqualified_paper.pdf",
        dense_score=0.28,
    )
    # c_qual matches only 1 keyword ("calibration"), but has high dense score (0.50 >= 0.38)
    c_qual = _make_dummy_chunk(
        "experimental calibration error measured at 1.2 mm across tests",
        source="target_paper.pdf",
        dense_score=0.50,
    )

    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c_unqual, c_qual]

    res = run_app_query_pipeline(mock_engine, "camera lidar sensor calibration")

    assert res["has_supporting_evidence"] is True
    # best MUST be c_qual, NOT c_unqual
    assert res["best"] == c_qual
    assert res["meta_pills"]["source"] == "target_paper.pdf"
    assert res["history_entry"]["source"] == "target_paper.pdf"
    assert res["history_entry"]["score"] == 0.50


# ==============================================================================
# R4: Abstention verdict suppresses supporting UI and history even if chunks pass threshold
# ==============================================================================
def test_r4_abstention_verdict_with_qualifying_dense_score_chunks():
    """
    When verdict.should_abstain=True (e.g. content word overlap is 0),
    has_supporting_evidence must be False.
    Threshold-passing chunks must NEVER be displayed or persisted as supporting evidence.
    """
    # Chunk has high dense score (0.60), but zero content word overlap with query
    c_high_dense = _make_dummy_chunk(
        "The quick brown fox jumps over the lazy dog.",
        source="fox_paper.pdf",
        dense_score=0.60,
    )

    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [c_high_dense]

    # Query with disjoint vocabulary
    res = run_app_query_pipeline(mock_engine, "semiconductor quantum computing qubits")

    # Gate should abstain due to zero content word overlap
    assert res["verdict"].should_abstain is True
    # has_supporting_evidence must be False
    assert res["has_supporting_evidence"] is False
    assert res["llm_called"] is False
    assert res["best"] is None
    assert res["meta_pills"] is None
    assert res["supporting_passages"] == []
    # History must have safe blanks, not the 0.60 chunk!
    assert res["history_entry"]["source"] == ""
    assert res["history_entry"]["score"] == 0.0
    assert res["history_entry"]["passages"] == []
    assert res["history_entry"]["citations"] == []


# ==============================================================================
# R5: LLM prompt in src/llm_engine.py contains approved recency and grounding instructions
# ==============================================================================
def test_r5_llm_prompt_contains_approved_recency_instruction():
    """Verify that src/llm_engine.py contains the exact approved recency instruction and rules."""
    from src.llm_engine import generate_answer
    src = inspect.getsource(generate_answer)

    expected_instruction = (
        "Answer using only the supplied passages. Do not refuse solely because a cited paper's "
        "publication date is newer than your training data. If the passages directly support the answer, "
        "answer from them. If they do not provide sufficient evidence, state that the information is unavailable."
    )
    assert expected_instruction in src, "llm_engine.py prompt must contain the exact approved recency instruction"

    # Verify citation and anti-speculation rules are preserved
    assert "[paper.pdf, p. 4]" in src, "llm_engine.py must preserve page-level citation format example"
    assert "Do NOT speculate, extrapolate, or assume information beyond the provided text" in src
