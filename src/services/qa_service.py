"""
IntelliAsk Research — QA Service

Encapsulates the exact question-answering flow currently embedded in app.py,
making it independently testable and callable by FastAPI endpoints without
any Streamlit, session-state, or UI rendering concerns.

Supports both:
- run_qa_query(): Non-streaming query execution returning full QAResult.
- stream_qa_query(): Server-Sent Events (SSE) progressive generator yielding
  typed events: ('metadata', payload), ('token', payload), ('final', payload),
  ('error', payload), and ('done', payload).

Design notes
------------
* Accepts an already-initialised QAEngine — does NOT construct or cache one
  internally.  The caller (endpoint dependency, test, CLI) is responsible for
  providing a ready engine or handling the "no engine" case.
* Exact replicas of the score helpers originally in app.py are placed
  here so the service is self-contained and app.py is not imported.
* LLM answer generation is injected as a callable so tests can supply a
  stub without a live Ollama server.
* Abstention gate runs BEFORE LLM invocation; if evidence is insufficient,
  abstention is returned immediately with zero LLM calls and zero token events.
* No model loading, database calls, or network requests are made at import time.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from src.citation import (
    build_abstention_message,
    check_evidence_sufficiency,
    format_citation,
    format_citations,
    EvidenceVerdict,
)
from src.llm_engine import generate_answer as _default_generate_answer


# ── Score helpers (mirror of app.py — duplicated so service has no Streamlit dep) ──

def _extract_dense_score(chunk: Any) -> Optional[float]:
    """
    Safely extract the dense cosine similarity score from a chunk.

    Returns a float strictly if the value is numeric, finite (not NaN, +inf,
    -inf), and > -1.0 (filtering out uncomputed sentinel values).  Returns
    None for missing, NaN, infinities, non-numeric values, or sentinels.
    """
    ds = (
        chunk.get("dense_score") if isinstance(chunk, dict)
        else getattr(chunk, "dense_score", None)
    )
    if ds is None:
        return None
    try:
        val = float(ds)
        if not math.isfinite(val) or val <= -1.0:
            return None
        return val
    except (TypeError, ValueError):
        return None


def _is_qualifying_chunk(chunk: Any, threshold: float = 0.38) -> bool:
    """Return True only if the chunk has a valid finite dense_score >= threshold."""
    score = _extract_dense_score(chunk)
    return score is not None and score >= threshold


def _get_display_score(chunk: Any) -> float:
    """Return the valid dense score, or 0.0 if missing / invalid."""
    score = _extract_dense_score(chunk)
    return score if score is not None else 0.0


# ── Result & Context Types ────────────────────────────────────────────────────

@dataclass
class PassageInfo:
    """A single retrieved passage with its citation and display score."""

    text: str
    citation: str
    score: float
    source_filename: str
    page_number: Optional[int]
    qualifies: bool  # True iff dense_score >= SCORE_THRESHOLD


@dataclass
class QAResult:
    """
    Structured result from run_qa_query().

    Attributes:
        query:                  The original query string.
        answer:                 Generated answer text, or the abstention message.
        answer_source:          "ollama" | "abstention" | "error".
        has_supporting_evidence: True iff verdict.should_abstain is False and
                                 at least one qualifying chunk exists.
        verdict:                Full EvidenceVerdict from check_evidence_sufficiency.
        best_source:            source_filename of the best chunk, or "" if abstaining.
        best_score:             Display score of the best chunk, or 0.0.
        citations:              Deduplicated citation list from the verdict.
        supporting_passages:    Qualifying chunks (dense_score >= threshold), up to k=10.
        all_retrieved:          All hybrid-search results with rank and qualify flag.
    """

    query: str
    answer: str
    answer_source: str  # "ollama" | "abstention" | "error"
    has_supporting_evidence: bool
    verdict: EvidenceVerdict
    best_source: str = ""
    best_score: float = 0.0
    citations: List[str] = field(default_factory=list)
    supporting_passages: List[PassageInfo] = field(default_factory=list)
    all_retrieved: List[PassageInfo] = field(default_factory=list)


@dataclass
class PreparedQAContext:
    """
    Intermediate representation of retrieval, sufficiency verdict, and ranked evidence.
    Shared between non-streaming and streaming execution paths to eliminate code duplication.
    """

    query: str
    results: List[Any]
    verdict: EvidenceVerdict
    qualifying_chunks: List[Any]
    has_supporting_evidence: bool
    best: Optional[Any]
    best_source: str
    best_score: float
    supporting_passages: List[PassageInfo]
    all_passages: List[PassageInfo]
    citations: List[str]


# ── Constants (mirrors of app.py) ────────────────────────────────────────────

SCORE_THRESHOLD: float = 0.38
RETRIEVAL_K: int = 10
MIN_CONTENT_WORD_OVERLAP: int = 1
MIN_SUFFICIENT_CHUNKS: int = 1


# ── Context Preparation Helper ───────────────────────────────────────────────

def prepare_qa_context(
    query: str,
    engine: Any,
    score_threshold: float = SCORE_THRESHOLD,
    retrieval_k: int = RETRIEVAL_K,
) -> PreparedQAContext:
    """
    Execute hybrid search, evidence sufficiency evaluation, and chunk ranking.
    Used by both non-streaming run_qa_query and streaming stream_qa_query.
    """
    query_words = [
        re.sub(r"[^\w]", "", w)
        for w in query.lower().split()
        if re.sub(r"[^\w]", "", w)
    ]
    content_words = [w for w in query_words if len(w) > 3] or query_words

    results: List[Any] = engine.search_hybrid(query, k=retrieval_k)

    if not results:
        verdict = check_evidence_sufficiency(
            [],
            query=query,
            score_threshold=score_threshold,
            min_content_word_overlap=MIN_CONTENT_WORD_OVERLAP,
            min_sufficient_chunks=MIN_SUFFICIENT_CHUNKS,
            use_dense_score=True,
        )
        return PreparedQAContext(
            query=query,
            results=[],
            verdict=verdict,
            qualifying_chunks=[],
            has_supporting_evidence=False,
            best=None,
            best_source="",
            best_score=0.0,
            supporting_passages=[],
            all_passages=[],
            citations=[],
        )

    verdict = check_evidence_sufficiency(
        results,
        query=query,
        score_threshold=score_threshold,
        min_content_word_overlap=MIN_CONTENT_WORD_OVERLAP,
        min_sufficient_chunks=MIN_SUFFICIENT_CHUNKS,
        use_dense_score=True,
    )

    qualifying_chunks: List[Any] = [
        r for r in results if _is_qualifying_chunk(r, score_threshold)
    ]
    has_supporting_evidence = not verdict.should_abstain and bool(qualifying_chunks)

    best: Optional[Any] = None
    if has_supporting_evidence:
        best = qualifying_chunks[0]
        best_combined = -1.0
        for r in qualifying_chunks:
            chunk_text = getattr(r, "text", "") if not isinstance(r, dict) else r.get("text", "")
            mc = sum(w in chunk_text.lower() for w in content_words)
            cs = _get_display_score(r) + 0.15 * mc
            if cs > best_combined:
                best_combined = cs
                best = r

    supporting_passages = _build_passage_list(qualifying_chunks, score_threshold)
    all_passages = _build_passage_list(results, score_threshold)
    citations = format_citations(qualifying_chunks[:3]) if has_supporting_evidence else []
    best_source = (
        str(getattr(best, "source", getattr(best, "source_filename", "")) or "")
        if best
        else ""
    )
    best_score = _get_display_score(best) if best else 0.0

    return PreparedQAContext(
        query=query,
        results=results,
        verdict=verdict,
        qualifying_chunks=qualifying_chunks,
        has_supporting_evidence=has_supporting_evidence,
        best=best,
        best_source=best_source,
        best_score=best_score,
        supporting_passages=supporting_passages,
        all_passages=all_passages,
        citations=citations,
    )


# ── Non-streaming service function ───────────────────────────────────────────

def run_qa_query(
    query: str,
    engine: Any,
    *,
    answer_generator: Optional[Callable[..., Generator[str, None, None]]] = None,
    ollama_model: str = "llama3.2:1b",
    score_threshold: float = SCORE_THRESHOLD,
    retrieval_k: int = RETRIEVAL_K,
) -> QAResult:
    """
    Execute the full QA pipeline for a single query against an initialised engine.

    Pipeline
    --------
    1. Hybrid search:  ``engine.search_hybrid(query, k=retrieval_k)``
    2. Evidence gate:  ``check_evidence_sufficiency(..., use_dense_score=True)``
    3. Qualifying filter: chunks with ``dense_score >= score_threshold``
    4. Best-chunk selection (dense_score + 0.15 × content-word hit count)
    5. Abstention or LLM answer generation (via injected ``answer_generator``)
    """
    if answer_generator is None:
        answer_generator = _default_generate_answer

    try:
        ctx = prepare_qa_context(
            query=query,
            engine=engine,
            score_threshold=score_threshold,
            retrieval_k=retrieval_k,
        )

        if not ctx.has_supporting_evidence:
            abstention_text = build_abstention_message(ctx.verdict, query)
            return QAResult(
                query=query,
                answer=abstention_text,
                answer_source="abstention",
                has_supporting_evidence=False,
                verdict=ctx.verdict,
                citations=[],
                supporting_passages=[],
                all_retrieved=ctx.all_passages,
            )

        passages_for_llm = ctx.qualifying_chunks
        token_gen = answer_generator(query, passages_for_llm, model=ollama_model)
        answer_tokens: List[str] = []
        try:
            for token in token_gen:
                answer_tokens.append(str(token))
        except Exception as gen_exc:
            return QAResult(
                query=query,
                answer=f"⚠️ Answer generation failed: {type(gen_exc).__name__}",
                answer_source="error",
                has_supporting_evidence=True,
                verdict=ctx.verdict,
            )
        llm_answer = "".join(answer_tokens)

        return QAResult(
            query=query,
            answer=llm_answer,
            answer_source="ollama",
            has_supporting_evidence=True,
            verdict=ctx.verdict,
            best_source=ctx.best_source,
            best_score=ctx.best_score,
            citations=ctx.citations,
            supporting_passages=ctx.supporting_passages,
            all_retrieved=ctx.all_passages,
        )

    except Exception as exc:
        return QAResult(
            query=query,
            answer=f"⚠️ Internal service error: {type(exc).__name__}",
            answer_source="error",
            has_supporting_evidence=False,
            verdict=EvidenceVerdict(
                sufficient=False,
                should_abstain=True,
                confidence=0.0,
                reason=f"Service error: {type(exc).__name__}",
            ),
        )


# ── Streaming service generator ──────────────────────────────────────────────

def stream_qa_query(
    query: str,
    engine: Any,
    *,
    answer_generator: Optional[Callable[..., Generator[str, None, None]]] = None,
    ollama_model: str = "llama3.2:1b",
    score_threshold: float = SCORE_THRESHOLD,
    retrieval_k: int = RETRIEVAL_K,
    query_id: Optional[str] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Execute streaming QA pipeline yielding typed (event_type, event_payload) tuples.

    Event Protocol
    --------------
    1. 'metadata' -> query_id, query, model, has_supporting_evidence, verdict, best_source, citations
    2. 'token'    -> incremental answer fragment (when evidence is sufficient)
    3. 'final'    -> complete answer, answer_source, citations, supporting_passages, completion_status
    4. 'error'    -> safe error name and message if LLM/retrieval generation fails
    5. 'done'     -> explicit stream termination (always the last event)

    When evidence is insufficient, 'final' is emitted with completion_status='abstention',
    and NO tokens or LLM calls are made.
    """
    if answer_generator is None:
        answer_generator = _default_generate_answer

    qid = query_id or f"q_{uuid.uuid4().hex[:12]}"

    try:
        ctx = prepare_qa_context(
            query=query,
            engine=engine,
            score_threshold=score_threshold,
            retrieval_k=retrieval_k,
        )
    except Exception as exc:
        yield (
            "error",
            {
                "error": "RetrievalError",
                "detail": f"Retrieval failed: {type(exc).__name__}",
            },
        )
        yield ("done", {"done": True})
        return

    # 1. Emit metadata event
    yield (
        "metadata",
        {
            "query_id": qid,
            "query": query,
            "model": ollama_model,
            "has_supporting_evidence": ctx.has_supporting_evidence,
            "verdict": _verdict_to_dict(ctx.verdict),
            "best_source": ctx.best_source,
            "best_score": ctx.best_score,
            "citations": ctx.citations,
        },
    )

    # 2. Abstention path: NO token events, NO LLM call
    if not ctx.has_supporting_evidence:
        abstention_text = build_abstention_message(ctx.verdict, query)
        yield (
            "final",
            {
                "query_id": qid,
                "query": query,
                "answer": abstention_text,
                "answer_source": "abstention",
                "has_supporting_evidence": False,
                "best_source": "",
                "best_score": 0.0,
                "citations": [],
                "supporting_passages": [],
                "all_retrieved": [_passage_to_dict(p) for p in ctx.all_passages],
                "verdict": _verdict_to_dict(ctx.verdict),
                "completion_status": "abstention",
            },
        )
        yield ("done", {"done": True})
        return

    # 3. Sufficient evidence: stream tokens progressively
    tokens: List[str] = []
    try:
        token_gen = answer_generator(query, ctx.qualifying_chunks, model=ollama_model)
        for token in token_gen:
            str_token = str(token)
            tokens.append(str_token)
            yield ("token", {"token": str_token})
    except Exception as gen_exc:
        yield (
            "error",
            {
                "error": "GenerationError",
                "detail": f"Answer generation failed: {type(gen_exc).__name__}",
            },
        )
        yield ("done", {"done": True})
        return

    # 4. Final summary event
    full_answer = "".join(tokens)
    yield (
        "final",
        {
            "query_id": qid,
            "query": query,
            "answer": full_answer,
            "answer_source": "ollama",
            "has_supporting_evidence": True,
            "best_source": ctx.best_source,
            "best_score": ctx.best_score,
            "citations": ctx.citations,
            "supporting_passages": [_passage_to_dict(p) for p in ctx.supporting_passages],
            "all_retrieved": [_passage_to_dict(p) for p in ctx.all_passages],
            "verdict": _verdict_to_dict(ctx.verdict),
            "completion_status": "complete",
        },
    )
    yield ("done", {"done": True})


# ── Internal serialization helpers ────────────────────────────────────────────

def _build_passage_list(
    chunks: Sequence[Any],
    score_threshold: float,
) -> List[PassageInfo]:
    """Convert raw retrieved chunks into PassageInfo dataclasses."""
    passages: List[PassageInfo] = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        source = (
            chunk.get("source_filename") or chunk.get("source", "")
            if isinstance(chunk, dict)
            else getattr(chunk, "source_filename", "") or getattr(chunk, "source", "")
        )
        page = (
            chunk.get("page_number") if isinstance(chunk, dict)
            else getattr(chunk, "page_number", None)
        )
        passages.append(
            PassageInfo(
                text=str(text),
                citation=format_citation(chunk),
                score=_get_display_score(chunk),
                source_filename=str(source),
                page_number=int(page) if page is not None else None,
                qualifies=_is_qualifying_chunk(chunk, score_threshold),
            )
        )
    return passages


def _verdict_to_dict(verdict: EvidenceVerdict) -> Dict[str, Any]:
    """Serialize an EvidenceVerdict into a clean, JSON-compatible dict."""
    return {
        "sufficient": verdict.sufficient,
        "should_abstain": verdict.should_abstain,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "top_score": verdict.top_score,
        "chunk_count": verdict.chunk_count,
        "scoped_chunk_count": verdict.scoped_chunk_count,
        "missing_entities": verdict.missing_entities,
        "unscored_chunk_count": verdict.unscored_chunk_count,
        "missing_sources": verdict.missing_sources,
        "citations": verdict.citations,
    }


def _passage_to_dict(p: PassageInfo) -> Dict[str, Any]:
    """Serialize a PassageInfo dataclass into a clean, JSON-compatible dict."""
    return {
        "text": p.text,
        "citation": p.citation,
        "score": p.score,
        "source_filename": p.source_filename,
        "page_number": p.page_number,
        "qualifies": p.qualifies,
    }
