"""
IntelliAsk Research — Citation and Evidence Sufficiency Module

Provides:
  - Page-level citation formatting:  [Filename, p. X]
  - Evidence sufficiency evaluation: determines whether retrieved chunks
    constitute sufficient grounding to generate a confident answer.
  - Abstention policy: returns a structured refusal signal when evidence
    is absent, ambiguous, or below a configurable confidence threshold.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


# ── Citation formatting ───────────────────────────────────────────────────────

def format_citation(chunk: Any) -> str:
    """
    Format a single chunk as a human-readable page-level citation.

    Format:  [Filename, p. X]

    Args:
        chunk: Chunk object or dict containing source_filename / source
               and page_number attributes.

    Returns:
        Citation string, e.g. "[attention_is_all_you_need.pdf, p. 4]"
    """
    if isinstance(chunk, dict):
        source = chunk.get("source_filename") or chunk.get("source", "Unknown")
        page = chunk.get("page_number")
    else:
        source = (
            getattr(chunk, "source_filename", None)
            or getattr(chunk, "source", "Unknown")
        )
        page = getattr(chunk, "page_number", None)

    # Use basename only (strip any directory prefix)
    source = os.path.basename(str(source))

    if page is not None:
        return f"[{source}, p. {page}]"
    return f"[{source}]"


def format_citations(chunks: Sequence[Any], deduplicate: bool = True) -> List[str]:
    """
    Format a sequence of chunks as deduplicated page-level citations.

    Args:
        chunks: Retrieved Chunk objects or dicts.
        deduplicate: If True, suppress duplicate (source, page) pairs.

    Returns:
        Ordered list of unique citation strings.
    """
    seen: set = set()
    citations: List[str] = []

    for chunk in chunks:
        citation = format_citation(chunk)
        if deduplicate:
            if citation in seen:
                continue
            seen.add(citation)
        citations.append(citation)

    return citations


def build_grounded_context(
    chunks: Sequence[Any],
    max_chunks: int = 5,
    max_chars_per_chunk: int = 800,
) -> str:
    """
    Build a grounded context string for LLM prompt injection, where each
    passage is prefixed with its page-level citation.

    Format:
        [source.pdf, p. 4]
        <passage text...>

    Args:
        chunks: Retrieved Chunk objects ordered by relevance score.
        max_chunks: Maximum number of passages to include.
        max_chars_per_chunk: Character budget per passage.

    Returns:
        Multi-passage grounded context string.
    """
    parts: List[str] = []
    for chunk in chunks[:max_chunks]:
        citation = format_citation(chunk)
        text = (
            chunk.get("text", "") if isinstance(chunk, dict)
            else getattr(chunk, "text", "")
        )
        text = text.strip()[:max_chars_per_chunk]
        parts.append(f"{citation}\n{text}")
    return "\n\n".join(parts)


# ── Evidence sufficiency and abstention ──────────────────────────────────────

@dataclass
class EvidenceVerdict:
    """
    Structured verdict from the evidence sufficiency check.

    Attributes:
        sufficient: True if retrieved evidence meets the confidence threshold.
        should_abstain: True if the system should refuse to answer.
        confidence: Aggregate evidence confidence score in [0, 1].
        citations: Formatted citation strings for passages used as evidence.
        reason: Human-readable explanation of the verdict.
        top_score: Best individual chunk cosine-similarity score.
        chunk_count: Number of candidate chunks evaluated.
    """
    sufficient: bool
    should_abstain: bool
    confidence: float
    citations: List[str] = field(default_factory=list)
    reason: str = ""
    top_score: float = 0.0
    chunk_count: int = 0


def check_evidence_sufficiency(
    chunks: Sequence[Any],
    query: str,
    score_threshold: float = 0.38,
    min_content_word_overlap: int = 1,
    min_sufficient_chunks: int = 1,
) -> EvidenceVerdict:
    """
    Evaluate whether retrieved chunks provide sufficient evidence to ground
    a trustworthy answer for the given query.

    Sufficiency requires ALL of:
      1. At least ``min_sufficient_chunks`` chunks exceed ``score_threshold``.
      2. At least ``min_content_word_overlap`` content words (len > 3) from the
         query appear in the combined text of the top chunks.

    Args:
        chunks: Retrieved Chunk objects ordered by descending score.
        query: Natural language question string.
        score_threshold: Minimum cosine similarity for a chunk to be considered
                         relevant evidence (default 0.38).
        min_content_word_overlap: Minimum number of query content words that
                                   must appear in the retrieved passages.
        min_sufficient_chunks: Minimum number of above-threshold chunks required.

    Returns:
        EvidenceVerdict describing whether the evidence is sufficient.
    """
    if not chunks:
        return EvidenceVerdict(
            sufficient=False,
            should_abstain=True,
            confidence=0.0,
            reason="No passages retrieved from the document collection.",
        )

    # Extract content words from query (len > 3 to skip stopwords)
    raw_words = re.sub(r"[^\w\s]", "", query.lower()).split()
    content_words = [w for w in raw_words if len(w) > 3]
    if not content_words:
        content_words = raw_words  # fallback for short queries

    scores = []
    for c in chunks:
        s = c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            scores.append(0.0)

    top_score = max(scores) if scores else 0.0

    # Count chunks that exceed the score threshold
    above_threshold = [
        c for c, s in zip(chunks, scores) if s >= score_threshold
    ]

    # Check content-word overlap in the union of above-threshold passage text
    combined_text = " ".join(
        (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")).lower()
        for c in above_threshold
    )
    word_hits = sum(w in combined_text for w in content_words)

    sufficient_chunks = len(above_threshold) >= min_sufficient_chunks
    sufficient_words = word_hits >= min_content_word_overlap

    sufficient = sufficient_chunks and sufficient_words

    # Aggregate confidence as the mean score of above-threshold chunks (clamped)
    if above_threshold:
        mean_score = sum(
            s for s in scores if s >= score_threshold
        ) / max(len(above_threshold), 1)
    else:
        mean_score = 0.0

    confidence = min(1.0, max(0.0, mean_score))

    # Build citations for above-threshold chunks only
    citations = format_citations(above_threshold)

    if sufficient:
        reason = (
            f"{len(above_threshold)} passage(s) exceed the relevance threshold "
            f"(score ≥ {score_threshold}); {word_hits}/{len(content_words)} "
            f"query term(s) matched."
        )
    elif not sufficient_chunks:
        reason = (
            f"No passages exceed the relevance threshold (score ≥ {score_threshold}). "
            f"Top score: {top_score:.3f}. The document may not contain information "
            f"relevant to this question."
        )
    else:
        reason = (
            f"Passages retrieved (top score {top_score:.3f}) but only {word_hits}/"
            f"{len(content_words)} query term(s) matched; evidence may not be "
            f"directly relevant to the question."
        )

    return EvidenceVerdict(
        sufficient=sufficient,
        should_abstain=not sufficient,
        confidence=confidence,
        citations=citations,
        reason=reason,
        top_score=top_score,
        chunk_count=len(chunks),
    )


# ── Abstention message generation ─────────────────────────────────────────────

def build_abstention_message(verdict: EvidenceVerdict, query: str = "") -> str:
    """
    Build a calibrated, informative refusal message for display when the
    system determines there is insufficient evidence to answer.

    Args:
        verdict: EvidenceVerdict from check_evidence_sufficiency.
        query: Optional original query string for context in the message.

    Returns:
        Formatted abstention string suitable for display in the Streamlit UI.
    """
    if query:
        ellipsis_mark = chr(0x2026) if len(query) > 80 else ''
        query_snippet = ' ' + chr(0x201C) + query[:80] + ellipsis_mark + chr(0x201D)
    else:
        query_snippet = ''  
    return (
        f"⚠️ **Insufficient evidence to answer confidently.**\n\n"
        f"The document collection does not appear to contain passages that "
        f"directly support{query_snippet}.\n\n"
        f"**Reason:** {verdict.reason}\n\n"
        f"*IntelliAsk Research declines to speculate beyond what the retrieved "
        f"evidence supports. Please upload additional relevant documents or "
        f"rephrase your question.*"
    )
