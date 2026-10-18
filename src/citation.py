"""
IntelliAsk Research — Citation and Evidence Sufficiency Module

Provides:
  - Page-level citation formatting:  [Filename, p. X]
  - Evidence sufficiency evaluation: determines whether retrieved chunks
    constitute sufficient grounding to generate a confident answer.
  - Abstention policy: returns a structured refusal signal when evidence
    is absent, ambiguous, or below a configurable confidence threshold.

Week 3 additions (backward-compatible):
  - Document-scoping: evidence can be restricted to chunks from specific
    source filenames, preventing cross-paper contamination.
  - Critical-entity validation: explicit technical identifiers (model names,
    hardware IDs, dataset names) must be present in the scoped evidence text.
    Entity presence alone does not guarantee propositional entailment.
  - Dual-score support: if a chunk carries a ``dense_score`` attribute (set by
    hybrid retrieval), that value is used for semantic gating instead of
    ``chunk.score`` (which holds the RRF rank score in hybrid mode).

Limitations (acknowledged):
  - These deterministic checks cover Levels 1-3 of the evidence hierarchy
    (topic relevance, lexical match, entity presence and document scope).
    They do NOT establish propositional entailment (Level 4). Whether a
    passage *asserts* the specific numerical value or relationship asked about
    remains the responsibility of the grounded LLM prompt.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Collection, List, Optional, Sequence


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


# ── Evidence sufficiency helpers (Week 3) ────────────────────────────────────

def _chunk_source_filename(chunk: Any) -> str:
    """Return the basename of the source filename for a Chunk object or dict."""
    if isinstance(chunk, dict):
        raw = chunk.get("source_filename") or chunk.get("source", "")
    else:
        raw = getattr(chunk, "source_filename", "") or getattr(chunk, "source", "")
    return os.path.basename(str(raw))


def _chunk_semantic_score(chunk: Any, use_dense_score: bool) -> Optional[float]:
    """
    Extract the semantic relevance score from a chunk.

    In dense mode (use_dense_score=False):
        Returns chunk.score (cosine similarity).

    In hybrid mode (use_dense_score=True):
        Requires chunk.dense_score (cosine similarity from the dense leg or
        computed from precomputed embeddings). Returns a score only if the
        value is a finite, non-NaN float (i.e. math.isfinite(val) is True).
        Rejects NaN, positive infinity, negative infinity, the legacy
        sentinel <= -1.0, non-numeric types, and missing scores (None).
        It NEVER falls back to chunk.score (which may hold an incomparable
        RRF rank score or BM25 score).
    """
    if use_dense_score:
        ds = (
            chunk.get("dense_score") if isinstance(chunk, dict)
            else getattr(chunk, "dense_score", None)
        )
        if ds is not None:
            try:
                val = float(ds)
                # Accept only finite values (rejects NaN, +inf, -inf)
                # and exclude the legacy uncomputable sentinel (<= -1.0)
                if math.isfinite(val) and val > -1.0:
                    return val
            except (TypeError, ValueError):
                pass
        return None

    # Dense-only mode: legacy behavior using chunk.score (cosine similarity)
    raw = chunk.get("score", 0.0) if isinstance(chunk, dict) else getattr(chunk, "score", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _tokenize_entity(text: str) -> List[str]:
    """
    Extract lowercase alphanumeric tokens from text, splitting on whitespace,
    hyphens, underscores, slashes, and punctuation.
    """
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _normalize_entity(text: str) -> str:
    """Normalize text into space-separated lowercase tokens."""
    return " ".join(_tokenize_entity(text))


def _entity_present_in_text(entity: str, combined_text: str) -> bool:
    """
    Return True if the entity's token sequence appears as a contiguous
    sub-sequence in the tokenized combined_text.

    This avoids false substring positives (e.g. 'ZED' matching inside
    'analyzed', 'OS1' matching inside 'OS10', 'DPT' matching inside 'adaptation')
    while preserving intended matches across case, whitespace, hyphens, and
    underscores (e.g. 'OS1-32' matching 'OS1 32', 'OS1_32', or 'OS1-32').
    """
    entity_tokens = _tokenize_entity(entity)
    if not entity_tokens:
        return False
    text_tokens = _tokenize_entity(combined_text)
    if len(text_tokens) < len(entity_tokens):
        return False

    n = len(entity_tokens)
    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i : i + n] == entity_tokens:
            return True
    return False


# ── Evidence sufficiency and abstention ──────────────────────────────────────

@dataclass
class EvidenceVerdict:
    """
    Structured verdict from the rule-based evidence sufficiency check.

    Attributes:
        sufficient: True if retrieved evidence meets the heuristic sufficiency criteria.
        should_abstain: True if the system should refuse to answer based on sufficiency rules.
        confidence: Aggregate heuristic evidence score in [0, 1] (mean cosine similarity of
                    above-threshold chunks; cosine similarity is a vector ranking/similarity
                    signal, not a calibrated posterior probability).
        citations: Formatted citation strings for passages used as evidence.
        reason: Human-readable explanation of the verdict.
        top_score: Best individual chunk cosine-similarity score.
        chunk_count: Number of candidate chunks evaluated.
        scoped_chunk_count: Chunks remaining after document-scope filtering (0 if no scoping).
        missing_entities: Critical entities not found in the scoped evidence text.
        unscored_chunk_count: Chunks that lacked valid semantic scores in hybrid mode.
        missing_sources: Required source documents missing from retrieved evidence.
    """
    sufficient: bool
    should_abstain: bool
    confidence: float
    citations: List[str] = field(default_factory=list)
    reason: str = ""
    top_score: float = 0.0
    chunk_count: int = 0
    scoped_chunk_count: int = 0
    missing_entities: List[str] = field(default_factory=list)
    unscored_chunk_count: int = 0
    missing_sources: List[str] = field(default_factory=list)


def check_evidence_sufficiency(
    chunks: Sequence[Any],
    query: str,
    score_threshold: float = 0.38,
    min_content_word_overlap: int = 1,
    min_sufficient_chunks: int = 1,
    scope_filenames: Optional[Collection[str]] = None,
    critical_entities: Optional[Collection[str]] = None,
    use_dense_score: bool = False,
    required_source_filenames: Optional[Collection[str]] = None,
) -> EvidenceVerdict:
    """
    Evaluate whether retrieved chunks provide sufficient evidence to ground
    an answer for the given query using deterministic, rule-based heuristics.

    Sufficiency requires ALL of:
      1. At least ``min_sufficient_chunks`` chunks exceed ``score_threshold``
         in the (optionally scoped) candidate set.
      2. At least ``min_content_word_overlap`` content words (len > 3) from the
         query appear in the combined text of the above-threshold chunks.
      3. (If ``critical_entities`` is provided) Every required entity must be
         present in the retrieved evidence text of a relevant scoped source.
      4. (If ``required_source_filenames`` is provided) Every required source
         document must be represented by at least one above-threshold passage.
         present in the combined text of the scoped above-threshold chunks.

    Document Scoping (Stage 1 — opt-in, backward-compatible):
        When ``scope_filenames`` is provided, only chunks whose source_filename
        basename is in ``scope_filenames`` are treated as evidence.  If scoping
        produces an empty set, the function abstains with a clear reason rather
        than falling back to unscoped results.

    Dual-Score Support (Stage 2 — opt-in, backward-compatible):
        When ``use_dense_score=True``, semantic gating uses ``chunk.dense_score``
        (cosine similarity from the dense leg of hybrid retrieval) instead of
        ``chunk.score`` (which holds the RRF rank score).  Sentinel -1.0 on
        ``dense_score`` signals an uncomputed value; the function falls back to
        ``chunk.score`` rather than using zero silently.

    Critical-Entity Validation (Stage 3 — opt-in, backward-compatible):
        ``critical_entities`` is an optional iterable of strings (e.g. "OS1-32",
        "ResNet-50", "KITTI") that must appear in the combined scoped evidence.
        Matching is case-insensitive with hyphen/underscore normalisation.
        Entity presence is necessary but NOT sufficient — it does not establish
        that the passage asserts the specific value or relationship being asked.

    Backward Compatibility:
        All three new parameters default to their disabled values (None / False),
        so all existing callers (app.py, run_benchmark.py, all tests) continue
        to receive exactly the same behaviour as before.

    Note:
        Cosine similarity scores serve as relative semantic proximity and ranking
        signals; they are not calibrated probabilities of factual correctness.
        This function implements a rule-based sufficiency safeguard rather than
        a statistically calibrated probability model.

    Args:
        chunks: Retrieved Chunk objects ordered by descending score.
        query: Natural language question string.
        score_threshold: Minimum cosine similarity for a chunk to be considered
                         relevant evidence (default 0.38).
        min_content_word_overlap: Minimum number of query content words that
                                   must appear in the retrieved passages.
        min_sufficient_chunks: Minimum number of above-threshold chunks required.
        scope_filenames: Optional collection of source_filename basenames to
                         restrict evidence to.  None disables scoping.
        critical_entities: Optional collection of technical entity strings that
                            must appear in the scoped evidence text.
        use_dense_score: If True, use chunk.dense_score for threshold comparison
                         instead of chunk.score (for hybrid retrieval mode).

    Returns:
        EvidenceVerdict describing whether the evidence is sufficient.
    """
    if not chunks:
        return EvidenceVerdict(
            sufficient=False,
            should_abstain=True,
            confidence=0.0,
            reason="No passages retrieved from the document collection.",
            chunk_count=0,
            scoped_chunk_count=0,
        )

    total_chunk_count = len(chunks)

    # ── Stage 1: Document-scope filtering (opt-in) ────────────────────────────
    if scope_filenames is not None:
        scope_set = {os.path.basename(str(fn)) for fn in scope_filenames}
        scoped_chunks: List[Any] = [c for c in chunks if _chunk_source_filename(c) in scope_set]

        if not scoped_chunks:
            scope_list = ", ".join(sorted(scope_set))
            return EvidenceVerdict(
                sufficient=False,
                should_abstain=True,
                confidence=0.0,
                reason=(
                    f"Document scoping is active for [{scope_list}] but none of the "
                    f"retrieved passages originated from the requested document(s). "
                    f"The corpus may not contain the relevant section."
                ),
                chunk_count=total_chunk_count,
                scoped_chunk_count=0,
            )
    else:
        scoped_chunks = list(chunks)

    scoped_count = len(scoped_chunks)

    # ── Stage 2: Semantic / lexical gating ───────────────────────────────────

    # Extract content words from query (len > 3 to skip stopwords)
    raw_words = re.sub(r"[^\w\s]", "", query.lower()).split()
    content_words = [w for w in raw_words if len(w) > 3]
    if not content_words:
        content_words = raw_words  # fallback for short queries

    scores: List[Optional[float]] = [_chunk_semantic_score(c, use_dense_score) for c in scoped_chunks]
    valid_scores: List[float] = [s for s in scores if s is not None]
    unscored_chunks: List[Any] = [c for c, s in zip(scoped_chunks, scores) if s is None]
    top_score = max(valid_scores) if valid_scores else 0.0

    # Count chunks that exceed the score threshold (only valid scored chunks!)
    above_threshold = [
        c for c, s in zip(scoped_chunks, scores) if s is not None and s >= score_threshold
    ]

    # Check content-word overlap in the union of above-threshold passage text
    combined_text = " ".join(
        (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")).lower()
        for c in above_threshold
    )
    word_hits = sum(w in combined_text for w in content_words)

    sufficient_chunks = len(above_threshold) >= min_sufficient_chunks
    sufficient_words = word_hits >= min_content_word_overlap

    # ── Stage 2.5: Required-source coverage (opt-in for multi-document synthesis) ──
    missing_sources: List[str] = []
    if required_source_filenames is not None:
        required_set = {os.path.basename(str(fn)) for fn in required_source_filenames}
        represented_sources = {_chunk_source_filename(c) for c in above_threshold}
        missing_sources = sorted([s for s in required_set if s not in represented_sources])

    # ── Stage 3: Critical-entity validation (opt-in) ─────────────────────────
    # Explicitly configured entities are checked across the relevant retrieved
    # evidence for the same required/scoped source, rather than strictly inside
    # a single high-scoring chunk (e.g. allowing numerical metrics in one chunk
    # and equipment identifiers in another to co-ground the proposition).
    # Entities from unrelated papers (documents with 0 above-threshold chunks
    # or outside scope) can NEVER satisfy requirements.
    missing_entities: List[str] = []
    if critical_entities:
        if above_threshold:
            # Sources that have proven semantic relevance by exceeding threshold
            relevant_sources = {_chunk_source_filename(c) for c in above_threshold}
            # Entity search context: all scoped candidate chunks belonging to those relevant sources
            entity_chunks = [c for c in scoped_chunks if _chunk_source_filename(c) in relevant_sources]
            entity_context_text = " ".join(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")).lower()
                for c in entity_chunks
            )
            for entity in critical_entities:
                if not _entity_present_in_text(entity, entity_context_text):
                    missing_entities.append(entity)
        else:
            # No above-threshold chunks; all entities are trivially absent
            missing_entities = list(critical_entities)

    sufficient_entities = len(missing_entities) == 0
    sufficient_sources = len(missing_sources) == 0

    # Overall sufficiency
    sufficient = sufficient_chunks and sufficient_words and sufficient_entities and sufficient_sources

    # Aggregate confidence as the mean score of above-threshold chunks (clamped)
    if above_threshold:
        above_scores = [s for s in valid_scores if s >= score_threshold]
        mean_score = sum(above_scores) / max(len(above_threshold), 1)
    else:
        mean_score = 0.0

    confidence = min(1.0, max(0.0, mean_score))

    # Build citations for above-threshold chunks only
    citations = format_citations(above_threshold)

    # ── Reason string ─────────────────────────────────────────────────────────
    if sufficient:
        reason = (
            f"{len(above_threshold)} passage(s) exceed the relevance threshold "
            f"(score \u2265 {score_threshold}); {word_hits}/{len(content_words)} "
            f"query term(s) matched."
        )
        if critical_entities:
            n = len(list(critical_entities))
            reason += f" All {n} required entity/entities confirmed present."
        if required_source_filenames is not None:
            reason += f" All {len(required_source_filenames)} required source(s) represented."
        if unscored_chunks:
            reason += f" ({len(unscored_chunks)} unscored passage(s) excluded from thresholding)."
    elif missing_sources:
        reason = (
            f"Required source document(s) missing from retrieved evidence: {missing_sources}. "
            f"Cross-document synthesis requires supporting evidence from each specified paper."
        )
    elif missing_entities:
        reason = (
            f"Required technical entity/entities not found in the scoped evidence: "
            f"{missing_entities}. Entity presence is necessary but not sufficient "
            f"for propositional grounding."
        )
    elif unscored_chunks and not above_threshold and not valid_scores:
        reason = (
            f"All {len(unscored_chunks)} retrieved passage(s) lacked valid semantic scores "
            f"for hybrid evaluation. Cannot verify semantic relevance."
        )
    elif not sufficient_chunks:
        reason = (
            f"No passages exceed the relevance threshold (score \u2265 {score_threshold}). "
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
        chunk_count=total_chunk_count,
        scoped_chunk_count=scoped_count,
        missing_entities=missing_entities,
        unscored_chunk_count=len(unscored_chunks),
        missing_sources=missing_sources,
    )


# ── Abstention message generation ─────────────────────────────────────────────

def build_abstention_message(verdict: EvidenceVerdict, query: str = "") -> str:
    """
    Build a structured, informative refusal message for display when the
    heuristic evidence-sufficiency check determines there is insufficient
    evidence to answer.

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
