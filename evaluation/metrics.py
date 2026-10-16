import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


def is_chunk_relevant(chunk: Any, question_item: Dict[str, Any]) -> bool:
    """
    Determines whether a retrieved chunk provides supporting evidence for a question.

    Relevance Convention:
    1. Unanswerable queries (answerable == False or query_type == "abstention")
       have no gold evidence; hence no retrieved chunk is considered relevant.
    2. For answerable queries, a chunk is relevant if:
       a) The chunk's document_id or source_filename matches one of the target
          relevant_document_ids, AND
       b) Either:
          - The chunk's physical page_number matches a page specified in gold_evidence (or gold_pages), OR
          - The chunk's text contains or overlaps significantly with an annotated evidence_span.

    Args:
        chunk: Chunk object or dict with document_id, page_number, text.
        question_item: Dict conforming to the benchmark schema.

    Returns:
        True if the chunk meets the relevance criteria, False otherwise.
    """
    if not question_item.get("answerable", True):
        return False

    if question_item.get("query_type") == "abstention":
        return False

    chunk_doc_id = getattr(chunk, "document_id", None) or (
        chunk.get("document_id") if isinstance(chunk, dict) else None
    )
    chunk_source = getattr(chunk, "source_filename", None) or getattr(chunk, "source", None) or (
        chunk.get("source_filename") or chunk.get("source") if isinstance(chunk, dict) else None
    )
    chunk_page = getattr(chunk, "page_number", None) or (
        chunk.get("page_number") if isinstance(chunk, dict) else None
    )
    chunk_text = (
        getattr(chunk, "text", "") if not isinstance(chunk, dict) else chunk.get("text", "")
    ).lower()

    # Document ID check
    rel_doc_ids = question_item.get("relevant_document_ids") or question_item.get("target_paper_ids") or []
    doc_matches = False
    if not rel_doc_ids:
        doc_matches = True
    else:
        for r_id in rel_doc_ids:
            if (chunk_doc_id and r_id in chunk_doc_id) or (chunk_source and r_id in chunk_source):
                doc_matches = True
                break

    if not doc_matches:
        return False

    # Extract gold evidence anchors
    gold_pages = set()
    evidence_spans = []

    # Check 'gold_evidence' list of dicts
    for item in question_item.get("gold_evidence", []):
        if isinstance(item, dict):
            if "page_number" in item:
                gold_pages.add(int(item["page_number"]))
            if "evidence_span" in item and item["evidence_span"]:
                evidence_spans.append(item["evidence_span"].strip().lower())

    # Fallback to direct 'gold_pages' and 'gold_evidence_spans' / 'gold_passages'
    for p in question_item.get("gold_pages", []):
        try:
            gold_pages.add(int(p))
        except (ValueError, TypeError):
            pass

    for span in question_item.get("gold_evidence_spans", []) or question_item.get("gold_passages", []):
        if span:
            evidence_spans.append(span.strip().lower())

    # Check page match
    if chunk_page is not None and int(chunk_page) in gold_pages:
        return True

    # Check evidence span match
    for span in evidence_spans:
        # If span is substantial, look for inclusion or partial overlap
        if len(span) > 20:
            words = [w for w in span.split() if len(w) > 3]
            if words:
                matches = sum(w in chunk_text for w in words)
                if matches / len(words) >= 0.6:
                    return True
        elif span in chunk_text:
            return True

    return False


def compute_recall_at_k(
    retrieved_chunks: Sequence[Any],
    question_item: Dict[str, Any],
    k: int = 10,
    relevance_fn: Optional[Callable[[Any, Dict[str, Any]], bool]] = None,
) -> float:
    """
    Computes Recall@k: Whether at least one relevant passage/page was captured
    within the top-k retrieved candidates for an answerable query.

    For queries requiring multiple distinct gold pages, computes the proportion
    of distinct gold pages captured in the top-k.
    """
    if relevance_fn is None:
        relevance_fn = is_chunk_relevant

    if not question_item.get("answerable", True) or question_item.get("query_type") == "abstention":
        # Negative tests have no recall requirement
        return 1.0

    gold_pages = set()
    for item in question_item.get("gold_evidence", []):
        if isinstance(item, dict) and "page_number" in item:
            gold_pages.add(int(item["page_number"]))
    for p in question_item.get("gold_pages", []):
        gold_pages.add(int(p))

    top_k = retrieved_chunks[:k]

    if not gold_pages:
        # Binary target: check if any chunk is relevant
        return 1.0 if any(relevance_fn(c, question_item) for c in top_k) else 0.0

    # Proportion of distinct required gold pages retrieved
    retrieved_pages = set()
    for c in top_k:
        if relevance_fn(c, question_item):
            p = getattr(c, "page_number", None) or (c.get("page_number") if isinstance(c, dict) else None)
            if p is not None:
                retrieved_pages.add(int(p))

    hit_count = len(gold_pages.intersection(retrieved_pages))
    return float(hit_count / len(gold_pages)) if gold_pages else 0.0


def compute_precision_at_k(
    retrieved_chunks: Sequence[Any],
    question_item: Dict[str, Any],
    k: int = 10,
    relevance_fn: Optional[Callable[[Any, Dict[str, Any]], bool]] = None,
) -> float:
    """
    Computes Precision@k: Proportion of top-k retrieved chunks that are relevant.
    """
    if relevance_fn is None:
        relevance_fn = is_chunk_relevant

    if not retrieved_chunks or k <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for c in top_k if relevance_fn(c, question_item))
    return float(relevant_count / len(top_k))


def compute_mrr(
    retrieved_chunks: Sequence[Any],
    question_item: Dict[str, Any],
    max_k: int = 10,
    relevance_fn: Optional[Callable[[Any, Dict[str, Any]], bool]] = None,
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for a query: 1 / rank of the first
    relevant chunk within the top max_k results. Returns 0.0 if no relevant chunk found.
    """
    if relevance_fn is None:
        relevance_fn = is_chunk_relevant

    if not question_item.get("answerable", True) or question_item.get("query_type") == "abstention":
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks[:max_k], start=1):
        if relevance_fn(chunk, question_item):
            return 1.0 / rank

    return 0.0


def compute_ndcg_at_k(
    relevance_scores: Sequence[Union[int, float]],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@k) from a sequence of
    graded relevance scores (e.g. [3, 2, 0, 1] or binary [1, 0, 1]).

    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    if not relevance_scores or k <= 0:
        return 0.0

    actual_scores = list(relevance_scores[:k])
    dcg = 0.0
    for i, rel in enumerate(actual_scores, start=1):
        dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """
    Computes mean, median, min, max, and 95th percentile (p95) latency statistics.
    """
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    mean = sum(sorted_lats) / n
    median = (
        sorted_lats[n // 2]
        if n % 2 != 0
        else (sorted_lats[n // 2 - 1] + sorted_lats[n // 2]) / 2.0
    )

    p95_idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    p95 = sorted_lats[p95_idx]

    return {
        "count": n,
        "mean_ms": round(float(mean), 2),
        "median_ms": round(float(median), 2),
        "p95_ms": round(float(p95), 2),
        "min_ms": round(float(sorted_lats[0]), 2),
        "max_ms": round(float(sorted_lats[-1]), 2),
    }
