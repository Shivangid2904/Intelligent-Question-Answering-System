import math
import pytest
from evaluation.metrics import (
    compute_latency_stats,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    is_chunk_relevant,
)
from src.document_processor import Chunk


def test_recall_and_mrr_on_explicit_constructed_cases():
    """Test 7: Verify Recall@k and MRR on known ground-truth cases."""
    question = {
        "question_id": "q1",
        "question": "What is the training batch size?",
        "answerable": True,
        "query_type": "factoid",
        "relevant_document_ids": ["doc_alpha"],
        "gold_evidence": [
            {"document_id": "doc_alpha", "page_number": 4, "evidence_span": "batch size of 64"}
        ],
    }

    # Case A: First chunk is relevant -> MRR = 1.0, Recall@3 = 1.0
    chunks_a = [
        Chunk("We utilized a batch size of 64 across 8 GPUs.", "doc_alpha.pdf", 4, "doc_alpha", 0),
        Chunk("Other unrelated text.", "doc_beta.pdf", 1, "doc_beta", 1),
        Chunk("More unrelated text.", "doc_gamma.pdf", 2, "doc_gamma", 2),
    ]

    assert compute_recall_at_k(chunks_a, question, k=3) == 1.0
    assert compute_recall_at_k(chunks_a, question, k=1) == 1.0
    assert compute_mrr(chunks_a, question, max_k=10) == 1.0
    assert compute_precision_at_k(chunks_a, question, k=3) == pytest.approx(1.0 / 3.0)

    # Case B: Second chunk is relevant -> MRR = 0.5, Recall@1 = 0.0, Recall@2 = 1.0
    chunks_b = [
        Chunk("Unrelated first chunk.", "doc_beta.pdf", 1, "doc_beta", 0),
        Chunk("We utilized a batch size of 64 across 8 GPUs.", "doc_alpha.pdf", 4, "doc_alpha", 1),
        Chunk("More unrelated text.", "doc_gamma.pdf", 2, "doc_gamma", 2),
    ]

    assert compute_recall_at_k(chunks_b, question, k=1) == 0.0
    assert compute_recall_at_k(chunks_b, question, k=2) == 1.0
    assert compute_mrr(chunks_b, question, max_k=10) == 0.5

    # Case C: Third chunk is relevant -> MRR = 1/3 ~ 0.3333
    chunks_c = [
        Chunk("Irrelevant 1.", "doc_beta.pdf", 1, "doc_beta", 0),
        Chunk("Irrelevant 2.", "doc_beta.pdf", 2, "doc_beta", 1),
        Chunk("We utilized a batch size of 64 across 8 GPUs.", "doc_alpha.pdf", 4, "doc_alpha", 2),
    ]
    assert compute_recall_at_k(chunks_c, question, k=2) == 0.0
    assert compute_recall_at_k(chunks_c, question, k=3) == 1.0
    assert compute_mrr(chunks_c, question, max_k=10) == pytest.approx(1.0 / 3.0)

    # Case D: No relevant chunk -> Recall = 0.0, MRR = 0.0
    chunks_d = [
        Chunk("Irrelevant 1.", "doc_beta.pdf", 1, "doc_beta", 0),
        Chunk("Irrelevant 2.", "doc_beta.pdf", 2, "doc_beta", 1),
    ]
    assert compute_recall_at_k(chunks_d, question, k=5) == 0.0
    assert compute_mrr(chunks_d, question, max_k=10) == 0.0


def test_ndcg_at_k_calculation():
    """Test 8: Verify nDCG calculations for perfect, degraded, and zero rankings."""
    # Perfect ranking: [3, 2, 1, 0]
    perfect_ranks = [3, 2, 1, 0]
    assert compute_ndcg_at_k(perfect_ranks, k=4) == 1.0

    # Inverted ranking: [0, 1, 2, 3] -> must be strictly less than 1.0
    inverted_ranks = [0, 1, 2, 3]
    score = compute_ndcg_at_k(inverted_ranks, k=4)
    assert 0.0 < score < 1.0

    # All zeros -> 0.0
    assert compute_ndcg_at_k([0, 0, 0], k=3) == 0.0

    # Binary relevance: [1, 0, 0] vs [0, 1, 0]
    ndcg_first = compute_ndcg_at_k([1, 0, 0], k=3)
    ndcg_second = compute_ndcg_at_k([0, 1, 0], k=3)
    assert ndcg_first == 1.0
    assert ndcg_second == pytest.approx((1.0 / math.log2(3)) / (1.0 / math.log2(2)))
    assert ndcg_first > ndcg_second


def test_unanswerable_and_missing_annotations_handling():
    """Test 9: Verify unanswerable and missing annotations produce documented behavior rather than crashes or fabricated scores."""
    unanswerable_q = {
        "question_id": "neg_01",
        "question": "What is the secret passphrase?",
        "answerable": False,
        "query_type": "abstention",
        "relevant_document_ids": ["doc_alpha"],
        "gold_evidence": [],
    }

    chunks = [
        Chunk("Text mentioning passphrase.", "doc_alpha.pdf", 1, "doc_alpha", 0),
    ]

    # For negative tests, no chunk is considered a valid gold match
    assert is_chunk_relevant(chunks[0], unanswerable_q) is False
    assert compute_mrr(chunks, unanswerable_q) == 0.0
    # Negative queries should have 1.0 recall (no required gold missed)
    assert compute_recall_at_k(chunks, unanswerable_q, k=3) == 1.0


def test_latency_statistics_calculation():
    """Verify latency calculation handles empty, single-element, and multi-element lists."""
    assert compute_latency_stats([])["count"] == 0

    single = compute_latency_stats([12.5])
    assert single["count"] == 1
    assert single["mean_ms"] == 12.5
    assert single["median_ms"] == 12.5
    assert single["p95_ms"] == 12.5

    lats = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = compute_latency_stats(lats)
    assert stats["count"] == 10
    assert stats["mean_ms"] == 55.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 100.0
    assert stats["p95_ms"] >= 90.0
