"""
IntelliAsk Research — Retrieval Evaluation Package
Provides metrics computation, relevance verification, and reproducible benchmarking
for dense and hybrid retrieval pipelines.
"""

from .metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_latency_stats,
    is_chunk_relevant,
)
from .evaluator import DenseRetrievalEvaluator

__all__ = [
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_latency_stats",
    "is_chunk_relevant",
    "DenseRetrievalEvaluator",
]
