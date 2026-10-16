import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from .metrics import (
    compute_latency_stats,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    is_chunk_relevant,
)


class DenseRetrievalEvaluator:
    """
    Evaluator for dense semantic retrieval pipelines against structured benchmark questions.
    Generates reproducible JSONL query logs, JSON aggregate metrics, and Markdown reports.
    """

    def __init__(
        self,
        qa_engine: Any,
        benchmark_questions: Union[str, Sequence[Dict[str, Any]]],
        max_k: int = 10,
    ):
        """
        Args:
            qa_engine: Instantiated QAEngine (or mock with .search(query, k)).
            benchmark_questions: List of question dicts or path to benchmark JSON file.
            max_k: Maximum candidate depth to retrieve for metric computation.
        """
        self.qa_engine = qa_engine
        self.max_k = max(10, max_k)

        if isinstance(benchmark_questions, (str, os.PathLike)):
            with open(benchmark_questions, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.questions = data.get("questions", data if isinstance(data, list) else [])
        else:
            self.questions = list(benchmark_questions)

        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Executes dense retrieval evaluation across all benchmark questions.

        Returns:
            Dictionary containing aggregate metrics and run metadata.
        """
        if not self.questions:
            self.summary = {
                "status": "skipped",
                "reason": "No benchmark questions provided",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_count": 0,
            }
            return self.summary

        self.results = []
        latencies: List[float] = []

        recalls_3: List[float] = []
        recalls_5: List[float] = []
        recalls_10: List[float] = []

        precisions_3: List[float] = []
        precisions_5: List[float] = []
        precisions_10: List[float] = []

        mrrs: List[float] = []
        ndcgs_10: List[float] = []

        answerable_count = 0
        unanswerable_count = 0

        for q in self.questions:
            q_id = q.get("question_id") or q.get("id", "unknown_qid")
            query_str = q.get("question", "")
            is_answerable = q.get("answerable", True) and q.get("query_type") != "abstention"

            if is_answerable:
                answerable_count += 1
            else:
                unanswerable_count += 1

            t0 = time.perf_counter()
            retrieved_chunks = self.qa_engine.search(query_str, k=self.max_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            # Compute per-query metrics
            r3 = compute_recall_at_k(retrieved_chunks, q, k=3)
            r5 = compute_recall_at_k(retrieved_chunks, q, k=5)
            r10 = compute_recall_at_k(retrieved_chunks, q, k=10)

            p3 = compute_precision_at_k(retrieved_chunks, q, k=3)
            p5 = compute_precision_at_k(retrieved_chunks, q, k=5)
            p10 = compute_precision_at_k(retrieved_chunks, q, k=10)

            mrr = compute_mrr(retrieved_chunks, q, max_k=10)

            # Binary relevance sequence for nDCG
            relevance_sequence = [
                1.0 if is_chunk_relevant(c, q) else 0.0 for c in retrieved_chunks
            ]
            ndcg10 = compute_ndcg_at_k(relevance_sequence, k=10)

            if is_answerable:
                recalls_3.append(r3)
                recalls_5.append(r5)
                recalls_10.append(r10)
                precisions_3.append(p3)
                precisions_5.append(p5)
                precisions_10.append(p10)
                mrrs.append(mrr)
                ndcgs_10.append(ndcg10)

            retrieved_summary = []
            for rank, c in enumerate(retrieved_chunks, start=1):
                chunk_id = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else f"rank_{rank}")
                page_num = getattr(c, "page_number", None) or (c.get("page_number") if isinstance(c, dict) else None)
                score = getattr(c, "score", 0.0) or (c.get("score", 0.0) if isinstance(c, dict) else 0.0)
                is_rel = is_chunk_relevant(c, q)

                retrieved_summary.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "score": round(float(score), 4),
                    "is_relevant": is_rel,
                })

            self.results.append({
                "question_id": q_id,
                "question": query_str,
                "answerable": is_answerable,
                "query_type": q.get("query_type", "factoid" if is_answerable else "abstention"),
                "latency_ms": round(elapsed_ms, 2),
                "metrics": {
                    "recall@3": round(r3, 4),
                    "recall@5": round(r5, 4),
                    "recall@10": round(r10, 4),
                    "precision@3": round(p3, 4),
                    "precision@5": round(p5, 4),
                    "precision@10": round(p10, 4),
                    "mrr": round(mrr, 4),
                    "ndcg@10": round(ndcg10, 4),
                },
                "retrieved": retrieved_summary,
            })

        def safe_mean(lst: List[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        latency_stats = compute_latency_stats(latencies)

        self.summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": "Dense Semantic Baseline (all-MiniLM-L6-v2 + FAISS IndexFlatIP)",
            "query_count": len(self.questions),
            "answerable_count": answerable_count,
            "unanswerable_count": unanswerable_count,
            "metrics": {
                "recall@3": safe_mean(recalls_3),
                "recall@5": safe_mean(recalls_5),
                "recall@10": safe_mean(recalls_10),
                "precision@3": safe_mean(precisions_3),
                "precision@5": safe_mean(precisions_5),
                "precision@10": safe_mean(precisions_10),
                "mrr": safe_mean(mrrs),
                "ndcg@10": safe_mean(ndcgs_10),
            },
            "latency": latency_stats,
        }

        return self.summary

    def save_jsonl_results(self, output_path: str) -> None:
        """Saves detailed per-query retrieval results to a JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.results:
                f.write(json.dumps(item) + "\n")

    def save_json_summary(self, output_path: str) -> None:
        """Saves aggregate metrics summary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)

    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generates a clean, transparent Markdown evaluation report.
        Optionally writes to output_path.
        """
        metrics = self.summary.get("metrics", {})
        latency = self.summary.get("latency", {})

        report_lines = [
            "# IntelliAsk Research — Dense Retrieval Evaluation Report",
            "",
            f"**Execution Timestamp**: {self.summary.get('timestamp', 'N/A')}",
            f"**Pipeline Configuration**: {self.summary.get('pipeline', 'N/A')}",
            f"**Total Queries Evaluated**: {self.summary.get('query_count', 0)} "
            f"({self.summary.get('answerable_count', 0)} answerable, "
            f"{self.summary.get('unanswerable_count', 0)} negative/abstention tests)",
            "",
            "## 1. Aggregate Retrieval Metrics",
            "",
            "| Metric | Score | Target Purpose |",
            "| :--- | :---: | :--- |",
            f"| **Recall@3** | {metrics.get('recall@3', 0.0)} | % gold evidence captured in top 3 |",
            f"| **Recall@5** | {metrics.get('recall@5', 0.0)} | % gold evidence captured in top 5 |",
            f"| **Recall@10** | {metrics.get('recall@10', 0.0)} | % gold evidence captured in top 10 |",
            f"| **Precision@3** | {metrics.get('precision@3', 0.0)} | Evidence density in top 3 |",
            f"| **Precision@5** | {metrics.get('precision@5', 0.0)} | Evidence density in top 5 |",
            f"| **Precision@10** | {metrics.get('precision@10', 0.0)} | Evidence density in top 10 |",
            f"| **Mean Reciprocal Rank (MRR)** | {metrics.get('mrr', 0.0)} | Rank position of first relevant passage |",
            f"| **nDCG@10** | {metrics.get('ndcg@10', 0.0)} | Graded ranking quality across top 10 |",
            "",
            "## 2. Retrieval Latency Statistics",
            "",
            "| Statistic | Value (ms) |",
            "| :--- | :---: |",
            f"| **Mean Latency** | {latency.get('mean_ms', 0.0)} ms |",
            f"| **Median Latency** | {latency.get('median_ms', 0.0)} ms |",
            f"| **p95 Latency** | {latency.get('p95_ms', 0.0)} ms |",
            f"| **Min / Max Latency** | {latency.get('min_ms', 0.0)} / {latency.get('max_ms', 0.0)} ms |",
            "",
            "## 3. Methodological Notes & Limitations",
            "- Evaluation metrics are computed strictly across answerable benchmark queries.",
            "- Gold relevance requires target document match and physical page/span overlap.",
            "- In compliance with Week 1 roadmap goals, answer generation (LLM judging) is evaluated in subsequent milestones.",
        ]

        report_content = "\n".join(report_lines)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content + "\n")

        return report_content
