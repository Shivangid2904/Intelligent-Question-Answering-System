import json
import os
import tempfile
import pytest
from evaluation.evaluator import DenseRetrievalEvaluator
from src.document_processor import Chunk


class DummyEngine:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query: str, k: int = 10):
        # Return chunks in predetermined order
        return self.chunks[:k]


def test_evaluator_runs_and_generates_outputs():
    chunks = [
        Chunk("Primary evidence passage on page 2.", "study.pdf", 2, "study", 0),
        Chunk("Secondary passage on page 3.", "study.pdf", 3, "study", 1),
    ]
    engine = DummyEngine(chunks)

    questions = [
        {
            "question_id": "eval_01",
            "question": "What is the primary evidence?",
            "answerable": True,
            "query_type": "factoid",
            "relevant_document_ids": ["study"],
            "gold_evidence": [{"document_id": "study", "page_number": 2, "evidence_span": "primary evidence"}],
        },
        {
            "question_id": "eval_02",
            "question": "What is the unanswerable claim?",
            "answerable": False,
            "query_type": "abstention",
            "relevant_document_ids": ["study"],
            "gold_evidence": [],
        },
    ]

    evaluator = DenseRetrievalEvaluator(engine, questions, max_k=5)
    summary = evaluator.run_evaluation()

    assert summary["query_count"] == 2
    assert summary["answerable_count"] == 1
    assert summary["unanswerable_count"] == 1
    assert summary["metrics"]["recall@3"] == 1.0
    assert summary["metrics"]["mrr"] == 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "results.jsonl")
        json_path = os.path.join(tmpdir, "summary.json")
        md_path = os.path.join(tmpdir, "report.md")

        evaluator.save_jsonl_results(jsonl_path)
        evaluator.save_json_summary(json_path)
        report = evaluator.generate_markdown_report(md_path)

        assert os.path.exists(jsonl_path)
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_summary = json.load(f)
            assert loaded_summary["query_count"] == 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) == 2
            assert lines[0]["question_id"] == "eval_01"

        assert "# IntelliAsk Research — Dense Retrieval Evaluation Report" in report
