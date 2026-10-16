import argparse
import os
import sys

# Ensure repository root is on sys.path when invoked directly as a script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.document_processor import DocumentProcessor, get_nlp
from src.qa_engine import QAEngine, get_embedding_model
from evaluation.evaluator import DenseRetrievalEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Run IntelliAsk Research Dense Retrieval Baseline Evaluation."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/corpus_manifest_template.json",
        help="Path to the corpus manifest JSON.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/benchmark_questions_template.json",
        help="Path to the benchmark questions JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation artifacts.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("IntelliAsk Research - Dense Retrieval Baseline Evaluation Runner")
    print("=" * 70)

    if not os.path.exists(args.benchmark):
        print(f"[ERROR] Benchmark questions file not found: {args.benchmark}")
        sys.exit(1)

    # Check for actual papers in data/papers/
    papers_dir = "data/papers"
    pdf_files = []
    if os.path.exists(papers_dir):
        pdf_files = [
            os.path.join(papers_dir, f)
            for f in os.listdir(papers_dir)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_files:
        print(f"[INFO] Notice: No PDF documents found in '{papers_dir}'.")
        print("Corpus selection and curation are pending Week 1 tasks.")
        print("To run evaluation on real papers, place PDFs in 'data/papers/' and update the manifest.")
        print("=" * 70)
        return

    print(f"[INFO] Ingesting {len(pdf_files)} PDF document(s)...")
    nlp = get_nlp()
    processor = DocumentProcessor()
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  * Parsing {os.path.basename(pdf_path)}...")
        chunks = processor.process_file(pdf_path, nlp)
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks extracted: {len(all_chunks)}")
    print("[INFO] Initializing dense embedding model (all-MiniLM-L6-v2)...")
    embedding_model = get_embedding_model()

    print("[INFO] Building in-memory FAISS vector index...")
    qa_engine = QAEngine(all_chunks, embedding_model)

    print(f"[INFO] Running evaluation against '{args.benchmark}'...")
    evaluator = DenseRetrievalEvaluator(qa_engine, args.benchmark)
    summary = evaluator.run_evaluation()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "query_results.jsonl")
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    report_path = os.path.join(args.output_dir, "baseline_evaluation_report.md")

    evaluator.save_jsonl_results(jsonl_path)
    evaluator.save_json_summary(json_path)
    evaluator.generate_markdown_report(report_path)

    print("=" * 70)
    print("Evaluation Complete. Artifacts saved:")
    print(f"  * Per-query log: {jsonl_path}")
    print(f"  * Summary JSON:  {json_path}")
    print(f"  * Report:        {report_path}")
    print("=" * 70)
    print(f"Recall@3:  {summary.get('metrics', {}).get('recall@3', 0.0)}")
    print(f"Recall@5:  {summary.get('metrics', {}).get('recall@5', 0.0)}")
    print(f"Recall@10: {summary.get('metrics', {}).get('recall@10', 0.0)}")
    print(f"MRR:       {summary.get('metrics', {}).get('mrr', 0.0)}")
    print(f"p95 Latency: {summary.get('latency', {}).get('p95_ms', 0.0)} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
