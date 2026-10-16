# IntelliAsk Research — Evaluation Corpus & Benchmark Directory

This directory houses the evaluation manifests, paper corpora, and ground-truth benchmark datasets for **IntelliAsk Research**.

---

## 1. Directory Structure

```text
data/
├── corpus_manifest_template.json   # JSON schema and template for corpus manifests
├── benchmark_questions_template.json # JSON schema and template for benchmark QA pairs
├── papers/                         # Local evaluation PDFs (queued for Week 1 curation)
└── README.md                       # Documentation on curation standards and schemas
```

---

## 2. Benchmark Design & Standards

As defined in [`docs/EVALUATION.md`](../docs/EVALUATION.md):
- **Corpus Scale**: Approximately 5–10 peer-reviewed scientific papers.
- **Domain Independence**: General scientific literature selected for evaluation suitability across retrieval, provenance, and synthesis.
- **Question Volume**: Approximately 20–30 manually curated questions.
- **Provenance Anchors**: Physical 1-based PDF page numbers (`page_number`) and exact evidence text excerpts (`evidence_span`).
- **Query Types**:
  1. `factoid`: Precise quantitative or hyperparameter queries.
  2. `synthesis`: Cross-paragraph methodological comparisons or explanations.
  3. `abstention`: Plausible but unanswerable questions testing refusal resistance.

---

## 3. Current Status

> [!IMPORTANT]
> **Pending Curation**: In strict compliance with Week 1 roadmap objectives, paper selection and manual question annotation are queued tasks. No synthetic results or fabricated paper records are stored in this directory. Real open-access scientific PDFs and annotated QA pairs will be populated during Week 1 execution.
