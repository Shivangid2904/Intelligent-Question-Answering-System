"""
Week 2 — Tests for src/citation.py

Covers:
  - format_citation: Chunk objects, dict chunks, missing page number
  - format_citations: deduplication
  - build_grounded_context: structure and character budget
  - check_evidence_sufficiency: sufficient, insufficient (score), insufficient (word match)
  - build_abstention_message: content completeness
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock

from src.citation import (
    format_citation,
    format_citations,
    build_grounded_context,
    check_evidence_sufficiency,
    build_abstention_message,
    EvidenceVerdict,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(text: str, source_filename: str, page_number: int, score: float = 0.45):
    """Create a minimal Chunk-like mock object with required attributes."""
    c = MagicMock()
    c.text = text
    c.source_filename = source_filename
    c.source = source_filename  # backwards-compat alias
    c.page_number = page_number
    c.score = score
    c.document_id = f"doc_{source_filename[:8]}"
    return c


# ── format_citation ───────────────────────────────────────────────────────────

class TestFormatCitation:
    def test_chunk_object_with_page(self):
        chunk = _make_chunk("Some text.", "attention.pdf", page_number=4)
        result = format_citation(chunk)
        assert result == "[attention.pdf, p. 4]"

    def test_chunk_object_without_page(self):
        chunk = _make_chunk("Some text.", "paper.pdf", page_number=None)
        result = format_citation(chunk)
        assert result == "[paper.pdf]"

    def test_dict_chunk_with_page(self):
        chunk_dict = {
            "source_filename": "transformer.pdf",
            "page_number": 7,
            "text": "Attention is all you need.",
        }
        result = format_citation(chunk_dict)
        assert result == "[transformer.pdf, p. 7]"

    def test_dict_chunk_uses_source_fallback(self):
        chunk_dict = {
            "source": "fallback.pdf",
            "page_number": 2,
            "text": "Some text.",
        }
        result = format_citation(chunk_dict)
        assert result == "[fallback.pdf, p. 2]"

    def test_basename_only_is_used(self):
        """Ensures full directory paths are stripped to basename."""
        chunk = _make_chunk("text", "C:/data/papers/bert.pdf", page_number=1)
        result = format_citation(chunk)
        assert result == "[bert.pdf, p. 1]"


# ── format_citations ──────────────────────────────────────────────────────────

class TestFormatCitations:
    def test_deduplication_removes_duplicate_citations(self):
        c1 = _make_chunk("Text A", "paper.pdf", page_number=3)
        c2 = _make_chunk("Text B", "paper.pdf", page_number=3)  # same citation
        c3 = _make_chunk("Text C", "paper.pdf", page_number=5)
        result = format_citations([c1, c2, c3], deduplicate=True)
        assert result == ["[paper.pdf, p. 3]", "[paper.pdf, p. 5]"]

    def test_no_deduplication_includes_all(self):
        c1 = _make_chunk("Text A", "paper.pdf", page_number=3)
        c2 = _make_chunk("Text B", "paper.pdf", page_number=3)
        result = format_citations([c1, c2], deduplicate=False)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        assert format_citations([]) == []


# ── build_grounded_context ────────────────────────────────────────────────────

class TestBuildGroundedContext:
    def test_context_contains_citation_and_text(self):
        c = _make_chunk("Transformers use self-attention.", "attention.pdf", page_number=4)
        ctx = build_grounded_context([c], max_chunks=1)
        assert "[attention.pdf, p. 4]" in ctx
        assert "Transformers use self-attention." in ctx

    def test_max_chunks_limits_output(self):
        chunks = [_make_chunk(f"Passage {i}.", "doc.pdf", page_number=i) for i in range(1, 6)]
        ctx = build_grounded_context(chunks, max_chunks=2)
        # Should contain only 2 citations
        assert ctx.count("[doc.pdf,") == 2

    def test_character_budget_per_chunk_is_respected(self):
        long_text = "A" * 2000
        c = _make_chunk(long_text, "paper.pdf", page_number=1)
        ctx = build_grounded_context([c], max_chunks=1, max_chars_per_chunk=100)
        # The text section should not exceed the budget
        lines = ctx.split("\n")
        text_line = lines[1] if len(lines) > 1 else ""
        assert len(text_line) <= 100


# ── check_evidence_sufficiency ────────────────────────────────────────────────

class TestCheckEvidenceSufficiency:
    def test_sufficient_evidence_returns_true(self):
        chunks = [
            _make_chunk(
                "Attention mechanisms allow the model to focus on relevant tokens.",
                "attention.pdf",
                page_number=4,
                score=0.72,
            ),
            _make_chunk(
                "The transformer architecture uses multi-head attention layers.",
                "attention.pdf",
                page_number=5,
                score=0.65,
            ),
        ]
        verdict = check_evidence_sufficiency(
            chunks,
            query="How does attention work in transformers?",
            score_threshold=0.38,
        )
        assert verdict.sufficient is True
        assert verdict.should_abstain is False
        assert verdict.top_score > 0.5
        assert len(verdict.citations) >= 1

    def test_insufficient_score_triggers_abstention(self):
        chunks = [
            _make_chunk("Some unrelated text.", "paper.pdf", page_number=1, score=0.22),
            _make_chunk("Another unrelated passage.", "paper.pdf", page_number=2, score=0.18),
        ]
        verdict = check_evidence_sufficiency(
            chunks,
            query="What is the capital of France?",
            score_threshold=0.38,
        )
        assert verdict.sufficient is False
        assert verdict.should_abstain is True

    def test_empty_chunks_triggers_abstention(self):
        verdict = check_evidence_sufficiency([], query="Any question?")
        assert verdict.should_abstain is True
        assert verdict.confidence == 0.0
        assert verdict.chunk_count == 0

    def test_confidence_is_between_0_and_1(self):
        chunks = [_make_chunk("relevant text about transformers", "doc.pdf", page_number=1, score=0.6)]
        verdict = check_evidence_sufficiency(chunks, query="transformers attention model")
        assert 0.0 <= verdict.confidence <= 1.0

    def test_verdict_citations_only_for_above_threshold_chunks(self):
        chunks = [
            _make_chunk("Good passage with sufficient score.", "good.pdf", page_number=1, score=0.7),
            _make_chunk("Poor passage below threshold.", "bad.pdf", page_number=1, score=0.1),
        ]
        verdict = check_evidence_sufficiency(
            chunks, query="sufficient score passage", score_threshold=0.38
        )
        # Only the good chunk should be cited
        assert any("good.pdf" in c for c in verdict.citations)
        assert not any("bad.pdf" in c for c in verdict.citations)


# ── build_abstention_message ──────────────────────────────────────────────────

class TestBuildAbstentionMessage:
    def test_message_contains_core_elements(self):
        verdict = EvidenceVerdict(
            sufficient=False,
            should_abstain=True,
            confidence=0.0,
            reason="No passages exceed threshold.",
            top_score=0.2,
            chunk_count=5,
        )
        msg = build_abstention_message(verdict, query="What is the activation function used?")
        assert "Insufficient evidence" in msg
        assert "No passages exceed threshold." in msg
        assert "What is the activation function" in msg

    def test_message_without_query_still_valid(self):
        verdict = EvidenceVerdict(
            sufficient=False,
            should_abstain=True,
            confidence=0.0,
            reason="No results found.",
        )
        msg = build_abstention_message(verdict)
        assert "Insufficient evidence" in msg
        assert "No results found." in msg
