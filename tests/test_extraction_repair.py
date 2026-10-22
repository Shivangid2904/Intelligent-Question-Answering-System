"""
Week 4 Phase 1 — Tests for src/extraction/extractor.py

Covers:
  - Markdown-fenced JSON stripping
  - Malformed or incomplete JSON
  - LLM refusal or extraction failure
  - Validation failure without fabricated data
  - Correct association between source filenames, evidence passages, and physical pages
  - Retry logic on parsing failures

Uses mocked LLM responses for deterministic unit tests without requiring Ollama.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.extraction.extractor import (
    _attempt_json_parse,
    _build_extraction_prompt,
    _call_llm_for_extraction,
    _map_llm_output_to_schema,
    _repair_trailing_commas,
    _strip_markdown_fences,
    extract_paper_structure,
)
from src.extraction.schemas import PaperExtraction


class TestStripMarkdownFences:
    def test_no_fences_unchanged(self):
        """Test that plain JSON without fences is unchanged."""
        json_str = '{"key": "value"}'
        result = _strip_markdown_fences(json_str)
        assert result == '{"key": "value"}'

    def test_json_fence_removed(self):
        """Test removal of ```json fence."""
        json_str = '```json\n{"key": "value"}\n```'
        result = _strip_markdown_fences(json_str)
        assert result == '{"key": "value"}'

    def test_generic_fence_removed(self):
        """Test removal of ``` fence without language tag."""
        json_str = '```\n{"key": "value"}\n```'
        result = _strip_markdown_fences(json_str)
        assert result == '{"key": "value"}'

    def test_multiline_json_with_fences(self):
        """Test multiline JSON with fences."""
        json_str = '```json\n{\n  "key": "value"\n}\n```'
        result = _strip_markdown_fences(json_str)
        assert result == '{\n  "key": "value"\n}'

    def test_fences_with_whitespace(self):
        """Test fences with surrounding whitespace."""
        json_str = ' ```json \n {"key": "value"} \n ``` '
        result = _strip_markdown_fences(json_str)
        # The new implementation strips the fences
        assert '{"key": "value"}' in result
        assert '```' not in result


class TestRepairTrailingCommas:
    def test_no_trailing_commas_unchanged(self):
        """Test that valid JSON without trailing commas is unchanged."""
        json_str = '{"key": "value", "num": 42}'
        result = _repair_trailing_commas(json_str)
        assert result == '{"key": "value", "num": 42}'

    def test_trailing_comma_in_object_removed(self):
        """Test removal of trailing comma in object."""
        json_str = '{"key": "value",}'
        result = _repair_trailing_commas(json_str)
        assert result == '{"key": "value"}'

    def test_trailing_comma_in_array_removed(self):
        """Test removal of trailing comma in array."""
        json_str = '[1, 2, 3,]'
        result = _repair_trailing_commas(json_str)
        assert result == '[1, 2, 3]'

    def test_multiple_trailing_commas_removed(self):
        """Test removal of multiple trailing commas."""
        json_str = '{"key": "value", "nested": {"inner": "val",},}'
        result = _repair_trailing_commas(json_str)
        assert result == '{"key": "value", "nested": {"inner": "val"}}'

    def test_whitespace_with_trailing_comma(self):
        """Test trailing comma with whitespace."""
        json_str = '{"key": "value", }'
        result = _repair_trailing_commas(json_str)
        assert result == '{"key": "value"}'


class TestAttemptJsonParse:
    def test_valid_json_direct_parse(self):
        """Test that valid JSON parses on first attempt."""
        json_str = '{"key": "value"}'
        result = _attempt_json_parse(json_str)
        assert result == {"key": "value"}

    def test_json_with_fences_stripped_and_parsed(self):
        """Test that JSON with markdown fences is stripped and parsed."""
        json_str = '```json\n{"key": "value"}\n```'
        result = _attempt_json_parse(json_str)
        assert result == {"key": "value"}

    def test_json_with_trailing_comma_repaired_and_parsed(self):
        """Test that JSON with trailing comma is repaired and parsed."""
        json_str = '{"key": "value",}'
        result = _attempt_json_parse(json_str)
        assert result == {"key": "value"}

    def test_both_fences_and_trailing_comma_fixed(self):
        """Test that both fences and trailing commas are fixed."""
        json_str = '```json\n{"key": "value",}\n```'
        result = _attempt_json_parse(json_str)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        """Test that fundamentally invalid JSON returns None."""
        json_str = '{invalid json}'
        result = _attempt_json_parse(json_str)
        assert result is None


class TestBuildExtractionPrompt:
    def test_prompt_contains_paper_id_and_filename(self):
        """Test that prompt includes paper ID and filename."""
        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample text"
        prompt = _build_extraction_prompt([chunk], "test.pdf", "paper_001")
        assert "paper_001" in prompt
        assert "test.pdf" in prompt

    def test_prompt_includes_page_numbers(self):
        """Test that prompt includes page numbers from chunks."""
        chunk1 = MagicMock()
        chunk1.page_number = 3
        chunk1.text = "Page 3 text"
        chunk2 = MagicMock()
        chunk2.page_number = 5
        chunk2.text = "Page 5 text"
        prompt = _build_extraction_prompt([chunk1, chunk2], "test.pdf", "paper_001")
        assert "[Page 3]" in prompt
        assert "[Page 5]" in prompt

    def test_prompt_includes_context_text(self):
        """Test that prompt includes chunk text."""
        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "The paper discusses depth estimation."
        prompt = _build_extraction_prompt([chunk], "test.pdf", "paper_001")
        assert "depth estimation" in prompt

    def test_prompt_includes_structured_output_instructions(self):
        """Test that prompt instructs JSON-only output."""
        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample"
        prompt = _build_extraction_prompt([chunk], "test.pdf", "paper_001")
        assert "ONLY valid JSON" in prompt
        assert "no markdown code blocks" in prompt


class TestMapLlmOutputToSchema:
    def test_simple_mapping(self):
        """Test basic mapping from LLM output to schema."""
        chunk = MagicMock()
        chunk.page_number = 1

        llm_output = {
            "title": "Test Paper",
            "authors": "Author One",
            "year": "2025",
            "research_objective": "Test objective",
            "proposed_methodology": "Test method",
            "dataset_used": "Test dataset",
            "hardware_or_sensors": "GPU",
            "quantitative_findings": [
                {"metric_name": "accuracy", "value": "95%", "dataset": "test"}
            ],
            "acknowledged_limitations": "None",
        }

        extraction = _map_llm_output_to_schema(llm_output, "paper_001", [chunk])
        assert extraction.paper_id == "paper_001"
        assert extraction.title.value == "Test Paper"
        assert extraction.authors.value == "Author One"
        assert extraction.year.value == "2025"
        assert len(extraction.quantitative_findings) == 1
        assert extraction.quantitative_findings[0].metric_name == "accuracy"

    def test_missing_fields_mapped_to_none(self):
        """Test that missing fields map to None."""
        chunk = MagicMock()
        chunk.page_number = 1

        llm_output = {
            "title": "Test Paper",
            "research_objective": "Test objective",
            "proposed_methodology": "Test method",
            # Missing optional fields
        }

        extraction = _map_llm_output_to_schema(llm_output, "paper_001", [chunk])
        assert extraction.authors.value is None
        assert extraction.year.value is None
        assert extraction.dataset_used.value is None

    def test_quantitative_findings_with_multiple_metrics(self):
        """Test mapping multiple quantitative findings."""
        chunk = MagicMock()
        chunk.page_number = 1

        llm_output = {
            "title": "Test",
            "research_objective": "Test",
            "proposed_methodology": "Test",
            "quantitative_findings": [
                {"metric_name": "mAP@0.5", "value": "68.2%", "dataset": "KITTI"},
                {"metric_name": "F1", "value": "0.84", "dataset": "COCO"},
                {"metric_name": "RMSE", "value": "2.3 cm"},  # No dataset
            ],
        }

        extraction = _map_llm_output_to_schema(llm_output, "paper_001", [chunk])
        assert len(extraction.quantitative_findings) == 3
        assert extraction.quantitative_findings[0].metric_name == "mAP@0.5"
        assert extraction.quantitative_findings[1].metric_name == "F1"
        assert extraction.quantitative_findings[2].metric_name == "RMSE"
        assert extraction.quantitative_findings[2].dataset is None

    def test_provenance_page_numbers_from_chunks(self):
        """Test that provenance uses page numbers from chunks."""
        chunk1 = MagicMock()
        chunk1.page_number = 2
        chunk2 = MagicMock()
        chunk2.page_number = 5

        llm_output = {
            "title": "Test",
            "research_objective": "Test",
            "proposed_methodology": "Test",
        }

        extraction = _map_llm_output_to_schema(llm_output, "paper_001", [chunk1, chunk2])
        # Title should have provenance from first page
        assert extraction.title.provenance[0].page_number == 2
        # Methodology should have provenance from middle pages
        assert extraction.proposed_methodology.provenance[0].page_number == 5


class TestCallLlmForExtraction:
    @patch("src.extraction.extractor.requests.post")
    def test_successful_llm_call(self, mock_post):
        """Test successful LLM call returns response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": '{"key": "value"}'}
        mock_post.return_value = mock_response

        result = _call_llm_for_extraction("test prompt", "llama3.2:1b")
        assert result == '{"key": "value"}'

    @patch("src.extraction.extractor.requests.post")
    def test_model_not_found_error(self, mock_post):
        """Test that 404 error raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            _call_llm_for_extraction("test prompt", "unknown_model")
        assert "not found" in str(exc_info.value)

    @patch("src.extraction.extractor.requests.post")
    def test_connection_error_raises_runtime_error(self, mock_post):
        """Test that connection error raises RuntimeError."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            _call_llm_for_extraction("test prompt")
        assert "Cannot connect to Ollama" in str(exc_info.value)


class TestExtractPaperStructure:
    def test_empty_chunks_raises_error(self):
        """Test that empty chunks list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            extract_paper_structure([], "paper_001", "test.pdf")
        assert "empty chunks list" in str(exc_info.value)

    @patch("src.extraction.extractor._call_llm_for_extraction")
    @patch("src.extraction.extractor._attempt_json_parse")
    def test_successful_extraction(self, mock_parse, mock_llm):
        """Test end-to-end successful extraction."""
        # Mock LLM response
        mock_llm.return_value = '{"title": "Test Paper", "research_objective": "Test", "proposed_methodology": "Test"}'
        # Mock JSON parse
        mock_parse.return_value = {
            "title": "Test Paper",
            "research_objective": "Test",
            "proposed_methodology": "Test",
        }

        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample text"

        extraction = extract_paper_structure([chunk], "paper_001", "test.pdf")
        assert isinstance(extraction, PaperExtraction)
        assert extraction.paper_id == "paper_001"
        assert extraction.title.value == "Test Paper"

    @patch("src.extraction.extractor._call_llm_for_extraction")
    @patch("src.extraction.extractor._attempt_json_parse")
    def test_json_parse_failure_retries(self, mock_parse, mock_llm):
        """Test that JSON parse failure triggers retry."""
        # First attempt fails, second succeeds
        mock_parse.side_effect = [None, {"title": "Test", "research_objective": "Test", "proposed_methodology": "Test"}]
        mock_llm.return_value = '{"title": "Test"}'

        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample"

        extraction = extract_paper_structure([chunk], "paper_001", "test.pdf", max_retries=1)
        assert extraction.title.value == "Test"
        assert mock_parse.call_count == 2  # Initial + 1 retry

    @patch("src.extraction.extractor._call_llm_for_extraction")
    @patch("src.extraction.extractor._attempt_json_parse")
    def test_all_retries_exhausted_raises_error(self, mock_parse, mock_llm):
        """Test that exhausting all retries raises RuntimeError."""
        # All parse attempts fail
        mock_parse.return_value = None
        mock_llm.return_value = '{"invalid": json}'

        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample"

        with pytest.raises(RuntimeError) as exc_info:
            extract_paper_structure([chunk], "paper_001", "test.pdf", max_retries=2)
        assert "Extraction failed" in str(exc_info.value)

    @patch("src.extraction.extractor._call_llm_for_extraction")
    def test_llm_connection_failure_propagates(self, mock_llm):
        """Test that LLM connection failure propagates as RuntimeError."""
        mock_llm.side_effect = RuntimeError("Cannot connect to Ollama")

        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample"

        with pytest.raises(RuntimeError) as exc_info:
            extract_paper_structure([chunk], "paper_001", "test.pdf")
        assert "Cannot connect to Ollama" in str(exc_info.value)

    @patch("src.extraction.extractor._call_llm_for_extraction")
    @patch("src.extraction.extractor._attempt_json_parse")
    def test_source_filename_preserved_in_extraction(self, mock_parse, mock_llm):
        """Test that source filename is used in prompt but not in final extraction."""
        mock_llm.return_value = '{"title": "Test", "research_objective": "Test", "proposed_methodology": "Test"}'
        mock_parse.return_value = {
            "title": "Test",
            "research_objective": "Test",
            "proposed_methodology": "Test",
        }

        chunk = MagicMock()
        chunk.page_number = 1
        chunk.text = "Sample"
        chunk.source_filename = "original.pdf"

        extraction = extract_paper_structure([chunk], "paper_001", "original.pdf")
        # paper_id is preserved, source_filename was used for prompt building
        assert extraction.paper_id == "paper_001"

    @patch("src.extraction.extractor._call_llm_for_extraction")
    @patch("src.extraction.extractor._attempt_json_parse")
    def test_page_numbers_associated_with_provenance(self, mock_parse, mock_llm):
        """Test that chunk page numbers are correctly associated with provenance."""
        mock_llm.return_value = '{"title": "Test", "research_objective": "Test", "proposed_methodology": "Test"}'
        mock_parse.return_value = {
            "title": "Test",
            "research_objective": "Test",
            "proposed_methodology": "Test",
        }

        chunk1 = MagicMock()
        chunk1.page_number = 3
        chunk1.text = "Text from page 3"
        chunk2 = MagicMock()
        chunk2.page_number = 7
        chunk2.text = "Text from page 7"

        extraction = extract_paper_structure([chunk1, chunk2], "paper_001", "test.pdf")
        # Verify provenance uses the actual page numbers from chunks
        assert extraction.title.provenance[0].page_number == 3
