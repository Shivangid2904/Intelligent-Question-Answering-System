"""
IntelliAsk Research — Structured Paper Extraction

Provides extraction logic for converting paper chunks into structured
PaperExtraction objects using the local LLM with JSON output validation.
"""

import json
import re
from typing import List, Optional

import requests

from src.extraction.schemas import (
    ExtractedField,
    MetricResult,
    PaperExtraction,
    ProvenanceSpan,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"
MAX_RETRIES = 2


def _strip_markdown_fences(json_str: str) -> str:
    """
    Remove markdown code block fences (```json ... ```) from LLM output.
    Handles common variations including triple backticks with and without language tags.
    """
    # Strip first to handle surrounding whitespace
    json_str = json_str.strip()
    # Remove opening fence with optional language tag and whitespace
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str, flags=re.MULTILINE)
    # Remove closing fence with optional whitespace
    json_str = re.sub(r"\s*```$", "", json_str, flags=re.MULTILINE)
    return json_str.strip()


def _repair_trailing_commas(json_str: str) -> str:
    """
    Remove trailing commas from JSON objects and arrays.
    This is a common issue with LLM-generated JSON.
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json_str


def _attempt_json_parse(json_str: str) -> Optional[dict]:
    """
    Attempt to parse JSON string with progressive repair strategies.
    Returns None if all strategies fail.
    """
    strategies = [
        lambda s: json.loads(s),  # Direct parse
        lambda s: json.loads(_strip_markdown_fences(s)),  # Strip markdown
        lambda s: json.loads(_repair_trailing_commas(s)),  # Repair trailing commas
        lambda s: json.loads(_repair_trailing_commas(_strip_markdown_fences(s))),  # Both
    ]

    for strategy in strategies:
        try:
            return strategy(json_str)
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _build_extraction_prompt(
    chunks: List, source_filename: str, paper_id: str
) -> str:
    """
    Build a structured prompt for the LLM to extract paper metadata.

    The prompt instructs the model to output valid JSON conforming to the
    PaperExtraction schema, with explicit instructions about provenance
    and not fabricating missing information.
    """
    # Build context from chunks, including page numbers
    context_parts = []
    for chunk in chunks:
        page = getattr(chunk, "page_number", "?")
        text = getattr(chunk, "text", str(chunk))
        context_parts.append(f"[Page {page}]\n{text.strip()[:500]}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a scientific literature extraction assistant. Extract structured metadata from the following paper text.

Paper ID: {paper_id}
Filename: {source_filename}

Context passages (with page numbers):
{context}

Extract the following fields and output ONLY valid JSON:
- title: Full paper title
- authors: Author list or "First Author et al."
- year: Publication year (4-digit)
- research_objective: 1-2 sentence summary of the research goal
- proposed_methodology: Brief description of the method/architecture
- dataset_used: Dataset names or "Not reported"
- hardware_or_sensors: Hardware/sensors or "Not reported"
- quantitative_findings: List of metrics with metric_name, value, dataset
- acknowledged_limitations: Limitations stated by authors or "None acknowledged"

IMPORTANT RULES:
1. Output ONLY raw JSON (no markdown code blocks, no explanatory text)
2. If a field is not mentioned in the text, set value to null or "Not reported"
3. Do NOT fabricate or guess values that are not explicitly stated
4. For quantitative_findings, include only metrics explicitly reported with numerical values
5. Each finding should include metric_name (e.g. "mAP@0.5"), value (e.g. "68.2%"), and optional dataset

JSON output:"""

    return prompt


def _call_llm_for_extraction(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Call Ollama LLM for extraction with streaming disabled for full response.
    Returns the complete response text.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )

        if response.status_code == 404:
            raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
        elif response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        data = response.json()
        return data.get("response", "")

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out during extraction.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error calling LLM: {str(e)}")


def _map_llm_output_to_schema(
    llm_output: dict,
    paper_id: str,
    chunks: List,
) -> PaperExtraction:
    """
    Map LLM JSON output to PaperExtraction schema with provenance.

    Args:
        llm_output: Parsed JSON dict from LLM.
        paper_id: Document identifier.
        chunks: Original chunks for provenance mapping.

    Returns:
        Validated PaperExtraction object.
    """
    # Helper to create ExtractedField with simple provenance
    def make_field(
        value: Optional[str], is_stated: bool = True, page_numbers: Optional[List[int]] = None
    ) -> ExtractedField:
        provenance = []
        if value and is_stated and page_numbers:
            provenance = [
                ProvenanceSpan(page_number=p, verbatim_quote=None) for p in page_numbers
            ]
        return ExtractedField(value=value, is_stated=is_stated, provenance=provenance)

    # Get page numbers from chunks for provenance
    all_pages = sorted(set(getattr(c, "page_number", 1) for c in chunks))

    # Map fields
    title = make_field(llm_output.get("title"), page_numbers=all_pages[:1])
    authors = make_field(llm_output.get("authors"), page_numbers=all_pages[:1])
    year = make_field(llm_output.get("year"), page_numbers=all_pages[:1])
    research_objective = make_field(
        llm_output.get("research_objective"), page_numbers=all_pages[:2]
    )
    proposed_methodology = make_field(
        llm_output.get("proposed_methodology"), page_numbers=all_pages[1:3]
    )
    dataset_used = make_field(
        llm_output.get("dataset_used"), page_numbers=all_pages[2:4]
    )
    hardware_or_sensors = make_field(
        llm_output.get("hardware_or_sensors"), page_numbers=all_pages[2:4]
    )
    acknowledged_limitations = make_field(
        llm_output.get("acknowledged_limitations"), page_numbers=all_pages[-2:]
    )

    # Map quantitative findings
    quantitative_findings = []
    raw_findings = llm_output.get("quantitative_findings", [])
    if isinstance(raw_findings, list):
        for finding in raw_findings:
            if isinstance(finding, dict):
                metric_name = finding.get("metric_name")
                value = finding.get("value")
                dataset = finding.get("dataset")

                if metric_name and value:
                    metric = MetricResult(
                        metric_name=metric_name,
                        value=value,
                        dataset=dataset,
                        provenance=ProvenanceSpan(page_number=all_pages[-1], verbatim_quote=None),
                    )
                    quantitative_findings.append(metric)

    return PaperExtraction(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        research_objective=research_objective,
        proposed_methodology=proposed_methodology,
        dataset_used=dataset_used,
        hardware_or_sensors=hardware_or_sensors,
        quantitative_findings=quantitative_findings,
        acknowledged_limitations=acknowledged_limitations,
    )


def extract_paper_structure(
    chunks: List,
    paper_id: str,
    source_filename: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
) -> PaperExtraction:
    """
    Extract structured paper metadata from document chunks using local LLM.

    This function:
    1. Builds a structured extraction prompt from chunks with page numbers
    2. Calls the local Ollama LLM for JSON output
    3. Repairs common JSON formatting issues (markdown fences, trailing commas)
    4. Validates output against PaperExtraction Pydantic schema
    5. Retries on failure up to max_retries
    6. Returns a validated PaperExtraction or raises a clear error

    Args:
        chunks: List of Chunk objects with text, page_number, source_filename.
        paper_id: Unique document identifier.
        source_filename: Original PDF filename.
        model: Ollama model name (default: llama3.2:1b).
        max_retries: Maximum retry attempts on parsing/validation failure.

    Returns:
        Validated PaperExtraction object.

    Raises:
        RuntimeError: If LLM connection fails or all retries are exhausted.
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot extract from empty chunks list")

    # Build prompt
    prompt = _build_extraction_prompt(chunks, source_filename, paper_id)

    # Attempt extraction with retries
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Call LLM
            llm_response = _call_llm_for_extraction(prompt, model)

            # Parse JSON with repair strategies
            parsed = _attempt_json_parse(llm_response)
            if parsed is None:
                raise ValueError("Failed to parse LLM output as JSON after repair attempts")

            # Map to schema
            extraction = _map_llm_output_to_schema(parsed, paper_id, chunks)

            # Validate with Pydantic
            validated = PaperExtraction.model_validate(extraction)

            return validated

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                continue  # Retry
            else:
                # All retries exhausted
                raise RuntimeError(
                    f"Extraction failed after {max_retries + 1} attempts. "
                    f"Last error: {str(last_error)}"
                ) from last_error
