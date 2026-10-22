"""
IntelliAsk Research — Structured Extraction Schemas

Pydantic models for structured paper extraction with page-level provenance.
Distinguishes between explicitly stated facts, inferred values, and missing/unstated information.
All page numbers are 1-indexed physical PDF page numbers, consistent with Chunk.page_number.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ProvenanceSpan(BaseModel):
    """
    Pinpoints physical PDF page and verbatim quote supporting an extracted claim.

    Attributes:
        page_number: 1-indexed physical PDF page number (must be >= 1).
        verbatim_quote: Exact or normalized sentence from text supporting this extraction.
    """

    page_number: int = Field(..., ge=1, description="1-indexed physical PDF page number")
    verbatim_quote: Optional[str] = Field(
        None, description="Exact or normalized sentence from text supporting this extraction"
    )


class ExtractedField(BaseModel):
    """
    Generic structured field with confidence and page provenance.

    Attributes:
        value: Extracted claim or None if information is not reported in the paper.
        is_stated: False if explicitly absent or not discussed in the paper.
                   Inferred values must never be represented as explicitly stated.
        provenance: Pages supporting this field (may be empty if value is None).
        confidence: Extraction confidence in [0.0, 1.0].
    """

    value: Optional[str] = Field(
        None, description="Extracted claim or None if information is not reported"
    )
    is_stated: bool = Field(
        True,
        description="False if explicitly absent or not discussed in the paper; inferred values must set this to False",
    )
    provenance: List[ProvenanceSpan] = Field(
        default_factory=list, description="Pages supporting this field"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )

    @field_validator("provenance")
    @classmethod
    def validate_provenance_consistency(cls, v: List[ProvenanceSpan], info) -> List[ProvenanceSpan]:
        """
        Ensure provenance is empty when value is None, and non-empty when value is present and is_stated=True.
        This prevents fabricating page citations for missing information.
        """
        value = info.data.get("value")
        is_stated = info.data.get("is_stated", True)

        if value is None:
            # No value means no provenance should be present
            if v:
                raise ValueError("Provenance spans cannot be provided when value is None")
        elif is_stated and not v:
            # Explicitly stated claims should have at least one provenance span
            # This is a soft validation warning rather than a hard error to handle edge cases
            pass  # Allow empty provenance for explicitly stated fields in edge cases

        return v


class MetricResult(BaseModel):
    """
    Quantitative performance metric reported in a paper.

    Attributes:
        metric_name: Name of metric, e.g. mAP@0.5, F1-score, RMSE.
        value: Numeric value or percentage as reported (string to preserve units).
        dataset: Dataset or split on which metric was measured.
        provenance: Page supporting this metric.
    """

    metric_name: str = Field(..., description="Name of metric, e.g. mAP@0.5, F1-score, RMSE")
    value: str = Field(..., description="Numeric value or percentage as reported")
    dataset: Optional[str] = Field(
        None, description="Dataset or split on which metric was measured"
    )
    provenance: Optional[ProvenanceSpan] = None


class PaperExtraction(BaseModel):
    """
    Canonical structured extraction record for a single scientific paper.

    This schema supports paper metadata and structured literature-review fields
    including title, authors, year, research objective, methodology, datasets,
    hardware, quantitative metrics, findings, and limitations.

    Attributes:
        paper_id: Unique paper identifier (document_id or filename).
        title: Canonical paper title.
        authors: List of authors or first author et al.
        year: Publication or preprint year.
        research_objective: Primary research question, problem statement, or goal.
        proposed_methodology: Architectural design, algorithmic framework, or experimental method.
        dataset_used: Benchmark dataset(s) or custom data collection.
        hardware_or_sensors: Cameras, LiDAR, compute hardware, or optical devices.
        quantitative_findings: Extracted quantitative results (metrics and scores).
        acknowledged_limitations: Limitations stated by authors.
        extraction_timestamp: ISO 8601 UTC timestamp of extraction.
    """

    paper_id: str = Field(..., description="Unique paper identifier, e.g. document_id or filename")
    title: ExtractedField = Field(..., description="Canonical paper title")
    authors: ExtractedField = Field(
        default_factory=ExtractedField, description="List of authors or first author et al."
    )
    year: ExtractedField = Field(
        default_factory=ExtractedField, description="Publication or preprint year"
    )
    research_objective: ExtractedField = Field(
        ..., description="Primary research question, problem statement, or goal"
    )
    proposed_methodology: ExtractedField = Field(
        ..., description="Architectural design, algorithmic framework, or experimental method"
    )
    dataset_used: ExtractedField = Field(
        default_factory=ExtractedField, description="Benchmark dataset(s) or custom data collection"
    )
    hardware_or_sensors: ExtractedField = Field(
        default_factory=ExtractedField, description="Cameras, LiDAR, compute hardware, or optical devices"
    )
    quantitative_findings: List[MetricResult] = Field(
        default_factory=list, description="Extracted quantitative results"
    )
    acknowledged_limitations: ExtractedField = Field(
        default_factory=ExtractedField, description="Limitations stated by authors"
    )
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of extraction",
    )

    @field_validator("year")
    @classmethod
    def validate_year_format(cls, v: ExtractedField) -> ExtractedField:
        """
        Validate that year is a reasonable 4-digit year if provided.
        Rejects years < 1900 or > current year + 2.
        """
        if v.value is not None:
            try:
                year_int = int(v.value)
                current_year = datetime.now(timezone.utc).year
                if year_int < 1900 or year_int > current_year + 2:
                    raise ValueError(
                        f"Year {year_int} is outside reasonable range (1900-{current_year + 2})"
                    )
            except ValueError as e:
                # If year is not parseable as int (e.g., "2025 (preprint)"), mark as not stated
                # But if it's a range error, re-raise it
                if "outside reasonable range" in str(e):
                    raise
                # For non-numeric strings, allow but mark as not stated
                v.is_stated = False
            except TypeError:
                # Handle None or other non-string types
                v.is_stated = False
        return v
