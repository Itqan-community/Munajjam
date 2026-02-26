"""
JSON output formatter for alignment results.

Provides standardized JSON structure for alignment output with metadata
and formatted ayah results.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field

from munajjam.models import AlignmentResult, Ayah


class FormattedAyahResult(BaseModel):
    """
    Standardized format for a single aligned ayah result.

    This model provides a consistent JSON structure for each ayah alignment,
    including timing, text, and quality metrics.

    Attributes:
        id: Unique identifier for the ayah result (ayah_number)
        sura_id: Surah number (1-114)
        ayah_index: Zero-based ayah index within the surah
        start: Start time in seconds
        end: End time in seconds
        transcribed_text: Text as transcribed from audio
        corrected_text: Original Quranic text
        similarity_score: Similarity between transcribed and corrected text (0.0-1.0)
        duration: Duration of the segment in seconds (computed)
        high_confidence: Whether similarity_score >= 0.8 (computed)
    """

    id: int = Field(
        ...,
        description="Ayah number within the surah (1-based)",
        ge=1,
    )
    sura_id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
    )
    ayah_index: int = Field(
        ...,
        description="Zero-based ayah index within the surah",
        ge=0,
    )
    start: float = Field(
        ...,
        description="Start time in seconds",
        ge=0.0,
    )
    end: float = Field(
        ...,
        description="End time in seconds",
        ge=0.0,
    )
    transcribed_text: str = Field(
        ...,
        description="Text as transcribed from audio",
    )
    corrected_text: str = Field(
        ...,
        description="Original Quranic text",
    )
    similarity_score: float = Field(
        ...,
        description="Similarity between transcribed and corrected text (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @computed_field
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return round(self.end - self.start, 2)

    @computed_field
    @property
    def high_confidence(self) -> bool:
        """Whether the alignment has high confidence (>= 0.8 similarity)."""
        return self.similarity_score >= 0.8

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "sura_id": 1,
                    "ayah_index": 0,
                    "start": 0.0,
                    "end": 5.32,
                    "transcribed_text": "بسم الله الرحمن الرحيم",
                    "corrected_text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                    "similarity_score": 0.95,
                    "duration": 5.32,
                    "high_confidence": True,
                }
            ]
        }
    }


class AlignmentOutput(BaseModel):
    """
    Standardized output format for alignment results.

    This model provides a canonical JSON structure for alignment output,
    including metadata and a list of formatted ayah results.

    Attributes:
        version: Output format version
        generated_at: ISO 8601 timestamp of when output was generated
        surah_id: Surah number (1-114)
        reciter: Name of the reciter
        total_ayahs: Total number of ayahs in output (computed)
        avg_similarity: Average similarity score across all ayahs (computed)
        high_confidence_count: Number of high-confidence alignments (computed)
        results: List of formatted ayah alignment results
        metadata: Optional additional metadata
    """

    version: str = Field(
        default="1.0.0",
        description="Output format version",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp of when output was generated",
    )
    surah_id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
    )
    reciter: str = Field(
        default="Unknown",
        description="Name of the reciter",
    )
    results: list[FormattedAyahResult] = Field(
        default_factory=list,
        description="List of formatted ayah alignment results",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional additional metadata",
    )

    @computed_field
    @property
    def total_ayahs(self) -> int:
        """Total number of ayahs in output."""
        return len(self.results)

    @computed_field
    @property
    def avg_similarity(self) -> float:
        """Average similarity score across all ayahs."""
        if not self.results:
            return 0.0
        return round(
            sum(r.similarity_score for r in self.results) / len(self.results),
            3
        )

    @computed_field
    @property
    def high_confidence_count(self) -> int:
        """Number of high-confidence alignments (>= 0.8 similarity)."""
        return sum(1 for r in self.results if r.high_confidence)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serialize output to JSON string.

        Args:
            indent: Indentation level for pretty-printing (None for compact)

        Returns:
            JSON string representation of the output
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert output to dictionary.

        Returns:
            Dictionary representation of the output
        """
        return self.model_dump()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "version": "1.0.0",
                    "generated_at": "2024-01-15T10:30:00.000000Z",
                    "surah_id": 1,
                    "reciter": "Mishary Al-Afasy",
                    "total_ayahs": 7,
                    "avg_similarity": 0.945,
                    "high_confidence_count": 7,
                    "results": [],
                    "metadata": None,
                }
            ]
        }
    }


def format_alignment_results(
    results: list[AlignmentResult],
    surah_id: int,
    reciter: str = "Unknown",
    metadata: Optional[dict[str, Any]] = None,
    version: str = "1.0.0",
) -> AlignmentOutput:
    """
    Format alignment results into standardized output structure.

    This is the primary API for converting raw alignment results into
    a standardized JSON format. It transforms AlignmentResult objects
    into FormattedAyahResult objects and wraps them in an AlignmentOutput.

    Args:
        results: List of AlignmentResult objects from the alignment process
        surah_id: Surah number (1-114)
        reciter: Name of the reciter (default: "Unknown")
        metadata: Optional additional metadata to include in output
        version: Output format version (default: "1.0.0")

    Returns:
        AlignmentOutput containing formatted results and metadata

    Example:
        >>> from munajjam.core import align
        >>> from munajjam.data import load_surah_ayahs
        >>> from munajjam.output import format_alignment_results
        >>>
        >>> ayahs = load_surah_ayahs(1)
        >>> results = align(segments, ayahs)
        >>> output = format_alignment_results(
        ...     results,
        ...     surah_id=1,
        ...     reciter="Mishary Al-Afasy"
        ... )
        >>> print(output.to_json())
    """
    formatted_results = [
        FormattedAyahResult(
            id=result.ayah.ayah_number,
            sura_id=result.ayah.surah_id,
            ayah_index=result.ayah.ayah_number - 1,
            start=round(result.start_time, 2),
            end=round(result.end_time, 2),
            transcribed_text=result.transcribed_text,
            corrected_text=result.ayah.text,
            similarity_score=round(result.similarity_score, 3),
        )
        for result in results
    ]

    return AlignmentOutput(
        version=version,
        surah_id=surah_id,
        reciter=reciter,
        results=formatted_results,
        metadata=metadata,
    )
