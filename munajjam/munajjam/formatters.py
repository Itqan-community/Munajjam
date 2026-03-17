"""
Standardized JSON output formatter for Munajjam.

This module provides a canonical JSON formatter that all consumers should use
instead of hand-rolling their own JSON output format.

Usage:
    from munajjam.formatters import format_alignment_results, AlignmentOutput

    # Format results to a standardized dict
    output = format_alignment_results(
        results=alignment_results,
        surah_id=1,
        reciter="Badr Al-Turki",
        audio_file="surah_001.wav",
    )

    # Convert to JSON string
    json_str = output.to_json()

    # Or get as dict
    data = output.to_dict()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from munajjam._version import __version__
from munajjam.models.result import AlignmentResult
from munajjam.models.surah import SURAH_NAMES


class FormattedAyahResult(BaseModel):
    """Standardized representation of a single aligned ayah."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(
        ...,
        ge=1,
        description="Unique ayah identifier (1-6236)",
    )
    surah_id: int = Field(
        ...,
        ge=1,
        le=114,
        description="Surah number (1-114)",
    )
    ayah_number: int = Field(
        ...,
        ge=1,
        description="Ayah number within the surah (1-based)",
    )
    ayah_index: int = Field(
        ...,
        ge=0,
        description="Zero-based ayah index within the surah",
    )
    start_time: float = Field(
        ...,
        ge=0.0,
        description="Start time in seconds (rounded to 2 decimal places)",
    )
    end_time: float = Field(
        ...,
        ge=0.0,
        description="End time in seconds (rounded to 2 decimal places)",
    )
    duration: float = Field(
        ...,
        ge=0.0,
        description="Duration in seconds (rounded to 2 decimal places)",
    )
    transcribed_text: str = Field(
        ...,
        description="The transcribed text from audio",
    )
    original_text: str = Field(
        ...,
        description="The original Quran text of the ayah",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between transcribed and original text (0.0-1.0)",
    )
    is_high_confidence: bool = Field(
        ...,
        description="Whether the alignment has high confidence (>= 0.8)",
    )
    overlap_detected: bool = Field(
        ...,
        description="Whether overlap with adjacent segment was detected",
    )


class AlignmentMetadata(BaseModel):
    """Metadata about the alignment process."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        default="1.0",
        description="Schema version of this output format",
    )
    munajjam_version: str = Field(
        ...,
        description="Version of Munajjam used for alignment",
    )
    generated_at: str = Field(
        ...,
        description="ISO 8601 timestamp of when the output was generated",
    )
    surah_id: int | None = Field(
        default=None,
        ge=1,
        le=114,
        description="Surah number (1-114)",
    )
    surah_name: str | None = Field(
        default=None,
        description="Arabic name of the surah (auto-detected from surah_id)",
    )
    reciter: str | None = Field(
        default=None,
        description="Name of the reciter",
    )
    audio_file: str | None = Field(
        default=None,
        description="Path or name of the audio file",
    )
    total_ayahs: int = Field(
        ...,
        description="Total number of aligned ayahs",
    )
    total_duration: float = Field(
        ...,
        description="Total duration of all aligned segments in seconds",
    )
    average_confidence: float = Field(
        ...,
        description="Average similarity score across all results",
    )
    high_confidence_count: int = Field(
        ...,
        description="Number of high-confidence alignments",
    )


class AlignmentOutput(BaseModel):
    """
    Standardized output format for Munajjam alignment results.

    This is the canonical output format that all consumers should use.
    It includes metadata about the alignment process and a list of
    formatted ayah results.
    """

    model_config = ConfigDict(frozen=True)

    metadata: AlignmentMetadata = Field(
        ...,
        description="Metadata about the alignment process",
    )
    results: list[FormattedAyahResult] = Field(
        ...,
        description="List of aligned ayah results",
    )

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """Serialize to a JSON string.

        Args:
            indent: Number of spaces for indentation.
            ensure_ascii: If True, escape non-ASCII characters.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=ensure_ascii)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a Python dictionary.

        Returns:
            Dictionary representation.
        """
        return self.model_dump()

    def to_file(self, path: str, indent: int = 2, ensure_ascii: bool = False) -> None:
        """Write JSON output to a file.

        Args:
            path: File path to write to.
            indent: Number of spaces for indentation.
            ensure_ascii: If True, escape non-ASCII characters.
        """
        from pathlib import Path as P

        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")


def _format_single_result(result: AlignmentResult) -> FormattedAyahResult:
    """Convert a single AlignmentResult to the standardized format.

    Args:
        result: An AlignmentResult instance.

    Returns:
        A FormattedAyahResult instance.
    """
    # Fix duration rounding inconsistency:
    # Calculate duration from raw values first, then round.
    rounded_start = round(result.start_time, 2)
    rounded_end = round(result.end_time, 2)
    duration = round(result.end_time - result.start_time, 2)

    return FormattedAyahResult(
        id=result.ayah.id,
        surah_id=result.ayah.surah_id,
        ayah_number=result.ayah.ayah_number,
        ayah_index=result.ayah.ayah_number - 1,
        start_time=rounded_start,
        end_time=rounded_end,
        duration=duration,
        transcribed_text=result.transcribed_text,
        original_text=result.ayah.text,
        similarity_score=round(result.similarity_score, 3),
        is_high_confidence=result.is_high_confidence,
        overlap_detected=result.overlap_detected,
    )


def format_alignment_results(
    results: list[AlignmentResult],
    surah_id: int | None = None,
    surah_name: str | None = None,
    reciter: str | None = None,
    audio_file: str | None = None,
) -> AlignmentOutput:
    """
    Format a list of AlignmentResult objects into the standardized JSON output.

    This is the main entry point for formatting alignment results. All consumers
    should use this function instead of building their own JSON output.

    Args:
        results: List of AlignmentResult objects from the alignment process.
        surah_id: Optional surah number (1-114).
        surah_name: Optional surah name. If None and surah_id is provided,
            auto-detected from SURAH_NAMES.
        reciter: Optional name of the reciter.
        audio_file: Optional path or name of the audio file.

    Returns:
        An AlignmentOutput instance with metadata and formatted results.

    Example:
        >>> from munajjam.formatters import format_alignment_results
        >>> output = format_alignment_results(results, surah_id=1)
        >>> print(output.to_json())
    """
    formatted = [_format_single_result(r) for r in results]

    total_duration = sum(r.duration for r in formatted)
    avg_confidence = (
        sum(r.similarity_score for r in formatted) / len(formatted) if formatted else 0.0
    )
    high_conf_count = sum(1 for r in formatted if r.is_high_confidence)

    # Auto-detect surah name from surah_id if not provided
    if surah_name is None and surah_id is not None and 1 <= surah_id <= 114:
        surah_name = SURAH_NAMES.get(surah_id)

    metadata = AlignmentMetadata(
        munajjam_version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
        surah_id=surah_id,
        surah_name=surah_name,
        reciter=reciter,
        audio_file=audio_file,
        total_ayahs=len(formatted),
        total_duration=round(total_duration, 2),
        average_confidence=round(avg_confidence, 3),
        high_confidence_count=high_conf_count,
    )

    return AlignmentOutput(metadata=metadata, results=formatted)
