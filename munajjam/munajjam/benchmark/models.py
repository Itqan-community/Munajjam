"""
Pydantic data models for the benchmark harness.

Defines ground-truth, strategy metrics, and report structures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GroundTruthAyah(BaseModel):
    """A single ayah with manually-verified timestamps."""

    ayah_number: int = Field(..., ge=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def end_after_start(self) -> GroundTruthAyah:
        if self.end_time <= self.start_time:
            msg = f"end_time ({self.end_time}) must be > start_time ({self.start_time})"
            raise ValueError(msg)
        return self


class GroundTruthSurah(BaseModel):
    """Ground-truth timestamps for one surah."""

    surah_id: int = Field(..., ge=1, le=114)
    reciter: str
    audio_filename: str
    ayahs: list[GroundTruthAyah]
    notes: str | None = None


class StrategyMetrics(BaseModel):
    """Per-strategy benchmark results for one surah."""

    strategy: str
    surah_id: int = Field(..., ge=1, le=114)
    mae_start: float = Field(..., ge=0.0)
    mae_end: float = Field(..., ge=0.0)
    avg_similarity: float = Field(..., ge=0.0, le=1.0)
    pct_high_confidence: float = Field(..., ge=0.0, le=100.0)
    runtime_seconds: float = Field(..., ge=0.0)
    ayah_count: int = Field(..., ge=0)
    timestamp: str


class BenchmarkReport(BaseModel):
    """Top-level container for all benchmark results."""

    generated_at: str
    munajjam_version: str
    results: list[StrategyMetrics]
