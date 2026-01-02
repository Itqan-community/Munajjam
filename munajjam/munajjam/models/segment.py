"""
Audio segment data model.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SegmentType(str, Enum):
    """Type of audio segment."""

    AYAH = "ayah"
    ISTI3AZA = "isti3aza"  # أعوذ بالله من الشيطان الرجيم
    BASMALA = "basmala"  # بسم الله الرحمن الرحيم


class Segment(BaseModel):
    """
    Represents a transcribed audio segment.

    A segment is a contiguous portion of audio that has been transcribed
    to text. Segments can be ayahs, isti'aza (seeking refuge), or basmala.

    Attributes:
        id: Segment identifier (0 for special segments, positive for ayahs)
        surah_id: Surah number this segment belongs to
        start: Start time in seconds
        end: End time in seconds
        text: Transcribed Arabic text
        type: Type of segment (ayah, isti3aza, basmala)
        confidence: Optional confidence score from transcription (0.0-1.0)
    """

    id: int = Field(
        ...,
        description="Segment identifier (0 for special segments)",
    )
    surah_id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
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
    text: str = Field(
        ...,
        description="Transcribed Arabic text",
    )
    type: SegmentType = Field(
        default=SegmentType.AYAH,
        description="Type of segment",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score from transcription (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end time must be >= start time")
        return v

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    @property
    def is_special(self) -> bool:
        """Whether this is a special segment (isti'aza or basmala)."""
        return self.type in (SegmentType.ISTI3AZA, SegmentType.BASMALA)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "surah_id": 1,
                    "start": 0.0,
                    "end": 5.32,
                    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                    "type": "ayah",
                    "confidence": 0.95,
                }
            ]
        }
    }

    def __str__(self) -> str:
        return f"Segment({self.start:.2f}s-{self.end:.2f}s, {self.type.value})"

