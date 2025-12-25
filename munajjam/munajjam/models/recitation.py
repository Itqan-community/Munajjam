"""
Recitation session data model.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from munajjam.models.result import AlignmentResult


class RecitationStatus(str, Enum):
    """Status of a recitation processing session."""

    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    COMPLETED = "completed"
    FAILED = "failed"


class Recitation(BaseModel):
    """
    A complete recitation processing session.

    Represents the full lifecycle of processing a Quran recitation,
    from audio input through transcription and alignment.

    Attributes:
        id: Unique identifier for this recitation session
        reciter_name: Name of the reciter
        surah_id: Surah number (1-114)
        audio_path: Path to the source audio file
        created_at: Timestamp when processing started
        completed_at: Timestamp when processing completed (if finished)
        status: Current processing status
        alignments: List of aligned ayahs (populated after alignment)
        error_message: Error message if status is FAILED
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this recitation session",
    )
    reciter_name: str = Field(
        ...,
        description="Name of the reciter",
    )
    surah_id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
    )
    audio_path: Optional[str] = Field(
        default=None,
        description="Path to the source audio file",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when processing started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing completed",
    )
    status: RecitationStatus = Field(
        default=RecitationStatus.PENDING,
        description="Current processing status",
    )
    alignments: list[AlignmentResult] = Field(
        default_factory=list,
        description="List of aligned ayahs",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is FAILED",
    )

    @property
    def is_complete(self) -> bool:
        """Whether the recitation processing is complete."""
        return self.status == RecitationStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Whether the recitation processing failed."""
        return self.status == RecitationStatus.FAILED

    @property
    def aligned_ayah_count(self) -> int:
        """Number of ayahs that have been aligned."""
        return len(self.alignments)

    @property
    def processing_duration(self) -> Optional[float]:
        """Processing duration in seconds (if completed)."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.created_at).total_seconds()

    def mark_completed(self) -> None:
        """Mark the recitation as completed."""
        self.status = RecitationStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_failed(self, error_message: str) -> None:
        """Mark the recitation as failed with an error message."""
        self.status = RecitationStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now()

    def to_json_output(self) -> list[dict]:
        """
        Convert alignments to JSON-serializable output format.

        Returns:
            List of dicts with ayah timing information
        """
        output = []
        for result in self.alignments:
            output.append({
                "id": result.ayah.ayah_number,
                "sura_id": result.ayah.surah_id,
                "ayah_index": result.ayah.ayah_number - 1,
                "start": result.start_time,
                "end": result.end_time,
                "transcribed_text": result.transcribed_text,
                "corrected_text": result.ayah.text,
                "similarity_score": result.similarity_score,
            })
        return output

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "reciter_name": "Badr Al-Turki",
                    "surah_id": 1,
                    "audio_path": "/path/to/audio/001.wav",
                    "created_at": "2024-01-15T10:30:00",
                    "status": "completed",
                    "alignments": [],
                }
            ]
        }
    }

    def __str__(self) -> str:
        return (
            f"Recitation({self.reciter_name}, Surah {self.surah_id}, "
            f"status={self.status.value}, ayahs={self.aligned_ayah_count})"
        )
