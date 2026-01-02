"""
Alignment result data model.
"""

from pydantic import BaseModel, Field, computed_field

from munajjam.models.ayah import Ayah


class AlignmentResult(BaseModel):
    """
    Result of aligning an audio segment to a Quran ayah.

    This represents a single aligned ayah with its timing information
    and quality metrics.

    Attributes:
        ayah: The aligned Ayah
        start_time: Start time in the audio (seconds)
        end_time: End time in the audio (seconds)
        transcribed_text: The transcribed text from audio
        similarity_score: Similarity between transcribed and original text (0.0-1.0)
        overlap_detected: Whether overlap with adjacent segment was detected
    """

    ayah: Ayah = Field(
        ...,
        description="The aligned Ayah",
    )
    start_time: float = Field(
        ...,
        description="Start time in the audio (seconds)",
        ge=0.0,
    )
    end_time: float = Field(
        ...,
        description="End time in the audio (seconds)",
        ge=0.0,
    )
    transcribed_text: str = Field(
        ...,
        description="The transcribed text from audio",
    )
    similarity_score: float = Field(
        ...,
        description="Similarity score between transcribed and original text (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    overlap_detected: bool = Field(
        default=False,
        description="Whether overlap with adjacent segment was detected and removed",
    )

    @computed_field
    @property
    def duration(self) -> float:
        """Duration of the aligned segment in seconds."""
        return self.end_time - self.start_time

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Whether the alignment has high confidence (>0.8 similarity)."""
        return self.similarity_score >= 0.8

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ayah": {
                        "id": 1,
                        "surah_id": 1,
                        "ayah_number": 1,
                        "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                    },
                    "start_time": 0.0,
                    "end_time": 5.32,
                    "transcribed_text": "بسم الله الرحمن الرحيم",
                    "similarity_score": 0.95,
                    "overlap_detected": False,
                }
            ]
        }
    }

    def __str__(self) -> str:
        return (
            f"AlignmentResult({self.ayah.surah_id}:{self.ayah.ayah_number}, "
            f"{self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"score={self.similarity_score:.2f})"
        )

