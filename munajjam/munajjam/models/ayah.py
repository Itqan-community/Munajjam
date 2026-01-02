"""
Ayah (verse) data model.
"""

from pydantic import BaseModel, Field


class Ayah(BaseModel):
    """
    Represents a single ayah (verse) from the Quran.

    Attributes:
        id: Unique identifier for the ayah (1-6236)
        surah_id: Surah number (1-114)
        ayah_number: Ayah number within the surah (1-based)
        text: The Arabic text of the ayah
    """

    id: int = Field(
        ...,
        description="Unique identifier for the ayah (1-6236)",
        ge=1,
    )
    surah_id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
    )
    ayah_number: int = Field(
        ...,
        description="Ayah number within the surah (1-based)",
        ge=1,
    )
    text: str = Field(
        ...,
        description="The Arabic text of the ayah",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "surah_id": 1,
                    "ayah_number": 1,
                    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                }
            ]
        }
    }

    def __str__(self) -> str:
        return f"Ayah({self.surah_id}:{self.ayah_number})"

    def __repr__(self) -> str:
        return f"Ayah(id={self.id}, surah_id={self.surah_id}, ayah_number={self.ayah_number})"

