"""
Pydantic data models for Munajjam library.

These models represent the core data structures used throughout the library:
- Ayah: A single verse from the Quran
- Segment: A transcribed audio segment
- Surah: Surah metadata
- AlignmentResult: Result of aligning a segment to an ayah
"""

from munajjam.models.ayah import Ayah
from munajjam.models.segment import Segment, SegmentType
from munajjam.models.surah import Surah
from munajjam.models.result import AlignmentResult

__all__ = [
    "Ayah",
    "Segment",
    "SegmentType",
    "Surah",
    "AlignmentResult",
]

