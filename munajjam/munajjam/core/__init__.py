"""
Core modules for Munajjam library.

This package contains the core business logic for:
- Alignment of transcribed segments to Quran ayahs
- Arabic text normalization
- Similarity matching algorithms

Primary API:
    from munajjam.core import Aligner, AlignmentStrategy, align

    # Simple usage
    results = align("001.mp3", segments, ayahs)

    # With configuration
    aligner = Aligner("001.mp3", fix_drift=True)
    results = aligner.align(segments, ayahs, silences_ms=silences)
"""

# Primary API - what most users need
from munajjam.core.aligner import Aligner, AlignmentStrategy, align

# Text utilities - commonly used
from munajjam.core.arabic import detect_segment_type, normalize_arabic

# Stats classes - for inspecting results
from munajjam.core.hybrid import HybridStats
from munajjam.core.matcher import similarity
from munajjam.core.zone_realigner import ProblemZone, ZoneStats

__all__ = [
    # Primary API
    "Aligner",
    "AlignmentStrategy",
    "align",
    # Text utilities
    "normalize_arabic",
    "detect_segment_type",
    "similarity",
    # Stats (for debugging/inspection)
    "HybridStats",
    "ProblemZone",
    "ZoneStats",
]
