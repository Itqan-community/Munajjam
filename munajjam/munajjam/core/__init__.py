"""
Core modules for Munajjam library.

This package contains the core business logic for:
- Arabic text normalization
- Similarity matching algorithms
- Overlap detection and removal
- Ayah-segment alignment
"""

from munajjam.core.arabic import normalize_arabic, detect_special_type
from munajjam.core.matcher import (
    similarity,
    get_first_words,
    get_last_words,
    get_first_last_words,
    compute_coverage_ratio,
    check_boundary_match,
)
from munajjam.core.overlap import (
    remove_overlap,
    apply_buffers,
    find_silence_gap_between,
    convert_silences_to_seconds,
)
from munajjam.core.aligner import align_segments, AlignmentContext, get_alignment_stats

__all__ = [
    # Arabic
    "normalize_arabic",
    "detect_special_type",
    # Matcher
    "similarity",
    "get_first_words",
    "get_last_words",
    "get_first_last_words",
    "compute_coverage_ratio",
    "check_boundary_match",
    # Overlap
    "remove_overlap",
    "apply_buffers",
    "find_silence_gap_between",
    "convert_silences_to_seconds",
    # Aligner
    "align_segments",
    "AlignmentContext",
    "get_alignment_stats",
]

