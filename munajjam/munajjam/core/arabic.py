"""
Arabic text normalization utilities.

This module provides functions for normalizing Arabic text, which is essential
for accurate comparison between transcribed audio and reference Quran text.

Uses high-performance Rust implementations when available, with
fallback to pure Python implementations.
"""

import re
from typing import Literal

from munajjam.models.segment import Segment, SegmentType


# Try to import Rust implementations for better performance
try:
    import munajjam_rs
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


# Regex patterns for special segments
# Istiadha pattern: handles أعوذ with various alef forms and optional waw prefix
ISTIADHA_PATTERN = re.compile(r"[اأإآٱو]?عوذ\s*بالله\s*من\s*الشيطان\s*الرجيم")

# Basmala pattern: بسم الله الرحمن الرحيم with variations
BASMALA_PATTERN = re.compile(r"(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم")


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for comparison.

    Performs the following normalizations:
    - Replace all alef variants (أ إ آ ا ٱ) with plain alef (ا)
    - Replace alef maqsura (ى) with ya (ي)
    - Replace ta marbuta (ة) with ha (ه)
    - Remove punctuation
    - Collapse multiple spaces

    When munajjam_rs is installed, uses high-performance Rust implementation.

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized text string

    Examples:
        >>> normalize_arabic("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
        'بسم الله الرحمن الرحيم'
        >>> normalize_arabic("أَعُوذُ")
        'اعوذ'
    """
    if not text:
        return ""

    # Use Rust implementation if available
    if _USE_RUST:
        return munajjam_rs.normalize_arabic(text)

    # Fallback to Python implementation
    # Normalize alef variants (including alef wasla ٱ U+0671)
    text = re.sub(r"[أإآاٱ]", "ا", text)

    # Normalize alef maqsura to ya
    text = re.sub(r"ى", "ي", text)

    # Normalize ta marbuta to ha
    text = re.sub(r"ة", "ه", text)

    # Remove punctuation and diacritics (keeping letters and spaces)
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.

    Removes: fatha, kasra, damma, shadda, sukun, tanween, etc.

    Args:
        text: Arabic text with diacritics

    Returns:
        Text without diacritics
    """
    # Arabic diacritics Unicode range: U+064B to U+0652
    diacritics_pattern = re.compile(r"[\u064B-\u0652]")
    return diacritics_pattern.sub("", text)


def detect_special_type(
    segment: Segment | dict,
) -> Literal["istiadha", "basmala"] | None:
    """
    Detect if a segment is a special type (istiadha or basmala).

    This function checks both the segment's explicit type field and
    performs text-based detection for cases where the type is missing.

    Args:
        segment: A Segment model or dict with 'type' and 'text' fields

    Returns:
        'istiadha', 'basmala', or None if not a special type

    Examples:
        >>> detect_special_type({"text": "أعوذ بالله من الشيطان الرجيم", "type": "ayah"})
        'istiadha'
    """
    # Handle both Segment model and dict
    if isinstance(segment, Segment):
        seg_type = segment.type.value if segment.type else None
        text = segment.text
    else:
        seg_type = segment.get("type")
        text = segment.get("text", "")

    # Check explicit type first
    special_types = {"istiadha", "basmala", "basmalah"}
    if seg_type in special_types:
        # Normalize spelling
        return "basmala" if seg_type == "basmalah" else seg_type  # type: ignore

    # Text-based detection
    normalized = normalize_arabic(text)

    if BASMALA_PATTERN.search(normalized):
        return "basmala"

    if ISTIADHA_PATTERN.search(normalized):
        return "istiadha"

    return None


def is_special_segment(segment: Segment | dict) -> bool:
    """
    Check if a segment is a special type (istiadha or basmala).

    Args:
        segment: A Segment model or dict

    Returns:
        True if segment is istiadha or basmala
    """
    return detect_special_type(segment) is not None


def word_count(text: str) -> int:
    """
    Count Arabic words in text.

    Args:
        text: Arabic text

    Returns:
        Number of words
    """
    normalized = normalize_arabic(text)
    if not normalized:
        return 0
    return len(normalized.split())


def detect_segment_type(text: str) -> tuple[SegmentType, int]:
    """
    Detect segment type from transcribed text.

    Used by transcribers to classify segments as ayah, istiadha, or basmala.
    This is canonical implementation - do not duplicate in other modules.

    Args:
        text: Transcribed Arabic text

    Returns:
        Tuple of (segment_type, segment_id)
        segment_id is 0 for special segments, 1 for ayah (to be renumbered later)

    Examples:
        >>> detect_segment_type("أعوذ بالله من الشيطان الرجيم")
        (SegmentType.ISTIADHA, 0)
        >>> detect_segment_type("بسم الله الرحمن الرحيم")
        (SegmentType.BASMALA, 0)
        >>> detect_segment_type("الحمد لله رب العالمين")
        (SegmentType.AYAH, 1)
    """
    normalized = normalize_arabic(text)

    if ISTIADHA_PATTERN.search(normalized):
        return SegmentType.ISTIADHA, 0

    if BASMALA_PATTERN.search(normalized):
        return SegmentType.BASMALA, 0

    return SegmentType.AYAH, 1

