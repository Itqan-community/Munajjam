"""
Similarity matching algorithms for Arabic text.

This module provides functions for computing similarity between
transcribed audio text and reference Quran text.
"""

from difflib import SequenceMatcher

from munajjam.core.arabic import normalize_arabic


def similarity(text1: str, text2: str, normalize: bool = True) -> float:
    """
    Compute similarity ratio between two strings.

    Uses SequenceMatcher to compute a ratio between 0.0 (no similarity)
    and 1.0 (identical strings).

    Args:
        text1: First string to compare
        text2: Second string to compare
        normalize: Whether to normalize Arabic text before comparison

    Returns:
        Similarity ratio between 0.0 and 1.0

    Examples:
        >>> similarity("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم")
        1.0
        >>> similarity("بِسْمِ اللَّهِ", "بسم الله", normalize=True)
        1.0
    """
    if normalize:
        text1 = normalize_arabic(text1)
        text2 = normalize_arabic(text2)

    return SequenceMatcher(None, text1, text2).ratio()


def get_first_words(text: str, n: int = 1, normalize: bool = True) -> str:
    """
    Get the first n words from text.

    Args:
        text: Input text
        n: Number of words to extract
        normalize: Whether to normalize Arabic text

    Returns:
        First n words joined by spaces
    """
    if normalize:
        text = normalize_arabic(text)

    words = text.split()
    return " ".join(words[:n]) if len(words) >= n else " ".join(words)


def get_last_words(text: str, n: int = 1, normalize: bool = True) -> str:
    """
    Get the last n words from text.

    Args:
        text: Input text
        n: Number of words to extract
        normalize: Whether to normalize Arabic text

    Returns:
        Last n words joined by spaces
    """
    if normalize:
        text = normalize_arabic(text)

    words = text.split()
    return " ".join(words[-n:]) if len(words) >= n else " ".join(words)


def get_first_last_words(
    text: str, n: int = 1, normalize: bool = True
) -> tuple[str, str]:
    """
    Get both first n and last n words from text.

    Args:
        text: Input text
        n: Number of words to extract from each end
        normalize: Whether to normalize Arabic text

    Returns:
        Tuple of (first_n_words, last_n_words)
    """
    if normalize:
        text = normalize_arabic(text)

    words = text.split()

    first = " ".join(words[:n]) if len(words) >= n else " ".join(words)
    last = " ".join(words[-n:]) if len(words) >= n else " ".join(words)

    return first, last


def compute_coverage_ratio(transcribed_text: str, ayah_text: str) -> float:
    """
    Compute the word coverage ratio of transcribed text vs ayah text.

    This helps determine if the transcription covers enough of the ayah
    to be considered a complete match.

    Args:
        transcribed_text: Text from transcription
        ayah_text: Reference ayah text

    Returns:
        Ratio of transcribed words to ayah words (can be > 1.0)
    """
    trans_words = len(normalize_arabic(transcribed_text).split())
    ayah_words = len(normalize_arabic(ayah_text).split())

    if ayah_words == 0:
        return 0.0

    return trans_words / ayah_words


def check_boundary_match(
    segment_text: str,
    ayah_text: str,
    position: str = "end",
    n_words: int = 3,
    threshold: float = 0.6,
) -> bool:
    """
    Check if segment text matches ayah at a boundary (start or end).

    Args:
        segment_text: Transcribed segment text
        ayah_text: Reference ayah text
        position: 'start' or 'end' boundary to check
        n_words: Number of words to compare
        threshold: Minimum similarity for a match

    Returns:
        True if boundary matches with sufficient similarity
    """
    # Adjust n_words based on ayah length
    ayah_words = normalize_arabic(ayah_text).split()
    actual_n = min(n_words, len(ayah_words))
    if actual_n == 0:
        return False

    if position == "start":
        seg_words = get_first_words(segment_text, actual_n)
        ayah_words_str = get_first_words(ayah_text, actual_n)
    else:  # end
        seg_words = get_last_words(segment_text, actual_n)
        ayah_words_str = get_last_words(ayah_text, actual_n)

    sim = similarity(seg_words, ayah_words_str, normalize=False)  # Already normalized
    return sim >= threshold

