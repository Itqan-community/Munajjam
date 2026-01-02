"""
Overlap detection and removal utilities.

This module provides functions for detecting and removing overlapping
text between adjacent segments, and for applying timing buffers.
"""

from collections import Counter

from munajjam.core.arabic import normalize_arabic


def remove_overlap(text1: str, text2: str) -> tuple[str, bool]:
    """
    Merge text2 into text1 while removing overlapping words.

    When segments are transcribed separately, there can be overlap at
    boundaries. This function removes duplicate words from the merge.

    Args:
        text1: First text (before)
        text2: Second text (to append)

    Returns:
        Tuple of (merged_text, overlap_found)

    Examples:
        >>> remove_overlap("بسم الله", "الله الرحمن")
        ('بسم الله الرحمن', True)
    """
    words1 = normalize_arabic(text1).split()
    words2 = text2.split()

    count1 = Counter(words1)
    cleaned_words2 = []

    for word in words2:
        normalized_word = normalize_arabic(word)
        if count1[normalized_word] > 0:
            count1[normalized_word] -= 1
            continue
        cleaned_words2.append(word)

    if not cleaned_words2:
        return text1.strip(), True

    merged = text1.strip() + " " + " ".join(cleaned_words2).strip()
    overlap_found = len(cleaned_words2) < len(words2)

    return merged, overlap_found


def apply_buffers(
    start_time: float,
    end_time: float,
    silences: list[list[int] | tuple[int, int]],
    prev_end: float | None = None,
    next_start: float | None = None,
    buffer: float = 0.3,
) -> tuple[float, float]:
    """
    Apply buffers to start and end times by extending into adjacent silence periods.

    This improves ayah boundaries by including slight pauses before/after
    the spoken text, creating more natural-sounding segments.

    Args:
        start_time: Original start time in seconds
        end_time: Original end time in seconds
        silences: List of silence periods [[start_ms, end_ms], ...]
        prev_end: Previous ayah's end time (to prevent overlap)
        next_start: Next ayah's start time (to prevent overlap)
        buffer: Buffer duration in seconds (default 0.3)

    Returns:
        Tuple of (new_start_time, new_end_time) with buffers applied
    """
    new_start = start_time
    new_end = end_time

    if not silences:
        return new_start, new_end

    # Convert silences to seconds and sort by start time
    silences_sec = [(s[0] / 1000, s[1] / 1000) for s in silences]
    silences_sec.sort(key=lambda x: x[0])

    # Find silence before start_time
    best_silence_before = None
    for silence_start, silence_end in silences_sec:
        if silence_end <= start_time:
            if best_silence_before is None or silence_end > best_silence_before[1]:
                best_silence_before = (silence_start, silence_end)
        elif silence_start > start_time:
            break

    if best_silence_before:
        silence_start, silence_end = best_silence_before
        available_buffer = start_time - silence_start
        buffer_to_apply = min(buffer, available_buffer)
        buffer_start = start_time - buffer_to_apply

        if prev_end is None:
            new_start = buffer_start
        elif buffer_start >= prev_end:
            new_start = buffer_start
        elif prev_end < start_time:
            new_start = max(buffer_start, prev_end)

    # Find silence after end_time
    best_silence_after = None
    for silence_start, silence_end in silences_sec:
        if silence_start >= end_time:
            if best_silence_after is None or silence_start < best_silence_after[0]:
                best_silence_after = (silence_start, silence_end)
        elif silence_end < end_time:
            continue

    if best_silence_after:
        silence_start, silence_end = best_silence_after
        available_buffer = silence_end - end_time
        buffer_to_apply = min(buffer, available_buffer)
        buffer_end = end_time + buffer_to_apply

        if next_start is None:
            new_end = buffer_end
        elif buffer_end <= next_start:
            new_end = buffer_end
        elif next_start > end_time:
            new_end = min(buffer_end, next_start)

    return new_start, new_end


def find_silence_gap_between(
    current_end: float,
    next_start: float | None,
    silences_sec: list[tuple[float, float]],
    min_gap: float = 0.18,
) -> tuple[float, float] | None:
    """
    Detect a silence gap between two consecutive segments.

    Used to identify natural boundaries between ayahs where there's
    a clear pause in the recitation.

    Args:
        current_end: End time (seconds) of the current segment
        next_start: Start time (seconds) of the next segment
        silences_sec: List of silence periods in seconds [(start, end), ...]
        min_gap: Minimum silence duration to be considered a gap (seconds)

    Returns:
        (silence_start, silence_end) if a qualifying gap exists, otherwise None
    """
    if not silences_sec or next_start is None:
        return None

    for silence_start, silence_end in silences_sec:
        # Skip silences that end before the current segment
        if silence_end <= current_end:
            continue

        # Stop early once we pass the next segment
        if silence_start >= next_start:
            break

        # Silence fully between current_end and next_start
        if silence_start >= current_end and silence_end <= next_start:
            if (silence_end - silence_start) >= min_gap:
                return silence_start, silence_end

    return None


def convert_silences_to_seconds(
    silences_ms: list[list[int] | tuple[int, int]],
) -> list[tuple[float, float]]:
    """
    Convert silence periods from milliseconds to seconds.

    Args:
        silences_ms: Silences in milliseconds [[start_ms, end_ms], ...]

    Returns:
        Silences in seconds [(start_sec, end_sec), ...]
    """
    silences_sec = [(s[0] / 1000, s[1] / 1000) for s in silences_ms]
    silences_sec.sort(key=lambda x: x[0])
    return silences_sec

