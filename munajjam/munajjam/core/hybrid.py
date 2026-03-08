"""
Hybrid alignment strategy.

Combines DP alignment with fallback to greedy alignment and
split-and-restitch for long ayahs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ..models import AlignmentResult, Ayah, Segment
from .matcher import similarity


@dataclass
class HybridStats:
    """Statistics from hybrid alignment."""

    total_ayahs: int = 0
    dp_kept: int = 0  # High quality from DP, kept as-is
    old_fallback: int = 0  # Fell back to old aligner
    split_improved: int = 0  # Improved via split-and-restitch
    still_low: int = 0  # Remained low quality after all attempts

    def __str__(self) -> str:
        return (
            f"HybridStats(total={self.total_ayahs}, dp_kept={self.dp_kept}, "
            f"old_fallback={self.old_fallback}, split_improved={self.split_improved}, "
            f"still_low={self.still_low})"
        )


def _find_silences_in_range(
    silences_sec: list[tuple[float, float]],
    start_time: float,
    end_time: float,
    min_duration: float = 0.2,
) -> list[tuple[float, float]]:
    """
    Find silence periods within a given time range.

    This is used to identify natural break points within a long ayah,
    allowing us to split it into smaller chunks for better alignment.

    Args:
        silences_sec: All detected silence periods in seconds [(start, end), ...]
        start_time: Start of the time range to search within
        end_time: End of the time range to search within
        min_duration: Minimum silence duration to consider (default 0.2s)

    Returns:
        List of silence periods that fall within the specified range
        and meet the minimum duration requirement
    """
    result = []
    for sil_start, sil_end in silences_sec:
        # Check if silence overlaps with our time range
        if sil_end > start_time and sil_start < end_time:
            # Clip silence to fit within the range
            clipped_start = max(sil_start, start_time)
            clipped_end = min(sil_end, end_time)
            duration = clipped_end - clipped_start

            # Only include if it's long enough to be a meaningful break
            if duration >= min_duration:
                result.append((clipped_start, clipped_end))
    return result


def _split_segments_at_silences(
    segments: list[Segment],
    silences_sec: list[tuple[float, float]],
    start_time: float,
    end_time: float,
) -> list[list[Segment]]:
    """
    Split segments into chunks based on silence boundaries.

    This is a key part of the "split-and-restitch" strategy for long ayahs.
    By splitting at silences, we can re-align each chunk independently and
    then merge them back together, often resulting in better alignment.

    Args:
        segments: All transcribed segments
        silences_sec: Silence periods in seconds
        start_time: Start of ayah time range
        end_time: End of ayah time range

    Returns:
        List of segment groups, where each group represents a chunk between silences.
        Example: [[seg1, seg2], [seg3], [seg4, seg5, seg6]]
    """
    # Step 1: Filter to only segments within the ayah's time range (with small buffer)
    range_segments = [
        s for s in segments if s.start >= start_time - 0.5 and s.end <= end_time + 0.5
    ]

    if not range_segments:
        return []

    # Step 2: Find silences within this range
    silences = _find_silences_in_range(silences_sec, start_time, end_time)

    # If no silences, return all segments as one chunk
    if not silences:
        return [range_segments]

    silences.sort(key=lambda x: x[0])

    # Step 3: Split segments at each silence boundary
    chunks = []
    current_chunk = []
    silence_idx = 0

    for seg in range_segments:
        if silence_idx < len(silences):
            sil_start, sil_end = silences[silence_idx]

            # Segment ends before this silence - add to current chunk
            if seg.end <= sil_start:
                current_chunk.append(seg)

            # Segment starts after this silence - finish current chunk and move to next
            elif seg.start >= sil_end:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                silence_idx += 1

            # Segment overlaps with silence - add it and finish chunk
            else:
                current_chunk.append(seg)
                chunks.append(current_chunk)
                current_chunk = []
                silence_idx += 1
        else:
            # No more silences, add remaining segments to current chunk
            current_chunk.append(seg)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [range_segments]


def _try_split_and_restitch(
    segments: list[Segment],
    ayah: Ayah,
    dp_result: AlignmentResult,
    silences_ms: list[tuple[int, int]] | None,
) -> AlignmentResult | None:
    """
    Try to improve alignment for a long ayah by splitting at silences.

    Strategy:
    1. Find silence boundaries within the ayah's time range
    2. Split into chunks
    3. Compute similarity for the merged text of all chunks
    4. If better than original, return improved result

    Returns:
        Improved AlignmentResult or None if no improvement
    """
    if not silences_ms:
        return None

    silences_sec = [(s / 1000.0, e / 1000.0) for s, e in silences_ms]

    chunks = _split_segments_at_silences(
        segments, silences_sec, dp_result.start_time, dp_result.end_time
    )

    if len(chunks) <= 1:
        return None

    all_texts = []
    for chunk in chunks:
        chunk_text = " ".join(seg.text for seg in chunk)
        if chunk_text.strip():
            all_texts.append(chunk_text)

    if not all_texts:
        return None

    merged_text = " ".join(all_texts)

    new_sim = similarity(merged_text, ayah.text)

    # Only accept if significantly better (at least 5% improvement)
    if new_sim > dp_result.similarity_score + 0.05:
        return AlignmentResult(
            ayah=ayah,
            start_time=dp_result.start_time,
            end_time=dp_result.end_time,
            transcribed_text=merged_text,
            similarity_score=new_sim,
            overlap_detected=dp_result.overlap_detected,
        )

    return None


def align_segments_hybrid(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    quality_threshold: float = 0.85,
    long_ayah_words: int = 30,
    long_ayah_duration: float = 30.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[AlignmentResult], HybridStats]:
    """
    Hybrid alignment combining DP and greedy aligner with smart fallback.

    This is the RECOMMENDED strategy for most use cases. It provides a good
    balance between accuracy and robustness by:

    1. Starting with DP for optimal alignment quality
    2. Identifying low-confidence ayahs that need improvement
    3. Trying multiple strategies to improve them (split-and-restitch, greedy)
    4. Keeping the best result for each ayah

    Strategy:
    1. Run DP aligner on all segments/ayahs (optimal but can fail on edge cases)
    2. For each ayah with similarity < quality_threshold:
       a. If long ayah (>30 words or >30s): try split-and-restitch
          - Split at silence boundaries
          - Realign each chunk
          - Merge results
       b. Try greedy aligner as fallback
          - Simple linear matching
          - Often works better for problematic cases
       c. Keep whichever result has highest similarity score

    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs (in order)
        silences_ms: Silence periods in milliseconds [(start, end), ...]
        quality_threshold: Similarity below which to try fallback (default 0.85)
        long_ayah_words: Word count threshold for "long" ayahs (default 30)
        long_ayah_duration: Duration threshold for "long" ayahs in seconds (default 30.0)
        on_progress: Optional progress callback (current, total)

    Returns:
        Tuple of:
        - List of AlignmentResult objects (one per ayah)
        - HybridStats object with breakdown of which strategies were used

    Example:
        >>> results, stats = align_segments_hybrid(segments, ayahs, silences_ms)
        >>> print(stats)
        HybridStats(total=7, dp_kept=5, old_fallback=1, split_improved=1, still_low=0)
    """
    # Import here to avoid circular dependency
    from .aligner_greedy import align_segments as align_segments_greedy
    from .dp_core import align_segments_dp_with_constraints

    stats = HybridStats(total_ayahs=len(ayahs))

    if not segments or not ayahs:
        return [], stats

    # ============================================================================
    # PHASE 1: Run DP Aligner (Primary Strategy)
    # ============================================================================
    # DP provides optimal alignment by considering all possible segment-to-ayah
    # mappings and choosing the path with minimum cost. This is our baseline.

    dp_results = align_segments_dp_with_constraints(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
        on_progress=on_progress,
    )

    # Edge case: If DP completely fails (rare), fall back entirely to greedy
    if not dp_results:
        greedy_silences = cast(list[list[int] | tuple[int, int]] | None, silences_ms)
        old_results = align_segments_greedy(
            segments=segments,
            ayahs=ayahs,
            silences_ms=greedy_silences,
        )
        stats.old_fallback = len(old_results)
        return old_results, stats

    # ============================================================================
    # PHASE 2: Run Greedy Aligner (Fallback Strategy)
    # ============================================================================
    # Greedy aligner uses simple linear matching. It's less optimal but more
    # robust to edge cases. We run it now so we can compare results later.

    greedy_silences = cast(list[list[int] | tuple[int, int]] | None, silences_ms)
    old_results = align_segments_greedy(
        segments=segments,
        ayahs=ayahs,
        silences_ms=greedy_silences,
    )

    # Build lookup table for quick access to greedy results by ayah number
    old_by_ayah: dict[int, AlignmentResult] = {}
    for r in old_results:
        old_by_ayah[r.ayah.ayah_number] = r

    # ============================================================================
    # PHASE 3: Quality Check and Intelligent Fallback
    # ============================================================================
    # For each ayah, we now decide which result to keep based on quality.
    # We try multiple strategies to improve low-quality alignments.

    final_results = []

    for dp_r in dp_results:
        ayah = dp_r.ayah
        ayah_word_count = len(ayah.text.split())
        ayah_duration = dp_r.end_time - dp_r.start_time

        # Identify "long" ayahs which benefit from split-and-restitch
        is_long_ayah = ayah_word_count > long_ayah_words or ayah_duration > long_ayah_duration

        # --- Decision Point 1: Is DP result good enough? ---
        if dp_r.similarity_score >= quality_threshold:
            # DP did great! Keep it and move on.
            final_results.append(dp_r)
            stats.dp_kept += 1
            continue

        # --- Decision Point 2: DP result is low quality, try to improve ---
        # We'll compare multiple strategies and keep the best one.
        best_result = dp_r
        best_source = "dp"

        # Strategy A: Split-and-Restitch (for long ayahs only)
        # Long ayahs often have timing drift. By splitting at silences and
        # realigning each chunk, we can often get much better results.
        if is_long_ayah:
            split_result = _try_split_and_restitch(segments, ayah, dp_r, silences_ms)
            if split_result and split_result.similarity_score > best_result.similarity_score:
                best_result = split_result
                best_source = "split"

        # Strategy B: Greedy Fallback
        # Check if the simple greedy aligner actually did better than DP.
        # This happens when DP made a suboptimal choice due to constraints.
        old_r = old_by_ayah.get(ayah.ayah_number)
        if old_r and old_r.similarity_score > best_result.similarity_score:
            best_result = old_r
            best_source = "old"

        # --- Record Statistics ---
        # Track which strategy we ended up using for this ayah
        if best_source == "old":
            stats.old_fallback += 1
        elif best_source == "split":
            stats.split_improved += 1
        elif best_result.similarity_score < quality_threshold:
            # Still low quality even after all attempts
            stats.still_low += 1
        else:
            stats.dp_kept += 1

        final_results.append(best_result)

    return final_results, stats
