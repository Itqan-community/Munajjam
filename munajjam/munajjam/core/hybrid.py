"""
Hybrid alignment strategy.

Combines DP alignment with fallback to greedy alignment and
split-and-restitch for long ayahs.
"""

from dataclasses import dataclass
from typing import Callable

from ..models import Segment, Ayah, AlignmentResult
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
    """Find silence periods within a given time range."""
    result = []
    for sil_start, sil_end in silences_sec:
        if sil_end > start_time and sil_start < end_time:
            clipped_start = max(sil_start, start_time)
            clipped_end = min(sil_end, end_time)
            duration = clipped_end - clipped_start
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

    Returns a list of segment groups, where each group represents
    a chunk between silences.
    """
    range_segments = [
        s for s in segments
        if s.start >= start_time - 0.5 and s.end <= end_time + 0.5
    ]

    if not range_segments:
        return []

    silences = _find_silences_in_range(silences_sec, start_time, end_time)

    if not silences:
        return [range_segments]

    silences.sort(key=lambda x: x[0])

    chunks = []
    current_chunk = []
    silence_idx = 0

    for seg in range_segments:
        if silence_idx < len(silences):
            sil_start, sil_end = silences[silence_idx]

            if seg.end <= sil_start:
                current_chunk.append(seg)
            elif seg.start >= sil_end:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                silence_idx += 1
            else:
                current_chunk.append(seg)
                chunks.append(current_chunk)
                current_chunk = []
                silence_idx += 1
        else:
            current_chunk.append(seg)

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
        segments,
        silences_sec,
        dp_result.start_time,
        dp_result.end_time
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

    Strategy:
    1. Run DP aligner on all segments/ayahs
    2. For each ayah with similarity < quality_threshold:
       a. If long ayah (>30 words or >30s): try split-and-restitch
       b. Try old aligner as fallback
       c. Keep whichever result is best

    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        silences_ms: Silence periods in milliseconds
        quality_threshold: Similarity below which to try fallback (default 0.85)
        long_ayah_words: Word count threshold for "long" ayahs
        long_ayah_duration: Duration threshold for "long" ayahs (seconds)
        on_progress: Optional progress callback (current, total)

    Returns:
        Tuple of (alignment_results, hybrid_stats)
    """
    # Import here to avoid circular dependency
    from .aligner_greedy import align_segments as align_segments_greedy
    from .dp_core import align_segments_dp_with_constraints

    stats = HybridStats(total_ayahs=len(ayahs))

    if not segments or not ayahs:
        return [], stats

    # Step 1: Run DP aligner
    dp_results = align_segments_dp_with_constraints(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
        on_progress=on_progress,
    )

    # If DP returned no results, fall back entirely to old aligner
    if not dp_results:
        old_results = align_segments_greedy(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
        )
        stats.old_fallback = len(old_results)
        return old_results, stats

    # Step 2: Run old aligner to have fallback options
    old_results = align_segments_greedy(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
    )

    # Build lookup for old results by ayah number
    old_by_ayah: dict[int, AlignmentResult] = {}
    for r in old_results:
        old_by_ayah[r.ayah.ayah_number] = r

    # Step 3: For each DP result, check quality and apply fallback if needed
    final_results = []

    for dp_r in dp_results:
        ayah = dp_r.ayah
        ayah_word_count = len(ayah.text.split())
        ayah_duration = dp_r.end_time - dp_r.start_time

        is_long_ayah = (
            ayah_word_count > long_ayah_words or
            ayah_duration > long_ayah_duration
        )

        # Check if DP result is good enough
        if dp_r.similarity_score >= quality_threshold:
            final_results.append(dp_r)
            stats.dp_kept += 1
            continue

        # DP result is low quality - try to improve
        best_result = dp_r
        best_source = "dp"

        # Try 1: For long ayahs, try split-and-restitch
        if is_long_ayah:
            split_result = _try_split_and_restitch(
                segments, ayah, dp_r, silences_ms
            )
            if split_result and split_result.similarity_score > best_result.similarity_score:
                best_result = split_result
                best_source = "split"

        # Try 2: Check if old aligner did better
        old_r = old_by_ayah.get(ayah.ayah_number)
        if old_r and old_r.similarity_score > best_result.similarity_score:
            best_result = old_r
            best_source = "old"

        # Record stats based on final choice
        if best_source == "old":
            stats.old_fallback += 1
        elif best_source == "split":
            stats.split_improved += 1
        elif best_result.similarity_score < quality_threshold:
            stats.still_low += 1
        else:
            stats.dp_kept += 1

        final_results.append(best_result)

    return final_results, stats
