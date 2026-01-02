"""
Ayah-segment alignment algorithm.

This module contains the core logic for aligning transcribed audio segments
with reference Quran ayahs (verses).
"""

from dataclasses import dataclass, field
from typing import Callable

from munajjam.core.arabic import normalize_arabic, detect_special_type
from munajjam.core.matcher import (
    similarity,
    get_first_last_words,
    compute_coverage_ratio,
)
from munajjam.core.overlap import (
    remove_overlap,
    apply_buffers,
    find_silence_gap_between,
    convert_silences_to_seconds,
)
from munajjam.models import Ayah, Segment, AlignmentResult
from munajjam.config import MunajjamSettings, get_settings


@dataclass
class AlignmentContext:
    """
    Context object for tracking alignment state.

    This holds all the state needed during the alignment process,
    including configuration, data, and progress tracking.
    """

    ayahs: list[Ayah]
    segments: list[Segment]
    silences_ms: list[list[int] | tuple[int, int]] = field(default_factory=list)
    settings: MunajjamSettings = field(default_factory=get_settings)

    # Progress tracking
    current_segment_idx: int = 0
    current_ayah_idx: int = 0
    prev_ayah_end: float | None = None

    # Results
    results: list[AlignmentResult] = field(default_factory=list)

    # Statistics
    segments_merged: int = 0
    overlaps_detected: int = 0

    @property
    def silences_sec(self) -> list[tuple[float, float]]:
        """Silences converted to seconds."""
        return convert_silences_to_seconds(self.silences_ms)

    @property
    def current_segment(self) -> Segment | None:
        """Current segment being processed."""
        if self.current_segment_idx < len(self.segments):
            return self.segments[self.current_segment_idx]
        return None

    @property
    def current_ayah(self) -> Ayah | None:
        """Current ayah being aligned."""
        if self.current_ayah_idx < len(self.ayahs):
            return self.ayahs[self.current_ayah_idx]
        return None

    @property
    def next_ayah(self) -> Ayah | None:
        """Next ayah (if exists)."""
        if self.current_ayah_idx + 1 < len(self.ayahs):
            return self.ayahs[self.current_ayah_idx + 1]
        return None

    @property
    def is_complete(self) -> bool:
        """Whether alignment is complete."""
        return (
            self.current_segment_idx >= len(self.segments)
            or self.current_ayah_idx >= len(self.ayahs)
        )


def _get_n_check_words(ayah_text: str) -> int:
    """Determine number of words to check based on ayah length."""
    words = ayah_text.strip().split()
    if len(words) >= 3:
        return 3
    elif len(words) == 2:
        return 2
    return 1


def _check_end_of_ayah(
    merged_text: str,
    ayah: Ayah,
    settings: MunajjamSettings,
) -> tuple[bool, float]:
    """
    Check if merged text represents the end of an ayah.

    Returns:
        Tuple of (is_end_of_ayah, similarity_score)
    """
    n_check = _get_n_check_words(ayah.text)
    _, seg_last = get_first_last_words(merged_text, n=n_check)
    _, ayah_last = get_first_last_words(ayah.text, n=n_check)

    last_words_sim = similarity(seg_last, ayah_last)

    if last_words_sim >= settings.similarity_threshold:
        # Check coverage ratio
        coverage = compute_coverage_ratio(merged_text, ayah.text)
        if coverage >= settings.coverage_threshold:
            return True, last_words_sim

    return False, last_words_sim


def _check_next_ayah_starts(
    next_segment: Segment,
    next_ayah: Ayah,
    settings: MunajjamSettings,
) -> bool:
    """Check if the next segment starts the next ayah.
    
    Uses adaptive word count (1-3) based on ayah length.
    The coverage check in the caller provides additional safeguard against
    false positives when consecutive ayahs share similar opening phrases.
    """
    n_check = _get_n_check_words(next_ayah.text)
    
    next_first_seg, _ = get_first_last_words(next_segment.text, n=n_check)
    next_first_ayah, _ = get_first_last_words(next_ayah.text, n=n_check)

    sim = similarity(next_first_seg, next_first_ayah)
    return sim > settings.similarity_threshold


def _finalize_ayah(
    ctx: AlignmentContext,
    merged_text: str,
    start_time: float,
    end_time: float,
    overlap_detected: bool,
    next_ayah_start: float | None = None,
) -> AlignmentResult:
    """
    Finalize alignment for an ayah and create result.

    Applies buffers and creates the AlignmentResult.
    """
    ayah = ctx.current_ayah
    if ayah is None:
        raise ValueError("No current ayah to finalize")

    # Apply buffers
    buffered_start, buffered_end = apply_buffers(
        start_time,
        end_time,
        ctx.silences_ms,
        prev_end=ctx.prev_ayah_end,
        next_start=next_ayah_start,
        buffer=ctx.settings.buffer_seconds,
    )

    # Compute full similarity
    full_sim = similarity(merged_text, ayah.text)

    result = AlignmentResult(
        ayah=ayah,
        start_time=buffered_start,
        end_time=buffered_end,
        transcribed_text=merged_text,
        similarity_score=full_sim,
        overlap_detected=overlap_detected,
    )

    # Update context
    ctx.results.append(result)
    ctx.prev_ayah_end = buffered_end
    ctx.current_ayah_idx += 1

    if overlap_detected:
        ctx.overlaps_detected += 1

    return result


def align_segments(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[list[int] | tuple[int, int]] | None = None,
    settings: MunajjamSettings | None = None,
    on_ayah_aligned: Callable[[AlignmentResult], None] | None = None,
    required_tokens_map: dict[tuple[int, int], list[str]] | None = None,
) -> list[AlignmentResult]:
    """
    Align transcribed segments with reference ayahs.

    This is the main alignment function that matches audio segments
    to their corresponding Quran verses.

    Args:
        segments: List of transcribed Segment objects
        ayahs: List of reference Ayah objects (in order)
        silences_ms: Optional silence periods in milliseconds
        settings: Optional settings override
        on_ayah_aligned: Optional callback for each aligned ayah
        required_tokens_map: Optional map of (surah_id, ayah_idx) to required tokens

    Returns:
        List of AlignmentResult objects
    """
    if settings is None:
        settings = get_settings()

    ctx = AlignmentContext(
        ayahs=ayahs,
        segments=segments,
        silences_ms=silences_ms or [],
        settings=settings,
    )

    # Pre-compute silences in seconds
    silences_sec = ctx.silences_sec

    # Process segments
    i = 0
    while i < len(segments) and ctx.current_ayah_idx < len(ayahs):
        segment = segments[i]

        # Determine if we should skip this segment
        special_type = detect_special_type(segment)
        
        # Always skip isti'aza
        if special_type == "isti3aza":
            i += 1
            continue
        
        # Skip basmala UNLESS it's Surah Al-Fatiha (surah 1) where basmala is ayah 1
        # In Al-Fatiha, the basmala is the first ayah, not a special segment
        if special_type == "basmala" and segment.surah_id != 1:
            i += 1
            continue
        
        # Skip other id=0 segments that aren't basmala in surah 1
        if segment.id == 0 and special_type is None:
            i += 1
            continue

        # Initialize for this ayah
        start_time = segment.start
        merged_text = segment.text
        end_time = segment.end
        overlap_flag = False

        # Process until we find ayah boundary
        while True:
            ayah = ctx.current_ayah
            if ayah is None:
                break

            # Check for required tokens (surah-specific guards)
            if required_tokens_map:
                lookup_key = (ayah.surah_id, ctx.current_ayah_idx)
                if lookup_key in required_tokens_map:
                    normalized_seg = normalize_arabic(merged_text)
                    missing_required = False
                    for tok in required_tokens_map[lookup_key]:
                        if tok not in normalized_seg:
                            missing_required = True
                            break

                    if missing_required and i + 1 < len(segments):
                        # Force merge next segment
                        merged_text, overlap_found = remove_overlap(
                            merged_text, segments[i + 1].text
                        )
                        if overlap_found:
                            overlap_flag = True
                            ctx.overlaps_detected += 1
                        end_time = segments[i + 1].end
                        i += 1
                        ctx.segments_merged += 1
                        continue

            # Check if reached end of ayah
            is_end, _ = _check_end_of_ayah(merged_text, ayah, settings)
            if is_end:
                result = _finalize_ayah(
                    ctx, merged_text, start_time, end_time, overlap_flag
                )
                if on_ayah_aligned:
                    on_ayah_aligned(result)
                break

            # Check if next segment starts next ayah
            if i + 1 < len(segments) and ctx.next_ayah is not None:
                next_segment = segments[i + 1]

                if _check_next_ayah_starts(next_segment, ctx.next_ayah, settings):
                    # Additional safeguard: verify current ayah is complete enough
                    # This prevents false positives when consecutive ayahs share opening phrases
                    # (e.g., ayahs 106-107 both contain "ألم تعلم أن الله")
                    coverage = compute_coverage_ratio(merged_text, ayah.text)
                    
                    # Check if current ayah's last words match
                    n_check = _get_n_check_words(ayah.text)
                    _, seg_last = get_first_last_words(merged_text, n=n_check)
                    _, ayah_last = get_first_last_words(ayah.text, n=n_check)
                    last_words_match = similarity(seg_last, ayah_last) >= settings.similarity_threshold
                    
                    # Finalize if: last words match, OR coverage is very high (>90%)
                    if last_words_match or coverage >= 0.9:
                    result = _finalize_ayah(
                        ctx,
                        merged_text,
                        start_time,
                        end_time,
                        overlap_flag,
                        next_ayah_start=next_segment.start,
                    )
                    if on_ayah_aligned:
                        on_ayah_aligned(result)
                    break
                    # If last words don't match and coverage < 90%, continue merging

                # Check for silence gap
                silence_gap = find_silence_gap_between(
                    end_time,
                    next_segment.start,
                    silences_sec,
                    min_gap=settings.min_silence_gap,
                )

                if silence_gap is not None:
                    # Check if next segment starts next ayah
                    if _check_next_ayah_starts(next_segment, ctx.next_ayah, settings):
                        # Additional safeguard: verify current ayah is complete enough
                        coverage = compute_coverage_ratio(merged_text, ayah.text)
                        
                        # Check if current ayah's last words match
                        n_check = _get_n_check_words(ayah.text)
                        _, seg_last = get_first_last_words(merged_text, n=n_check)
                        _, ayah_last = get_first_last_words(ayah.text, n=n_check)
                        last_words_match = similarity(seg_last, ayah_last) >= settings.similarity_threshold
                        
                        # Finalize if: last words match, OR coverage is very high (>90%)
                        if last_words_match or coverage >= 0.9:
                        gap_start, _ = silence_gap
                        result = _finalize_ayah(
                            ctx,
                            merged_text,
                            start_time,
                            end_time,
                            overlap_flag,
                            next_ayah_start=gap_start,
                        )
                        if on_ayah_aligned:
                            on_ayah_aligned(result)
                        break
                        # If last words don't match and coverage < 90%, continue merging

                # Merge next segment
                merged_text, overlap_found = remove_overlap(
                    merged_text, next_segment.text
                )
                if overlap_found:
                    overlap_flag = True
                    ctx.overlaps_detected += 1
                end_time = next_segment.end
                i += 1
                ctx.segments_merged += 1

            else:
                # End of segments - finalize last ayah
                result = _finalize_ayah(
                    ctx, merged_text, start_time, end_time, overlap_flag
                )
                if on_ayah_aligned:
                    on_ayah_aligned(result)
                break

        i += 1

    return ctx.results


def get_alignment_stats(ctx: AlignmentContext) -> dict:
    """
    Get statistics from an alignment context.

    Returns:
        Dictionary with alignment statistics
    """
    if not ctx.results:
        return {
            "total_ayahs": len(ctx.ayahs),
            "aligned_ayahs": 0,
            "segments_merged": ctx.segments_merged,
            "overlaps_detected": ctx.overlaps_detected,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
        }

    similarities = [r.similarity_score for r in ctx.results]

    return {
        "total_ayahs": len(ctx.ayahs),
        "aligned_ayahs": len(ctx.results),
        "segments_merged": ctx.segments_merged,
        "overlaps_detected": ctx.overlaps_detected,
        "avg_similarity": sum(similarities) / len(similarities),
        "min_similarity": min(similarities),
    }

