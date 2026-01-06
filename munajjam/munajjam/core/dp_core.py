"""
Core Dynamic Programming alignment algorithm.

Contains the fundamental DP components for aligning transcribed segments
to reference Quran ayahs.
"""

from dataclasses import dataclass
from typing import Callable

from ..models import Ayah, Segment, AlignmentResult, SegmentType
from .matcher import similarity, compute_coverage_ratio


@dataclass
class DPCell:
    """A cell in the DP matrix."""
    cost: float  # Accumulated cost to reach this cell
    merged_text: str  # Text merged so far for current ayah
    seg_start_idx: int  # Index of first segment in current ayah
    parent: tuple[int, int] | None  # Previous cell for backtracking


def compute_alignment_cost(
    merged_text: str,
    ayah_text: str,
    critical_threshold: float = 0.4,
    critical_penalty: float = 0.5,
) -> float:
    """
    Compute the cost of aligning merged segments to an ayah.
    Lower cost = better alignment.

    Returns value between 0 (perfect match) and higher values for worse matches.

    Args:
        merged_text: The merged segment text
        ayah_text: The reference ayah text
        critical_threshold: Similarity below this triggers heavy penalty
        critical_penalty: Extra penalty added when below critical threshold
    """
    if not merged_text.strip():
        return 1.5  # High cost for empty text

    # Primary: text similarity
    sim = similarity(merged_text, ayah_text)

    # Secondary: coverage ratio penalty
    coverage = compute_coverage_ratio(merged_text, ayah_text)

    # Penalize under-coverage (missing words) and over-coverage (extra words)
    if coverage < 0.7:
        coverage_penalty = (0.7 - coverage) * 0.5  # Up to 0.35 penalty
    elif coverage > 1.3:
        coverage_penalty = (coverage - 1.3) * 0.3  # Penalty for too much text
    else:
        coverage_penalty = 0

    # Cost = 1 - similarity + coverage penalty
    cost = (1 - sim) + coverage_penalty

    # CRITICAL THRESHOLD: Heavy penalty for very low similarity
    # This discourages the DP from choosing paths with misaligned ayahs
    if sim < critical_threshold:
        cost += critical_penalty * (critical_threshold - sim) / critical_threshold

    return max(0.0, cost)


def _filter_special_segments(segments: list[Segment], ayahs: list[Ayah]) -> list[Segment]:
    """Filter out non-ayah segments (isti3aza, basmala for non-Fatiha)."""
    filtered = []
    for seg in segments:
        if seg.type == SegmentType.ISTI3AZA:
            continue
        if seg.type == SegmentType.BASMALA and ayahs[0].surah_id != 1:
            continue
        filtered.append(seg)
    return filtered if filtered else segments


def _align_greedy_multi_ayah(
    segments: list[Segment],
    ayahs: list[Ayah],
) -> list[AlignmentResult]:
    """
    Bidirectional greedy alignment for when segments != ayahs.

    Handles two cases:
    1. Reciter merges multiple ayahs into one segment (segment > ayahs)
    2. Reciter splits one ayah across multiple segments (ayah > segments)

    Strategy: Try both merging segments and merging ayahs to find best match.
    """
    if not segments or not ayahs:
        return []

    results = []
    seg_idx = 0
    ayah_idx = 0

    while seg_idx < len(segments) and ayah_idx < len(ayahs):
        best_match_ayahs = None
        best_match_segs = None
        best_score = 0
        best_seg_end = seg_idx
        best_ayah_end = ayah_idx

        # Search window for ayahs (look ahead)
        ayah_search_limit = min(ayah_idx + 4, len(ayahs))
        seg_search_limit = min(seg_idx + 4, len(segments))

        for num_segs in range(1, seg_search_limit - seg_idx + 1):
            merged_segs = segments[seg_idx:seg_idx + num_segs]
            merged_seg_text = " ".join(s.text for s in merged_segs)

            for num_ayahs in range(1, ayah_search_limit - ayah_idx + 1):
                concat_ayahs = ayahs[ayah_idx:ayah_idx + num_ayahs]
                concat_text = " ".join(a.text for a in concat_ayahs)

                score = similarity(merged_seg_text, concat_text)

                # Prefer matches that consume more content
                ratio_penalty = 0
                if num_segs > 1 and num_ayahs > 1:
                    ratio_penalty = 0.05

                adjusted_score = score - ratio_penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match_ayahs = concat_ayahs
                    best_match_segs = merged_segs
                    best_seg_end = seg_idx + num_segs
                    best_ayah_end = ayah_idx + num_ayahs

        # Apply match if score is acceptable
        if best_match_ayahs and best_score > 0.35:
            start_time = best_match_segs[0].start
            end_time = best_match_segs[-1].end
            merged_text = " ".join(s.text for s in best_match_segs)

            total_words = sum(len(a.text.split()) for a in best_match_ayahs)
            current_time = start_time
            segment_duration = end_time - start_time

            for a in best_match_ayahs:
                ayah_words = len(a.text.split())
                ayah_duration = (
                    (ayah_words / total_words) * segment_duration
                    if total_words > 0
                    else segment_duration / len(best_match_ayahs)
                )

                result = AlignmentResult(
                    ayah=a,
                    start_time=round(current_time, 2),
                    end_time=round(current_time + ayah_duration, 2),
                    transcribed_text=merged_text if len(best_match_ayahs) == 1 else f"[{len(best_match_segs)} segs -> {len(best_match_ayahs)} ayahs]",
                    similarity_score=best_score,
                    overlap_detected=len(best_match_ayahs) > 1 or len(best_match_segs) > 1,
                )
                results.append(result)
                current_time += ayah_duration

            seg_idx = best_seg_end
            ayah_idx = best_ayah_end
        else:
            seg_idx += 1

    return results


def align_segments_dp(
    segments: list[Segment],
    ayahs: list[Ayah],
    max_segments_per_ayah: int = 15,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    Align segments to ayahs using dynamic programming.

    Finds the globally optimal alignment that minimizes total cost,
    allowing multiple segments to be merged into each ayah.

    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        max_segments_per_ayah: Maximum segments that can be merged into one ayah
        on_progress: Optional callback for progress updates

    Returns:
        List of AlignmentResult objects
    """
    if not segments or not ayahs:
        return []

    segments = _filter_special_segments(segments, ayahs)

    n_seg = len(segments)
    n_ayah = len(ayahs)

    INF = float('inf')
    dp: dict[tuple[int, int], DPCell] = {}
    dp[(0, 0)] = DPCell(cost=0, merged_text="", seg_start_idx=0, parent=None)

    # Fill DP table
    for j in range(1, n_ayah + 1):
        ayah = ayahs[j - 1]

        if on_progress:
            on_progress(j, n_ayah)

        for i in range(j, n_seg + 1):
            best_cell = None
            best_cost = INF

            for k in range(1, min(max_segments_per_ayah, i - j + 2)):
                prev_i = i - k
                prev_j = j - 1

                if (prev_i, prev_j) not in dp:
                    continue

                prev_cell = dp[(prev_i, prev_j)]

                merged_segments = segments[prev_i:i]
                merged_text = " ".join(seg.text for seg in merged_segments)

                cost = compute_alignment_cost(merged_text, ayah.text)
                total_cost = prev_cell.cost + cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cell = DPCell(
                        cost=total_cost,
                        merged_text=merged_text,
                        seg_start_idx=prev_i,
                        parent=(prev_i, prev_j),
                    )

            if best_cell is not None:
                dp[(i, j)] = best_cell

    # Find best ending point
    best_end = None
    best_end_cost = INF

    for i in range(n_ayah, n_seg + 1):
        if (i, n_ayah) in dp:
            if dp[(i, n_ayah)].cost < best_end_cost:
                best_end_cost = dp[(i, n_ayah)].cost
                best_end = (i, n_ayah)

    if best_end is None:
        for j in range(n_ayah, 0, -1):
            for i in range(n_seg, 0, -1):
                if (i, j) in dp:
                    best_end = (i, j)
                    break
            if best_end:
                break

    if best_end is None:
        return []

    # Backtrack to reconstruct alignment
    path = []
    current = best_end

    while current and current in dp:
        cell = dp[current]
        i, j = current

        if cell.parent is not None:
            prev_i, prev_j = cell.parent
            path.append((prev_i, i, j, cell.merged_text))

        current = cell.parent

    path.reverse()

    # Convert path to AlignmentResult objects
    results = []
    for seg_start, seg_end, ayah_idx, merged_text in path:
        if seg_start >= len(segments) or seg_end > len(segments):
            continue

        ayah = ayahs[ayah_idx - 1]
        start_time = segments[seg_start].start
        end_time = segments[seg_end - 1].end

        sim_score = similarity(merged_text, ayah.text)

        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        results.append(result)

    return results


def align_segments_dp_with_constraints(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    max_segments_per_ayah: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    DP alignment with optimizations.

    Optimizations:
    - Pre-compute all merged texts
    - Cache similarity computations
    - Limit search window (beam search style)
    - Skip impossible paths early

    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        silences_ms: List of (start_ms, end_ms) silence periods (for post-processing)
        max_segments_per_ayah: Maximum segments per ayah
        on_progress: Optional progress callback

    Returns:
        List of AlignmentResult objects
    """
    # Import here to avoid circular dependency
    from .cascade_recovery import apply_cascade_recovery

    if not segments or not ayahs:
        return []

    segments = _filter_special_segments(segments, ayahs)

    n_seg = len(segments)
    n_ayah = len(ayahs)

    # Handle case where we have fewer segments than ayahs
    if n_seg < n_ayah:
        return _align_greedy_multi_ayah(segments, ayahs)

    # Pre-compute merged texts
    merged_cache: dict[tuple[int, int], str] = {}

    def get_merged_text(start: int, end: int) -> str:
        if (start, end) not in merged_cache:
            merged_cache[(start, end)] = " ".join(seg.text for seg in segments[start:end])
        return merged_cache[(start, end)]

    # Cache similarity computations
    sim_cache: dict[tuple[str, int], float] = {}

    def get_cost(merged_text: str, ayah_idx: int) -> float:
        cache_key = (merged_text[:100], ayah_idx)
        if cache_key not in sim_cache:
            sim_cache[cache_key] = compute_alignment_cost(merged_text, ayahs[ayah_idx].text)
        return sim_cache[cache_key]

    INF = float('inf')
    dp: dict[tuple[int, int], DPCell] = {}
    dp[(0, 0)] = DPCell(cost=0, merged_text="", seg_start_idx=0, parent=None)

    for j in range(1, n_ayah + 1):
        if on_progress:
            on_progress(j, n_ayah)

        # Relaxed windowing
        min_i = max(1, j)
        remaining_ayahs = n_ayah - j
        min_segs_needed_for_remaining = max(1, remaining_ayahs)
        max_i = min(n_seg + 1, n_seg - min_segs_needed_for_remaining + 1 + max_segments_per_ayah)
        max_i = max(min_i + 1, max_i)

        for i in range(min_i, max_i):
            best_cell = None
            best_cost = INF

            max_k = min(max_segments_per_ayah, i)

            for k in range(1, max_k + 1):
                prev_i = i - k
                prev_j = j - 1

                if (prev_i, prev_j) not in dp:
                    continue

                prev_cell = dp[(prev_i, prev_j)]

                if prev_cell.cost > best_cost:
                    continue

                merged_text = get_merged_text(prev_i, i)
                cost = get_cost(merged_text, j - 1)

                total_cost = prev_cell.cost + cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cell = DPCell(
                        cost=total_cost,
                        merged_text=merged_text,
                        seg_start_idx=prev_i,
                        parent=(prev_i, prev_j),
                    )

            if best_cell is not None:
                dp[(i, j)] = best_cell

    # Find best ending
    best_end = None
    best_end_cost = INF

    for i in range(n_ayah, n_seg + 1):
        if (i, n_ayah) in dp:
            if dp[(i, n_ayah)].cost < best_end_cost:
                best_end_cost = dp[(i, n_ayah)].cost
                best_end = (i, n_ayah)

    if best_end is None:
        for j in range(n_ayah, 0, -1):
            for i in range(n_seg, 0, -1):
                if (i, j) in dp:
                    best_end = (i, j)
                    break
            if best_end:
                break

    if best_end is None:
        return []

    # Backtrack
    path = []
    current = best_end

    while current and current in dp:
        cell = dp[current]
        i, j = current

        if cell.parent is not None:
            prev_i, prev_j = cell.parent
            path.append((prev_i, i, j, cell.merged_text))

        current = cell.parent

    path.reverse()

    # Convert to results
    results = []
    for seg_start, seg_end, ayah_idx, merged_text in path:
        if seg_start >= len(segments) or seg_end > len(segments):
            continue

        ayah = ayahs[ayah_idx - 1]
        start_time = segments[seg_start].start
        end_time = segments[seg_end - 1].end

        sim_score = similarity(merged_text, ayah.text)

        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        results.append(result)

    # POST-PROCESSING: Cascade Recovery
    results = apply_cascade_recovery(
        segments=segments,
        ayahs=ayahs,
        results=results,
        silences_ms=silences_ms,
        cascade_threshold=0.7,
        min_cascade_length=2,
    )

    return results
