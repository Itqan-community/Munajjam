"""
Cascade recovery for alignment drift.

Detects sequences of consecutive low-scoring ayahs and attempts
to re-align them using silence boundaries for better sync.
"""

from ..models import Segment, Ayah, AlignmentResult
from .matcher import similarity
from .dp_core import compute_alignment_cost


def find_cascade_sequences(
    results: list[AlignmentResult],
    threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[tuple[int, int]]:
    """
    Find sequences of consecutive low-scoring ayahs (cascades).

    Args:
        results: List of alignment results
        threshold: Similarity threshold below which is considered low
        min_cascade_length: Minimum length to be considered a cascade

    Returns:
        List of (start_idx, end_idx) tuples for each cascade
    """
    cascades = []
    i = 0

    while i < len(results):
        if results[i].similarity_score < threshold:
            start = i
            while i < len(results) and results[i].similarity_score < threshold:
                i += 1
            end = i

            if end - start >= min_cascade_length:
                cascades.append((start, end))
        else:
            i += 1

    return cascades


def _recover_cascade_with_resync(
    segments: list[Segment],
    ayahs: list[Ayah],
    results: list[AlignmentResult],
    cascade_start: int,
    cascade_end: int,
    silences_sec: list[tuple[float, float]],
    context_ayahs: int = 1,
) -> list[AlignmentResult] | None:
    """
    Attempt to recover a cascade by re-aligning using silence boundaries.

    Strategy:
    1. Extend the cascade range by 1 ayah on each side for context
    2. Find the segment range covered by these ayahs
    3. Find silence boundaries within this range
    4. Re-run DP alignment on just this portion with emphasis on silence breaks

    Returns:
        New alignment results for the cascade region, or None if recovery failed
    """
    # Extend range for context
    extended_start = max(0, cascade_start - context_ayahs)
    extended_end = min(len(results), cascade_end + context_ayahs)

    # Get segment range for the extended ayah range
    seg_start_time = results[extended_start].start_time
    seg_end_time = results[extended_end - 1].end_time

    # Find segments in this time range
    seg_indices = []
    for idx, seg in enumerate(segments):
        if seg.start >= seg_start_time - 0.5 and seg.end <= seg_end_time + 0.5:
            seg_indices.append(idx)

    if not seg_indices:
        return None

    seg_range_start = min(seg_indices)
    seg_range_end = max(seg_indices) + 1

    # Extract the segments and ayahs for re-alignment
    sub_segments = segments[seg_range_start:seg_range_end]
    sub_ayahs = [results[i].ayah for i in range(extended_start, extended_end)]

    if len(sub_segments) < len(sub_ayahs):
        return None

    # Find silences in this time range
    relevant_silences = []
    for sil_start, sil_end in silences_sec:
        if seg_start_time <= sil_start <= seg_end_time:
            relevant_silences.append((sil_start, sil_end))

    n_sub_seg = len(sub_segments)
    n_sub_ayah = len(sub_ayahs)

    INF = float('inf')
    dp: dict[tuple[int, int], tuple[float, str, int, tuple | None]] = {}
    dp[(0, 0)] = (0.0, "", 0, None)

    # Build set of segment indices that align with silences
    silence_aligned_ends = set()
    for idx, seg in enumerate(sub_segments):
        for sil_start, sil_end in relevant_silences:
            if abs(seg.end - sil_start) < 0.3:
                silence_aligned_ends.add(idx + 1)

    max_segs = min(6, n_sub_seg)

    for j in range(1, n_sub_ayah + 1):
        for i in range(j, n_sub_seg + 1):
            best = None
            best_cost = INF

            for k in range(1, min(max_segs, i) + 1):
                prev_i = i - k
                prev_j = j - 1

                if (prev_i, prev_j) not in dp:
                    continue

                prev_cost, _, _, _ = dp[(prev_i, prev_j)]

                merged_text = " ".join(seg.text for seg in sub_segments[prev_i:i])
                cost = compute_alignment_cost(merged_text, sub_ayahs[j - 1].text)

                # Bonus for ending at silence boundary
                if i in silence_aligned_ends:
                    cost -= 0.15

                total_cost = prev_cost + cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best = (total_cost, merged_text, prev_i, (prev_i, prev_j))

            if best is not None:
                dp[(i, j)] = best

    # Find best ending
    best_end = None
    best_end_cost = INF

    for i in range(n_sub_ayah, n_sub_seg + 1):
        if (i, n_sub_ayah) in dp:
            if dp[(i, n_sub_ayah)][0] < best_end_cost:
                best_end_cost = dp[(i, n_sub_ayah)][0]
                best_end = (i, n_sub_ayah)

    if best_end is None:
        return None

    # Backtrack and build new results
    path = []
    current = best_end

    while current and current in dp:
        cost, merged_text, seg_start, parent = dp[current]
        i, j = current

        if parent is not None:
            path.append((seg_start, i, j, merged_text))

        current = parent

    path.reverse()

    # Convert to results
    new_results = []
    for seg_start_idx, seg_end_idx, ayah_idx, merged_text in path:
        if seg_start_idx >= len(sub_segments) or seg_end_idx > len(sub_segments):
            continue

        ayah = sub_ayahs[ayah_idx - 1]
        start_time = sub_segments[seg_start_idx].start
        end_time = sub_segments[seg_end_idx - 1].end

        sim_score = similarity(merged_text, ayah.text)

        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        new_results.append(result)

    # Check if recovery improved the results
    if len(new_results) != extended_end - extended_start:
        return None

    old_results_range = results[extended_start:extended_end]

    # Conservative check: Don't accept recovery if ANY ayah degrades significantly
    for old, new in zip(old_results_range, new_results):
        drop = old.similarity_score - new.similarity_score

        # Strict protection for good ayahs (>= 75%): max 8% drop
        if old.similarity_score >= 0.75 and drop > 0.08:
            return None

        # Protect mediocre ayahs (50-75%): max 12% drop
        if old.similarity_score >= 0.5 and drop > 0.12:
            return None

        # Never let a good ayah (>=75%) drop below 70%
        if old.similarity_score >= 0.75 and new.similarity_score < 0.70:
            return None

    # Check overall improvement in the cascade region
    context = 1
    cascade_old_start = max(0, context)
    cascade_old_end = (
        min(len(old_results_range), len(old_results_range) - context)
        if len(old_results_range) > 2
        else len(old_results_range)
    )

    cascade_new_start = cascade_old_start
    cascade_new_end = cascade_old_end

    old_cascade_sim = sum(r.similarity_score for r in old_results_range[cascade_old_start:cascade_old_end])
    new_cascade_sim = sum(r.similarity_score for r in new_results[cascade_new_start:cascade_new_end])

    cascade_len = cascade_old_end - cascade_old_start
    if cascade_len == 0:
        return None

    old_avg = old_cascade_sim / cascade_len
    new_avg = new_cascade_sim / cascade_len

    # Require significant improvement in the cascade region
    if new_avg > old_avg + 0.08:
        return new_results

    return None


def apply_cascade_recovery(
    segments: list[Segment],
    ayahs: list[Ayah],
    results: list[AlignmentResult],
    silences_ms: list[tuple[int, int]] | None = None,
    cascade_threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[AlignmentResult]:
    """
    Post-process alignment results to recover cascaded failures.

    Detects sequences of consecutive low-scoring ayahs and attempts
    to re-align them using silence boundaries for better sync.

    Args:
        segments: Original segments
        ayahs: Original ayahs
        results: Initial alignment results
        silences_ms: Silence periods in milliseconds
        cascade_threshold: Similarity below which triggers cascade detection
        min_cascade_length: Minimum consecutive failures to be a cascade

    Returns:
        Improved alignment results
    """
    if not results:
        return results

    # Convert silences to seconds
    silences_sec = []
    if silences_ms:
        for start_ms, end_ms in silences_ms:
            silences_sec.append((start_ms / 1000.0, end_ms / 1000.0))

    # Find cascades
    cascades = find_cascade_sequences(results, cascade_threshold, min_cascade_length)

    if not cascades:
        return results

    # Process each cascade (in reverse order to maintain indices)
    improved_results = list(results)

    for cascade_start, cascade_end in reversed(cascades):
        recovery = _recover_cascade_with_resync(
            segments,
            ayahs,
            improved_results,
            cascade_start,
            cascade_end,
            silences_sec,
        )

        if recovery:
            context = 1
            ext_start = max(0, cascade_start - context)
            ext_end = min(len(improved_results), cascade_end + context)

            improved_results = (
                improved_results[:ext_start] +
                recovery +
                improved_results[ext_end:]
            )

    return improved_results
