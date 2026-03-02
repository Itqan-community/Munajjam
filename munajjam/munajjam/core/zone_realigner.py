"""
Zone Re-alignment Module

Fixes drift issues in long surahs by identifying and re-aligning problematic regions.

WHAT IS DRIFT?
--------------
In long surahs (especially 30+ minutes), timing can gradually drift away from
the true alignment. This happens because:
1. Small errors accumulate over time
2. The reciter's pace may change throughout the recording
3. Pauses and emphasis patterns vary

Example of drift:
- Ayahs 1-50: Good alignment (similarity ~0.95)
- Ayahs 51-75: Drift begins (similarity drops to ~0.75)
- Ayahs 76-100: Severe drift (similarity ~0.60)
- Ayahs 101-120: Good again after a natural break

SOLUTION: ZONE RE-ALIGNMENT
---------------------------
Instead of re-aligning the entire surah (expensive), we:
1. Identify "problem zones" - consecutive low-confidence ayahs
2. Re-align ONLY those zones using segments in their time range
3. Keep the better result for each ayah (old vs. new)

This is much faster and preserves good alignments while fixing drift.

TWO STRATEGIES:
--------------
1. Problem Zone Re-alignment: Fixes obvious low-confidence regions
2. Anchor-based Re-alignment: Uses high-confidence ayahs as anchors
   and re-aligns the gaps between them
"""

from dataclasses import dataclass

from munajjam.models import AlignmentResult, Ayah, Segment

from .arabic import normalize_arabic
from .dp_core import align_segments_dp


@dataclass
class ZoneStats:
    """Statistics from zone re-alignment."""

    zones_found: int = 0
    zones_improved: int = 0
    ayahs_improved: int = 0
    ayahs_unchanged: int = 0
    ayahs_degraded: int = 0


@dataclass
class ProblemZone:
    """A zone of consecutive low-confidence ayahs."""

    start_idx: int  # Index in results list
    end_idx: int  # Index in results list (exclusive)
    start_ayah: int
    end_ayah: int
    avg_similarity: float
    start_time: float
    end_time: float


def adaptive_quality_threshold(ayah_text: str, base_threshold: float = 0.85) -> float:
    """Return a length-adaptive quality threshold for an ayah.

    Short ayahs are inherently harder to match with high similarity,
    so we relax the threshold proportionally.

    - < 5 words  → 0.60
    - 5-15 words → 0.75
    - > 15 words → base_threshold (default 0.85)
    """
    word_count = len(normalize_arabic(ayah_text).split())
    if word_count < 5:
        return 0.60
    elif word_count <= 15:
        return 0.75
    return base_threshold


def identify_problem_zones(
    results: list[AlignmentResult],
    min_consecutive: int = 3,
    quality_threshold: float = 0.85,
    adaptive: bool = False,
) -> list[ProblemZone]:
    """
    Find sequences of consecutive low-confidence ayahs (drift zones).

    A "problem zone" is a region where alignment quality has degraded,
    typically due to timing drift in long recordings.

    Example scenario:
    ```
    Ayah 48: similarity 0.92 ✓
    Ayah 49: similarity 0.88 ✓
    Ayah 50: similarity 0.78 ← START OF ZONE
    Ayah 51: similarity 0.75 ← IN ZONE
    Ayah 52: similarity 0.72 ← IN ZONE
    Ayah 53: similarity 0.90 ✓ ← END OF ZONE
    ```

    If min_consecutive=3 and quality_threshold=0.85, ayahs 50-52 form a zone.

    Args:
        results: List of alignment results (in order)
        min_consecutive: Minimum consecutive low-confidence ayahs to form a zone.
                        Default 3 prevents single outliers from triggering re-alignment.
        quality_threshold: Similarity below which an ayah is "low confidence".
                          Default 0.85 means 85% similarity is the cutoff.

    Returns:
        List of ProblemZone objects, each containing:
        - start_idx, end_idx: Slice indices in results list
        - start_ayah, end_ayah: Ayah numbers
        - avg_similarity: Average similarity in the zone
        - start_time, end_time: Time boundaries for re-alignment
    """
    zones = []
    current_zone_start = None  # Index where current zone started
    current_zone_sims = []  # Similarity scores in current zone

    for i, result in enumerate(results):
        if adaptive:
            thresh = adaptive_quality_threshold(result.ayah.text, quality_threshold)
        else:
            thresh = quality_threshold
        is_low = result.similarity_score < thresh

        if is_low:
            # This ayah is low-confidence
            if current_zone_start is None:
                # Start a new zone
                current_zone_start = i
                current_zone_sims = [result.similarity_score]
            else:
                # Continue existing zone
                current_zone_sims.append(result.similarity_score)
        else:
            # This ayah is high-confidence - potential end of zone
            if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
                # We have a valid zone (enough consecutive low-conf ayahs)
                zone_results = results[current_zone_start:i]
                zones.append(
                    ProblemZone(
                        start_idx=current_zone_start,
                        end_idx=i,
                        start_ayah=zone_results[0].ayah.ayah_number,
                        end_ayah=zone_results[-1].ayah.ayah_number,
                        avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
                        start_time=zone_results[0].start_time,
                        end_time=zone_results[-1].end_time,
                    )
                )
            # Reset zone tracking
            current_zone_start = None
            current_zone_sims = []

    # Handle zone that extends to the end of the surah
    if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
        zone_results = results[current_zone_start:]
        zones.append(
            ProblemZone(
                start_idx=current_zone_start,
                end_idx=len(results),
                start_ayah=zone_results[0].ayah.ayah_number,
                end_ayah=zone_results[-1].ayah.ayah_number,
                avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
                start_time=zone_results[0].start_time,
                end_time=zone_results[-1].end_time,
            )
        )

    return zones


def find_segments_for_zone(
    segments: list[Segment],
    zone: ProblemZone,
    buffer_seconds: float = 10.0,
) -> list[Segment]:
    """
    Find all segments that fall within a problem zone's time range.

    Args:
        segments: All segments
        zone: The problem zone
        buffer_seconds: Extra time to include at boundaries

    Returns:
        List of segments in the zone's time range
    """
    zone_start = zone.start_time - buffer_seconds
    zone_end = zone.end_time + buffer_seconds

    return [s for s in segments if s.start >= zone_start and s.end <= zone_end]


def realign_zone(
    zone: ProblemZone,
    zone_segments: list[Segment],
    ayahs: list[Ayah],
) -> list[AlignmentResult]:
    """
    Re-align a problem zone using the DP aligner.

    Args:
        zone: The problem zone
        zone_segments: Segments in the zone's time range
        ayahs: All ayahs (will filter to zone's ayahs)

    Returns:
        New alignment results for the zone
    """
    # Get the ayahs for this zone (1-indexed)
    zone_ayahs = [a for a in ayahs if zone.start_ayah <= a.ayah_number <= zone.end_ayah]

    if not zone_ayahs or not zone_segments:
        return []

    # Run DP alignment on this small zone
    results = align_segments_dp(zone_segments, zone_ayahs)

    return results


def realign_problem_zones(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    min_consecutive: int = 3,
    quality_threshold: float = 0.85,
    buffer_seconds: float = 10.0,
    adaptive: bool = False,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Re-align problem zones in the results to fix timing drift.

    This is the MAIN ENTRY POINT for zone-based drift correction.

    How it works:
    1. Scan through results to find "problem zones" (consecutive low-conf ayahs)
    2. For each zone:
       a. Extract segments in that time range (with buffer)
       b. Extract ayahs in that zone
       c. Re-run DP alignment on just that small region
       d. Compare old vs new results for each ayah
       e. Keep whichever result is better
    3. Return updated results with statistics

    Example:
    ```
    Initial results: [0.95, 0.92, 0.78, 0.75, 0.72, 0.91, 0.94]
                                    ↑____________↑
                                    Problem zone (ayahs 3-5)

    After re-alignment: [0.95, 0.92, 0.89, 0.87, 0.86, 0.91, 0.94]
                                         ↑____________↑
                                         Improved!
    ```

    Args:
        results: Initial alignment results (potentially with drift)
        segments: All transcribed segments (needed for re-alignment)
        ayahs: All reference ayahs (needed for re-alignment)
        min_consecutive: Min consecutive low-conf ayahs to form a zone (default 3)
        quality_threshold: Similarity below which is "low confidence" (default 0.85)
        buffer_seconds: Extra time to include at zone boundaries for context (default 10s)

    Returns:
        Tuple of:
        - Updated alignment results (with improved zones)
        - ZoneStats object with statistics about what was improved
    """
    stats = ZoneStats()

    # ============================================================================
    # STEP 1: Identify Problem Zones
    # ============================================================================
    # Scan through all results to find consecutive low-confidence ayahs
    zones = identify_problem_zones(results, min_consecutive, quality_threshold, adaptive=adaptive)
    stats.zones_found = len(zones)

    # If no problem zones, return original results unchanged
    if not zones:
        return results, stats

    # Create a mutable copy of results (we'll update it in-place)
    new_results = list(results)

    # Create quick lookup table: ayah_number -> Ayah object
    ayah_by_num = {a.ayah_number: a for a in ayahs}

    # ============================================================================
    # STEP 2: Re-align Each Problem Zone
    # ============================================================================
    for zone in zones:
        # --- Extract segments for this zone (with buffer for context) ---
        zone_segments = find_segments_for_zone(segments, zone, buffer_seconds)

        # Safety check: If we don't have enough segments, try a larger buffer
        # This can happen if the zone boundaries are tight
        if len(zone_segments) < (zone.end_ayah - zone.start_ayah + 1):
            zone_segments = find_segments_for_zone(segments, zone, buffer_seconds * 3)

        if not zone_segments:
            # Still no segments - skip this zone
            continue

        # --- Extract ayahs for this zone ---
        zone_ayahs = [
            ayah_by_num[n] for n in range(zone.start_ayah, zone.end_ayah + 1) if n in ayah_by_num
        ]

        # --- Re-run DP alignment on just this small region ---
        # This is much faster than re-aligning the entire surah
        new_zone_results = align_segments_dp(zone_segments, zone_ayahs)

        if not new_zone_results:
            # Re-alignment failed - skip this zone
            continue

        # ============================================================================
        # STEP 3: Compare Old vs New Results (Keep Better)
        # ============================================================================
        zone_improved = False
        new_by_ayah = {r.ayah.ayah_number: r for r in new_zone_results}

        # For each ayah in the zone, compare old vs new
        for i in range(zone.start_idx, zone.end_idx):
            if i >= len(new_results):
                break

            old_result = new_results[i]
            ayah_num = old_result.ayah.ayah_number

            if ayah_num in new_by_ayah:
                new_result = new_by_ayah[ayah_num]

                # Keep whichever result has higher similarity
                if new_result.similarity_score > old_result.similarity_score:
                    new_results[i] = new_result
                    stats.ayahs_improved += 1
                    zone_improved = True
                elif new_result.similarity_score < old_result.similarity_score:
                    # New result is worse - keep old result
                    stats.ayahs_degraded += 1
                else:
                    # Same score - keep old result
                    stats.ayahs_unchanged += 1

        if zone_improved:
            stats.zones_improved += 1

    return new_results, stats


def iterative_realign_problem_zones(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    passes: int = 3,
    initial_threshold: float = 0.85,
    buffer_seconds: float = 10.0,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Run multiple passes of zone realignment with decreasing thresholds.

    Each pass fixes the worst zones, improving context for the next pass.
    Uses adaptive thresholds so short ayahs aren't penalised.

    Args:
        results: Initial alignment results
        segments: All transcribed segments
        ayahs: All reference ayahs
        passes: Number of realignment passes (default 3)
        initial_threshold: Starting quality threshold (default 0.85)
        buffer_seconds: Extra time at zone boundaries

    Returns:
        Tuple of (updated results, cumulative stats)
    """
    cumulative = ZoneStats()
    current = list(results)

    for pass_idx in range(passes):
        # First pass: use original threshold with adaptive; later passes
        # lower the threshold but require more consecutive bad ayahs to avoid
        # touching areas that are already acceptable.
        if pass_idx == 0:
            thresh = initial_threshold
            min_consec = 3
        else:
            thresh = max(0.5, initial_threshold - pass_idx * 0.1)
            min_consec = max(3, 4 + pass_idx)  # 5, 6 — very conservative

        current, stats = realign_problem_zones(
            results=current,
            segments=segments,
            ayahs=ayahs,
            min_consecutive=min_consec,
            quality_threshold=thresh,
            buffer_seconds=buffer_seconds,
            adaptive=True,
        )
        cumulative.zones_found += stats.zones_found
        cumulative.zones_improved += stats.zones_improved
        cumulative.ayahs_improved += stats.ayahs_improved
        cumulative.ayahs_unchanged += stats.ayahs_unchanged
        cumulative.ayahs_degraded += stats.ayahs_degraded

        if stats.zones_found == 0:
            break  # No more problem zones

    return current, cumulative


def find_anchors(
    results: list[AlignmentResult],
    min_similarity: float = 0.95,
    min_wps: float = 0.8,
    max_wps: float = 2.0,
    confidence_weighted: bool = True,
) -> list[tuple[int, AlignmentResult]]:
    """
    Find anchor points - high-confidence ayahs with normal recitation pace.

    Anchors are ayahs we're very confident about. We use them as reference
    points and re-align the regions between them.

    When confidence_weighted is True, also considers timestamp consistency
    with neighbors as part of the anchor quality score.

    Args:
        results: List of alignment results
        min_similarity: Minimum similarity to be an anchor (default 0.95)
        min_wps: Minimum words per second (default 0.8)
        max_wps: Maximum words per second (default 2.0)
        confidence_weighted: Also check neighbor timestamp consistency

    Returns:
        List of (index, result) tuples for anchor points
    """
    anchors = []
    for i, r in enumerate(results):
        words = len(r.ayah.text.split())
        duration = r.end_time - r.start_time
        wps = words / duration if duration > 0 else 0

        if r.similarity_score < min_similarity or not (min_wps <= wps <= max_wps):
            continue

        if confidence_weighted and len(results) > 2:
            # Check that neighbors also have reasonable similarity
            neighbor_ok = True
            for ni in (i - 1, i + 1):
                if 0 <= ni < len(results):
                    if results[ni].similarity_score < 0.5:
                        neighbor_ok = False
                        break
                    # Check for timestamp ordering consistency
                    if ni == i - 1 and results[ni].end_time > r.start_time + 1.0:
                        neighbor_ok = False
                        break
                    if ni == i + 1 and results[ni].start_time < r.end_time - 1.0:
                        neighbor_ok = False
                        break
            if not neighbor_ok:
                continue

        anchors.append((i, r))

    return anchors


def realign_from_anchors(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    min_gap_size: int = 3,
    buffer_seconds: float = 5.0,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Re-align regions between anchor points (anchor-based drift correction).

    This is a SECOND PASS after problem zone re-alignment. It's more aggressive
    and uses a different strategy:

    STRATEGY:
    1. Find "anchor" ayahs (very high confidence, normal pace)
    2. Identify gaps between anchors (3+ ayahs)
    3. Re-align each gap independently
    4. Keep better results

    Example:
    ```
    Ayah 1:  0.97 ← Anchor
    Ayah 2:  0.82
    Ayah 3:  0.79  ← Gap (low confidence)
    Ayah 4:  0.81
    Ayah 5:  0.96 ← Anchor
    Ayah 6:  0.75
    Ayah 7:  0.98 ← Anchor
    ```

    Gaps identified: [2-4] between anchors 1 and 5
    We re-align ayahs 2-4 using segments in that time range.

    This complements problem zone re-alignment by:
    - Using different criteria (anchors vs consecutive low-conf)
    - Being more targeted (gaps between known-good points)
    - Catching drift that doesn't form long consecutive zones

    Args:
        results: Alignment results (ideally after problem zone re-alignment)
        segments: All transcribed segments
        ayahs: All reference ayahs
        min_gap_size: Minimum gap size to re-align (default 3 ayahs)
        buffer_seconds: Extra time at boundaries (default 5s)

    Returns:
        Tuple of (updated results, statistics)
    """
    stats = ZoneStats()

    # Find anchors
    anchors = find_anchors(results)

    if len(anchors) < 2:
        return results, stats

    # Create mutable copy
    new_results = list(results)
    ayah_by_num = {a.ayah_number: a for a in ayahs}

    # Find gaps between anchors
    gaps = []
    prev_idx = -1
    prev_result = None

    for idx, result in anchors:
        if prev_idx >= 0:
            gap_size = idx - prev_idx - 1
            if gap_size >= min_gap_size:
                gaps.append(
                    {
                        "start_idx": prev_idx + 1,
                        "end_idx": idx,
                        "start_time": prev_result.end_time - buffer_seconds,
                        "end_time": result.start_time + buffer_seconds,
                    }
                )
        prev_idx = idx
        prev_result = result

    # Check for gap at the end (after last anchor to end of surah)
    if anchors:
        last_idx, last_result = anchors[-1]
        if len(results) - last_idx - 1 >= min_gap_size:
            gaps.append(
                {
                    "start_idx": last_idx + 1,
                    "end_idx": len(results),
                    "start_time": last_result.end_time - buffer_seconds,
                    "end_time": segments[-1].end + buffer_seconds if segments else 0,
                }
            )

    stats.zones_found = len(gaps)

    # Re-align each gap
    for gap in gaps:
        start_idx = gap["start_idx"]
        end_idx = gap["end_idx"]
        time_start = max(0, gap["start_time"])
        time_end = gap["end_time"]

        # Get segments and ayahs for this gap
        gap_segments = [s for s in segments if s.start >= time_start and s.end <= time_end]
        gap_ayah_nums = [
            new_results[i].ayah.ayah_number
            for i in range(start_idx, min(end_idx, len(new_results)))
        ]
        gap_ayahs = [ayah_by_num[n] for n in gap_ayah_nums if n in ayah_by_num]

        if not gap_segments or not gap_ayahs:
            continue

        # Re-align
        new_gap_results = align_segments_dp(gap_segments, gap_ayahs)

        if not new_gap_results:
            continue

        # Update results where improved
        new_by_ayah = {r.ayah.ayah_number: r for r in new_gap_results}
        gap_improved = False

        for i in range(start_idx, min(end_idx, len(new_results))):
            old_result = new_results[i]
            ayah_num = old_result.ayah.ayah_number

            if ayah_num in new_by_ayah:
                new_result = new_by_ayah[ayah_num]

                if new_result.similarity_score > old_result.similarity_score + 0.01:
                    new_results[i] = new_result
                    stats.ayahs_improved += 1
                    gap_improved = True
                elif new_result.similarity_score < old_result.similarity_score - 0.01:
                    stats.ayahs_degraded += 1
                else:
                    stats.ayahs_unchanged += 1

        if gap_improved:
            stats.zones_improved += 1

    return new_results, stats


def fix_overlaps(results: list[AlignmentResult], min_gap: float = 0.0) -> int:
    """
    Fix any overlapping ayah timings by adjusting boundaries.

    Args:
        results: List of alignment results to fix in-place
        min_gap: Minimum gap in seconds between consecutive ayahs (default 0.0)

    Returns:
        Number of overlaps/zero-gaps fixed.
    """
    if len(results) < 2:
        return 0

    # Sort by ayah number
    results.sort(key=lambda r: r.ayah.ayah_number)

    fixes = 0
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]

        actual_gap = curr.start_time - prev.end_time

        # Fix if there's an overlap OR if gap is less than minimum
        if actual_gap < min_gap:
            # Calculate how much adjustment is needed
            gap_needed = min_gap - actual_gap

            # Split the adjustment: take half from previous end, add half to current start
            half_adjust = gap_needed / 2

            new_prev_end = prev.end_time - half_adjust
            new_curr_start = curr.start_time + half_adjust

            # Ensure we don't create invalid timings
            if new_prev_end < prev.start_time:
                new_prev_end = prev.start_time + 0.1
            if new_curr_start > curr.end_time:
                new_curr_start = curr.end_time - 0.1

            # Update timing (create new results since AlignmentResult might be immutable)
            results[i - 1] = AlignmentResult(
                ayah=prev.ayah,
                start_time=prev.start_time,
                end_time=round(new_prev_end, 2),
                transcribed_text=prev.transcribed_text,
                similarity_score=prev.similarity_score,
                overlap_detected=prev.overlap_detected,
            )
            results[i] = AlignmentResult(
                ayah=curr.ayah,
                start_time=round(new_curr_start, 2),
                end_time=curr.end_time,
                transcribed_text=curr.transcribed_text,
                similarity_score=curr.similarity_score,
                overlap_detected=curr.overlap_detected,
            )
            fixes += 1

    return fixes


def snap_boundaries_to_silences(
    results: list[AlignmentResult],
    silences_ms: list[tuple[int, int]] | None,
    max_snap_distance: float = 2.0,
) -> int:
    """
    Snap ayah start/end boundaries to nearest silence periods.

    This fixes timestamp drift by aligning ayah boundaries to actual
    pauses in the audio rather than relying on potentially drifted
    segment timestamps.

    Args:
        results: List of alignment results to fix in-place
        silences_ms: Silence periods in milliseconds [(start_ms, end_ms), ...]
        max_snap_distance: Maximum distance in seconds to snap (default 2.0)

    Returns:
        Number of boundaries snapped.
    """
    if not silences_ms or len(results) < 2:
        return 0

    # Convert silences to seconds and create lookup structure
    silences_sec = [(s / 1000.0, e / 1000.0) for s, e in silences_ms]
    silences_sec.sort(key=lambda x: x[0])

    # Create list of silence midpoints (natural ayah boundary points)
    [(s + e) / 2 for s, e in silences_sec]

    # Sort results by ayah number
    results.sort(key=lambda r: r.ayah.ayah_number)

    snaps = 0

    # For each ayah boundary (except first start and last end),
    # find the nearest silence and snap to it
    for i in range(len(results) - 1):
        curr = results[i]
        next_r = results[i + 1]

        # Current boundary is at curr.end_time / next_r.start_time
        boundary_time = (curr.end_time + next_r.start_time) / 2

        # Find nearest silence midpoint
        best_silence_mid = None
        best_distance = float("inf")

        for sil_start, sil_end in silences_sec:
            sil_mid = (sil_start + sil_end) / 2
            distance = abs(sil_mid - boundary_time)

            if distance < best_distance and distance <= max_snap_distance:
                best_distance = distance
                best_silence_mid = sil_mid
                best_silence = (sil_start, sil_end)

        if best_silence_mid is not None:
            # Snap curr.end to silence start, next.start to silence end
            sil_start, sil_end = best_silence

            # Only snap if it doesn't create invalid timings
            if sil_start > curr.start_time and sil_end < next_r.end_time:
                # Update curr to end at silence start
                results[i] = AlignmentResult(
                    ayah=curr.ayah,
                    start_time=curr.start_time,
                    end_time=round(sil_start, 2),
                    transcribed_text=curr.transcribed_text,
                    similarity_score=curr.similarity_score,
                    overlap_detected=curr.overlap_detected,
                )

                # Update next to start at silence end
                results[i + 1] = AlignmentResult(
                    ayah=next_r.ayah,
                    start_time=round(sil_end, 2),
                    end_time=next_r.end_time,
                    transcribed_text=next_r.transcribed_text,
                    similarity_score=next_r.similarity_score,
                    overlap_detected=next_r.overlap_detected,
                )

                snaps += 1

    return snaps


def snap_boundaries_to_energy(
    results: list[AlignmentResult],
    energy_envelope: list[tuple[float, float]],
    max_snap_distance: float = 1.0,
) -> int:
    """
    Snap ayah boundaries to local energy minima for precise timing.

    Uses the RMS energy envelope to find the lowest-energy point near
    each boundary, which typically corresponds to a brief pause between
    ayahs.

    Args:
        results: List of alignment results to fix in-place
        energy_envelope: Output of compute_energy_envelope()
        max_snap_distance: Maximum distance in seconds to snap (default 1.0)

    Returns:
        Number of boundaries snapped.
    """
    if not energy_envelope or len(results) < 2:
        return 0

    from ..transcription.silence import find_energy_minima

    results.sort(key=lambda r: r.ayah.ayah_number)
    snaps = 0

    for i in range(len(results) - 1):
        curr = results[i]
        next_r = results[i + 1]

        boundary_time = (curr.end_time + next_r.start_time) / 2
        search_start = boundary_time - max_snap_distance
        search_end = boundary_time + max_snap_distance

        minima = find_energy_minima(
            energy_envelope,
            search_start,
            search_end,
            top_n=1,
        )

        if not minima:
            continue

        snap_point = minima[0]

        # Only snap if it doesn't create invalid timings
        if snap_point > curr.start_time + 0.1 and snap_point < next_r.end_time - 0.1:
            results[i] = AlignmentResult(
                ayah=curr.ayah,
                start_time=curr.start_time,
                end_time=round(snap_point, 3),
                transcribed_text=curr.transcribed_text,
                similarity_score=curr.similarity_score,
                overlap_detected=curr.overlap_detected,
            )
            results[i + 1] = AlignmentResult(
                ayah=next_r.ayah,
                start_time=round(snap_point, 3),
                end_time=next_r.end_time,
                transcribed_text=next_r.transcribed_text,
                similarity_score=next_r.similarity_score,
                overlap_detected=next_r.overlap_detected,
            )
            snaps += 1

    return snaps


def identify_drift_zones(
    results: list[AlignmentResult],
    min_consecutive: int = 5,
    max_pace_ratio: float = 2.5,
) -> list[ProblemZone]:
    """
    Find zones where timestamps drift even though similarity is high.

    Traditional zone detection checks similarity, but repetitive short ayahs
    can have high similarity at the WRONG audio position.  This function
    checks *pace consistency*: if an ayah's duration per word is far from
    the surah-wide median, it's likely misplaced.

    Args:
        results: Alignment results (in order)
        min_consecutive: Minimum consecutive drifted ayahs to form a zone
        max_pace_ratio: Flag an ayah when its pace deviates by more than
                        this factor from the median pace

    Returns:
        List of ProblemZone objects for zones with pace drift.
    """
    if len(results) < min_consecutive:
        return []

    # Compute per-ayah duration-per-word (pace)
    paces: list[float] = []
    for r in results:
        duration = r.end_time - r.start_time
        n_words = max(len(normalize_arabic(r.ayah.text).split()), 1)
        paces.append(duration / n_words if duration > 0 else 0.0)

    # Median pace (robust to outliers)
    sorted_paces = sorted(p for p in paces if p > 0)
    if not sorted_paces:
        return []
    median_pace = sorted_paces[len(sorted_paces) // 2]

    if median_pace <= 0:
        return []

    # Flag ayahs with abnormal pace
    is_drifted = []
    for p in paces:
        if p <= 0:
            is_drifted.append(True)
        else:
            ratio = max(p / median_pace, median_pace / p)
            is_drifted.append(ratio > max_pace_ratio)

    # Find consecutive runs
    zones: list[ProblemZone] = []
    run_start = None

    for i, drifted in enumerate(is_drifted):
        if drifted:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_consecutive:
                zone_results = results[run_start:i]
                avg_sim = sum(r.similarity_score for r in zone_results) / len(zone_results)
                zones.append(
                    ProblemZone(
                        start_idx=run_start,
                        end_idx=i,
                        start_ayah=zone_results[0].ayah.ayah_number,
                        end_ayah=zone_results[-1].ayah.ayah_number,
                        avg_similarity=avg_sim,
                        start_time=zone_results[0].start_time,
                        end_time=zone_results[-1].end_time,
                    )
                )
            run_start = None

    # Trailing zone
    if run_start is not None and (len(results) - run_start) >= min_consecutive:
        zone_results = results[run_start:]
        avg_sim = sum(r.similarity_score for r in zone_results) / len(zone_results)
        zones.append(
            ProblemZone(
                start_idx=run_start,
                end_idx=len(results),
                start_ayah=zone_results[0].ayah.ayah_number,
                end_ayah=zone_results[-1].ayah.ayah_number,
                avg_similarity=avg_sim,
                start_time=zone_results[0].start_time,
                end_time=zone_results[-1].end_time,
            )
        )

    return zones


def _find_problem_runs(
    results: list[AlignmentResult],
    similarity_threshold: float = 0.75,
    min_consecutive: int = 2,
    max_pace_ratio: float = 2.5,
) -> list[tuple[int, int]]:
    """
    Find consecutive ayah runs that are low-confidence or pace-abnormal.

    Returns:
        List of (start_idx, end_idx) runs where end_idx is exclusive.
    """
    if len(results) < min_consecutive:
        return []

    # Per-ayah pace: seconds per word
    paces: list[float] = []
    for r in results:
        duration = r.end_time - r.start_time
        words = max(len(normalize_arabic(r.ayah.text).split()), 1)
        paces.append(duration / words if duration > 0 else 0.0)

    valid_paces = sorted(p for p in paces if p > 0)
    median_pace = valid_paces[len(valid_paces) // 2] if valid_paces else 0.0

    is_problem: list[bool] = []
    for i, r in enumerate(results):
        low_similarity = r.similarity_score < similarity_threshold
        pace = paces[i]

        pace_outlier = False
        if median_pace > 0 and pace > 0:
            ratio = max(pace / median_pace, median_pace / pace)
            pace_outlier = ratio > max_pace_ratio

        is_problem.append(low_similarity or pace_outlier)

    runs: list[tuple[int, int]] = []
    run_start = None

    for i, flag in enumerate(is_problem):
        if flag:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start >= min_consecutive:
                runs.append((run_start, i))
            run_start = None

    if run_start is not None and (len(results) - run_start) >= min_consecutive:
        runs.append((run_start, len(results)))

    return runs


def realign_drift_zones_word_dp(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    min_consecutive: int = 5,
    max_pace_ratio: float = 2.5,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Detect and re-align zones with timestamp drift using word-level DP.

    Unlike similarity-based zone realignment, this detects drift via pace
    analysis and re-runs word-DP on the zone with CORRECT time bounds
    derived from surrounding anchors.

    Args:
        results: Initial alignment results
        segments: All transcribed segments
        ayahs: All reference ayahs
        min_consecutive: Min consecutive drifted ayahs to form a zone
        max_pace_ratio: Pace deviation threshold

    Returns:
        Tuple of (updated results, statistics)
    """
    from .matcher import similarity as _sim
    from .word_level_dp import (
        align_words_dp,
        build_reference_words,
        build_word_stream,
    )

    stats = ZoneStats()
    drift_zones = identify_drift_zones(results, min_consecutive, max_pace_ratio)
    stats.zones_found = len(drift_zones)

    if not drift_zones:
        return results, stats

    new_results = list(results)
    ayah_by_num = {a.ayah_number: a for a in ayahs}

    # Build full word stream once
    from .dp_core import _filter_special_segments

    filtered = _filter_special_segments(segments, ayahs)
    all_words = build_word_stream(filtered)
    all_ref_words = build_reference_words(ayahs)

    if not all_words:
        return results, stats

    for zone in drift_zones:
        # Find anchor times from surrounding well-placed ayahs.
        # Look for the last non-drifted ayah before the zone and the
        # first non-drifted ayah after.
        anchor_start_time = 0.0
        anchor_end_time = all_words[-1].estimated_end

        # Search backwards from zone start for a reliable anchor
        for i in range(zone.start_idx - 1, -1, -1):
            r = new_results[i]
            dur = r.end_time - r.start_time
            n_w = max(len(normalize_arabic(r.ayah.text).split()), 1)
            if dur > 0 and r.similarity_score >= 0.7:
                anchor_start_time = r.end_time
                break

        # Search forwards from zone end for a reliable anchor
        for i in range(zone.end_idx, len(new_results)):
            r = new_results[i]
            dur = r.end_time - r.start_time
            n_w = max(len(normalize_arabic(r.ayah.text).split()), 1)
            if dur > 0 and r.similarity_score >= 0.7:
                anchor_end_time = r.start_time
                break

        # Generous buffer
        buffer = 5.0
        t_lo = max(0.0, anchor_start_time - buffer)
        t_hi = anchor_end_time + buffer

        # Extract words within the anchor time bounds
        zone_word_indices = [
            i
            for i, w in enumerate(all_words)
            if w.estimated_start >= t_lo and w.estimated_end <= t_hi
        ]

        if not zone_word_indices:
            continue

        w_lo = zone_word_indices[0]
        w_hi = zone_word_indices[-1] + 1
        zone_words = all_words[w_lo:w_hi]

        # Extract ayahs for this zone
        zone_ayah_nums = list(range(zone.start_ayah, zone.end_ayah + 1))
        zone_ayahs = [ayah_by_num[n] for n in zone_ayah_nums if n in ayah_by_num]
        zone_ayah_indices = [n - 1 for n in zone_ayah_nums if n in ayah_by_num]
        zone_ref_words = [all_ref_words[i] for i in zone_ayah_indices]

        if not zone_ayahs or not zone_words:
            continue

        # Run word-DP on just this zone
        assignments = align_words_dp(
            zone_words,
            zone_ayahs,
            zone_ref_words,
            max_word_ratio=3.0,
            beam_width=80,
        )

        if not assignments:
            continue

        # Convert assignments to results and compare
        zone_improved = False
        new_by_ayah: dict[int, AlignmentResult] = {}

        for word_start, word_end, ayah_idx in assignments:
            ayah = zone_ayahs[ayah_idx]
            start_time = zone_words[word_start].estimated_start
            end_time = zone_words[word_end - 1].estimated_end
            transcribed = " ".join(w.text for w in zone_words[word_start:word_end])
            sim = _sim(transcribed, ayah.text)

            new_by_ayah[ayah.ayah_number] = AlignmentResult(
                ayah=ayah,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                transcribed_text=transcribed,
                similarity_score=round(sim, 4),
                overlap_detected=False,
            )

        # Replace results for zone ayahs — accept if overall zone improves.
        # For drift zones, we check BOTH similarity and pace consistency.
        # A result with slightly lower sim but much better pace is preferred.
        sorted_paces = []
        for r in new_results:
            dur = r.end_time - r.start_time
            nw = max(len(normalize_arabic(r.ayah.text).split()), 1)
            if dur > 0:
                sorted_paces.append(dur / nw)
        sorted_paces.sort()
        median_pace = sorted_paces[len(sorted_paces) // 2] if sorted_paces else 1.0

        for i in range(zone.start_idx, min(zone.end_idx, len(new_results))):
            old_r = new_results[i]
            anum = old_r.ayah.ayah_number
            if anum not in new_by_ayah:
                continue

            new_r = new_by_ayah[anum]

            # Compute pace quality for old and new
            old_dur = old_r.end_time - old_r.start_time
            new_dur = new_r.end_time - new_r.start_time
            n_w = max(len(normalize_arabic(old_r.ayah.text).split()), 1)

            old_pace = old_dur / n_w if old_dur > 0 else 0
            new_pace = new_dur / n_w if new_dur > 0 else 0

            old_pace_dev = abs(old_pace - median_pace) / median_pace if median_pace > 0 else 999
            new_pace_dev = abs(new_pace - median_pace) / median_pace if median_pace > 0 else 999

            # Accept new result if:
            # 1. Similarity is at least 90% of old, AND
            # 2. Pace is significantly closer to median
            sim_ok = new_r.similarity_score >= old_r.similarity_score * 0.9
            pace_improved = new_pace_dev < old_pace_dev * 0.8

            if sim_ok and pace_improved:
                new_results[i] = new_r
                stats.ayahs_improved += 1
                zone_improved = True
            elif new_r.similarity_score > old_r.similarity_score + 0.02:
                new_results[i] = new_r
                stats.ayahs_improved += 1
                zone_improved = True
            else:
                stats.ayahs_unchanged += 1

        if zone_improved:
            stats.zones_improved += 1

    return new_results, stats


# Export for use in other modules
__all__ = [
    "realign_problem_zones",
    "iterative_realign_problem_zones",
    "identify_problem_zones",
    "identify_drift_zones",
    "realign_drift_zones_word_dp",
    "adaptive_quality_threshold",
    "realign_from_anchors",
    "find_anchors",
    "fix_overlaps",
    "snap_boundaries_to_silences",
    "snap_boundaries_to_energy",
    "ProblemZone",
    "ZoneStats",
]
