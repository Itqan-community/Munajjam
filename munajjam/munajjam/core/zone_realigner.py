"""
Zone Re-alignment Module

Fixes drift issues in long surahs by:
1. Identifying "problem zones" - consecutive low-confidence ayahs
2. Re-aligning just those zones using segments in the time range
3. Keeping the better result for each ayah
"""

from dataclasses import dataclass
from munajjam.models import Segment, Ayah, AlignmentResult
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
    end_idx: int    # Index in results list (exclusive)
    start_ayah: int
    end_ayah: int
    avg_similarity: float
    start_time: float
    end_time: float


def identify_problem_zones(
    results: list[AlignmentResult],
    min_consecutive: int = 3,
    quality_threshold: float = 0.85,
) -> list[ProblemZone]:
    """
    Find sequences of consecutive low-confidence ayahs.
    
    Args:
        results: List of alignment results
        min_consecutive: Minimum consecutive low-conf ayahs to form a zone
        quality_threshold: Similarity threshold below which is "low confidence"
    
    Returns:
        List of ProblemZone objects
    """
    zones = []
    current_zone_start = None
    current_zone_sims = []
    
    for i, result in enumerate(results):
        is_low = result.similarity_score < quality_threshold
        
        if is_low:
            if current_zone_start is None:
                current_zone_start = i
                current_zone_sims = [result.similarity_score]
            else:
                current_zone_sims.append(result.similarity_score)
        else:
            # End of potential zone
            if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
                zone_results = results[current_zone_start:i]
                zones.append(ProblemZone(
                    start_idx=current_zone_start,
                    end_idx=i,
                    start_ayah=zone_results[0].ayah.ayah_number,
                    end_ayah=zone_results[-1].ayah.ayah_number,
                    avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
                    start_time=zone_results[0].start_time,
                    end_time=zone_results[-1].end_time,
                ))
            current_zone_start = None
            current_zone_sims = []
    
    # Handle zone at the end
    if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
        zone_results = results[current_zone_start:]
        zones.append(ProblemZone(
            start_idx=current_zone_start,
            end_idx=len(results),
            start_ayah=zone_results[0].ayah.ayah_number,
            end_ayah=zone_results[-1].ayah.ayah_number,
            avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
            start_time=zone_results[0].start_time,
            end_time=zone_results[-1].end_time,
        ))
    
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
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Re-align problem zones in the results.
    
    This is the main function to call for drift fix.
    
    Args:
        results: Initial alignment results
        segments: All segments
        ayahs: All ayahs
        min_consecutive: Min consecutive low-conf ayahs to form a zone
        quality_threshold: Similarity threshold for low confidence
        buffer_seconds: Extra time at zone boundaries
    
    Returns:
        Tuple of (updated results, statistics)
    """
    stats = ZoneStats()
    
    # Identify problem zones
    zones = identify_problem_zones(results, min_consecutive, quality_threshold)
    stats.zones_found = len(zones)
    
    if not zones:
        return results, stats
    
    # Create a mutable copy of results
    new_results = list(results)
    
    # Create ayah lookup
    ayah_by_num = {a.ayah_number: a for a in ayahs}
    
    for zone in zones:
        # Find segments for this zone
        zone_segments = find_segments_for_zone(segments, zone, buffer_seconds)
        
        if len(zone_segments) < (zone.end_ayah - zone.start_ayah + 1):
            # Not enough segments, try with larger buffer
            zone_segments = find_segments_for_zone(segments, zone, buffer_seconds * 3)
        
        if not zone_segments:
            continue
        
        # Re-align the zone
        zone_ayahs = [ayah_by_num[n] for n in range(zone.start_ayah, zone.end_ayah + 1) 
                      if n in ayah_by_num]
        
        new_zone_results = align_segments_dp(zone_segments, zone_ayahs)
        
        if not new_zone_results:
            continue
        
        # Compare and keep better results
        zone_improved = False
        new_by_ayah = {r.ayah.ayah_number: r for r in new_zone_results}
        
        for i in range(zone.start_idx, zone.end_idx):
            if i >= len(new_results):
                break
                
            old_result = new_results[i]
            ayah_num = old_result.ayah.ayah_number
            
            if ayah_num in new_by_ayah:
                new_result = new_by_ayah[ayah_num]
                
                if new_result.similarity_score > old_result.similarity_score:
                    new_results[i] = new_result
                    stats.ayahs_improved += 1
                    zone_improved = True
                elif new_result.similarity_score < old_result.similarity_score:
                    stats.ayahs_degraded += 1
                else:
                    stats.ayahs_unchanged += 1
        
        if zone_improved:
            stats.zones_improved += 1
    
    return new_results, stats


def find_anchors(
    results: list[AlignmentResult],
    min_similarity: float = 0.95,
    min_wps: float = 0.8,
    max_wps: float = 2.0,
) -> list[tuple[int, AlignmentResult]]:
    """
    Find anchor points - high-confidence ayahs with normal WPS.
    
    Returns list of (index, result) tuples.
    """
    anchors = []
    for i, r in enumerate(results):
        words = len(r.ayah.text.split())
        duration = r.end_time - r.start_time
        wps = words / duration if duration > 0 else 0
        
        if r.similarity_score >= min_similarity and min_wps <= wps <= max_wps:
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
    Re-align regions between anchor points.
    
    This is more aggressive than zone re-alignment - it uses high-confidence
    ayahs as anchors and re-aligns everything between them.
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
                gaps.append({
                    'start_idx': prev_idx + 1,
                    'end_idx': idx,
                    'start_time': prev_result.end_time - buffer_seconds,
                    'end_time': result.start_time + buffer_seconds,
                })
        prev_idx = idx
        prev_result = result
    
    # Check for gap at the end (after last anchor to end of surah)
    if anchors:
        last_idx, last_result = anchors[-1]
        if len(results) - last_idx - 1 >= min_gap_size:
            gaps.append({
                'start_idx': last_idx + 1,
                'end_idx': len(results),
                'start_time': last_result.end_time - buffer_seconds,
                'end_time': segments[-1].end + buffer_seconds if segments else 0,
            })
    
    stats.zones_found = len(gaps)
    
    # Re-align each gap
    for gap in gaps:
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']
        time_start = max(0, gap['start_time'])
        time_end = gap['end_time']
        
        # Get segments and ayahs for this gap
        gap_segments = [s for s in segments if s.start >= time_start and s.end <= time_end]
        gap_ayah_nums = [new_results[i].ayah.ayah_number for i in range(start_idx, min(end_idx, len(new_results)))]
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


def fix_overlaps(results: list[AlignmentResult]) -> int:
    """
    Fix any overlapping ayah timings by adjusting boundaries.
    
    Returns number of overlaps fixed.
    """
    if len(results) < 2:
        return 0
    
    # Sort by ayah number
    results.sort(key=lambda r: r.ayah.ayah_number)
    
    fixes = 0
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        if curr.start_time < prev.end_time:
            # Fix by meeting at midpoint
            midpoint = (prev.end_time + curr.start_time) / 2
            
            # Update timing (create new results since AlignmentResult might be immutable)
            results[i-1] = AlignmentResult(
                ayah=prev.ayah,
                start_time=prev.start_time,
                end_time=round(midpoint - 0.05, 2),
                transcribed_text=prev.transcribed_text,
                similarity_score=prev.similarity_score,
                overlap_detected=prev.overlap_detected,
            )
            results[i] = AlignmentResult(
                ayah=curr.ayah,
                start_time=round(midpoint + 0.05, 2),
                end_time=curr.end_time,
                transcribed_text=curr.transcribed_text,
                similarity_score=curr.similarity_score,
                overlap_detected=curr.overlap_detected,
            )
            fixes += 1
    
    return fixes


# Export for use in other modules
__all__ = [
    'realign_problem_zones',
    'identify_problem_zones',
    'realign_from_anchors',
    'find_anchors',
    'fix_overlaps',
    'ProblemZone',
    'ZoneStats',
]
