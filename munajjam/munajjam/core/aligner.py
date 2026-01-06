"""
Unified Aligner class for Quran audio alignment.

This module provides a single, simple interface for aligning transcribed
audio segments with reference Quran ayahs. It supports multiple alignment
strategies and handles all post-processing (zone realignment, overlap fixing).

Usage:
    from munajjam.core import Aligner

    aligner = Aligner(strategy="hybrid")
    results = aligner.align(segments, ayahs, silences_ms=silences)
"""

from enum import Enum
from typing import Callable

from ..models import Segment, Ayah, AlignmentResult
from .hybrid import HybridStats


class AlignmentStrategy(str, Enum):
    """Available alignment strategies."""
    GREEDY = "greedy"  # Fast, simple greedy matching
    DP = "dp"  # Dynamic programming for optimal alignment
    HYBRID = "hybrid"  # DP with fallback to greedy (recommended)


class Aligner:
    """
    Unified alignment interface.

    Provides a single entry point for all alignment operations with
    configurable strategy and automatic post-processing.

    Attributes:
        strategy: The alignment strategy to use
        quality_threshold: Similarity threshold for high-quality alignment (0.0-1.0)
        fix_drift: Whether to run zone realignment to fix timing drift
        fix_overlaps: Whether to fix overlapping ayah timings

    Example:
        >>> aligner = Aligner(strategy="hybrid")
        >>> results = aligner.align(segments, ayahs)
        >>> for r in results:
        ...     print(f"Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s")
    """

    def __init__(
        self,
        strategy: str | AlignmentStrategy = AlignmentStrategy.HYBRID,
        quality_threshold: float = 0.85,
        fix_drift: bool = True,
        fix_overlaps: bool = True,
    ):
        """
        Initialize the Aligner.

        Args:
            strategy: Alignment strategy ("greedy", "dp", or "hybrid")
            quality_threshold: Similarity threshold for quality checks (0.0-1.0)
            fix_drift: Run zone realignment to fix timing drift in long surahs
            fix_overlaps: Fix any overlapping ayah timings
        """
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.fix_drift = fix_drift
        self.fix_overlaps = fix_overlaps

        # Stats from last alignment (only populated for hybrid strategy)
        self._last_stats: HybridStats | None = None

    @property
    def last_stats(self) -> HybridStats | None:
        """Get stats from the last hybrid alignment, or None if not applicable."""
        return self._last_stats

    def align(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[AlignmentResult]:
        """
        Align transcribed segments to reference ayahs.

        This is the main method for performing alignment. It automatically
        applies the configured strategy and post-processing steps.

        Args:
            segments: List of transcribed Segment objects
            ayahs: List of reference Ayah objects (in order)
            silences_ms: Optional silence periods in milliseconds [(start, end), ...]
            on_progress: Optional callback for progress updates (current, total)

        Returns:
            List of AlignmentResult objects with timing and similarity info
        """
        if not segments or not ayahs:
            return []

        # Clear previous stats
        self._last_stats = None

        # Run alignment based on strategy
        if self.strategy == AlignmentStrategy.GREEDY:
            results = self._align_greedy(segments, ayahs, silences_ms)
        elif self.strategy == AlignmentStrategy.DP:
            results = self._align_dp(segments, ayahs, silences_ms, on_progress)
        else:  # HYBRID
            results = self._align_hybrid(segments, ayahs, silences_ms, on_progress)

        # Post-processing
        if self.fix_drift and results:
            results = self._apply_drift_fix(results, segments, ayahs)

        if self.fix_overlaps and results:
            self._apply_overlap_fix(results)

        return results

    def _align_greedy(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
    ) -> list[AlignmentResult]:
        """Run greedy alignment."""
        from .aligner_greedy import align_segments

        return align_segments(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
        )

    def _align_dp(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run DP alignment."""
        from .dp_core import align_segments_dp_with_constraints

        return align_segments_dp_with_constraints(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            on_progress=on_progress,
        )

    def _align_hybrid(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run hybrid alignment (DP + fallback)."""
        from .hybrid import align_segments_hybrid

        results, stats = align_segments_hybrid(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            quality_threshold=self.quality_threshold,
            on_progress=on_progress,
        )
        self._last_stats = stats
        return results

    def _apply_drift_fix(
        self,
        results: list[AlignmentResult],
        segments: list[Segment],
        ayahs: list[Ayah],
    ) -> list[AlignmentResult]:
        """Apply zone realignment to fix timing drift."""
        from .zone_realigner import realign_problem_zones, realign_from_anchors

        # First pass: realign problem zones
        results, _ = realign_problem_zones(
            results=results,
            segments=segments,
            ayahs=ayahs,
            min_consecutive=3,
            quality_threshold=self.quality_threshold,
            buffer_seconds=10.0,
        )

        # Second pass: anchor-based realignment
        results, _ = realign_from_anchors(
            results=results,
            segments=segments,
            ayahs=ayahs,
            min_gap_size=3,
            buffer_seconds=5.0,
        )

        return results

    def _apply_overlap_fix(self, results: list[AlignmentResult]) -> int:
        """Fix overlapping ayah timings in-place."""
        from .zone_realigner import fix_overlaps

        return fix_overlaps(results)


# Convenience function for simple usage
def align(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    strategy: str = "hybrid",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    Convenience function for alignment with default settings.

    This is equivalent to:
        Aligner(strategy=strategy).align(segments, ayahs, silences_ms)

    Args:
        segments: List of transcribed Segment objects
        ayahs: List of reference Ayah objects
        silences_ms: Optional silence periods in milliseconds
        strategy: Alignment strategy ("greedy", "dp", or "hybrid")
        on_progress: Optional progress callback

    Returns:
        List of AlignmentResult objects
    """
    aligner = Aligner(strategy=strategy)
    return aligner.align(segments, ayahs, silences_ms, on_progress)
