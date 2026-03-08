"""
Unit tests for benchmark metric computation (Issue #45).

Tests cover:
- compute_mae with identical, offset, empty, and mismatched lists
- compute_avg_similarity with known scores
- compute_pct_high_confidence with all/none/mixed high-confidence
- compute_strategy_metrics assembly function
"""

import pytest

from munajjam.benchmark.metrics import (
    compute_avg_similarity,
    compute_mae,
    compute_pct_high_confidence,
    compute_strategy_metrics,
)
from munajjam.benchmark.models import GroundTruthAyah, GroundTruthSurah
from munajjam.models import AlignmentResult, Ayah


def _make_result(
    ayah_number: int,
    start: float,
    end: float,
    similarity: float,
) -> AlignmentResult:
    """Helper to create an AlignmentResult with minimal boilerplate."""
    return AlignmentResult(
        ayah=Ayah(id=ayah_number, surah_id=1, ayah_number=ayah_number, text="test"),
        start_time=start,
        end_time=end,
        transcribed_text="test",
        similarity_score=similarity,
    )


class TestComputeMae:
    """Test mean absolute error computation."""

    def test_identical_lists(self) -> None:
        assert compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_offset(self) -> None:
        # |1.0-1.5| + |2.0-2.5| = 0.5 + 0.5 = 1.0, / 2 = 0.5
        assert compute_mae([1.0, 2.0], [1.5, 2.5]) == pytest.approx(0.5)

    def test_empty_predicted(self) -> None:
        assert compute_mae([], [1.0, 2.0]) == 0.0

    def test_empty_actual(self) -> None:
        assert compute_mae([1.0, 2.0], []) == 0.0

    def test_both_empty(self) -> None:
        assert compute_mae([], []) == 0.0

    def test_mismatched_lengths_uses_min(self) -> None:
        # Only compare first 2 pairs: |1.0-1.1| + |2.0-2.2| = 0.1 + 0.2 = 0.3 / 2 = 0.15
        result = compute_mae([1.0, 2.0, 3.0], [1.1, 2.2])
        assert result == pytest.approx(0.15)

    def test_single_element(self) -> None:
        assert compute_mae([5.0], [3.0]) == pytest.approx(2.0)


class TestComputeAvgSimilarity:
    """Test average similarity computation."""

    def test_empty_results(self) -> None:
        assert compute_avg_similarity([]) == 0.0

    def test_known_scores(self) -> None:
        results = [
            _make_result(1, 0.0, 5.0, 0.9),
            _make_result(2, 5.0, 10.0, 0.8),
        ]
        assert compute_avg_similarity(results) == pytest.approx(0.85)

    def test_single_result(self) -> None:
        results = [_make_result(1, 0.0, 5.0, 0.95)]
        assert compute_avg_similarity(results) == pytest.approx(0.95)

    def test_all_perfect(self) -> None:
        results = [_make_result(i, 0.0, 1.0, 1.0) for i in range(1, 6)]
        assert compute_avg_similarity(results) == pytest.approx(1.0)


class TestComputePctHighConfidence:
    """Test percentage high-confidence computation."""

    def test_empty_results(self) -> None:
        assert compute_pct_high_confidence([]) == 0.0

    def test_all_high_confidence(self) -> None:
        # is_high_confidence = similarity_score >= 0.8
        results = [_make_result(i, 0.0, 1.0, 0.9) for i in range(1, 4)]
        assert compute_pct_high_confidence(results) == pytest.approx(100.0)

    def test_none_high_confidence(self) -> None:
        results = [_make_result(i, 0.0, 1.0, 0.5) for i in range(1, 4)]
        assert compute_pct_high_confidence(results) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        results = [
            _make_result(1, 0.0, 1.0, 0.9),  # high
            _make_result(2, 1.0, 2.0, 0.5),  # low
            _make_result(3, 2.0, 3.0, 0.85),  # high
            _make_result(4, 3.0, 4.0, 0.7),  # low
        ]
        assert compute_pct_high_confidence(results) == pytest.approx(50.0)

    def test_boundary_at_0_8(self) -> None:
        """Exactly 0.8 should be high confidence."""
        results = [_make_result(1, 0.0, 1.0, 0.8)]
        assert compute_pct_high_confidence(results) == pytest.approx(100.0)

    def test_just_below_0_8(self) -> None:
        """0.79 should NOT be high confidence."""
        results = [_make_result(1, 0.0, 1.0, 0.79)]
        assert compute_pct_high_confidence(results) == pytest.approx(0.0)


class TestComputeStrategyMetrics:
    """Test the assembly function that computes all metrics."""

    def test_produces_valid_strategy_metrics(self) -> None:
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="test.wav",
            ayahs=[
                GroundTruthAyah(ayah_number=1, start_time=0.0, end_time=5.0, text="a"),
                GroundTruthAyah(ayah_number=2, start_time=5.5, end_time=10.0, text="b"),
            ],
        )
        results = [
            _make_result(1, 0.1, 5.2, 0.95),
            _make_result(2, 5.3, 10.1, 0.85),
        ]
        metrics = compute_strategy_metrics(
            strategy="greedy",
            surah_id=1,
            results=results,
            ground_truth=gt,
            runtime_seconds=0.042,
        )
        assert metrics.strategy == "greedy"
        assert metrics.surah_id == 1
        assert metrics.mae_start == pytest.approx(0.15, abs=0.01)
        assert metrics.mae_end == pytest.approx(0.15, abs=0.01)
        assert metrics.avg_similarity == pytest.approx(0.9)
        assert metrics.pct_high_confidence == pytest.approx(100.0)
        assert metrics.runtime_seconds == pytest.approx(0.042)
        assert metrics.ayah_count == 2
        assert metrics.timestamp  # non-empty

    def test_empty_results(self) -> None:
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="test.wav",
            ayahs=[
                GroundTruthAyah(ayah_number=1, start_time=0.0, end_time=5.0, text="a"),
            ],
        )
        metrics = compute_strategy_metrics(
            strategy="dp",
            surah_id=1,
            results=[],
            ground_truth=gt,
            runtime_seconds=0.01,
        )
        assert metrics.mae_start == 0.0
        assert metrics.avg_similarity == 0.0
        assert metrics.ayah_count == 0
