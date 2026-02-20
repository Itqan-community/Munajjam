"""
Unit tests for the benchmark harness.

Tests cover:
- Ground-truth data integrity
- Synthetic segment generation
- Metrics computation
- JSON output format
- Leaderboard generation
- Full benchmark run (without GPU)
"""

import json
import time
from pathlib import Path

import pytest

from benchmarks.benchmark import (
    GROUND_TRUTH,
    STRATEGIES,
    SURAH_IDS,
    StrategyMetrics,
    _compute_metrics,
    _make_ayahs_from_ground_truth,
    _make_segments_from_ground_truth,
    _rank_strategies,
    generate_leaderboard,
    run_benchmark,
    save_results_json,
)
from munajjam.core import Aligner
from munajjam.models import AlignmentResult, Ayah, Segment, SegmentType


# ---------------------------------------------------------------------------
# Ground-truth data integrity
# ---------------------------------------------------------------------------


class TestGroundTruthData:
    """Verify that ground-truth data is well-formed."""

    def test_ground_truth_has_two_surahs(self):
        assert len(GROUND_TRUTH) == 2

    def test_surah_1_has_seven_ayahs(self):
        assert len(GROUND_TRUTH["surah_1_fatiha"]) == 7

    def test_surah_112_has_four_ayahs(self):
        assert len(GROUND_TRUTH["surah_112_ikhlas"]) == 4

    def test_all_ayahs_have_required_fields(self):
        required = {"ayah_number", "text", "start_time", "end_time"}
        for surah_name, ayahs in GROUND_TRUTH.items():
            for ayah in ayahs:
                assert required.issubset(ayah.keys()), (
                    f"Missing fields in {surah_name}: {required - ayah.keys()}"
                )

    def test_timestamps_are_non_negative(self):
        for surah_name, ayahs in GROUND_TRUTH.items():
            for ayah in ayahs:
                assert ayah["start_time"] >= 0, f"Negative start_time in {surah_name}"
                assert ayah["end_time"] > ayah["start_time"], (
                    f"end_time <= start_time in {surah_name} ayah {ayah['ayah_number']}"
                )

    def test_timestamps_are_sequential(self):
        """Each ayah should start after the previous one ends."""
        for surah_name, ayahs in GROUND_TRUTH.items():
            for i in range(1, len(ayahs)):
                assert ayahs[i]["start_time"] >= ayahs[i - 1]["end_time"], (
                    f"Overlapping timestamps in {surah_name} at ayah {i + 1}"
                )

    def test_all_texts_are_non_empty(self):
        for surah_name, ayahs in GROUND_TRUTH.items():
            for ayah in ayahs:
                assert ayah["text"].strip(), f"Empty text in {surah_name}"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestSyntheticDataGeneration:
    """Test that synthetic segments and ayahs are generated correctly."""

    def test_segments_count_matches_ground_truth(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        segments = _make_segments_from_ground_truth(gt, surah_id=1)
        assert len(segments) == len(gt)

    def test_segments_have_valid_times(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        segments = _make_segments_from_ground_truth(gt, surah_id=1)
        for seg in segments:
            assert seg.start >= 0
            assert seg.end > seg.start

    def test_segments_have_correct_surah_id(self):
        gt = GROUND_TRUTH["surah_112_ikhlas"]
        segments = _make_segments_from_ground_truth(gt, surah_id=112)
        for seg in segments:
            assert seg.surah_id == 112

    def test_segments_preserve_text(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        segments = _make_segments_from_ground_truth(gt, surah_id=1)
        for seg, g in zip(segments, gt):
            assert seg.text == g["text"]

    def test_ayahs_count_matches_ground_truth(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        ayahs = _make_ayahs_from_ground_truth(gt, surah_id=1)
        assert len(ayahs) == len(gt)

    def test_ayahs_have_correct_surah_id(self):
        gt = GROUND_TRUTH["surah_112_ikhlas"]
        ayahs = _make_ayahs_from_ground_truth(gt, surah_id=112)
        for ayah in ayahs:
            assert ayah.surah_id == 112

    def test_ayahs_have_sequential_numbers(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        ayahs = _make_ayahs_from_ground_truth(gt, surah_id=1)
        for i, ayah in enumerate(ayahs):
            assert ayah.ayah_number == gt[i]["ayah_number"]

    def test_zero_noise_preserves_exact_timestamps(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        segments = _make_segments_from_ground_truth(gt, surah_id=1, noise_factor=0.0)
        for seg, g in zip(segments, gt):
            assert abs(seg.start - g["start_time"]) < 1e-9
            assert abs(seg.end - g["end_time"]) < 1e-9


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


class TestMetricsComputation:
    """Test that metrics are computed correctly."""

    def _make_perfect_results(self, gt: list[dict], surah_id: int) -> list[AlignmentResult]:
        """Build AlignmentResult objects that perfectly match ground truth."""
        ayahs = _make_ayahs_from_ground_truth(gt, surah_id)
        return [
            AlignmentResult(
                ayah=ayah,
                start_time=g["start_time"],
                end_time=g["end_time"],
                transcribed_text=g["text"],
                similarity_score=1.0,
            )
            for ayah, g in zip(ayahs, gt)
        ]

    def test_perfect_results_give_zero_mae(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        results = self._make_perfect_results(gt, surah_id=1)
        metrics = _compute_metrics(results, gt, "greedy", "surah_1_fatiha", 0.1)
        assert metrics.mae_start == 0.0
        assert metrics.mae_end == 0.0
        assert metrics.mae_combined == 0.0

    def test_perfect_results_give_full_similarity(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        results = self._make_perfect_results(gt, surah_id=1)
        metrics = _compute_metrics(results, gt, "greedy", "surah_1_fatiha", 0.1)
        assert metrics.avg_similarity == 1.0
        assert metrics.pct_high_confidence == 100.0

    def test_metrics_record_runtime(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        results = self._make_perfect_results(gt, surah_id=1)
        metrics = _compute_metrics(results, gt, "greedy", "surah_1_fatiha", 1.234)
        assert metrics.runtime_seconds == 1.234

    def test_empty_results_return_zero_metrics(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        metrics = _compute_metrics([], gt, "greedy", "surah_1_fatiha", 0.0)
        assert metrics.num_ayahs == 0
        assert metrics.aligned_count == 0

    def test_mae_is_non_negative(self):
        gt = GROUND_TRUTH["surah_1_fatiha"]
        results = self._make_perfect_results(gt, surah_id=1)
        # Shift all results by 0.5 seconds
        shifted = [
            AlignmentResult(
                ayah=r.ayah,
                start_time=r.start_time + 0.5,
                end_time=r.end_time + 0.5,
                transcribed_text=r.transcribed_text,
                similarity_score=r.similarity_score,
            )
            for r in results
        ]
        metrics = _compute_metrics(shifted, gt, "greedy", "surah_1_fatiha", 0.1)
        assert metrics.mae_start >= 0
        assert metrics.mae_end >= 0
        assert abs(metrics.mae_start - 0.5) < 0.001


# ---------------------------------------------------------------------------
# Strategy ranking
# ---------------------------------------------------------------------------


class TestStrategyRanking:
    """Test that strategy ranking is consistent and correct."""

    def _make_metrics_list(self) -> list[StrategyMetrics]:
        return [
            StrategyMetrics("greedy", "surah_1_fatiha", 7, mae_combined=0.5, avg_similarity=0.7, pct_high_confidence=60.0, runtime_seconds=0.01),
            StrategyMetrics("dp", "surah_1_fatiha", 7, mae_combined=0.2, avg_similarity=0.9, pct_high_confidence=90.0, runtime_seconds=0.05),
            StrategyMetrics("hybrid", "surah_1_fatiha", 7, mae_combined=0.3, avg_similarity=0.85, pct_high_confidence=80.0, runtime_seconds=0.04),
            StrategyMetrics("auto", "surah_1_fatiha", 7, mae_combined=0.25, avg_similarity=0.88, pct_high_confidence=85.0, runtime_seconds=0.04),
        ]

    def test_ranking_returns_all_strategies(self):
        metrics = self._make_metrics_list()
        ranked = _rank_strategies(metrics)
        strategies = [r[0] for r in ranked]
        assert set(strategies) == {"greedy", "dp", "hybrid", "auto"}

    def test_ranking_is_sorted_descending(self):
        metrics = self._make_metrics_list()
        ranked = _rank_strategies(metrics)
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_best_strategy_has_highest_score(self):
        metrics = self._make_metrics_list()
        ranked = _rank_strategies(metrics)
        # dp has best similarity and MAE in our test data
        assert ranked[0][0] == "dp"


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Test JSON serialization of benchmark results."""

    def test_json_output_is_valid(self, tmp_path):
        metrics = [
            StrategyMetrics("greedy", "surah_1_fatiha", 7, mae_combined=0.3, avg_similarity=0.85, pct_high_confidence=80.0, runtime_seconds=0.01),
        ]
        output = tmp_path / "results" / "benchmark_results.json"
        save_results_json(metrics, output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "generated_at" in data
        assert "results" in data
        assert len(data["results"]) == 1

    def test_json_contains_all_metric_fields(self, tmp_path):
        metrics = [
            StrategyMetrics("greedy", "surah_1_fatiha", 7),
        ]
        output = tmp_path / "results" / "benchmark_results.json"
        save_results_json(metrics, output)
        data = json.loads(output.read_text())
        result = data["results"][0]
        expected_fields = {
            "strategy", "surah_name", "num_ayahs",
            "mae_start", "mae_end", "mae_combined",
            "avg_similarity", "pct_high_confidence",
            "runtime_seconds", "aligned_count",
        }
        assert expected_fields.issubset(result.keys())


# ---------------------------------------------------------------------------
# Leaderboard generation
# ---------------------------------------------------------------------------


class TestLeaderboardGeneration:
    """Test Markdown leaderboard output."""

    def _sample_metrics(self) -> list[StrategyMetrics]:
        return [
            StrategyMetrics("greedy", "surah_1_fatiha", 7, mae_combined=0.5, avg_similarity=0.7, pct_high_confidence=60.0, runtime_seconds=0.01),
            StrategyMetrics("dp", "surah_1_fatiha", 7, mae_combined=0.2, avg_similarity=0.9, pct_high_confidence=90.0, runtime_seconds=0.05),
            StrategyMetrics("greedy", "surah_112_ikhlas", 4, mae_combined=0.4, avg_similarity=0.75, pct_high_confidence=70.0, runtime_seconds=0.01),
            StrategyMetrics("dp", "surah_112_ikhlas", 4, mae_combined=0.15, avg_similarity=0.92, pct_high_confidence=95.0, runtime_seconds=0.04),
        ]

    def test_leaderboard_file_is_created(self, tmp_path):
        metrics = self._sample_metrics()
        output = tmp_path / "LEADERBOARD.md"
        generate_leaderboard(metrics, output)
        assert output.exists()

    def test_leaderboard_contains_all_strategies(self, tmp_path):
        metrics = self._sample_metrics()
        output = tmp_path / "LEADERBOARD.md"
        generate_leaderboard(metrics, output)
        content = output.read_text()
        assert "`greedy`" in content
        assert "`dp`" in content

    def test_leaderboard_contains_both_surahs(self, tmp_path):
        metrics = self._sample_metrics()
        output = tmp_path / "LEADERBOARD.md"
        generate_leaderboard(metrics, output)
        content = output.read_text()
        assert "Surah 1 Fatiha" in content or "surah_1_fatiha" in content.lower()
        assert "Surah 112 Ikhlas" in content or "surah_112_ikhlas" in content.lower()

    def test_leaderboard_has_metric_definitions(self, tmp_path):
        metrics = self._sample_metrics()
        output = tmp_path / "LEADERBOARD.md"
        generate_leaderboard(metrics, output)
        content = output.read_text()
        assert "MAE" in content
        assert "Avg Similarity" in content
        assert "High-Confidence" in content

    def test_leaderboard_has_how_to_run_section(self, tmp_path):
        metrics = self._sample_metrics()
        output = tmp_path / "LEADERBOARD.md"
        generate_leaderboard(metrics, output)
        content = output.read_text()
        assert "How to Run" in content
        assert "python -m benchmarks.benchmark" in content


# ---------------------------------------------------------------------------
# Full benchmark run (no GPU required)
# ---------------------------------------------------------------------------


class TestFullBenchmarkRun:
    """Integration test: run the full benchmark end-to-end."""

    def test_benchmark_returns_correct_number_of_results(self):
        """Should return len(STRATEGIES) * len(GROUND_TRUTH) results."""
        metrics = run_benchmark()
        expected = len(STRATEGIES) * len(GROUND_TRUTH)
        assert len(metrics) == expected

    def test_benchmark_all_strategies_present(self):
        metrics = run_benchmark()
        strategies = {m.strategy for m in metrics}
        assert strategies == {s.value for s in STRATEGIES}

    def test_benchmark_all_surahs_present(self):
        metrics = run_benchmark()
        surahs = {m.surah_name for m in metrics}
        assert surahs == set(GROUND_TRUTH.keys())

    def test_benchmark_metrics_are_valid(self):
        metrics = run_benchmark()
        for m in metrics:
            assert 0.0 <= m.avg_similarity <= 1.0
            assert 0.0 <= m.pct_high_confidence <= 100.0
            assert m.mae_start >= 0.0
            assert m.mae_end >= 0.0
            assert m.runtime_seconds >= 0.0
            assert m.aligned_count > 0

    def test_benchmark_runs_without_gpu(self):
        """Benchmark must complete without requiring a GPU."""
        start = time.perf_counter()
        metrics = run_benchmark()
        elapsed = time.perf_counter() - start
        # Should complete in under 30 seconds on any CPU
        assert elapsed < 30.0
        assert len(metrics) > 0

    def test_full_pipeline_json_and_leaderboard(self, tmp_path):
        """End-to-end: run benchmark, save JSON, generate leaderboard."""
        metrics = run_benchmark()
        json_path = tmp_path / "results" / "benchmark_results.json"
        md_path = tmp_path / "LEADERBOARD.md"
        save_results_json(metrics, json_path)
        generate_leaderboard(metrics, md_path)
        assert json_path.exists()
        assert md_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data["results"]) == len(metrics)
        content = md_path.read_text()
        assert "Leaderboard" in content
