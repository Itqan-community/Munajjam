"""
Unit tests for benchmark runner (Issue #45).

Tests cover:
- BenchmarkRunner.run_surah produces StrategyMetrics per strategy
- BenchmarkRunner._ground_truth_to_segments conversion
- BenchmarkRunner.run_all with fixture files
- BenchmarkRunner.save_json round-trip
"""

import json

import pytest

from munajjam.benchmark.models import (
    BenchmarkReport,
    GroundTruthAyah,
    GroundTruthSurah,
    StrategyMetrics,
)
from munajjam.benchmark.runner import BenchmarkRunner
from munajjam.core.aligner import AlignmentStrategy
from munajjam.models import Segment, SegmentType


def _make_ground_truth_surah_1() -> GroundTruthSurah:
    """Create a minimal 3-ayah ground truth for Surah 1."""
    return GroundTruthSurah(
        surah_id=1,
        reciter="test",
        audio_filename="001.wav",
        ayahs=[
            GroundTruthAyah(
                ayah_number=1, start_time=0.0, end_time=5.0,
                text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            ),
            GroundTruthAyah(
                ayah_number=2, start_time=5.5, end_time=10.0,
                text="ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَٰلَمِينَ",
            ),
            GroundTruthAyah(
                ayah_number=3, start_time=10.5, end_time=14.0,
                text="ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ",
            ),
        ],
    )


class TestGroundTruthToSegments:
    """Test conversion of ground-truth to Segment objects."""

    def test_converts_to_segments(self, tmp_path) -> None:
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
        )
        gt = _make_ground_truth_surah_1()
        segments = runner._ground_truth_to_segments(gt)

        assert len(segments) == 3
        assert all(isinstance(s, Segment) for s in segments)
        assert segments[0].surah_id == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0
        assert segments[0].type == SegmentType.AYAH
        assert segments[0].text == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

    def test_segments_sorted_by_ayah_number(self, tmp_path) -> None:
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
        )
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="001.wav",
            ayahs=[
                GroundTruthAyah(ayah_number=3, start_time=10.0, end_time=14.0, text="c"),
                GroundTruthAyah(ayah_number=1, start_time=0.0, end_time=5.0, text="a"),
                GroundTruthAyah(ayah_number=2, start_time=5.0, end_time=10.0, text="b"),
            ],
        )
        segments = runner._ground_truth_to_segments(gt)
        assert segments[0].text == "a"
        assert segments[1].text == "b"
        assert segments[2].text == "c"


class TestRunSurah:
    """Test BenchmarkRunner.run_surah."""

    def test_returns_metrics_per_strategy(self, tmp_path) -> None:
        strategies = [AlignmentStrategy.GREEDY, AlignmentStrategy.DP, AlignmentStrategy.HYBRID]
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
            strategies=strategies,
        )
        gt = _make_ground_truth_surah_1()
        metrics_list = runner.run_surah(gt)

        assert len(metrics_list) == 3
        strategy_names = {m.strategy for m in metrics_list}
        assert strategy_names == {"greedy", "dp", "hybrid"}

    def test_metrics_have_valid_ranges(self, tmp_path) -> None:
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
            strategies=[AlignmentStrategy.GREEDY],
        )
        gt = _make_ground_truth_surah_1()
        metrics_list = runner.run_surah(gt)

        assert len(metrics_list) == 1
        m = metrics_list[0]
        assert m.mae_start >= 0.0
        assert m.mae_end >= 0.0
        assert 0.0 <= m.avg_similarity <= 1.0
        assert 0.0 <= m.pct_high_confidence <= 100.0
        assert m.runtime_seconds >= 0.0
        assert m.ayah_count >= 0

    def test_single_strategy(self, tmp_path) -> None:
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
            strategies=[AlignmentStrategy.DP],
        )
        gt = _make_ground_truth_surah_1()
        metrics_list = runner.run_surah(gt)

        assert len(metrics_list) == 1
        assert metrics_list[0].strategy == "dp"


class TestRunAll:
    """Test BenchmarkRunner.run_all with fixture files."""

    def test_loads_and_runs_all_ground_truth_files(self, tmp_path) -> None:
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write two ground truth files
        for surah_id, ayah_count in [(1, 3), (112, 2)]:
            data = {
                "surah_id": surah_id,
                "reciter": "test",
                "audio_filename": f"{surah_id:03d}.wav",
                "ayahs": [
                    {
                        "ayah_number": i + 1,
                        "start_time": float(i * 5),
                        "end_time": float(i * 5 + 4),
                        "text": f"ayah {i + 1}",
                    }
                    for i in range(ayah_count)
                ],
            }
            (gt_dir / f"surah_{surah_id:03d}.json").write_text(
                json.dumps(data), encoding="utf-8",
            )

        runner = BenchmarkRunner(
            ground_truth_dir=gt_dir,
            results_dir=results_dir,
            strategies=[AlignmentStrategy.GREEDY, AlignmentStrategy.DP],
        )
        report = runner.run_all()

        assert isinstance(report, BenchmarkReport)
        # 2 surahs * 2 strategies = 4 results
        assert len(report.results) == 4
        assert report.munajjam_version

    def test_empty_ground_truth_dir(self, tmp_path) -> None:
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        runner = BenchmarkRunner(
            ground_truth_dir=gt_dir,
            results_dir=results_dir,
        )
        report = runner.run_all()
        assert report.results == []


class TestSaveJson:
    """Test BenchmarkRunner.save_json."""

    def test_writes_valid_json(self, tmp_path) -> None:
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0a1",
            results=[
                StrategyMetrics(
                    strategy="greedy",
                    surah_id=1,
                    mae_start=0.1,
                    mae_end=0.2,
                    avg_similarity=0.9,
                    pct_high_confidence=80.0,
                    runtime_seconds=0.01,
                    ayah_count=7,
                    timestamp="2026-02-26T12:00:00+00:00",
                ),
            ],
        )
        output_path = tmp_path / "results" / "benchmark_results.json"
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            results_dir=tmp_path,
        )
        runner.save_json(report, output_path)

        assert output_path.exists()
        restored = BenchmarkReport.model_validate_json(
            output_path.read_text(encoding="utf-8"),
        )
        assert restored.results[0].strategy == "greedy"
