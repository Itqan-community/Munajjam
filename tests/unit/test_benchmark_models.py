"""
Unit tests for benchmark data models (Issue #45).

Tests cover:
- GroundTruthAyah creation and validation
- GroundTruthSurah creation with nested ayahs
- StrategyMetrics validation
- BenchmarkReport JSON round-trip
"""

import pytest

from munajjam.benchmark.models import (
    BenchmarkReport,
    GroundTruthAyah,
    GroundTruthSurah,
    StrategyMetrics,
)


class TestGroundTruthAyah:
    """Test GroundTruthAyah model."""

    def test_valid_creation(self) -> None:
        ayah = GroundTruthAyah(
            ayah_number=1,
            start_time=5.72,
            end_time=9.74,
            text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        )
        assert ayah.ayah_number == 1
        assert ayah.start_time == 5.72
        assert ayah.end_time == 9.74

    def test_rejects_negative_start_time(self) -> None:
        with pytest.raises(ValueError):
            GroundTruthAyah(
                ayah_number=1,
                start_time=-1.0,
                end_time=5.0,
                text="بسم الله",
            )

    def test_rejects_zero_ayah_number(self) -> None:
        with pytest.raises(ValueError):
            GroundTruthAyah(
                ayah_number=0,
                start_time=0.0,
                end_time=5.0,
                text="بسم الله",
            )

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError):
            GroundTruthAyah(
                ayah_number=1,
                start_time=0.0,
                end_time=5.0,
                text="",
            )

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError, match="end_time.*must be > start_time"):
            GroundTruthAyah(
                ayah_number=1,
                start_time=5.0,
                end_time=1.0,
                text="بسم الله",
            )

    def test_rejects_equal_start_end(self) -> None:
        with pytest.raises(ValueError, match="end_time.*must be > start_time"):
            GroundTruthAyah(
                ayah_number=1,
                start_time=5.0,
                end_time=5.0,
                text="بسم الله",
            )


class TestGroundTruthSurah:
    """Test GroundTruthSurah model."""

    def test_valid_creation(self) -> None:
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="sample",
            audio_filename="001.wav",
            ayahs=[
                GroundTruthAyah(
                    ayah_number=1,
                    start_time=0.0,
                    end_time=5.0,
                    text="بسم الله",
                ),
            ],
        )
        assert gt.surah_id == 1
        assert len(gt.ayahs) == 1
        assert gt.notes is None

    def test_rejects_invalid_surah_id(self) -> None:
        with pytest.raises(ValueError):
            GroundTruthSurah(
                surah_id=0,
                reciter="test",
                audio_filename="test.wav",
                ayahs=[
                    GroundTruthAyah(
                        ayah_number=1,
                        start_time=0.0,
                        end_time=5.0,
                        text="test",
                    ),
                ],
            )

    def test_rejects_surah_id_above_114(self) -> None:
        with pytest.raises(ValueError):
            GroundTruthSurah(
                surah_id=115,
                reciter="test",
                audio_filename="test.wav",
                ayahs=[
                    GroundTruthAyah(
                        ayah_number=1,
                        start_time=0.0,
                        end_time=5.0,
                        text="test",
                    ),
                ],
            )

    def test_optional_notes(self) -> None:
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="sample",
            audio_filename="001.wav",
            ayahs=[
                GroundTruthAyah(
                    ayah_number=1,
                    start_time=0.0,
                    end_time=5.0,
                    text="بسم الله",
                ),
            ],
            notes="Test notes",
        )
        assert gt.notes == "Test notes"


class TestStrategyMetrics:
    """Test StrategyMetrics model."""

    def test_valid_creation(self) -> None:
        m = StrategyMetrics(
            strategy="greedy",
            surah_id=1,
            mae_start=0.15,
            mae_end=0.20,
            avg_similarity=0.95,
            pct_high_confidence=85.7,
            runtime_seconds=0.042,
            ayah_count=7,
            timestamp="2026-02-26T12:00:00+00:00",
        )
        assert m.strategy == "greedy"
        assert m.mae_start == 0.15
        assert m.pct_high_confidence == 85.7

    def test_rejects_negative_mae(self) -> None:
        with pytest.raises(ValueError):
            StrategyMetrics(
                strategy="greedy",
                surah_id=1,
                mae_start=-0.1,
                mae_end=0.0,
                avg_similarity=0.9,
                pct_high_confidence=80.0,
                runtime_seconds=0.01,
                ayah_count=7,
                timestamp="2026-02-26T12:00:00+00:00",
            )

    def test_rejects_similarity_above_one(self) -> None:
        with pytest.raises(ValueError):
            StrategyMetrics(
                strategy="dp",
                surah_id=1,
                mae_start=0.0,
                mae_end=0.0,
                avg_similarity=1.1,
                pct_high_confidence=100.0,
                runtime_seconds=0.01,
                ayah_count=7,
                timestamp="2026-02-26T12:00:00+00:00",
            )

    def test_rejects_pct_above_100(self) -> None:
        with pytest.raises(ValueError):
            StrategyMetrics(
                strategy="hybrid",
                surah_id=1,
                mae_start=0.0,
                mae_end=0.0,
                avg_similarity=0.9,
                pct_high_confidence=100.1,
                runtime_seconds=0.01,
                ayah_count=7,
                timestamp="2026-02-26T12:00:00+00:00",
            )


class TestBenchmarkReport:
    """Test BenchmarkReport model."""

    def test_valid_creation(self) -> None:
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0a1",
            results=[],
        )
        assert report.munajjam_version == "2.0.0a1"
        assert report.results == []

    def test_json_round_trip(self) -> None:
        metrics = StrategyMetrics(
            strategy="greedy",
            surah_id=1,
            mae_start=0.15,
            mae_end=0.20,
            avg_similarity=0.95,
            pct_high_confidence=85.7,
            runtime_seconds=0.042,
            ayah_count=7,
            timestamp="2026-02-26T12:00:00+00:00",
        )
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0a1",
            results=[metrics],
        )
        json_str = report.model_dump_json()
        restored = BenchmarkReport.model_validate_json(json_str)
        assert restored.results[0].strategy == "greedy"
        assert restored.results[0].mae_start == 0.15

    def test_report_with_multiple_results(self) -> None:
        results = [
            StrategyMetrics(
                strategy=s,
                surah_id=1,
                mae_start=0.1,
                mae_end=0.2,
                avg_similarity=0.9,
                pct_high_confidence=80.0,
                runtime_seconds=0.01,
                ayah_count=7,
                timestamp="2026-02-26T12:00:00+00:00",
            )
            for s in ("greedy", "dp", "hybrid")
        ]
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0a1",
            results=results,
        )
        assert len(report.results) == 3
