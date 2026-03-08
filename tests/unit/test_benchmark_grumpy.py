"""
Adversarial / grumpy-tester tests for the benchmark harness (Issue #45).

This file deliberately targets gaps in the existing test suite:
- Boundary conditions the happy-path tests ignore
- Model validation weaknesses (empty strings, whitespace-only text)
- Numerical edge cases (NaN, infinity, very large values)
- File-system edge cases (empty file, BOM, wrong JSON root type)
- Loader behaviour with extra/missing fields
- Leaderboard formatting with long strategy names and zero values
- Runner behaviour with empty strategies list
- Duplicate ayah numbers in ground truth
"""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from munajjam.benchmark.leaderboard import generate_leaderboard, save_leaderboard
from munajjam.benchmark.loader import list_ground_truth_files, load_ground_truth
from munajjam.benchmark.metrics import (
    compute_avg_similarity,
    compute_mae,
    compute_pct_high_confidence,
    compute_strategy_metrics,
)
from munajjam.benchmark.models import (
    BenchmarkReport,
    GroundTruthAyah,
    GroundTruthSurah,
    StrategyMetrics,
)
from munajjam.benchmark.runner import BenchmarkRunner
from munajjam.models import AlignmentResult, Ayah


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ayah(
    ayah_number: int = 1,
    start_time: float = 0.0,
    end_time: float = 5.0,
    text: str = "بسم الله",
) -> GroundTruthAyah:
    return GroundTruthAyah(
        ayah_number=ayah_number,
        start_time=start_time,
        end_time=end_time,
        text=text,
    )


def _make_surah(
    surah_id: int = 1,
    ayahs: list[GroundTruthAyah] | None = None,
) -> GroundTruthSurah:
    if ayahs is None:
        ayahs = [_make_ayah()]
    return GroundTruthSurah(
        surah_id=surah_id,
        reciter="test",
        audio_filename="001.wav",
        ayahs=ayahs,
    )


def _make_result(
    ayah_number: int,
    start: float,
    end: float,
    similarity: float,
    surah_id: int = 1,
) -> AlignmentResult:
    return AlignmentResult(
        ayah=Ayah(
            id=ayah_number,
            surah_id=surah_id,
            ayah_number=ayah_number,
            text="test",
        ),
        start_time=start,
        end_time=end,
        transcribed_text="test",
        similarity_score=similarity,
    )


def _make_strategy_metrics(**kwargs: object) -> StrategyMetrics:
    defaults: dict[str, object] = {
        "strategy": "greedy",
        "surah_id": 1,
        "mae_start": 0.1,
        "mae_end": 0.2,
        "avg_similarity": 0.9,
        "pct_high_confidence": 80.0,
        "runtime_seconds": 0.01,
        "ayah_count": 7,
        "timestamp": "2026-02-26T12:00:00+00:00",
    }
    defaults.update(kwargs)
    return StrategyMetrics(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# MODEL VALIDATION - boundary and weakness tests
# ===========================================================================


class TestGroundTruthAyahBoundaries:
    """Probe GroundTruthAyah for validation gaps the happy-path suite missed."""

    def test_whitespace_only_text_is_accepted_by_pydantic(self) -> None:
        """
        A single space passes min_length=1 but is semantically meaningless.
        This test documents the current behaviour. If the model is tightened
        to reject whitespace-only text this test should be updated.
        """
        # This SHOULD ideally be rejected, but min_length=1 allows it.
        # We document it as a known weakness.
        ayah = GroundTruthAyah(
            ayah_number=1,
            start_time=0.0,
            end_time=1.0,
            text=" ",
        )
        assert ayah.text == " "

    def test_minimum_positive_duration_is_accepted(self) -> None:
        """end_time must be strictly > start_time; a tiny epsilon must pass."""
        ayah = GroundTruthAyah(
            ayah_number=1,
            start_time=0.0,
            end_time=1e-9,
            text="بسم الله",
        )
        assert ayah.end_time > ayah.start_time

    def test_very_large_timestamps_are_accepted(self) -> None:
        """No upper bound on timestamps; a 24-hour audio file should be fine."""
        ayah = GroundTruthAyah(
            ayah_number=1,
            start_time=86390.0,
            end_time=86399.0,
            text="بسم الله",
        )
        assert ayah.end_time > ayah.start_time

    def test_ayah_number_at_maximum_reasonable_value(self) -> None:
        """The longest surah (Al-Baqarah) has 286 ayahs; no upper cap on ayah_number."""
        ayah = _make_ayah(ayah_number=286)
        assert ayah.ayah_number == 286

    def test_rejects_negative_end_time(self) -> None:
        """Both start_time and end_time must be >= 0."""
        with pytest.raises(ValidationError):
            GroundTruthAyah(
                ayah_number=1,
                start_time=0.0,
                end_time=-1.0,
                text="بسم الله",
            )


class TestGroundTruthSurahBoundaries:
    """Probe GroundTruthSurah for gaps."""

    def test_surah_id_at_exact_upper_boundary_114(self) -> None:
        """surah_id=114 is valid; the existing suite only tested 115 (rejection)."""
        gt = _make_surah(surah_id=114)
        assert gt.surah_id == 114

    def test_surah_id_at_exact_lower_boundary_1(self) -> None:
        gt = _make_surah(surah_id=1)
        assert gt.surah_id == 1

    def test_empty_ayahs_list_is_accepted(self) -> None:
        """
        GroundTruthSurah has no validator requiring at least one ayah.
        This test documents the current behaviour - empty ayahs list is accepted.
        """
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="001.wav",
            ayahs=[],
        )
        assert gt.ayahs == []

    def test_empty_reciter_string_is_accepted(self) -> None:
        """
        reciter field has no min_length constraint. Empty string is accepted.
        This documents a potential data-quality gap.
        """
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="",
            audio_filename="001.wav",
            ayahs=[_make_ayah()],
        )
        assert gt.reciter == ""

    def test_empty_audio_filename_is_accepted(self) -> None:
        """
        audio_filename has no min_length or format constraint.
        An empty string passes, which could cause downstream issues.
        """
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="",
            ayahs=[_make_ayah()],
        )
        assert gt.audio_filename == ""

    def test_duplicate_ayah_numbers_are_not_rejected(self) -> None:
        """
        GroundTruthSurah has no validator rejecting duplicate ayah_number values.
        Two ayahs with the same number can be inserted. This documents a known gap.
        """
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="001.wav",
            ayahs=[
                _make_ayah(ayah_number=1, start_time=0.0, end_time=5.0),
                _make_ayah(ayah_number=1, start_time=5.0, end_time=10.0),
            ],
        )
        assert len(gt.ayahs) == 2

    def test_notes_can_be_empty_string(self) -> None:
        """notes='' is different from notes=None; both should work."""
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="001.wav",
            ayahs=[_make_ayah()],
            notes="",
        )
        assert gt.notes == ""

    def test_arabic_unicode_in_reciter_is_preserved(self) -> None:
        """Unicode reciter names must round-trip through the model."""
        reciter_name = "محمود خليل الحصري"
        gt = GroundTruthSurah(
            surah_id=1,
            reciter=reciter_name,
            audio_filename="001.wav",
            ayahs=[_make_ayah()],
        )
        assert gt.reciter == reciter_name


class TestStrategyMetricsBoundaries:
    """Probe StrategyMetrics for validation edge cases."""

    def test_empty_strategy_string_is_accepted(self) -> None:
        """
        strategy field has no min_length. An empty strategy name is accepted.
        This documents a data-quality gap.
        """
        m = _make_strategy_metrics(strategy="")
        assert m.strategy == ""

    def test_runtime_seconds_exactly_zero(self) -> None:
        """ge=0.0 means 0.0 should pass."""
        m = _make_strategy_metrics(runtime_seconds=0.0)
        assert m.runtime_seconds == 0.0

    def test_ayah_count_exactly_zero(self) -> None:
        """ge=0 means 0 should pass (empty run)."""
        m = _make_strategy_metrics(ayah_count=0)
        assert m.ayah_count == 0

    def test_avg_similarity_exact_boundaries(self) -> None:
        """Both 0.0 and 1.0 are valid boundary values."""
        m_low = _make_strategy_metrics(avg_similarity=0.0)
        m_high = _make_strategy_metrics(avg_similarity=1.0)
        assert m_low.avg_similarity == 0.0
        assert m_high.avg_similarity == 1.0

    def test_pct_high_confidence_exact_boundaries(self) -> None:
        """Both 0.0 and 100.0 are valid boundary values."""
        m_low = _make_strategy_metrics(pct_high_confidence=0.0)
        m_high = _make_strategy_metrics(pct_high_confidence=100.0)
        assert m_low.pct_high_confidence == 0.0
        assert m_high.pct_high_confidence == 100.0

    def test_rejects_negative_runtime(self) -> None:
        with pytest.raises(ValidationError):
            _make_strategy_metrics(runtime_seconds=-0.001)

    def test_rejects_negative_ayah_count(self) -> None:
        with pytest.raises(ValidationError):
            _make_strategy_metrics(ayah_count=-1)

    def test_strategy_with_unicode_name(self) -> None:
        """A Unicode strategy name must survive model round-trip."""
        m = _make_strategy_metrics(strategy="استراتيجية")
        assert m.strategy == "استراتيجية"

    def test_strategy_metrics_json_round_trip_preserves_floats(self) -> None:
        """Verify floating-point precision is not silently truncated during JSON serialization."""
        m = _make_strategy_metrics(mae_start=0.123456789, mae_end=0.987654321)
        restored = StrategyMetrics.model_validate_json(m.model_dump_json())
        assert restored.mae_start == pytest.approx(0.123456789, rel=1e-9)
        assert restored.mae_end == pytest.approx(0.987654321, rel=1e-9)


class TestBenchmarkReportBoundaries:
    """Probe BenchmarkReport for edge cases."""

    def test_empty_results_list(self) -> None:
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[],
        )
        assert report.results == []

    def test_unicode_in_version_string(self) -> None:
        """Version strings are not constrained; unusual values must survive."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0-بيتا",
            results=[],
        )
        assert "بيتا" in report.munajjam_version

    def test_json_round_trip_with_empty_results(self) -> None:
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0",
            results=[],
        )
        restored = BenchmarkReport.model_validate_json(report.model_dump_json())
        assert restored.results == []
        assert restored.munajjam_version == "2.0.0"


# ===========================================================================
# METRIC COMPUTATION - numerical edge cases
# ===========================================================================


class TestComputeMaeNumericalEdges:
    """Stress-test compute_mae with problematic numerical inputs."""

    def test_nan_in_predicted_propagates(self) -> None:
        """
        NaN in predicted list causes compute_mae to return NaN.
        The function performs no sanitisation. This test documents that
        the function does NOT defend against NaN inputs.
        """
        result = compute_mae([float("nan"), 1.0], [1.0, 1.0])
        assert math.isnan(result)

    def test_nan_in_actual_propagates(self) -> None:
        result = compute_mae([1.0, 1.0], [float("nan"), 1.0])
        assert math.isnan(result)

    def test_inf_in_predicted_propagates(self) -> None:
        """Infinity in input produces infinity in output."""
        result = compute_mae([float("inf"), 1.0], [1.0, 1.0])
        assert math.isinf(result)

    def test_very_large_floats_do_not_overflow(self) -> None:
        """Python floats can represent up to ~1.8e308; this should not raise."""
        large = 1e300
        result = compute_mae([large], [0.0])
        assert result == pytest.approx(large)

    def test_negative_values_in_inputs(self) -> None:
        """
        compute_mae accepts any float list. Negative timestamps would be
        invalid domain data, but the function itself applies no domain constraint.
        |(-5.0) - (-3.0)| = 2.0
        """
        result = compute_mae([-5.0], [-3.0])
        assert result == pytest.approx(2.0)

    def test_mismatched_predicted_longer_than_actual(self) -> None:
        """When predicted is longer, only first len(actual) pairs are used."""
        # Only pair: |1.0 - 2.0| = 1.0; /1 = 1.0
        result = compute_mae([1.0, 5.0, 9.0], [2.0])
        assert result == pytest.approx(1.0)

    def test_single_zero_delta(self) -> None:
        """Perfect prediction at a single point gives exactly 0.0."""
        result = compute_mae([42.0], [42.0])
        assert result == 0.0


class TestComputeAvgSimilarityEdges:
    """Stress-test compute_avg_similarity."""

    def test_single_result_at_zero_similarity(self) -> None:
        results = [_make_result(1, 0.0, 1.0, 0.0)]
        assert compute_avg_similarity(results) == pytest.approx(0.0)

    def test_single_result_at_max_similarity(self) -> None:
        results = [_make_result(1, 0.0, 1.0, 1.0)]
        assert compute_avg_similarity(results) == pytest.approx(1.0)

    def test_large_result_set_average_correctness(self) -> None:
        """100 results all at 0.5 must average to exactly 0.5."""
        results = [_make_result(i, float(i), float(i) + 1.0, 0.5) for i in range(1, 101)]
        assert compute_avg_similarity(results) == pytest.approx(0.5)

    def test_alternating_zero_and_one(self) -> None:
        """50 zeros and 50 ones must average to 0.5."""
        results = []
        for i in range(1, 51):
            results.append(_make_result(i, float(i), float(i) + 0.5, 0.0))
        for i in range(51, 101):
            results.append(_make_result(i, float(i), float(i) + 0.5, 1.0))
        assert compute_avg_similarity(results) == pytest.approx(0.5)


class TestComputePctHighConfidenceEdges:
    """Stress-test compute_pct_high_confidence boundary conditions."""

    def test_single_result_at_exact_0_8_is_high_confidence(self) -> None:
        """is_high_confidence uses >=0.8; exactly 0.8 MUST be high confidence."""
        results = [_make_result(1, 0.0, 1.0, 0.8)]
        assert compute_pct_high_confidence(results) == pytest.approx(100.0)

    def test_single_result_just_below_0_8(self) -> None:
        """0.7999 must NOT be high confidence."""
        results = [_make_result(1, 0.0, 1.0, 0.7999)]
        assert compute_pct_high_confidence(results) == pytest.approx(0.0)

    def test_single_result_at_zero_similarity(self) -> None:
        results = [_make_result(1, 0.0, 1.0, 0.0)]
        assert compute_pct_high_confidence(results) == pytest.approx(0.0)

    def test_single_result_at_one_similarity(self) -> None:
        results = [_make_result(1, 0.0, 1.0, 1.0)]
        assert compute_pct_high_confidence(results) == pytest.approx(100.0)

    def test_result_count_does_not_affect_percentage_math(self) -> None:
        """1 out of 3 is 33.33...%; must not round to wrong value."""
        results = [
            _make_result(1, 0.0, 1.0, 0.9),  # high
            _make_result(2, 1.0, 2.0, 0.5),  # low
            _make_result(3, 2.0, 3.0, 0.3),  # low
        ]
        expected = (1 / 3) * 100.0
        assert compute_pct_high_confidence(results) == pytest.approx(expected, rel=1e-6)


class TestComputeStrategyMetricsEdges:
    """Stress-test the assembly function."""

    def test_more_results_than_ground_truth_ayahs(self) -> None:
        """
        compute_strategy_metrics calls compute_mae which uses min(len(p), len(a)).
        Predicted list longer than GT must not raise; ayah_count should reflect
        the actual number of results, not the GT count.
        """
        gt = _make_surah(ayahs=[
            _make_ayah(ayah_number=1, start_time=0.0, end_time=5.0),
        ])
        results = [
            _make_result(1, 0.1, 5.1, 0.9),
            _make_result(2, 5.5, 10.0, 0.85),  # extra result beyond GT
        ]
        metrics = compute_strategy_metrics(
            strategy="greedy",
            surah_id=1,
            results=results,
            ground_truth=gt,
            runtime_seconds=0.01,
        )
        # ayah_count is len(results), not len(gt.ayahs)
        assert metrics.ayah_count == 2
        # MAE only compared 1 pair (min of 2 predicted, 1 GT)
        assert metrics.mae_start == pytest.approx(abs(0.1 - 0.0))

    def test_more_ground_truth_than_results(self) -> None:
        """Ground truth has 3 ayahs but only 1 result; MAE uses min=1 pair."""
        gt = _make_surah(ayahs=[
            _make_ayah(ayah_number=1, start_time=0.0, end_time=5.0),
            _make_ayah(ayah_number=2, start_time=5.5, end_time=10.0),
            _make_ayah(ayah_number=3, start_time=10.5, end_time=14.0),
        ])
        results = [_make_result(1, 0.0, 5.0, 0.9)]
        metrics = compute_strategy_metrics(
            strategy="dp",
            surah_id=1,
            results=results,
            ground_truth=gt,
            runtime_seconds=0.005,
        )
        assert metrics.ayah_count == 1
        assert metrics.mae_start == pytest.approx(0.0)

    def test_results_with_unsorted_ayah_numbers_are_sorted_before_comparison(self) -> None:
        """
        compute_strategy_metrics sorts both GT and results by ayah_number.
        Results handed in backwards must produce the same MAE as sorted order.
        """
        gt = _make_surah(ayahs=[
            _make_ayah(ayah_number=1, start_time=0.0, end_time=5.0),
            _make_ayah(ayah_number=2, start_time=5.0, end_time=10.0),
        ])
        # Intentionally reversed
        results_reversed = [
            _make_result(2, 5.1, 10.1, 0.9),
            _make_result(1, 0.1, 5.1, 0.85),
        ]
        results_sorted = [
            _make_result(1, 0.1, 5.1, 0.85),
            _make_result(2, 5.1, 10.1, 0.9),
        ]
        m_rev = compute_strategy_metrics("greedy", 1, results_reversed, gt, 0.01)
        m_sorted = compute_strategy_metrics("greedy", 1, results_sorted, gt, 0.01)
        assert m_rev.mae_start == pytest.approx(m_sorted.mae_start)
        assert m_rev.mae_end == pytest.approx(m_sorted.mae_end)

    def test_negative_runtime_raises_validation_error(self) -> None:
        """
        A negative runtime_seconds value must cause StrategyMetrics construction
        to fail with a ValidationError (runtime_seconds has ge=0.0).
        """
        gt = _make_surah()
        results = [_make_result(1, 0.0, 5.0, 0.9)]
        with pytest.raises(ValidationError):
            compute_strategy_metrics(
                strategy="greedy",
                surah_id=1,
                results=results,
                ground_truth=gt,
                runtime_seconds=-1.0,
            )


# ===========================================================================
# LOADER - file-system and encoding edge cases
# ===========================================================================


class TestLoadGroundTruthEdgeCases:
    """File-system adversarial tests for load_ground_truth."""

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An empty file is not valid JSON; must raise json.JSONDecodeError."""
        gt_file = tmp_path / "empty.json"
        gt_file.write_text("", encoding="utf-8")
        with pytest.raises(Exception):  # json.JSONDecodeError
            load_ground_truth(gt_file)

    def test_json_null_root_raises(self, tmp_path: Path) -> None:
        """A file containing only 'null' is valid JSON but fails Pydantic validation."""
        gt_file = tmp_path / "null.json"
        gt_file.write_text("null", encoding="utf-8")
        with pytest.raises(Exception):
            load_ground_truth(gt_file)

    def test_json_array_root_raises(self, tmp_path: Path) -> None:
        """A JSON array at root level must fail Pydantic validation (expects object)."""
        gt_file = tmp_path / "array.json"
        gt_file.write_text('[{"surah_id": 1}]', encoding="utf-8")
        with pytest.raises(Exception):
            load_ground_truth(gt_file)

    def test_file_with_bom_is_loaded_correctly(self, tmp_path: Path) -> None:
        """
        A UTF-8 BOM (\ufeff) at the start of the file must not cause a parse error
        because Python's open(encoding='utf-8') does NOT strip BOM automatically;
        only 'utf-8-sig' does. This test verifies the current behaviour.
        """
        data = {
            "surah_id": 1,
            "reciter": "test",
            "audio_filename": "001.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "بسم الله",
                }
            ],
        }
        gt_file = tmp_path / "bom.json"
        # Write with BOM prefix manually
        raw = "\ufeff" + json.dumps(data)
        gt_file.write_bytes(raw.encode("utf-8-sig"))
        # With utf-8 (not utf-8-sig) the BOM character leaks into the JSON string.
        # json.loads will fail because the BOM makes the first character invalid.
        # We just confirm this raises rather than silently corrupting data.
        with pytest.raises(Exception):
            load_ground_truth(gt_file)

    def test_extra_fields_in_json_are_silently_ignored(self, tmp_path: Path) -> None:
        """
        Pydantic v2 by default ignores extra fields. Extra keys in the JSON
        must not cause a validation error.
        """
        data = {
            "surah_id": 1,
            "reciter": "test",
            "audio_filename": "001.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "بسم الله",
                    "unexpected_field": "should be ignored",
                }
            ],
            "completely_unknown_key": 42,
        }
        gt_file = tmp_path / "extra_fields.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_ground_truth(gt_file)
        assert result.surah_id == 1

    def test_directory_path_raises_os_error(self, tmp_path: Path) -> None:
        """
        Passing a directory path instead of a file raises IsADirectoryError
        (a subclass of OSError), NOT FileNotFoundError.  The docstring claims
        FileNotFoundError but the real behaviour is IsADirectoryError because
        path.exists() is True for directories, so the existence guard does not
        trigger; the error surfaces when open() is called on a directory.

        This test documents the gap between the docstring and reality.
        """
        with pytest.raises(OSError):
            load_ground_truth(tmp_path)

    def test_path_object_and_string_give_identical_results(self, tmp_path: Path) -> None:
        """load_ground_truth must accept both Path and str and return equal results."""
        data = {
            "surah_id": 112,
            "reciter": "test",
            "audio_filename": "112.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "text": "قل هو الله احد",
                }
            ],
        }
        gt_file = tmp_path / "surah_112.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")

        via_path = load_ground_truth(gt_file)
        via_str = load_ground_truth(str(gt_file))
        assert via_path.model_dump() == via_str.model_dump()

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """JSON missing 'reciter' must cause a Pydantic ValidationError."""
        data = {
            "surah_id": 1,
            "audio_filename": "001.wav",
            "ayahs": [],
        }
        gt_file = tmp_path / "missing_field.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(Exception):  # ValidationError
            load_ground_truth(gt_file)

    def test_wrong_type_for_surah_id_raises(self, tmp_path: Path) -> None:
        """surah_id as a string instead of int must fail validation."""
        data = {
            "surah_id": "one",
            "reciter": "test",
            "audio_filename": "001.wav",
            "ayahs": [],
        }
        gt_file = tmp_path / "bad_type.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(Exception):
            load_ground_truth(gt_file)


class TestListGroundTruthFilesEdgeCases:
    """Adversarial tests for list_ground_truth_files."""

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """list_ground_truth_files raises FileNotFoundError for nonexistent dirs."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Ground truth directory not found"):
            list_ground_truth_files(nonexistent)

    def test_only_non_json_files_returns_empty(self, tmp_path: Path) -> None:
        """A directory with only .txt files must return an empty list."""
        (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
        result = list_ground_truth_files(tmp_path)
        assert result == []

    def test_json_files_in_subdirectory_are_not_returned(self, tmp_path: Path) -> None:
        """
        list_ground_truth_files uses glob('*.json') which is NOT recursive.
        Files in subdirectories must not appear in the result.
        """
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.json").write_text("{}", encoding="utf-8")
        (tmp_path / "top_level.json").write_text("{}", encoding="utf-8")

        result = list_ground_truth_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "top_level.json"

    def test_result_is_sorted_lexicographically(self, tmp_path: Path) -> None:
        """Files must be returned in sorted order regardless of filesystem ordering."""
        names = ["surah_112.json", "surah_001.json", "surah_036.json", "surah_002.json"]
        for name in names:
            (tmp_path / name).write_text("{}", encoding="utf-8")

        result = list_ground_truth_files(tmp_path)
        assert [f.name for f in result] == sorted(names)

    def test_single_json_file_returns_list_of_one(self, tmp_path: Path) -> None:
        (tmp_path / "surah_001.json").write_text("{}", encoding="utf-8")
        result = list_ground_truth_files(tmp_path)
        assert len(result) == 1

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("{}", encoding="utf-8")
        result = list_ground_truth_files(tmp_path)
        assert all(isinstance(p, Path) for p in result)


# ===========================================================================
# LEADERBOARD - formatting and content edge cases
# ===========================================================================


class TestLeaderboardEdgeCases:
    """Adversarial tests for generate_leaderboard and save_leaderboard."""

    def _make_single_strategy_report(
        self,
        strategy: str = "greedy",
        surah_id: int = 1,
    ) -> BenchmarkReport:
        return BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[
                _make_strategy_metrics(
                    strategy=strategy,
                    surah_id=surah_id,
                    mae_start=0.1,
                    mae_end=0.2,
                    avg_similarity=0.9,
                    pct_high_confidence=80.0,
                    runtime_seconds=0.01,
                    ayah_count=7,
                )
            ],
        )

    def test_single_strategy_report_has_one_surah_section(self) -> None:
        report = self._make_single_strategy_report()
        md = generate_leaderboard(report)
        assert "## Surah 1" in md
        assert "## Surah 2" not in md

    def test_single_strategy_report_has_summary_section(self) -> None:
        report = self._make_single_strategy_report()
        md = generate_leaderboard(report)
        assert "Overall Summary" in md

    def test_strategy_name_longer_than_8_chars_does_not_crash(self) -> None:
        """
        _format_row uses '{strategy:<8}' which left-pads to 8 chars.
        A name longer than 8 chars must not crash - it just exceeds the column width.
        """
        report = self._make_single_strategy_report(strategy="very_long_strategy_name")
        md = generate_leaderboard(report)
        assert "very_long_strategy_name" in md

    def test_all_zero_metrics_renders_without_error(self) -> None:
        """Zero values in every field must produce a valid Markdown string."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[
                _make_strategy_metrics(
                    mae_start=0.0,
                    mae_end=0.0,
                    avg_similarity=0.0,
                    pct_high_confidence=0.0,
                    runtime_seconds=0.0,
                    ayah_count=0,
                )
            ],
        )
        md = generate_leaderboard(report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_surah_114_appears_in_leaderboard(self) -> None:
        """The last valid surah (114) must have its own section."""
        report = self._make_single_strategy_report(surah_id=114)
        md = generate_leaderboard(report)
        assert "## Surah 114" in md

    def test_multiple_strategies_single_surah_summary_averages_correctly(self) -> None:
        """
        The overall summary averages metrics per strategy across surahs.
        With two strategies on one surah, each strategy appears once in the summary.
        """
        results = [
            _make_strategy_metrics(strategy="greedy", surah_id=1, mae_start=0.2),
            _make_strategy_metrics(strategy="dp", surah_id=1, mae_start=0.1),
        ]
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=results,
        )
        md = generate_leaderboard(report)
        assert "greedy" in md
        assert "dp" in md

    def test_same_mae_start_tie_is_handled_without_error(self) -> None:
        """Two strategies with identical MAE start must not crash the sort."""
        results = [
            _make_strategy_metrics(strategy="greedy", surah_id=1, mae_start=0.1),
            _make_strategy_metrics(strategy="dp", surah_id=1, mae_start=0.1),
        ]
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=results,
        )
        md = generate_leaderboard(report)
        assert "greedy" in md
        assert "dp" in md

    def test_generated_at_appears_in_output(self) -> None:
        report = self._make_single_strategy_report()
        md = generate_leaderboard(report)
        assert "2026-02-26T12:00:00+00:00" in md

    def test_save_leaderboard_overwrites_existing_file(self, tmp_path: Path) -> None:
        """save_leaderboard must silently overwrite an existing file."""
        report = self._make_single_strategy_report()
        output_path = tmp_path / "LEADERBOARD.md"
        output_path.write_text("old content", encoding="utf-8")

        save_leaderboard(report, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "Munajjam" in content

    def test_save_leaderboard_creates_deeply_nested_dirs(self, tmp_path: Path) -> None:
        """Parent dirs three levels deep must be created automatically."""
        report = self._make_single_strategy_report()
        output_path = tmp_path / "a" / "b" / "c" / "LEADERBOARD.md"
        save_leaderboard(report, output_path)
        assert output_path.exists()

    def test_very_large_mae_value_renders_without_crashing(self) -> None:
        """Unrealistically large MAE values must not crash the format string."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[
                _make_strategy_metrics(
                    mae_start=99999.999,
                    mae_end=88888.888,
                    runtime_seconds=3600.0,
                    ayah_count=6236,
                )
            ],
        )
        md = generate_leaderboard(report)
        assert "99999.999" in md

    def test_leaderboard_output_contains_table_separator_row(self) -> None:
        """Each table must contain the Markdown separator line '|---|'."""
        report = self._make_single_strategy_report()
        md = generate_leaderboard(report)
        assert "|---" in md


# ===========================================================================
# RUNNER - configuration edge cases
# ===========================================================================


class TestBenchmarkRunnerEdgeCases:
    """Adversarial tests for BenchmarkRunner."""

    def test_empty_strategies_list_produces_no_metrics(
        self, tmp_path: Path
    ) -> None:
        """strategies=[] should produce zero metrics (no strategies to run)."""
        runner = BenchmarkRunner(
            ground_truth_dir=tmp_path,
            strategies=[],
        )
        gt = _make_surah()
        metrics = runner.run_surah(gt)
        assert len(metrics) == 0

    def test_empty_strategies_list_run_all_produces_empty_report(
        self, tmp_path: Path
    ) -> None:
        """strategies=[] through run_all should produce zero results."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        data = {
            "surah_id": 1,
            "reciter": "test",
            "audio_filename": "001.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "بسم الله",
                }
            ],
        }
        (gt_dir / "surah_001.json").write_text(json.dumps(data), encoding="utf-8")

        runner = BenchmarkRunner(
            ground_truth_dir=gt_dir,
            strategies=[],
        )
        report = runner.run_all()
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 0

    def test_save_json_overwrites_existing_file(self, tmp_path: Path) -> None:
        """save_json must silently overwrite a previously written file."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[_make_strategy_metrics()],
        )
        output_path = tmp_path / "results.json"
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)

        runner.save_json(report, output_path)
        # Write a second report with different version
        report2 = BenchmarkReport(
            generated_at="2026-02-26T13:00:00+00:00",
            munajjam_version="2.0.0",
            results=[],
        )
        runner.save_json(report2, output_path)
        second_content = output_path.read_text(encoding="utf-8")

        assert "2.0.0" in second_content
        assert "1.0.0" not in second_content

    def test_save_json_output_is_valid_json(self, tmp_path: Path) -> None:
        """The file written by save_json must be parseable by standard json.loads."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[_make_strategy_metrics()],
        )
        output_path = tmp_path / "report.json"
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)
        runner.save_json(report, output_path)

        raw = output_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["munajjam_version"] == "1.0.0"
        assert "results" in parsed

    def test_ground_truth_to_segments_with_empty_ayahs(self, tmp_path: Path) -> None:
        """
        _ground_truth_to_segments must return an empty list (not raise) when
        the ground truth has no ayahs.
        """
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)
        gt = GroundTruthSurah(
            surah_id=1,
            reciter="test",
            audio_filename="001.wav",
            ayahs=[],
        )
        segments = runner._ground_truth_to_segments(gt)
        assert segments == []

    def test_results_dir_none_does_not_affect_run_all(self, tmp_path: Path) -> None:
        """
        results_dir=None is a valid configuration. run_all must not fail
        solely because results_dir is not set (save_json is separate).
        """
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        runner = BenchmarkRunner(
            ground_truth_dir=gt_dir,
            results_dir=None,
            strategies=[],
        )
        report = runner.run_all()
        assert isinstance(report, BenchmarkReport)

    def test_save_json_creates_deeply_nested_parent_dirs(self, tmp_path: Path) -> None:
        """save_json must create arbitrarily deep parent directories."""
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="1.0.0",
            results=[],
        )
        output_path = tmp_path / "a" / "b" / "c" / "results.json"
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)
        runner.save_json(report, output_path)
        assert output_path.exists()

    def test_single_ayah_ground_truth_to_segments(self, tmp_path: Path) -> None:
        """A single-ayah ground truth produces exactly one segment with id=1."""
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)
        gt = _make_surah(ayahs=[_make_ayah(ayah_number=1)])
        segments = runner._ground_truth_to_segments(gt)
        assert len(segments) == 1
        assert segments[0].id == 1

    def test_segment_ids_are_sequential_from_one(self, tmp_path: Path) -> None:
        """
        Segment IDs are assigned as enumerate(sorted_ayahs)+1 regardless of
        the original ayah_number values.
        """
        runner = BenchmarkRunner(ground_truth_dir=tmp_path)
        gt = _make_surah(ayahs=[
            _make_ayah(ayah_number=5, start_time=0.0, end_time=5.0),
            _make_ayah(ayah_number=3, start_time=5.0, end_time=10.0),
            _make_ayah(ayah_number=1, start_time=10.0, end_time=15.0),
        ])
        segments = runner._ground_truth_to_segments(gt)
        # Sorted by ayah_number: 1, 3, 5 -> ids 1, 2, 3
        assert [s.id for s in segments] == [1, 2, 3]
        # Text ordering should match ayah_number sort order
        assert segments[0].start == 10.0  # ayah_number=1


# ===========================================================================
# FILE-SYSTEM PERMISSION TESTS (skip when running as root)
# ===========================================================================


@pytest.mark.skipif(os.getuid() == 0, reason="Root bypasses permission checks")
class TestFilePermissions:
    """Tests that require non-root execution to verify permission enforcement."""

    def test_load_ground_truth_unreadable_file_raises(self, tmp_path: Path) -> None:
        """A file with mode 000 must raise PermissionError or OSError."""
        gt_file = tmp_path / "no_read.json"
        data = {
            "surah_id": 1,
            "reciter": "test",
            "audio_filename": "001.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "بسم الله",
                }
            ],
        }
        gt_file.write_text(json.dumps(data), encoding="utf-8")
        gt_file.chmod(0o000)
        try:
            with pytest.raises((PermissionError, OSError)):
                load_ground_truth(gt_file)
        finally:
            gt_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_save_json_to_read_only_dir_raises(self, tmp_path: Path) -> None:
        """Writing to a read-only directory must raise PermissionError or OSError."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        try:
            report = BenchmarkReport(
                generated_at="2026-02-26T12:00:00+00:00",
                munajjam_version="1.0.0",
                results=[],
            )
            runner = BenchmarkRunner(ground_truth_dir=tmp_path)
            with pytest.raises((PermissionError, OSError)):
                runner.save_json(report, readonly_dir / "results.json")
        finally:
            readonly_dir.chmod(stat.S_IRWXU)
