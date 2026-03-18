"""
Tests for the standardized JSON output formatter.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from munajjam.formatters import (
    AlignmentMetadata,
    AlignmentOutput,
    FormattedAyahResult,
    _format_single_result,
    format_alignment_results,
)
from munajjam.models.ayah import Ayah
from munajjam.models.result import AlignmentResult

# --- Fixtures ---


def _make_ayah(
    id: int = 1, surah_id: int = 1, ayah_number: int = 1, text: str = "بِسْمِ اللَّهِ"
) -> Ayah:
    return Ayah(id=id, surah_id=surah_id, ayah_number=ayah_number, text=text)


def _make_result(
    ayah: Ayah | None = None,
    start: float = 0.0,
    end: float = 5.0,
    transcribed: str = "بسم الله",
    score: float = 0.9,
    overlap: bool = False,
) -> AlignmentResult:
    if ayah is None:
        ayah = _make_ayah()
    return AlignmentResult(
        ayah=ayah,
        start_time=start,
        end_time=end,
        transcribed_text=transcribed,
        similarity_score=score,
        overlap_detected=overlap,
    )


@pytest.fixture
def sample_ayah():
    return _make_ayah()


@pytest.fixture
def sample_result():
    return _make_result()


@pytest.fixture
def sample_results():
    ayahs = [
        _make_ayah(id=1, surah_id=1, ayah_number=1, text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
        _make_ayah(id=2, surah_id=1, ayah_number=2, text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"),
        _make_ayah(id=3, surah_id=1, ayah_number=3, text="الرَّحْمَٰنِ الرَّحِيمِ"),
        _make_ayah(id=4, surah_id=1, ayah_number=4, text="مَالِكِ يَوْمِ الدِّينِ"),
    ]
    return [
        _make_result(
            ayah=ayahs[0],
            start=0.0,
            end=5.32,
            transcribed="بسم الله الرحمن الرحيم",
            score=0.95,
        ),
        _make_result(
            ayah=ayahs[1],
            start=5.32,
            end=10.15,
            transcribed="الحمد لله رب العالمين",
            score=0.92,
        ),
        _make_result(
            ayah=ayahs[2],
            start=10.15,
            end=13.44,
            transcribed="الرحمن الرحيم",
            score=0.88,
        ),
        _make_result(
            ayah=ayahs[3],
            start=13.44,
            end=17.00,
            transcribed="مالك يوم الدين",
            score=0.70,
            overlap=True,
        ),
    ]


# --- FormattedAyahResult Tests ---


class TestFormattedAyahResult:
    def test_basic_creation(self):
        result = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.32,
            duration=5.32,
            transcribed_text="بسم الله",
            original_text="بِسْمِ اللَّهِ",
            similarity_score=0.95,
            is_high_confidence=True,
            overlap_detected=False,
        )
        assert result.id == 1
        assert result.ayah_index == 0
        assert result.duration == 5.32
        assert result.is_high_confidence is True

    def test_serialization(self):
        result = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="test",
            original_text="test",
            similarity_score=0.9,
            is_high_confidence=True,
            overlap_detected=False,
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert "id" in data
        assert "surah_id" in data
        assert "similarity_score" in data


# --- _format_single_result Tests ---


class TestFormatSingleResult:
    def test_converts_alignment_result(self, sample_result):
        formatted = _format_single_result(sample_result)
        assert isinstance(formatted, FormattedAyahResult)
        assert formatted.id == sample_result.ayah.id
        assert formatted.surah_id == sample_result.ayah.surah_id
        assert formatted.ayah_number == sample_result.ayah.ayah_number
        assert formatted.ayah_index == sample_result.ayah.ayah_number - 1
        assert formatted.start_time == round(sample_result.start_time, 2)
        assert formatted.end_time == round(sample_result.end_time, 2)
        assert formatted.transcribed_text == sample_result.transcribed_text
        assert formatted.original_text == sample_result.ayah.text
        assert formatted.similarity_score == round(sample_result.similarity_score, 3)
        assert formatted.is_high_confidence == sample_result.is_high_confidence
        assert formatted.overlap_detected == sample_result.overlap_detected

    def test_rounds_times(self):
        result = _make_result(start=1.23456, end=5.67891, score=0.12345)
        formatted = _format_single_result(result)
        assert formatted.start_time == 1.23
        assert formatted.end_time == 5.68
        assert formatted.similarity_score == 0.123

    def test_duration_calculation(self):
        result = _make_result(start=1.0, end=6.0)
        formatted = _format_single_result(result)
        assert formatted.duration == 5.0

    def test_high_confidence_threshold(self):
        high = _make_result(score=0.85)
        low = _make_result(score=0.75)
        assert _format_single_result(high).is_high_confidence is True
        assert _format_single_result(low).is_high_confidence is False

    def test_high_confidence_boundary(self):
        at_threshold = _make_result(score=0.8)
        just_below = _make_result(score=0.7999)
        assert _format_single_result(at_threshold).is_high_confidence is True
        assert _format_single_result(just_below).is_high_confidence is False

    def test_overlap_flag(self):
        with_overlap = _make_result(overlap=True)
        without_overlap = _make_result(overlap=False)
        assert _format_single_result(with_overlap).overlap_detected is True
        assert _format_single_result(without_overlap).overlap_detected is False


# --- format_alignment_results Tests ---


class TestFormatAlignmentResults:
    def test_basic_formatting(self, sample_results):
        output = format_alignment_results(sample_results)
        assert isinstance(output, AlignmentOutput)
        assert len(output.results) == 4
        assert output.metadata.total_ayahs == 4

    def test_metadata_fields(self, sample_results):
        output = format_alignment_results(
            sample_results,
            surah_id=1,
            reciter="Test Reciter",
            audio_file="test.wav",
        )
        meta = output.metadata
        assert meta.surah_id == 1
        assert meta.reciter == "Test Reciter"
        assert meta.audio_file == "test.wav"
        assert meta.total_ayahs == 4
        assert meta.munajjam_version is not None
        assert meta.generated_at is not None

    def test_metadata_optional_fields(self, sample_results):
        output = format_alignment_results(sample_results)
        meta = output.metadata
        assert meta.surah_id is None
        assert meta.reciter is None
        assert meta.audio_file is None

    def test_total_duration(self, sample_results):
        output = format_alignment_results(sample_results)
        expected_duration = round(5.32 + 4.83 + 3.29 + 3.56, 2)
        assert output.metadata.total_duration == expected_duration

    def test_average_confidence(self, sample_results):
        output = format_alignment_results(sample_results)
        expected_avg = round((0.95 + 0.92 + 0.88 + 0.70) / 4, 3)
        assert output.metadata.average_confidence == expected_avg

    def test_high_confidence_count(self, sample_results):
        output = format_alignment_results(sample_results)
        # 0.95, 0.92, 0.88 are >= 0.8, 0.70 is not
        assert output.metadata.high_confidence_count == 3

    def test_empty_results(self):
        output = format_alignment_results([])
        assert len(output.results) == 0
        assert output.metadata.total_ayahs == 0
        assert output.metadata.total_duration == 0.0
        assert output.metadata.average_confidence == 0.0
        assert output.metadata.high_confidence_count == 0

    def test_single_result(self, sample_result):
        output = format_alignment_results([sample_result])
        assert len(output.results) == 1
        assert output.metadata.total_ayahs == 1

    def test_results_order_preserved(self, sample_results):
        output = format_alignment_results(sample_results)
        for i, result in enumerate(output.results):
            assert result.ayah_number == i + 1


# --- AlignmentOutput Tests ---


class TestAlignmentOutput:
    def test_to_json(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        json_str = output.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "results" in parsed
        assert len(parsed["results"]) == 4

    def test_to_dict(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        data = output.to_dict()
        assert isinstance(data, dict)
        assert "metadata" in data
        assert "results" in data

    def test_to_file(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        output.to_file(path)
        content = Path(path).read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert parsed["metadata"]["surah_id"] == 1
        assert len(parsed["results"]) == 4
        Path(path).unlink()

    def test_json_contains_arabic(self, sample_results):
        output = format_alignment_results(sample_results)
        json_str = output.to_json()
        # Arabic text should not be escaped
        assert "بسم" in json_str or "الحمد" in json_str

    def test_json_schema_consistency(self, sample_results):
        """Ensure every result has the same set of keys."""
        output = format_alignment_results(sample_results)
        data = output.to_dict()
        expected_keys = {
            "id",
            "surah_id",
            "ayah_number",
            "ayah_index",
            "start_time",
            "end_time",
            "duration",
            "transcribed_text",
            "original_text",
            "similarity_score",
            "is_high_confidence",
            "overlap_detected",
        }
        for result in data["results"]:
            assert set(result.keys()) == expected_keys

    def test_metadata_schema(self, sample_results):
        """Ensure metadata has all expected keys."""
        output = format_alignment_results(sample_results)
        data = output.to_dict()
        expected_keys = {
            "schema_version",
            "munajjam_version",
            "generated_at",
            "surah_id",
            "surah_name",
            "reciter",
            "audio_file",
            "total_ayahs",
            "total_duration",
            "average_confidence",
            "high_confidence_count",
        }
        assert set(data["metadata"].keys()) == expected_keys


# --- Frozen Model Tests ---


class TestFrozenModels:
    """Verify all models are immutable (ConfigDict frozen=True)."""

    def test_formatted_ayah_result_is_frozen(self):
        result = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="test",
            original_text="test",
            similarity_score=0.9,
            is_high_confidence=True,
            overlap_detected=False,
        )
        with pytest.raises(ValidationError):
            result.id = 999

    def test_alignment_metadata_is_frozen(self):
        meta = AlignmentMetadata(
            munajjam_version="0.1.0",
            generated_at="2024-01-01T00:00:00+00:00",
            total_ayahs=1,
            total_duration=5.0,
            average_confidence=0.9,
            high_confidence_count=1,
        )
        with pytest.raises(ValidationError):
            meta.total_ayahs = 999

    def test_alignment_output_is_frozen(self, sample_results):
        output = format_alignment_results(sample_results)
        with pytest.raises(ValidationError):
            output.results = []


# --- Schema Versioning Tests ---


class TestSchemaVersioning:
    """Test schema_version field on AlignmentMetadata."""

    def test_default_schema_version(self, sample_results):
        output = format_alignment_results(sample_results)
        assert output.metadata.schema_version == "1.0"

    def test_schema_version_in_json(self, sample_results):
        output = format_alignment_results(sample_results)
        data = json.loads(output.to_json())
        assert data["metadata"]["schema_version"] == "1.0"

    def test_schema_version_in_dict(self, sample_results):
        output = format_alignment_results(sample_results)
        data = output.to_dict()
        assert data["metadata"]["schema_version"] == "1.0"


# --- Field Validation Tests ---


class TestFieldValidation:
    """Test pydantic Field(ge=, le=) constraints on FormattedAyahResult."""

    def test_reject_negative_id(self):
        with pytest.raises(ValidationError, match="id"):
            FormattedAyahResult(
                id=-1,
                surah_id=1,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_zero_id(self):
        with pytest.raises(ValidationError, match="id"):
            FormattedAyahResult(
                id=0,
                surah_id=1,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_surah_id_out_of_range(self):
        with pytest.raises(ValidationError, match="surah_id"):
            FormattedAyahResult(
                id=1,
                surah_id=115,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_zero_surah_id(self):
        with pytest.raises(ValidationError, match="surah_id"):
            FormattedAyahResult(
                id=1,
                surah_id=0,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_negative_start_time(self):
        with pytest.raises(ValidationError, match="start_time"):
            FormattedAyahResult(
                id=1,
                surah_id=1,
                ayah_number=1,
                ayah_index=0,
                start_time=-1.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_similarity_above_one(self):
        with pytest.raises(ValidationError, match="similarity_score"):
            FormattedAyahResult(
                id=1,
                surah_id=1,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=1.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_negative_similarity(self):
        with pytest.raises(ValidationError, match="similarity_score"):
            FormattedAyahResult(
                id=1,
                surah_id=1,
                ayah_number=1,
                ayah_index=0,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=-0.1,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_reject_negative_ayah_index(self):
        with pytest.raises(ValidationError, match="ayah_index"):
            FormattedAyahResult(
                id=1,
                surah_id=1,
                ayah_number=1,
                ayah_index=-1,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                transcribed_text="t",
                original_text="t",
                similarity_score=0.5,
                is_high_confidence=False,
                overlap_detected=False,
            )

    def test_accept_boundary_values(self):
        """Ensure min/max boundary values are accepted."""
        result = FormattedAyahResult(
            id=1,
            surah_id=114,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=0.0,
            duration=0.0,
            transcribed_text="",
            original_text="",
            similarity_score=0.0,
            is_high_confidence=False,
            overlap_detected=False,
        )
        assert result.surah_id == 114
        assert result.similarity_score == 0.0

    def test_accept_max_similarity(self):
        result = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="t",
            original_text="t",
            similarity_score=1.0,
            is_high_confidence=True,
            overlap_detected=False,
        )
        assert result.similarity_score == 1.0


# --- Surah Name Auto-Detection Tests ---


class TestSurahNameAutoDetection:
    """Test surah_name auto-detection from surah_id."""

    def test_auto_detect_al_fatiha(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        assert output.metadata.surah_name == "الفاتحة"

    def test_auto_detect_an_nas(self):
        result = _make_result()
        output = format_alignment_results([result], surah_id=114)
        assert output.metadata.surah_name == "الناس"

    def test_auto_detect_al_baqara(self):
        result = _make_result()
        output = format_alignment_results([result], surah_id=2)
        assert output.metadata.surah_name == "البقرة"

    def test_explicit_surah_name_overrides_auto(self, sample_results):
        output = format_alignment_results(
            sample_results, surah_id=1, surah_name="Custom Name"
        )
        assert output.metadata.surah_name == "Custom Name"

    def test_no_surah_id_means_no_surah_name(self, sample_results):
        output = format_alignment_results(sample_results)
        assert output.metadata.surah_name is None

    def test_surah_name_in_json(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        data = json.loads(output.to_json())
        assert data["metadata"]["surah_name"] == "الفاتحة"


# --- Round-Trip Serialization Tests ---


class TestRoundTrip:
    """Test JSON round-trip: format → to_json → parse → verify."""

    def test_json_round_trip(self, sample_results):
        output = format_alignment_results(
            sample_results,
            surah_id=1,
            reciter="Test",
            audio_file="test.wav",
        )
        json_str = output.to_json()
        parsed = json.loads(json_str)

        # Verify metadata round-trips
        meta = parsed["metadata"]
        assert meta["schema_version"] == "1.0"
        assert meta["surah_id"] == 1
        assert meta["surah_name"] == "الفاتحة"
        assert meta["reciter"] == "Test"
        assert meta["audio_file"] == "test.wav"
        assert meta["total_ayahs"] == 4
        assert isinstance(meta["total_duration"], float)
        assert isinstance(meta["average_confidence"], float)
        assert isinstance(meta["high_confidence_count"], int)

        # Verify results round-trip
        assert len(parsed["results"]) == 4
        first = parsed["results"][0]
        assert first["id"] == 1
        assert first["surah_id"] == 1
        assert first["ayah_number"] == 1
        assert first["ayah_index"] == 0
        assert isinstance(first["start_time"], float)
        assert isinstance(first["end_time"], float)
        assert isinstance(first["duration"], float)
        assert isinstance(first["similarity_score"], float)
        assert isinstance(first["is_high_confidence"], bool)
        assert isinstance(first["overlap_detected"], bool)

    def test_dict_round_trip(self, sample_results):
        output = format_alignment_results(sample_results, surah_id=1)
        data = output.to_dict()

        # Re-parse back into AlignmentOutput
        rebuilt = AlignmentOutput(**data)
        assert rebuilt.metadata.schema_version == output.metadata.schema_version
        assert rebuilt.metadata.surah_name == output.metadata.surah_name
        assert len(rebuilt.results) == len(output.results)
        for orig, rebuilt_r in zip(output.results, rebuilt.results, strict=True):
            assert orig.id == rebuilt_r.id
            assert orig.similarity_score == rebuilt_r.similarity_score

    def test_file_round_trip(self, sample_results, tmp_path):
        output = format_alignment_results(sample_results, surah_id=1)
        path = tmp_path / "round_trip.json"
        output.to_file(str(path))
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["metadata"]["schema_version"] == "1.0"
        assert parsed["metadata"]["surah_name"] == "الفاتحة"
        assert len(parsed["results"]) == 4


# ---------------------------------------------------------------------------
# ADVERSARIAL EDGE CASE TESTS (Grumpy Tester)
# ---------------------------------------------------------------------------


class TestAdversarialMetadataSurahIdConstraint:
    """
    FIXED: AlignmentMetadata.surah_id now has ge=1, le=114 constraints,
    consistent with FormattedAyahResult.surah_id. Invalid values are rejected.
    """

    def test_metadata_rejects_surah_id_zero(self):
        """surah_id=0 is rejected by AlignmentMetadata (ge=1)."""
        with pytest.raises(ValidationError, match="surah_id"):
            AlignmentMetadata(
                munajjam_version="0.1.0",
                generated_at="2024-01-01T00:00:00+00:00",
                surah_id=0,
                total_ayahs=1,
                total_duration=5.0,
                average_confidence=0.9,
                high_confidence_count=1,
            )

    def test_metadata_rejects_surah_id_negative(self):
        """surah_id=-1 is rejected by AlignmentMetadata (ge=1)."""
        with pytest.raises(ValidationError, match="surah_id"):
            AlignmentMetadata(
                munajjam_version="0.1.0",
                generated_at="2024-01-01T00:00:00+00:00",
                surah_id=-1,
                total_ayahs=1,
                total_duration=5.0,
                average_confidence=0.9,
                high_confidence_count=1,
            )

    def test_metadata_rejects_surah_id_above_114(self):
        """surah_id=115 is rejected by AlignmentMetadata (le=114)."""
        with pytest.raises(ValidationError, match="surah_id"):
            AlignmentMetadata(
                munajjam_version="0.1.0",
                generated_at="2024-01-01T00:00:00+00:00",
                surah_id=115,
                total_ayahs=1,
                total_duration=5.0,
                average_confidence=0.9,
                high_confidence_count=1,
            )

    def test_metadata_accepts_surah_id_none(self):
        """surah_id=None is valid (optional field)."""
        meta = AlignmentMetadata(
            munajjam_version="0.1.0",
            generated_at="2024-01-01T00:00:00+00:00",
            total_ayahs=1,
            total_duration=5.0,
            average_confidence=0.9,
            high_confidence_count=1,
        )
        assert meta.surah_id is None

    def test_format_alignment_results_rejects_invalid_surah_id(self):
        """
        format_alignment_results() now rejects invalid surah_id values
        via AlignmentMetadata validation.
        """
        result = _make_result()
        for bad_id in (0, -1, 115, 999):
            with pytest.raises(ValidationError):
                format_alignment_results([result], surah_id=bad_id)


class TestAdversarialDurationVsTimestampInconsistency:
    """
    FINDING (MEDIUM): duration is computed from raw (unrounded) timestamps, then
    rounded independently of start_time and end_time. Rounding each of the three
    values separately can produce: duration != end_time - start_time.

    Example: start=0.005, end=5.005
      round(0.005, 2) = 0.01  (rounds up)
      round(5.005, 2) = 5.0   (rounds down due to float repr)
      duration = round(5.005 - 0.005, 2) = round(5.0, 2) = 5.0
      But stored: end_time - start_time = 5.0 - 0.01 = 4.99  ≠  5.0
    """

    def test_duration_arithmetic_inconsistency_case1(self):
        """start=0.005, end=5.005: stored duration != end_time - start_time."""
        result = _make_result(start=0.005, end=5.005)
        formatted = _format_single_result(result)
        recomputed = round(formatted.end_time - formatted.start_time, 2)
        # Document the inconsistency: duration=5.0 but end-start=4.99
        assert formatted.duration != recomputed, (
            f"Expected inconsistency: duration={formatted.duration}, "
            f"end_time - start_time = {recomputed}. "
            "This shows the three values are independently rounded and can disagree."
        )

    def test_duration_arithmetic_inconsistency_case2(self):
        """start=1.125, end=4.375: stored duration != end_time - start_time."""
        result = _make_result(start=1.125, end=4.375)
        formatted = _format_single_result(result)
        recomputed = round(formatted.end_time - formatted.start_time, 2)
        assert formatted.duration != recomputed, (
            f"duration={formatted.duration}, end_time - start_time = {recomputed}. Inconsistent."
        )

    def test_end_before_start_raises_on_format(self):
        """
        AlignmentResult accepts end_time < start_time (no cross-field validation).
        _format_single_result raises ValidationError via FormattedAyahResult(duration ge=0).
        """
        ayah = _make_ayah()
        ar = AlignmentResult(
            ayah=ayah,
            start_time=10.0,
            end_time=1.0,  # end before start
            transcribed_text="t",
            similarity_score=0.5,
        )
        assert ar.duration == -9.0, "AlignmentResult itself does not guard end > start"
        with pytest.raises(ValidationError):
            _format_single_result(ar)  # FormattedAyahResult(duration=-9.0) must fail


class TestAdversarialHighConfidenceVsScoreContradiction:
    """
    FINDING (LOW): is_high_confidence is derived from the PRE-ROUNDING score
    (inside AlignmentResult), while similarity_score is stored ROUNDED to 3 places.
    A score of 0.79951 rounds to 0.800 but is_high_confidence is False.
    The JSON output then reads: similarity_score=0.8, is_high_confidence=false.
    This looks contradictory to consumers reading the JSON.
    """

    def test_score_rounds_to_point8_but_confidence_is_false(self):
        """
        0.79951 rounds to 0.800 (3 places) but raw score < 0.8 -> is_high_confidence=False.
        JSON shows similarity_score=0.8 alongside is_high_confidence=false.
        """
        result = _make_result(score=0.79951)
        formatted = _format_single_result(result)
        assert formatted.similarity_score == 0.8
        assert formatted.is_high_confidence is False, (
            "similarity_score=0.8 and is_high_confidence=False. "
            "Looks contradictory to JSON consumers."
        )

    def test_score_rounds_to_point8_from_just_above(self):
        """0.8004 also rounds to 0.8 but is_high_confidence=True - consistent."""
        result = _make_result(score=0.8004)
        formatted = _format_single_result(result)
        assert formatted.similarity_score == 0.8
        assert formatted.is_high_confidence is True


class TestAdversarialMissingUpperBounds:
    """
    FINDING (MEDIUM): FormattedAyahResult.id description says 1-6236 but
    the field only enforces ge=1. Similarly ayah_number has no upper bound.
    Garbage values like id=99999 silently pass.
    """

    def test_ayah_id_above_6236_accepted(self):
        """id=99999 exceeds the documented maximum of 6236 but is accepted."""
        r = FormattedAyahResult(
            id=99999,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="t",
            original_text="t",
            similarity_score=0.5,
            is_high_confidence=False,
            overlap_detected=False,
        )
        assert r.id == 99999, (
            "id=99999 accepted despite description saying max is 6236. "
            "Field needs le=6236 to enforce the documented constraint."
        )

    def test_ayah_number_has_no_upper_bound(self):
        """ayah_number=99999 is accepted - no upper bound constraint."""
        r = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=99999,
            ayah_index=99998,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="t",
            original_text="t",
            similarity_score=0.5,
            is_high_confidence=False,
            overlap_detected=False,
        )
        assert r.ayah_number == 99999, "ayah_number=99999 accepted with no upper bound."

    def test_empty_string_text_fields_accepted(self):
        """FormattedAyahResult has no min_length on text fields."""
        r = FormattedAyahResult(
            id=1,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="",
            original_text="",
            similarity_score=0.0,
            is_high_confidence=False,
            overlap_detected=False,
        )
        assert r.transcribed_text == ""
        assert r.original_text == ""


class TestAdversarialSchemaVersionNoConstraint:
    """
    FINDING (LOW): schema_version has a sensible default of '1.0' but no
    validation prevents setting it to an empty string or arbitrary value
    via direct AlignmentMetadata construction.
    """

    def test_schema_version_accepts_empty_string(self):
        """Empty string schema_version is accepted - no min_length constraint."""
        meta = AlignmentMetadata(
            schema_version="",
            munajjam_version="0.1.0",
            generated_at="2024-01-01T00:00:00+00:00",
            total_ayahs=1,
            total_duration=5.0,
            average_confidence=0.9,
            high_confidence_count=1,
        )
        assert meta.schema_version == "", (
            "Empty schema_version accepted. No min_length=1 constraint."
        )

    def test_schema_version_accepts_arbitrary_string(self):
        """Arbitrary schema_version string is accepted - no pattern validation."""
        meta = AlignmentMetadata(
            schema_version="99.99",
            munajjam_version="0.1.0",
            generated_at="2024-01-01T00:00:00+00:00",
            total_ayahs=1,
            total_duration=5.0,
            average_confidence=0.9,
            high_confidence_count=1,
        )
        assert meta.schema_version == "99.99"

    def test_schema_version_cannot_be_none(self):
        """None schema_version IS rejected (field type is str, not Optional[str])."""
        with pytest.raises(ValidationError):
            AlignmentMetadata(
                schema_version=None,  # type: ignore[arg-type]
                munajjam_version="0.1.0",
                generated_at="2024-01-01T00:00:00+00:00",
                total_ayahs=1,
                total_duration=5.0,
                average_confidence=0.9,
                high_confidence_count=1,
            )


class TestAdversarialFrozenModelListMutability:
    """
    FINDING (MEDIUM): AlignmentOutput.results is a list[FormattedAyahResult].
    frozen=True prevents reassigning output.results = [], but it does NOT
    prevent in-place mutation of the list itself (append, index assignment).
    True immutability requires tuple or other immutable container.
    """

    def test_results_list_append_succeeds_despite_frozen(self):
        """list.append() mutates the list even though AlignmentOutput is frozen."""
        result = _make_result()
        output = format_alignment_results([result])
        original_len = len(output.results)
        output.results.append(output.results[0])  # This should be blocked but isn't
        assert len(output.results) == original_len + 1, (
            "list.append() succeeded on a frozen model's list field. "
            "Use tuple[FormattedAyahResult, ...] to enforce true immutability."
        )

    def test_results_list_index_assignment_succeeds(self):
        """results[0] = x mutates the list even though AlignmentOutput is frozen."""
        result = _make_result()
        output = format_alignment_results([result])
        new_entry = FormattedAyahResult(
            id=999,
            surah_id=1,
            ayah_number=1,
            ayah_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            transcribed_text="mutated",
            original_text="mutated",
            similarity_score=0.5,
            is_high_confidence=False,
            overlap_detected=False,
        )
        output.results[0] = new_entry
        assert output.results[0].id == 999, (
            "Index assignment on list succeeded on a frozen AlignmentOutput."
        )

    def test_object_setattr_bypasses_frozen_on_metadata(self):
        """
        object.__setattr__ bypasses pydantic's frozen enforcement entirely.
        This is a Python-level reality, not a pydantic bug, but it means
        'frozen' is a convention, not a guarantee against determined mutation.
        """
        result = _make_result()
        output = format_alignment_results([result], surah_id=1)
        assert output.metadata.surah_id == 1
        object.__setattr__(output.metadata, "surah_id", 999)
        assert output.metadata.surah_id == 999, (
            "object.__setattr__ bypassed frozen constraint. "
            "Pydantic frozen is not cryptographically enforced."
        )


class TestAdversarialCrossSurahResults:
    """
    FINDING (LOW): format_alignment_results() does not validate that all results
    belong to the declared surah_id. Mixed-surah results are silently accepted.
    """

    def test_results_from_different_surahs_accepted(self):
        """Results from surah 1 and surah 2 accepted when surah_id=1 declared."""
        ayah1 = _make_ayah(id=1, surah_id=1, ayah_number=1)
        ayah2 = _make_ayah(id=8, surah_id=2, ayah_number=1)
        r1 = _make_result(ayah=ayah1)
        r2 = _make_result(ayah=ayah2, start=5.0, end=10.0)
        output = format_alignment_results([r1, r2], surah_id=1)
        assert output.metadata.surah_id == 1
        surah_ids_in_results = {r.surah_id for r in output.results}
        assert surah_ids_in_results == {1, 2}, (
            "Output declares surah_id=1 but results contain surahs 1 and 2. "
            "No cross-surah validation exists."
        )

    def test_no_surah_id_but_results_have_surah_id(self):
        """When no surah_id is passed, metadata.surah_id=None but results have surah_ids."""
        result = _make_result()
        output = format_alignment_results([result])
        assert output.metadata.surah_id is None
        assert output.results[0].surah_id == 1  # Result still has its own surah_id


class TestAdversarialInfinityInTimes:
    """
    FINDING (MEDIUM): AlignmentResult does not reject float('inf') for
    start_time or end_time. This propagates through formatting and produces
    non-standard JSON with 'Infinity' literals (not RFC 7159 compliant).
    """

    def test_inf_end_time_accepted_by_alignment_result(self):
        """float('inf') end_time passes AlignmentResult validation."""
        ayah = _make_ayah()
        r = AlignmentResult(
            ayah=ayah,
            start_time=0.0,
            end_time=float("inf"),
            transcribed_text="t",
            similarity_score=0.5,
        )
        assert r.end_time == float("inf")

    def test_inf_propagates_to_formatted_result(self):
        """float('inf') end_time passes through _format_single_result."""
        ayah = _make_ayah()
        r = AlignmentResult(
            ayah=ayah,
            start_time=0.0,
            end_time=float("inf"),
            transcribed_text="t",
            similarity_score=0.5,
        )
        formatted = _format_single_result(r)
        assert math.isinf(formatted.end_time)
        assert math.isinf(formatted.duration)

    def test_inf_produces_infinity_literal_in_json(self):
        """
        to_json() with Inf values produces 'Infinity' which is not standard JSON.
        Python's json module writes it (allow_nan=True by default), but strict
        parsers and the RFC 7159 spec do not allow it.
        """
        ayah = _make_ayah()
        r = AlignmentResult(
            ayah=ayah,
            start_time=0.0,
            end_time=float("inf"),
            transcribed_text="t",
            similarity_score=0.5,
        )
        output = format_alignment_results([r])
        json_str = output.to_json()
        assert "Infinity" in json_str, (
            "to_json() writes 'Infinity' which is not valid per RFC 7159. "
            "The formatter should either reject Inf inputs or use allow_nan=False."
        )
        assert math.isinf(output.metadata.total_duration)
