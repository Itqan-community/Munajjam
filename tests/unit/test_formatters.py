"""
Tests for the standardized JSON output formatter.
"""

import json
import tempfile
from pathlib import Path

import pytest

from munajjam.formatters import (
    AlignmentOutput,
    FormattedAyahResult,
    format_alignment_results,
    _format_single_result,
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
        _make_result(ayah=ayahs[0], start=0.0, end=5.32, transcribed="بسم الله الرحمن الرحيم", score=0.95),
        _make_result(ayah=ayahs[1], start=5.32, end=10.15, transcribed="الحمد لله رب العالمين", score=0.92),
        _make_result(ayah=ayahs[2], start=10.15, end=13.44, transcribed="الرحمن الرحيم", score=0.88),
        _make_result(ayah=ayahs[3], start=13.44, end=17.00, transcribed="مالك يوم الدين", score=0.70, overlap=True),
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
            "id", "surah_id", "ayah_number", "ayah_index",
            "start_time", "end_time", "duration",
            "transcribed_text", "original_text",
            "similarity_score", "is_high_confidence", "overlap_detected",
        }
        for result in data["results"]:
            assert set(result.keys()) == expected_keys

    def test_metadata_schema(self, sample_results):
        """Ensure metadata has all expected keys."""
        output = format_alignment_results(sample_results)
        data = output.to_dict()
        expected_keys = {
            "munajjam_version", "generated_at", "surah_id", "reciter",
            "audio_file", "total_ayahs", "total_duration",
            "average_confidence", "high_confidence_count",
        }
        assert set(data["metadata"].keys()) == expected_keys
