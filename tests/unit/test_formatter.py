"""
Unit tests for the JSON output formatter.
"""

import json
from datetime import datetime

import pytest

from munajjam.models import AlignmentResult, Ayah
from munajjam.output import AlignmentOutput, FormattedAyahResult, format_alignment_results


class TestFormattedAyahResult:
    """Tests for the FormattedAyahResult model."""

    def test_basic_creation(self):
        """Test creating a FormattedAyahResult with valid data."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=5.32,
            transcribed_text="بسم الله الرحمن الرحيم",
            corrected_text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            similarity_score=0.95,
        )
        
        assert result.id == 1
        assert result.sura_id == 1
        assert result.ayah_index == 0
        assert result.start == 0.0
        assert result.end == 5.32
        assert result.transcribed_text == "بسم الله الرحمن الرحيم"
        assert result.corrected_text == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        assert result.similarity_score == 0.95

    def test_computed_duration(self):
        """Test that duration is computed correctly."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=10.0,
            end=15.5,
            transcribed_text="test",
            corrected_text="test",
            similarity_score=0.9,
        )
        
        assert result.duration == 5.5

    def test_computed_high_confidence_true(self):
        """Test high_confidence is True when similarity >= 0.8."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=1.0,
            transcribed_text="test",
            corrected_text="test",
            similarity_score=0.8,
        )
        
        assert result.high_confidence is True

    def test_computed_high_confidence_false(self):
        """Test high_confidence is False when similarity < 0.8."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=1.0,
            transcribed_text="test",
            corrected_text="test",
            similarity_score=0.79,
        )
        
        assert result.high_confidence is False

    def test_validation_id_positive(self):
        """Test that id must be positive."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=0,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )

    def test_validation_surah_id_range(self):
        """Test that surah_id must be in range 1-114."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=0,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )
        
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=115,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )

    def test_validation_ayah_index_non_negative(self):
        """Test that ayah_index must be non-negative."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=-1,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )

    def test_validation_start_non_negative(self):
        """Test that start time must be non-negative."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=-1.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )

    def test_validation_end_non_negative(self):
        """Test that end time must be non-negative."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=-1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=0.9,
            )

    def test_validation_similarity_range(self):
        """Test that similarity_score must be in range 0.0-1.0."""
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=-0.1,
            )
        
        with pytest.raises(ValueError):
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test",
                corrected_text="test",
                similarity_score=1.1,
            )

    def test_json_serialization(self):
        """Test JSON serialization."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=5.32,
            transcribed_text="بسم الله",
            corrected_text="بِسْمِ اللَّهِ",
            similarity_score=0.95,
        )
        
        json_str = result.model_dump_json()
        data = json.loads(json_str)
        
        assert data["id"] == 1
        assert data["sura_id"] == 1
        assert data["ayah_index"] == 0
        assert data["start"] == 0.0
        assert data["end"] == 5.32
        assert data["transcribed_text"] == "بسم الله"
        assert data["corrected_text"] == "بِسْمِ اللَّهِ"
        assert data["similarity_score"] == 0.95
        assert data["duration"] == 5.32
        assert data["high_confidence"] is True


class TestAlignmentOutput:
    """Tests for the AlignmentOutput model."""

    def test_basic_creation(self):
        """Test creating an AlignmentOutput with valid data."""
        result1 = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=5.0,
            transcribed_text="test1",
            corrected_text="test1",
            similarity_score=0.9,
        )
        result2 = FormattedAyahResult(
            id=2,
            sura_id=1,
            ayah_index=1,
            start=5.0,
            end=10.0,
            transcribed_text="test2",
            corrected_text="test2",
            similarity_score=0.85,
        )
        
        output = AlignmentOutput(
            version="1.0.0",
            surah_id=1,
            reciter="Test Reciter",
            results=[result1, result2],
        )
        
        assert output.version == "1.0.0"
        assert output.surah_id == 1
        assert output.reciter == "Test Reciter"
        assert len(output.results) == 2
        assert output.generated_at is not None

    def test_computed_total_ayahs(self):
        """Test that total_ayahs is computed correctly."""
        results = [
            FormattedAyahResult(
                id=i,
                sura_id=1,
                ayah_index=i-1,
                start=float(i),
                end=float(i+1),
                transcribed_text=f"test{i}",
                corrected_text=f"test{i}",
                similarity_score=0.9,
            )
            for i in range(1, 4)
        ]
        
        output = AlignmentOutput(surah_id=1, results=results)
        
        assert output.total_ayahs == 3

    def test_computed_total_ayahs_empty(self):
        """Test that total_ayahs is 0 for empty results."""
        output = AlignmentOutput(surah_id=1, results=[])
        
        assert output.total_ayahs == 0

    def test_computed_avg_similarity(self):
        """Test that avg_similarity is computed correctly."""
        results = [
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test1",
                corrected_text="test1",
                similarity_score=0.9,
            ),
            FormattedAyahResult(
                id=2,
                sura_id=1,
                ayah_index=1,
                start=1.0,
                end=2.0,
                transcribed_text="test2",
                corrected_text="test2",
                similarity_score=0.8,
            ),
        ]
        
        output = AlignmentOutput(surah_id=1, results=results)
        
        assert output.avg_similarity == 0.85

    def test_computed_avg_similarity_empty(self):
        """Test that avg_similarity is 0.0 for empty results."""
        output = AlignmentOutput(surah_id=1, results=[])
        
        assert output.avg_similarity == 0.0

    def test_computed_high_confidence_count(self):
        """Test that high_confidence_count is computed correctly."""
        results = [
            FormattedAyahResult(
                id=1,
                sura_id=1,
                ayah_index=0,
                start=0.0,
                end=1.0,
                transcribed_text="test1",
                corrected_text="test1",
                similarity_score=0.9,  # high confidence
            ),
            FormattedAyahResult(
                id=2,
                sura_id=1,
                ayah_index=1,
                start=1.0,
                end=2.0,
                transcribed_text="test2",
                corrected_text="test2",
                similarity_score=0.7,  # not high confidence
            ),
            FormattedAyahResult(
                id=3,
                sura_id=1,
                ayah_index=2,
                start=2.0,
                end=3.0,
                transcribed_text="test3",
                corrected_text="test3",
                similarity_score=0.8,  # high confidence (boundary)
            ),
        ]
        
        output = AlignmentOutput(surah_id=1, results=results)
        
        assert output.high_confidence_count == 2

    def test_to_json(self):
        """Test to_json method."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=5.0,
            transcribed_text="test",
            corrected_text="test",
            similarity_score=0.9,
        )
        
        output = AlignmentOutput(
            version="1.0.0",
            surah_id=1,
            reciter="Test",
            results=[result],
        )
        
        json_str = output.to_json()
        data = json.loads(json_str)
        
        assert data["version"] == "1.0.0"
        assert data["surah_id"] == 1
        assert data["reciter"] == "Test"
        assert data["total_ayahs"] == 1
        assert data["avg_similarity"] == 0.9
        assert data["high_confidence_count"] == 1
        assert len(data["results"]) == 1

    def test_to_json_compact(self):
        """Test to_json with no indentation."""
        output = AlignmentOutput(surah_id=1, results=[])
        json_str = output.to_json(indent=None)
        
        # Should be a single line
        assert "\n" not in json_str

    def test_to_dict(self):
        """Test to_dict method."""
        result = FormattedAyahResult(
            id=1,
            sura_id=1,
            ayah_index=0,
            start=0.0,
            end=5.0,
            transcribed_text="test",
            corrected_text="test",
            similarity_score=0.9,
        )
        
        output = AlignmentOutput(
            version="1.0.0",
            surah_id=1,
            reciter="Test",
            results=[result],
        )
        
        data = output.to_dict()
        
        assert data["version"] == "1.0.0"
        assert data["surah_id"] == 1
        assert data["reciter"] == "Test"
        assert data["total_ayahs"] == 1
        assert len(data["results"]) == 1

    def test_with_metadata(self):
        """Test AlignmentOutput with metadata."""
        metadata = {
            "audio_file": "test.wav",
            "processed_at": datetime.utcnow().isoformat(),
            "custom_field": "custom_value",
        }
        
        output = AlignmentOutput(
            surah_id=1,
            reciter="Test",
            results=[],
            metadata=metadata,
        )
        
        assert output.metadata == metadata
        assert output.metadata["audio_file"] == "test.wav"

    def test_validation_surah_id_range(self):
        """Test that surah_id must be in range 1-114."""
        with pytest.raises(ValueError):
            AlignmentOutput(surah_id=0, results=[])
        
        with pytest.raises(ValueError):
            AlignmentOutput(surah_id=115, results=[])

    def test_default_version(self):
        """Test that default version is set."""
        output = AlignmentOutput(surah_id=1, results=[])
        
        assert output.version == "1.0.0"

    def test_default_reciter(self):
        """Test that default reciter is 'Unknown'."""
        output = AlignmentOutput(surah_id=1, results=[])
        
        assert output.reciter == "Unknown"

    def test_default_generated_at(self):
        """Test that generated_at is auto-generated."""
        output = AlignmentOutput(surah_id=1, results=[])
        
        assert output.generated_at is not None
        assert "T" in output.generated_at  # ISO 8601 format indicator


class TestFormatAlignmentResults:
    """Tests for the format_alignment_results function."""

    def create_alignment_result(
        self,
        ayah_id: int,
        surah_id: int,
        ayah_number: int,
        text: str,
        start_time: float,
        end_time: float,
        transcribed_text: str,
        similarity_score: float,
    ) -> AlignmentResult:
        """Helper to create AlignmentResult objects."""
        ayah = Ayah(
            id=ayah_id,
            surah_id=surah_id,
            ayah_number=ayah_number,
            text=text,
        )
        return AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=transcribed_text,
            similarity_score=similarity_score,
        )

    def test_format_single_result(self):
        """Test formatting a single alignment result."""
        result = self.create_alignment_result(
            ayah_id=1,
            surah_id=1,
            ayah_number=1,
            text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            start_time=0.0,
            end_time=5.32,
            transcribed_text="بسم الله الرحمن الرحيم",
            similarity_score=0.95,
        )
        
        output = format_alignment_results(
            results=[result],
            surah_id=1,
            reciter="Test Reciter",
        )
        
        assert isinstance(output, AlignmentOutput)
        assert output.surah_id == 1
        assert output.reciter == "Test Reciter"
        assert len(output.results) == 1
        
        formatted = output.results[0]
        assert formatted.id == 1
        assert formatted.sura_id == 1
        assert formatted.ayah_index == 0
        assert formatted.start == 0.0
        assert formatted.end == 5.32
        assert formatted.transcribed_text == "بسم الله الرحمن الرحيم"
        assert formatted.corrected_text == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        assert formatted.similarity_score == 0.95

    def test_format_multiple_results(self):
        """Test formatting multiple alignment results."""
        results = [
            self.create_alignment_result(
                ayah_id=i,
                surah_id=1,
                ayah_number=i,
                text=f"Ayah {i} text",
                start_time=float(i),
                end_time=float(i + 1),
                transcribed_text=f"Transcribed {i}",
                similarity_score=0.9,
            )
            for i in range(1, 4)
        ]
        
        output = format_alignment_results(
            results=results,
            surah_id=1,
            reciter="Test",
        )
        
        assert len(output.results) == 3
        assert output.total_ayahs == 3

    def test_format_empty_results(self):
        """Test formatting empty results list."""
        output = format_alignment_results(
            results=[],
            surah_id=1,
            reciter="Test",
        )
        
        assert len(output.results) == 0
        assert output.total_ayahs == 0
        assert output.avg_similarity == 0.0

    def test_ayah_index_calculation(self):
        """Test that ayah_index is calculated correctly (ayah_number - 1)."""
        result = self.create_alignment_result(
            ayah_id=5,
            surah_id=2,
            ayah_number=5,  # ayah_index should be 4
            text="test",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="test",
            similarity_score=0.9,
        )
        
        output = format_alignment_results(
            results=[result],
            surah_id=2,
        )
        
        assert output.results[0].ayah_index == 4

    def test_rounding(self):
        """Test that times and scores are rounded correctly."""
        result = self.create_alignment_result(
            ayah_id=1,
            surah_id=1,
            ayah_number=1,
            text="test",
            start_time=1.123456,  # Should round to 1.12
            end_time=5.678901,    # Should round to 5.68
            transcribed_text="test",
            similarity_score=0.987654,  # Should round to 0.988
        )
        
        output = format_alignment_results(
            results=[result],
            surah_id=1,
        )
        
        formatted = output.results[0]
        assert formatted.start == 1.12
        assert formatted.end == 5.68
        assert formatted.similarity_score == 0.988

    def test_with_metadata(self):
        """Test formatting with metadata."""
        result = self.create_alignment_result(
            ayah_id=1,
            surah_id=1,
            ayah_number=1,
            text="test",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="test",
            similarity_score=0.9,
        )
        
        metadata = {"audio_file": "test.wav", "quality": "high"}
        
        output = format_alignment_results(
            results=[result],
            surah_id=1,
            reciter="Test",
            metadata=metadata,
        )
        
        assert output.metadata == metadata

    def test_custom_version(self):
        """Test formatting with custom version."""
        result = self.create_alignment_result(
            ayah_id=1,
            surah_id=1,
            ayah_number=1,
            text="test",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="test",
            similarity_score=0.9,
        )
        
        output = format_alignment_results(
            results=[result],
            surah_id=1,
            version="2.0.0",
        )
        
        assert output.version == "2.0.0"

    def test_default_reciter(self):
        """Test that default reciter is 'Unknown'."""
        result = self.create_alignment_result(
            ayah_id=1,
            surah_id=1,
            ayah_number=1,
            text="test",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="test",
            similarity_score=0.9,
        )
        
        output = format_alignment_results(
            results=[result],
            surah_id=1,
        )
        
        assert output.reciter == "Unknown"

    def test_end_to_end_json_output(self):
        """Test end-to-end JSON output generation."""
        results = [
            self.create_alignment_result(
                ayah_id=i,
                surah_id=1,
                ayah_number=i,
                text=f"Text {i}",
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                transcribed_text=f"Transcribed {i}",
                similarity_score=0.85 + (i * 0.05),
            )
            for i in range(1, 4)
        ]
        
        output = format_alignment_results(
            results=results,
            surah_id=1,
            reciter="Test Reciter",
            metadata={"test": True},
        )
        
        json_str = output.to_json()
        data = json.loads(json_str)
        
        # Verify structure
        assert "version" in data
        assert "generated_at" in data
        assert data["surah_id"] == 1
        assert data["reciter"] == "Test Reciter"
        assert data["total_ayahs"] == 3
        assert "avg_similarity" in data
        assert "high_confidence_count" in data
        assert "results" in data
        assert "metadata" in data
        
        # Verify results
        assert len(data["results"]) == 3
        for i, result in enumerate(data["results"]):
            assert "id" in result
            assert "sura_id" in result
            assert "ayah_index" in result
            assert "start" in result
            assert "end" in result
            assert "transcribed_text" in result
            assert "corrected_text" in result
            assert "similarity_score" in result
            assert "duration" in result
            assert "high_confidence" in result
