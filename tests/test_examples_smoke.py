"""
Smoke tests for example scripts with mocked transcriber.

Validates that all example workflows (01-04) execute without errors
when the WhisperTranscriber and detect_silences are mocked out.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from munajjam.core import Aligner, AlignmentStrategy, align
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results
from munajjam.models import Segment, SegmentType

DUMMY_AUDIO = "test_audio.wav"
SURAH_NUMBER = 114


@pytest.fixture
def surah_114_ayahs():
    return load_surah_ayahs(SURAH_NUMBER)


@pytest.fixture
def mock_segments(surah_114_ayahs):
    segments = [
        Segment(
            id=0,
            surah_id=SURAH_NUMBER,
            start=0.0,
            end=2.0,
            text="أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            type=SegmentType.ISTIADHA,
        ),
        Segment(
            id=1,
            surah_id=SURAH_NUMBER,
            start=2.5,
            end=5.0,
            text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            type=SegmentType.BASMALA,
        ),
    ]
    time_cursor = 5.5
    for i, ayah in enumerate(surah_114_ayahs):
        start = time_cursor
        end = start + 2.5
        segments.append(
            Segment(
                id=i + 2,
                surah_id=SURAH_NUMBER,
                start=start,
                end=end,
                text=ayah.text,
                type=SegmentType.AYAH,
            )
        )
        time_cursor = end + 0.3
    return segments


@pytest.fixture
def mock_silences():
    return [(2000, 2500), (5000, 5500), (8300, 8600), (11100, 11400)]


class TestExample01BasicUsage:
    def test_basic_workflow(self, mock_segments, surah_114_ayahs, tmp_path):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_segments
        mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
        mock_transcriber.__exit__ = MagicMock(return_value=False)

        with patch(
            "munajjam.transcription.WhisperTranscriber",
            return_value=mock_transcriber,
        ):
            from munajjam.transcription import WhisperTranscriber

            with WhisperTranscriber() as transcriber:
                segments = transcriber.transcribe(DUMMY_AUDIO)

            assert len(segments) > 0
            assert segments[-1].end > 0

            ayahs = load_surah_ayahs(SURAH_NUMBER)
            assert len(ayahs) > 0

            results = align(DUMMY_AUDIO, segments, ayahs)
            assert len(results) > 0

            output = format_alignment_results(
                results=results,
                surah_id=SURAH_NUMBER,
                reciter="Test Reciter",
                audio_file=DUMMY_AUDIO,
            )

            json_str = output.to_json()
            parsed = json.loads(json_str)
            assert "metadata" in parsed
            assert "results" in parsed
            assert parsed["metadata"]["surah_id"] == SURAH_NUMBER

            output_file = tmp_path / "surah_114.json"
            output.to_file(str(output_file))
            assert output_file.exists()

            meta = output.metadata
            assert meta.average_confidence >= 0.0
            assert meta.high_confidence_count >= 0
            assert meta.total_duration >= 0.0


class TestExample02ComparingStrategies:
    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid", "auto"])
    def test_strategy_alignment(self, strategy, mock_segments, surah_114_ayahs):
        aligner = Aligner(
            audio_path=DUMMY_AUDIO,
            strategy=strategy,
            fix_drift=True,
            fix_overlaps=True,
            energy_snap=False,
        )
        results = aligner.align(mock_segments, surah_114_ayahs)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert hasattr(result, "ayah")
            assert hasattr(result, "start_time")
            assert hasattr(result, "end_time")
            assert hasattr(result, "similarity_score")
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.start_time >= 0
            assert result.end_time > result.start_time

    def test_strategy_comparison_workflow(self, mock_segments, surah_114_ayahs, tmp_path):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_segments
        mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
        mock_transcriber.__exit__ = MagicMock(return_value=False)

        strategies = ["greedy", "dp", "hybrid", "auto"]
        results_map = {}

        for strategy in strategies:
            aligner = Aligner(
                audio_path=DUMMY_AUDIO,
                strategy=strategy,
                fix_drift=True,
                fix_overlaps=True,
                energy_snap=False,
            )
            results = aligner.align(mock_segments, surah_114_ayahs)
            avg_similarity = sum(r.similarity_score for r in results) / len(results)
            high_confidence = len([r for r in results if r.is_high_confidence])
            overlaps = sum(r.overlap_detected for r in results)

            results_map[strategy] = {
                "results": results,
                "avg_similarity": avg_similarity,
                "high_confidence": high_confidence,
                "overlaps": overlaps,
            }

        assert len(results_map) == len(strategies)
        for strategy, data in results_map.items():
            assert len(data["results"]) > 0
            assert 0.0 <= data["avg_similarity"] <= 1.0

        most_accurate = max(strategies, key=lambda s: results_map[s]["avg_similarity"])
        best_results = results_map[most_accurate]["results"]
        output = format_alignment_results(
            results=best_results,
            surah_id=SURAH_NUMBER,
            audio_file=DUMMY_AUDIO,
        )
        output_path = tmp_path / f"surah_{SURAH_NUMBER:03d}_best_alignment.json"
        output.to_file(str(output_path))
        assert output_path.exists()


class TestExample03AdvancedConfiguration:
    def test_configure_and_silence_detection(
        self, mock_segments, surah_114_ayahs, mock_silences
    ):
        from munajjam.config import configure

        configure(
            model_id="OdyAsh/faster-whisper-base-ar-quran",
            device="auto",
            model_type="faster-whisper",
            silence_threshold_db=-30,
            min_silence_ms=300,
            buffer_seconds=0.3,
        )

        with patch(
            "munajjam.transcription.detect_silences",
            return_value=mock_silences,
        ):
            from munajjam.transcription import detect_silences

            silences_ms = detect_silences(
                audio_path=DUMMY_AUDIO,
                min_silence_len=300,
                silence_thresh=-30,
            )
            assert len(silences_ms) > 0
            total_silence = sum(end - start for start, end in silences_ms) / 1000
            assert total_silence > 0

    def test_advanced_alignment_with_silences(
        self, mock_segments, surah_114_ayahs, mock_silences, tmp_path
    ):
        ayah_segments = [s for s in mock_segments if s.type == SegmentType.AYAH]
        istiadha_segments = [s for s in mock_segments if s.type == SegmentType.ISTIADHA]
        basmala_segments = [s for s in mock_segments if s.type == SegmentType.BASMALA]

        assert len(ayah_segments) > 0
        assert len(istiadha_segments) >= 0
        assert len(basmala_segments) >= 0

        aligner = Aligner(
            audio_path=DUMMY_AUDIO,
            strategy="auto",
            quality_threshold=0.85,
            fix_drift=True,
            fix_overlaps=True,
            energy_snap=False,
        )

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        results = aligner.align(
            segments=mock_segments,
            ayahs=surah_114_ayahs,
            silences_ms=mock_silences,
            on_progress=progress_callback,
        )

        assert len(results) > 0

        excellent = [r for r in results if r.similarity_score >= 0.95]
        good = [r for r in results if 0.85 <= r.similarity_score < 0.95]
        fair = [r for r in results if 0.70 <= r.similarity_score < 0.85]
        poor = [r for r in results if r.similarity_score < 0.70]
        assert len(excellent) + len(good) + len(fair) + len(poor) == len(results)

        output = format_alignment_results(
            results=results,
            surah_id=SURAH_NUMBER,
            audio_file=DUMMY_AUDIO,
        )
        output_path = tmp_path / f"surah_{SURAH_NUMBER:03d}_alignment.json"
        output.to_file(str(output_path))
        assert output_path.exists()

    def test_hybrid_stats_available(self, mock_segments, surah_114_ayahs):
        aligner = Aligner(
            audio_path=DUMMY_AUDIO,
            strategy="hybrid",
            energy_snap=False,
        )
        aligner.align(mock_segments, surah_114_ayahs)
        stats = aligner.last_stats
        if stats is not None:
            assert hasattr(stats, "total_ayahs")
            assert hasattr(stats, "dp_kept")
            assert hasattr(stats, "old_fallback")
            assert hasattr(stats, "split_improved")
            assert hasattr(stats, "still_low")


class TestExample04BatchProcessing:
    def test_batch_single_surah(self, mock_segments, surah_114_ayahs, tmp_path):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_segments
        mock_transcriber.load = MagicMock()
        mock_transcriber.unload = MagicMock()

        surahs_to_process = [SURAH_NUMBER]
        output_directory = tmp_path / "output_examples"
        output_directory.mkdir(exist_ok=True)

        all_stats = []

        for surah_number in surahs_to_process:
            segments = mock_transcriber.transcribe(DUMMY_AUDIO)
            ayahs = load_surah_ayahs(surah_number)

            aligner = Aligner(audio_path=DUMMY_AUDIO, energy_snap=False)
            results = aligner.align(segments, ayahs)

            avg_similarity = sum(r.similarity_score for r in results) / len(results)
            high_confidence = len([r for r in results if r.is_high_confidence])
            overlaps = sum(r.overlap_detected for r in results)

            stats = {
                "surah_number": surah_number,
                "total_ayahs": len(results),
                "avg_similarity": round(avg_similarity, 4),
                "high_confidence_count": high_confidence,
                "high_confidence_pct": round(high_confidence / len(results), 4),
                "overlaps": overlaps,
            }
            all_stats.append(stats)

            output = format_alignment_results(
                results=results,
                surah_id=surah_number,
                audio_file=DUMMY_AUDIO,
            )
            output_file = output_directory / f"surah_{surah_number:03d}_alignment.json"
            output.to_file(str(output_file))

        mock_transcriber.unload.assert_not_called()
        mock_transcriber.unload()
        mock_transcriber.unload.assert_called_once()

        assert len(all_stats) == len(surahs_to_process)
        for stats in all_stats:
            assert stats["total_ayahs"] > 0
            assert 0.0 <= stats["avg_similarity"] <= 1.0

        total_ayahs = sum(s["total_ayahs"] for s in all_stats)
        assert total_ayahs > 0

        summary_data = {
            "processed_surahs": len(all_stats),
            "failed_surahs": [],
            "overall_stats": {
                "total_ayahs": total_ayahs,
                "avg_similarity": round(
                    sum(s["avg_similarity"] for s in all_stats) / len(all_stats), 4
                ),
                "total_overlaps": sum(s["overlaps"] for s in all_stats),
            },
            "per_surah_stats": all_stats,
        }

        summary_file = output_directory / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        assert summary_file.exists()

        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        assert loaded["processed_surahs"] == 1

    def test_batch_error_handling(self, mock_segments, tmp_path):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = RuntimeError("Model not loaded")

        failed_surahs = []
        try:
            mock_transcriber.transcribe(DUMMY_AUDIO)
        except Exception:
            failed_surahs.append(SURAH_NUMBER)

        assert SURAH_NUMBER in failed_surahs
