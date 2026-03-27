"""
Integration tests for Munajjam library using real Quran data.
"""

import pytest
import torch
from pathlib import Path
from munajjam.data import load_surah_ayahs
from munajjam.core import Aligner
from munajjam.transcription.whisperFactory import WhisperFactory, WhisperBackend


@pytest.fixture
def factory():
    return WhisperFactory()


@pytest.fixture
def real_audio():
    """Returns path to real Surah 1 audio fixture."""
    path = Path(__file__).parent.parent / "fixtures" / "surah_001.mp3"
    if not path.exists():
        pytest.skip(
            f"Real audio fixture not found at {path}. Run download_fixtures.py first."
        )
    return path


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataAlignment:
    """Integration tests with real Quran data."""

    @pytest.mark.parametrize(
        "surah_id,expected_count",
        [
            (1, 7),
            (114, 6),
        ],
    )
    def test_load_real_surah(self, surah_id, expected_count):
        """Test loading real surah ayahs."""
        ayahs = load_surah_ayahs(surah_id)

        assert len(ayahs) == expected_count
        assert ayahs[0].surah_id == surah_id
        assert ayahs[0].ayah_number == 1

    def test_alignment_whisperx_end_to_end(self, factory, real_audio):
        """Test alignment with real data using WhisperX backend."""
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is required for WhisperX but not found in PATH.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float32"

        transcriber = factory.create_whisper(
            backend=WhisperBackend.WHISPERX,
            model_name="OdyAsh/faster-whisper-base-ar-quran",
            device=device,
            compute_type=compute_type,
        )

        ayahs = load_surah_ayahs(1)
        segments = transcriber.transcribe(str(real_audio), surah_id=1)

        aligner = Aligner(audio_path=str(real_audio), strategy="hybrid")
        # Align just the first ayah for speed in integration tests
        results = aligner.align(segments, ayahs[:1])

        assert results is not None
        assert len(results) > 0
        assert 0.5 <= results[0].similarity_score <= 1.0

    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid"])
    def test_strategies_end_to_end(self, strategy, factory, real_audio):
        """Test all strategies produce results with real data through FasterWhisper."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        transcriber = factory.create_whisper(
            backend=WhisperBackend.FASTERWHISPER,
            model_name="OdyAsh/faster-whisper-base-ar-quran",
            device=device,
        )

        ayahs = load_surah_ayahs(1)
        segments = transcriber.transcribe(str(real_audio), surah_id=1)

        aligner = Aligner(audio_path=str(real_audio), strategy=strategy)
        results = aligner.align(segments, ayahs[:1])

        assert results is not None
        assert len(results) > 0
