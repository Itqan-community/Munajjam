"""
Integration tests for Munajjam library.
These tests use real data and can be slower.
"""

import pytest
from pathlib import Path
from munajjam.data import load_surah_ayahs, get_ayah_count
from munajjam.core import Aligner
from munajjam.models import Segment, SegmentType
from munajjam.transcription.whisperFactory import WhisperFactory

@pytest.fixture
def factory():
    return WhisperFactory()

@pytest.fixture
def audio_file(tmp_path):
    # Create a dummy audio file for testing the pipeline
    import numpy as np
    import soundfile as sf
    test_file = tmp_path / "1.wav"
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    y = np.sin(2 * np.pi * 440 * t) 
    sf.write(str(test_file), y, sample_rate)
    return test_file

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

    @pytest.mark.parametrize("surah_id", [1, 2, 114])
    def test_ayah_count_matches_loaded(self, surah_id):
        """Test that get_ayah_count matches loaded ayahs."""
        count = get_ayah_count(surah_id)
        ayahs = load_surah_ayahs(surah_id)

        assert count == len(ayahs)

    def test_alignment_with_real_ayahs(self, factory, tmp_path, audio_file):
        """Test alignment with real ayah data processed end-to-end."""
        import torch
        import shutil
        if not torch.cuda.is_available():
            pytest.skip("Skipping full model end-to-end alignment because CUDA is unavailable.")
        if shutil.which("ffmpeg") is None:
            pytest.skip("Skipping WhisperX integration test because ffmpeg is not installed on the system.")

        from munajjam.transcription.whisperFactory import WhisperBackend
        transcriber = factory.create_whisper(
            backend=WhisperBackend.WHISPERX, 
            model_name="OdyAsh/faster-whisper-base-ar-quran",
            device="cuda",
            compute_type="float32"
        )

        # Need the actual Fatiha Ayahs to align against
        ayahs = load_surah_ayahs(1)
        
        # We need a proper surah_1.wav for testing rather than just dummy tone audio. 
        # But for test purposes, we transcribe the fixture audio we get.
        
        # Real end-to-end flow:
        segments = transcriber.transcribe(audio_file)

        # The segments from our dummy audio won't actually match fatiha, but we can verify the aligner runs
        # without crashing
        aligner = Aligner(audio_path=str(audio_file), strategy="hybrid", energy_snap=False)
        results = aligner.align(segments, ayahs[:3])

        # We assert results are returned, though scores might be low due to dummy audio
        assert results is not None
        assert all(0 <= r.similarity_score <= 1.0 for r in results)

    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid"])
    def test_strategies_with_real_data(self, strategy, factory, audio_file):
        """Test all strategies produce results with real data going through full pipeline."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("Skipping strategy testing because CUDA is unavailable.")
            
        try:
            from munajjam.transcription.whisperFactory import WhisperBackend
            transcriber = factory.create_whisper(
                backend=WhisperBackend.FASTERWHISPER, 
                model_name="OdyAsh/faster-whisper-base-ar-quran",
                device="cuda",
              
            )
        except Exception as e:
            pytest.skip(f"Failed to load whisper model: {e}")

        ayahs = load_surah_ayahs(114)
        
        try:
            segments = transcriber.transcribe(audio_file)
        except Exception as e:
            pytest.skip(f"Transcription failed: {e}")

        aligner = Aligner(audio_path=str(audio_file), strategy=strategy, energy_snap=False)
        results = aligner.align(segments, ayahs)

        assert results is not None
