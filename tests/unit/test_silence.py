"""
Unit tests for silence detection.
"""


import pytest
from unittest.mock import patch
from munajjam.transcription.silence import detect_silences, detect_non_silent_chunks




class TestDetectSilences:
    """Test silence detection functions."""


    def test_detect_silences_file_not_found(self):
        """Test silence detection with non-existent file."""
        with pytest.raises(Exception):
            detect_silences("nonexistent_file.wav")


    def test_silence_tuple_format(self, sample_silences):
        """Test that silences are in (start_ms, end_ms) format."""
        for start, end in sample_silences:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end




class TestAdaptiveSilenceDetection:
    """Tests for adaptive silence detection retry logic."""


    def _make_chunks(self, n: int) -> list[tuple[int, int]]:
        """Helper: create n dummy non-silent chunks."""
        return [(i * 1000, i * 1000 + 500) for i in range(n)]


    def test_adaptive_disabled_by_default(self):
        """Non-adaptive call must not trigger retries regardless of chunk count."""
        few_chunks = self._make_chunks(1)


        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ):
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=False,
                expected_chunks=10,
            )


        # Only one call – no retries
        assert result == few_chunks


    def test_adaptive_no_retry_when_enough_chunks(self):
        """When chunk count already meets the threshold, no retry should occur."""
        enough_chunks = self._make_chunks(8)  # 8 >= 0.5 * 10 = 5


        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=enough_chunks,
        ):
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )


        assert result == enough_chunks


    def test_adaptive_retries_when_too_few_chunks(self):
        """When initial detection finds too few chunks, retries with relaxed thresholds."""
