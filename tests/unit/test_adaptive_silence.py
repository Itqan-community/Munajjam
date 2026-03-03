"""
Unit tests for adaptive silence detection.

Tests the retry logic without requiring real audio files by mocking
the underlying detection functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from munajjam.transcription.silence import (
    detect_non_silent_chunks_adaptive,
    detect_silences_adaptive,
)


class TestDetectNonSilentChunksAdaptive:
    """Tests for detect_non_silent_chunks_adaptive."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_sufficient_chunks_on_first_attempt(self, mock_detect):
        """When enough chunks are found immediately, no retry happens."""
        mock_detect.return_value = [(0, 1000), (1500, 3000), (3500, 5000)]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav", expected_chunks=3
        )

        assert len(chunks) == 3
        assert meta["retries_used"] == 0
        assert meta["adapted"] is False
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_retry_relaxes_thresholds(self, mock_detect):
        """When too few chunks, retries with relaxed thresholds."""
        mock_detect.side_effect = [
            [(0, 5000)],  # 1 chunk - too few
            [(0, 2000), (3000, 5000)],  # 2 chunks - still too few
            [(0, 1000), (1500, 3000), (3500, 5000), (5500, 7000)],  # 4 chunks
        ]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=4,
            min_chunk_ratio=1.0,
            min_silence_len=300,
            silence_thresh=-30,
        )

        assert len(chunks) == 4
        assert meta["retries_used"] == 2
        assert meta["adapted"] is True

        calls = mock_detect.call_args_list
        assert calls[0].kwargs["silence_thresh"] == -30
        assert calls[0].kwargs["min_silence_len"] == 300
        assert calls[1].kwargs["silence_thresh"] == -25
        assert calls[1].kwargs["min_silence_len"] == 250
        assert calls[2].kwargs["silence_thresh"] == -20
        assert calls[2].kwargs["min_silence_len"] == 200

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_max_retries_exhausted(self, mock_detect):
        """When max retries are exhausted, returns best result found."""
        mock_detect.side_effect = [
            [(0, 5000)],
            [(0, 2500), (3000, 5000)],
            [(0, 1500), (2000, 3500), (4000, 5000)],
            [(0, 1000), (1500, 3000), (3500, 5000)],
        ]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=10,
            min_chunk_ratio=1.0,
            max_retries=3,
        )

        assert meta["retries_used"] == 3
        assert meta["adapted"] is True
        assert meta["chunk_count"] == 3
        assert len(chunks) == 3

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_min_chunk_ratio(self, mock_detect):
        """Accepts result when chunks >= expected * min_chunk_ratio."""
        mock_detect.return_value = [(0, 1000), (2000, 3000), (4000, 5000)]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=5,
            min_chunk_ratio=0.5,
        )

        assert len(chunks) == 3
        assert meta["retries_used"] == 0
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_threshold_floor_respected(self, mock_detect):
        """Silence threshold should not exceed -5 dB."""
        mock_detect.return_value = [(0, 5000)]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=20,
            silence_thresh=-10,
            silence_thresh_step=10,
            max_retries=3,
        )

        calls = mock_detect.call_args_list
        for call in calls:
            assert call.kwargs["silence_thresh"] <= -5

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_min_silence_len_floor_respected(self, mock_detect):
        """Min silence length should not go below 100 ms."""
        mock_detect.return_value = [(0, 5000)]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=20,
            min_silence_len=150,
            silence_len_step=100,
            max_retries=3,
        )

        calls = mock_detect.call_args_list
        for call in calls:
            assert call.kwargs["min_silence_len"] >= 100

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_keeps_best_result(self, mock_detect):
        """Returns the result with the most chunks even if it wasn't the last."""
        mock_detect.side_effect = [
            [(0, 1000)],
            [(0, 500), (1000, 1500), (2000, 2500)],  # best: 3 chunks
            [(0, 1000), (2000, 3000)],  # worse: 2 chunks
        ]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=10,
            max_retries=2,
        )

        assert len(chunks) == 3
        assert meta["chunk_count"] == 3

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_custom_step_values(self, mock_detect):
        """Custom step values are applied correctly."""
        mock_detect.side_effect = [
            [(0, 5000)],
            [(0, 2000), (3000, 5000)],
        ]

        chunks, meta = detect_non_silent_chunks_adaptive(
            "test.wav",
            expected_chunks=2,
            min_chunk_ratio=1.0,
            min_silence_len=400,
            silence_thresh=-40,
            silence_len_step=100,
            silence_thresh_step=10,
        )

        calls = mock_detect.call_args_list
        assert calls[1].kwargs["silence_thresh"] == -30
        assert calls[1].kwargs["min_silence_len"] == 300


class TestDetectSilencesAdaptive:
    """Tests for detect_silences_adaptive."""

    @patch("munajjam.transcription.silence.detect_silences")
    def test_sufficient_silences_on_first_attempt(self, mock_detect):
        """No retry when enough silences found immediately."""
        mock_detect.return_value = [
            (1000, 1500),
            (3000, 3500),
            (5000, 5500),
        ]

        silences, meta = detect_silences_adaptive(
            "test.wav", expected_silences=3
        )

        assert len(silences) == 3
        assert meta["retries_used"] == 0
        assert meta["adapted"] is False
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_silences")
    def test_retry_with_relaxed_thresholds(self, mock_detect):
        """Retries with progressively relaxed thresholds."""
        mock_detect.side_effect = [
            [(1000, 1500)],
            [(1000, 1500), (3000, 3300), (5000, 5200)],
        ]

        silences, meta = detect_silences_adaptive(
            "test.wav",
            expected_silences=3,
            min_silence_ratio=1.0,
            min_silence_len=300,
            silence_thresh=-30,
        )

        assert len(silences) == 3
        assert meta["retries_used"] == 1
        assert meta["adapted"] is True

        calls = mock_detect.call_args_list
        assert calls[0].kwargs["silence_thresh"] == -30
        assert calls[1].kwargs["silence_thresh"] == -25
        assert calls[1].kwargs["min_silence_len"] == 250

    @patch("munajjam.transcription.silence.detect_silences")
    def test_max_retries_returns_best(self, mock_detect):
        """Returns best result when max retries exhausted."""
        mock_detect.side_effect = [
            [],
            [(1000, 1500)],
            [(1000, 1500), (3000, 3200)],
            [(1000, 1300)],
        ]

        silences, meta = detect_silences_adaptive(
            "test.wav",
            expected_silences=10,
            max_retries=3,
        )

        assert len(silences) == 2
        assert meta["silence_count"] == 2
        assert meta["retries_used"] == 3


class TestAdaptiveBackwardsCompatibility:
    """Verify adaptive functions are backwards-compatible opt-in."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_default_params_match_original(self, mock_detect):
        """Default params use the same values as the original function."""
        mock_detect.return_value = [(0, 5000)]

        detect_non_silent_chunks_adaptive("test.wav", expected_chunks=1)

        call = mock_detect.call_args_list[0]
        assert call.kwargs["min_silence_len"] == 300
        assert call.kwargs["silence_thresh"] == -30
        assert call.kwargs["use_fast"] is True

    @patch("munajjam.transcription.silence.detect_silences")
    def test_silences_default_params_match(self, mock_detect):
        """Default params use the same values as the original function."""
        mock_detect.return_value = [(1000, 1500)]

        detect_silences_adaptive("test.wav", expected_silences=1)

        call = mock_detect.call_args_list[0]
        assert call.kwargs["min_silence_len"] == 300
        assert call.kwargs["silence_thresh"] == -30
        assert call.kwargs["use_fast"] is True
