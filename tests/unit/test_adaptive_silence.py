"""
Unit tests for adaptive silence detection.
"""

from unittest.mock import patch

import pytest
from munajjam.transcription.silence import (
    _RELAXATION_STEPS,
    _adaptive_retry,
    detect_non_silent_chunks,
)


class TestAdaptiveRetry:
    """Test the adaptive retry logic for silence detection."""

    def test_no_retry_when_enough_chunks(self):
        initial_chunks = [(0, 1000), (1500, 2500), (3000, 4000)]
        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial_chunks,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=4,
            min_chunk_ratio=0.5,
        )
        assert result == initial_chunks

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_retry_with_relaxed_thresholds(self, mock_detect):
        initial = [(0, 5000)]
        relaxed = [(0, 2000), (2500, 4000), (4500, 5000)]
        mock_detect.return_value = relaxed

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=6,
            min_chunk_ratio=0.5,
        )
        assert result == relaxed
        assert mock_detect.called

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_keeps_best_result(self, mock_detect):
        initial = [(0, 5000)]
        step1 = [(0, 2000), (3000, 5000)]
        step2 = [(0, 1000), (1500, 2500), (3000, 4000), (4500, 5000)]
        mock_detect.side_effect = [step1, step2]

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=8,
            min_chunk_ratio=0.5,
        )
        assert result == step2

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_stops_when_target_reached(self, mock_detect):
        initial = [(0, 5000)]
        good_result = [(0, 1000), (1500, 2500), (3000, 4000)]
        mock_detect.return_value = good_result

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=4,
            min_chunk_ratio=0.5,
        )
        assert result == good_result
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_max_retry_steps(self, mock_detect):
        mock_detect.return_value = [(0, 5000)]

        _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=[(0, 5000)],
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=10,
            min_chunk_ratio=0.5,
        )
        assert mock_detect.call_count == len(_RELAXATION_STEPS)

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_relaxation_parameters(self, mock_detect):
        mock_detect.return_value = [(0, 5000)]

        _adaptive_retry(
            audio_path="test.wav",
            initial_chunks=[(0, 5000)],
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=10,
            min_chunk_ratio=0.5,
        )

        calls = mock_detect.call_args_list
        assert calls[0].args[1] == 210  # 300 * 0.7
        assert calls[0].args[2] == -25  # -30 + 5
        assert calls[1].args[1] == 150  # 300 * 0.5
        assert calls[1].args[2] == -20  # -30 + 10
        assert calls[2].args[1] == 90   # 300 * 0.3
        assert calls[2].args[2] == -15  # -30 + 15


class TestDetectNonSilentChunksAdaptive:
    """Test the adaptive parameter on detect_non_silent_chunks."""

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_adaptive_false_skips_retry(self, mock_detect):
        mock_detect.return_value = [(0, 5000)]
        result = detect_non_silent_chunks(
            "dummy.wav", adaptive=False, expected_chunks=10
        )
        assert result == [(0, 5000)]
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_adaptive_without_expected_skips_retry(self, mock_detect):
        mock_detect.return_value = [(0, 5000)]
        result = detect_non_silent_chunks("dummy.wav", adaptive=True)
        assert result == [(0, 5000)]
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_adaptive_triggers_retry(self, mock_detect):
        initial = [(0, 5000)]
        better = [(0, 2000), (2500, 3500), (4000, 5000)]
        mock_detect.side_effect = [initial, better]

        result = detect_non_silent_chunks(
            "dummy.wav", adaptive=True, expected_chunks=4, min_chunk_ratio=0.5
        )
        assert result == better

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_min_silence_len_floor(self, mock_detect):
        mock_detect.return_value = [(0, 5000)]

        _adaptive_retry(
            audio_path="test.wav",
            initial_chunks=[(0, 5000)],
            initial_min_silence_len=100,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=10,
            min_chunk_ratio=0.5,
        )

        for call in mock_detect.call_args_list:
            assert call.args[1] >= 50


class TestAdaptiveEdgeCases:
    """Regression tests for odd/single-chunk edge cases."""

    def test_single_chunk_below_target_triggers_retry(self):
        single = [(0, 10000)]
        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=single,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=1,
            min_chunk_ratio=0.5,
        )
        assert result == single

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_single_expected_chunk_no_retry(self, mock_detect):
        single = [(0, 10000)]
        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=single,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=1,
            min_chunk_ratio=0.5,
        )
        assert result == single
        mock_detect.assert_not_called()

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_odd_expected_chunks_ceil_target(self, mock_detect):
        initial = [(0, 5000)]
        relaxed = [(0, 2000), (2500, 4000), (4500, 5000)]
        mock_detect.return_value = relaxed

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=5,
            min_chunk_ratio=0.5,
        )
        # math.ceil(5 * 0.5) = 3, relaxed has 3 chunks -> should meet target
        assert result == relaxed
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_three_expected_chunks_needs_two(self, mock_detect):
        initial = [(0, 5000)]
        two_chunks = [(0, 2000), (3000, 5000)]
        mock_detect.return_value = two_chunks

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=initial,
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=3,
            min_chunk_ratio=0.5,
        )
        # math.ceil(3 * 0.5) = 2, two_chunks has 2 -> meets target
        assert result == two_chunks
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_empty_initial_chunks_retries(self, mock_detect):
        mock_detect.return_value = []

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=[],
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=1,
            min_chunk_ratio=0.5,
        )
        assert result == []
        assert mock_detect.call_count == len(_RELAXATION_STEPS)

    @patch("munajjam.transcription.silence._run_non_silent_detection")
    def test_empty_initial_retries_when_expected_nonzero(self, mock_detect):
        better = [(0, 3000), (4000, 6000)]
        mock_detect.return_value = better

        result = _adaptive_retry(
            audio_path="dummy.wav",
            initial_chunks=[],
            initial_min_silence_len=300,
            initial_silence_thresh=-30,
            use_fast=True,
            expected_chunks=4,
            min_chunk_ratio=0.5,
        )
        assert result == better
