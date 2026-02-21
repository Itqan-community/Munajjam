"""
Unit tests for silence detection.
"""

import pytest
from unittest.mock import patch, MagicMock
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
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=False,
                expected_chunks=10,
            )

        # Only one call – no retries
        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_no_retry_when_enough_chunks(self):
        """When chunk count already meets the threshold, no retry should occur."""
        enough_chunks = self._make_chunks(8)  # 8 >= 0.5 * 10 = 5

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=enough_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        assert mock_raw.call_count == 1
        assert result == enough_chunks

    def test_adaptive_retries_when_too_few_chunks(self):
        """When initial detection finds too few chunks, retries with relaxed thresholds."""
        few_chunks = self._make_chunks(2)   # 2 < 0.5 * 10 = 5  → retry
        enough_chunks = self._make_chunks(6)  # 6 >= 5            → stop

        call_results = [few_chunks, enough_chunks]
        call_index = {"i": 0}

        def side_effect(*args, **kwargs):
            result = call_results[min(call_index["i"], len(call_results) - 1)]
            call_index["i"] += 1
            return result

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        # First call (initial) + exactly one retry
        assert mock_raw.call_count == 2
        assert result == enough_chunks

    def test_adaptive_retry_uses_relaxed_thresholds(self):
        """Retry calls must use a higher (less negative) silence_thresh."""
        few_chunks = self._make_chunks(1)

        recorded_thresholds: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_thresholds.append(silence_thresh)
            return few_chunks  # Always return few chunks to exhaust all retries

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                silence_thresh=-30,
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        # First threshold is the original; subsequent thresholds must be higher
        assert recorded_thresholds[0] == -30
        for i in range(1, len(recorded_thresholds)):
            assert recorded_thresholds[i] > recorded_thresholds[i - 1], (
                f"Retry {i} threshold {recorded_thresholds[i]} is not higher than "
                f"previous {recorded_thresholds[i - 1]}"
            )

    def test_adaptive_retry_uses_shorter_min_silence_len(self):
        """Retry calls must use a shorter min_silence_len."""
        few_chunks = self._make_chunks(1)

        recorded_lens: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_lens.append(min_silence_len)
            return few_chunks

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                min_silence_len=300,
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        assert recorded_lens[0] == 300
        for i in range(1, len(recorded_lens)):
            assert recorded_lens[i] < recorded_lens[i - 1], (
                f"Retry {i} min_silence_len {recorded_lens[i]} is not shorter than "
                f"previous {recorded_lens[i - 1]}"
            )

    def test_adaptive_min_silence_len_never_below_50ms(self):
        """min_silence_len must never drop below 50 ms during retries."""
        few_chunks = self._make_chunks(1)

        recorded_lens: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_lens.append(min_silence_len)
            return few_chunks

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                min_silence_len=100,  # Very short initial value
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        for length in recorded_lens:
            assert length >= 50, f"min_silence_len {length} dropped below 50 ms"

    def test_adaptive_exhausts_all_retries_gracefully(self):
        """When all retries fail to find enough chunks, return the best result."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=100,
                min_chunks_ratio=0.9,  # Very strict: need 90 chunks
            )

        # Should have tried initial + 4 retry levels = 5 total calls
        assert mock_raw.call_count == 5
        # Returns whatever was found (graceful degradation)
        assert result == few_chunks

    def test_adaptive_ignored_when_expected_chunks_none(self):
        """When expected_chunks is None, adaptive mode is effectively disabled."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=None,
            )

        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_ignored_when_expected_chunks_zero(self):
        """When expected_chunks is 0, adaptive mode is effectively disabled."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=0,
            )

        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_backwards_compatible_signature(self):
        """Calling detect_non_silent_chunks without new args must behave as before."""
        chunks = self._make_chunks(5)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks("dummy.wav")

        assert mock_raw.call_count == 1
        assert result == chunks

    def test_adaptive_min_required_never_zero(self):
        """min_required should be at least 1 to prevent the retry from being silently skipped."""
        # Without the max(1, ...) fix, int(0.5 * 1) = 0, so len([]) >= 0 is
        # always True and retries never fire.  With the fix, min_required = 1
        # and an empty initial result triggers retries.
        empty_chunks: list[tuple[int, int]] = []

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=empty_chunks,
        ) as mock_raw:
            detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=1,       # int(0.5 * 1) = 0 → retry never fires without the fix
                min_chunks_ratio=0.5,
            )

        # Should still attempt retries even for tiny expected_chunks
        # because min_required is clamped to at least 1
        assert mock_raw.call_count > 1

    def test_adaptive_returns_best_result(self):
        """Adaptive retry should return the best result (most chunks) across all attempts."""
        initial_chunks = self._make_chunks(1)
        better_chunks = self._make_chunks(3)
        worse_chunks = self._make_chunks(2)

        call_index = {"i": 0}
        call_results = [initial_chunks, better_chunks, worse_chunks, worse_chunks, worse_chunks]

        def side_effect(*args, **kwargs):
            result = call_results[min(call_index["i"], len(call_results) - 1)]
            call_index["i"] += 1
            return result

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        # Should return the best result (3 chunks), not the last one (2 chunks)
        assert result == better_chunks

    def test_adaptive_thresh_capped_at_minus_10(self):
        """Relaxed threshold should never exceed -10 dBFS to avoid classifying speech as silence."""
        few_chunks = self._make_chunks(1)

        recorded_thresholds: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_thresholds.append(silence_thresh)
            return few_chunks

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                silence_thresh=-15,  # Close to cap
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        for thresh in recorded_thresholds[1:]:  # Skip initial call
            assert thresh >= -10 or thresh <= -10, "Threshold should be capped"
            assert thresh <= -10, (
                f"Threshold {thresh} exceeded -10 dBFS cap"
            )

    def test_adaptive_raises_on_invalid_min_chunks_ratio(self):
        """Adaptive mode should raise ValueError when min_chunks_ratio <= 0."""
        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=self._make_chunks(1),
        ):
            with pytest.raises(ValueError, match="min_chunks_ratio must be > 0"):
                detect_non_silent_chunks(
                    "dummy.wav",
                    adaptive=True,
                    expected_chunks=10,
                    min_chunks_ratio=0,
                )
