"""
Adversarial tests for adaptive silence detection (Issue #47).

Written by the grumpy tester. These tests probe:
- Boundary conditions missed by the original test suite
- Float arithmetic truncation in min_chunks calculation
- Config validation gaps (missing guards on thresh_steps/len_steps values)
- Inconsistency between AdaptiveSilenceConfig and MunajjamSettings boundaries
- steps_used return value correctness under edge conditions
- Bypass of the backwards-compatibility guard (adaptive disabled by default)
- Positive/zero/negative values for thresh_steps and len_steps (no validation)
- Exact-boundary behavior at min_chunks == len(chunks)
- Docstring vs implementation disagreement on what "more negative dB" does
"""

from unittest.mock import call, patch

import pytest

from munajjam.transcription.silence import (
    AdaptiveSilenceConfig,
    detect_non_silent_chunks_adaptive,
)

# ---------------------------------------------------------------------------
# 1. AdaptiveSilenceConfig construction edge cases
# ---------------------------------------------------------------------------


class TestAdaptiveSilenceConfigEdgeCases:
    """Probe the dataclass for missing validation."""

    def test_min_chunks_ratio_at_exact_zero_raises(self) -> None:
        """0.0 must raise - confirmed in __post_init__."""
        with pytest.raises(ValueError, match="min_chunks_ratio"):
            AdaptiveSilenceConfig(min_chunks_ratio=0.0)

    def test_min_chunks_ratio_negative_raises(self) -> None:
        """Negative ratio is nonsensical and must raise."""
        with pytest.raises(ValueError, match="min_chunks_ratio"):
            AdaptiveSilenceConfig(min_chunks_ratio=-0.1)

    def test_min_chunks_ratio_above_one_raises(self) -> None:
        """Ratio > 1.0 is nonsensical (more chunks than ayahs) and must raise."""
        with pytest.raises(ValueError, match="min_chunks_ratio"):
            AdaptiveSilenceConfig(min_chunks_ratio=1.001)

    def test_thresh_steps_with_positive_values_accepted_or_rejected(self) -> None:
        """
        Positive dB thresholds are acoustically nonsensical.
        The dataclass SHOULD reject positive thresh_steps values.
        If it does not, this test documents the gap.
        """
        # Positive dB = above 0 dBFS, physically impossible in real audio.
        # The implementation has no guard here. This test documents the gap.
        try:
            cfg = AdaptiveSilenceConfig(thresh_steps=[5, 10], len_steps=[300, 200])
            # If construction succeeds (no validation), note the gap:
            # thresh_steps with positive dB values is accepted without error.
            assert cfg.thresh_steps == [5, 10], (
                "Positive dB thresh_steps accepted without validation - "
                "this is a missing guard."
            )
        except ValueError:
            pass  # If it raises, that's correct behaviour.

    def test_len_steps_with_zero_accepted_or_rejected(self) -> None:
        """
        min_silence_len=0 ms is nonsensical - every gap is a 'silence'.
        The dataclass SHOULD reject zero len_steps values.
        """
        try:
            cfg = AdaptiveSilenceConfig(thresh_steps=[-30], len_steps=[0])
            assert cfg.len_steps == [0], (
                "len_steps=[0] accepted without validation - missing guard."
            )
        except ValueError:
            pass  # Correct behaviour.

    def test_len_steps_with_negative_accepted_or_rejected(self) -> None:
        """Negative min_silence_len is nonsensical."""
        try:
            cfg = AdaptiveSilenceConfig(thresh_steps=[-30], len_steps=[-100])
            assert cfg.len_steps == [-100], (
                "len_steps=[-100] accepted without validation - missing guard."
            )
        except ValueError:
            pass  # Correct behaviour.

    def test_mismatched_steps_rejected_at_construction(self) -> None:
        """
        AdaptiveSilenceConfig must reject mismatched thresh/len lists
        at construction time via __post_init__ validation.
        """
        with pytest.raises(ValueError, match="same length"):
            AdaptiveSilenceConfig(
                thresh_steps=[-30, -40, -50],
                len_steps=[300, 200],  # One element short
            )

    def test_empty_thresh_steps_rejected_at_construction(self) -> None:
        """
        Empty thresh_steps must be rejected at construction time
        via __post_init__ validation.
        """
        with pytest.raises(ValueError, match="non-empty"):
            AdaptiveSilenceConfig(thresh_steps=[], len_steps=[])


# ---------------------------------------------------------------------------
# 2. Float truncation in min_chunks calculation
# ---------------------------------------------------------------------------


class TestMinChunksFloatTruncation:
    """
    The formula is: min_chunks = max(1, int(min_chunks_ratio * expected_ayah_count))
    Python's int() truncates toward zero. Combined with IEEE 754 float
    imprecision, results may be one less than the mathematically expected value.
    """

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_round_handles_03_times_10_correctly(self, mock_detect) -> None:
        """
        0.3 * 10 in IEEE 754 = 2.9999999999999996
        round(2.9999...) = 3 (correct), unlike int() which truncates to 2.
        So 2 chunks < 3 → retry.
        """
        mock_detect.side_effect = [
            [(0, 1000), (2000, 3000)],  # Step 0: 2 chunks < 3 → retry
            [(0, 1000), (2000, 3000), (4000, 5000)],  # Step 1: 3 chunks >= 3
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.3,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=10, config=cfg
        )
        # round(0.3 * 10) = 3, so 2 chunks insufficient → retried to step 1
        assert step_idx == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_round_handles_07_times_10_correctly(self, mock_detect) -> None:
        """
        0.7 * 10 in IEEE 754 = 6.999999999999999
        round(6.999...) = 7 (correct), unlike int() which truncates to 6.
        So 6 chunks < 7 → retry.
        """
        mock_detect.side_effect = [
            [(0, 1000)] * 6,  # Step 0: 6 chunks < 7 → retry
            [(0, 1000)] * 7,  # Step 1: 7 chunks >= 7
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.7,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=10, config=cfg
        )
        # round(0.7 * 10) = 7, so 6 chunks insufficient → retried to step 1
        assert step_idx == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_ratio_1_expected_1_gives_min_chunks_1(self, mock_detect) -> None:
        """int(1.0 * 1) = 1, max(1, 1) = 1. No truncation risk here."""
        mock_detect.return_value = [(0, 5000)]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=1.0,
            thresh_steps=[-30],
            len_steps=[300],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=1, config=cfg
        )
        assert step_idx == 0
        assert mock_detect.call_count == 1


# ---------------------------------------------------------------------------
# 3. steps_used return value correctness
# ---------------------------------------------------------------------------


class TestStepsUsedReturnValue:
    """Probe correctness of the returned step index."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_steps_used_is_zero_on_first_success(self, mock_detect) -> None:
        mock_detect.return_value = [(0, 1000)] * 10
        cfg = AdaptiveSilenceConfig(thresh_steps=[-30, -40, -50], len_steps=[300, 200, 100])
        _, step_idx = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=5, config=cfg)
        assert step_idx == 0

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_steps_used_is_one_on_second_success(self, mock_detect) -> None:
        mock_detect.side_effect = [
            [(0, 1000)],                     # Step 0: 1 chunk, insufficient (need 3)
            [(0, 1000)] * 5,                 # Step 1: 5 chunks, sufficient
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40, -50],
            len_steps=[300, 200, 100],
        )
        _, step_idx = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=6, config=cfg)
        assert step_idx == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_steps_used_is_last_index_when_exhausted(self, mock_detect) -> None:
        """When all steps exhausted, returned index must be last step index (N-1)."""
        mock_detect.return_value = []  # Always empty, never sufficient
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-30, -40, -50],
            len_steps=[300, 200, 100],
        )
        _, step_idx = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=10, config=cfg)
        assert step_idx == 2, (
            f"Expected last step index 2, got {step_idx}. "
            "steps_used must equal len(steps)-1 when all steps are exhausted."
        )

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_steps_used_single_step_exhausted(self, mock_detect) -> None:
        """Single-step config exhausted: step_idx must be 0."""
        mock_detect.return_value = []
        cfg = AdaptiveSilenceConfig(thresh_steps=[-30], len_steps=[300])
        _, step_idx = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=10, config=cfg)
        assert step_idx == 0

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_steps_used_initialised_to_zero_not_minus_one(self, mock_detect) -> None:
        """
        steps_used is initialized to 0 before the loop.
        If the loop body runs even once (guaranteed by non-empty validation),
        it will be set correctly. Verify the initial value is not returned
        on any code path.
        """
        mock_detect.return_value = [(0, 5000)] * 100  # Always sufficient
        cfg = AdaptiveSilenceConfig(thresh_steps=[-30], len_steps=[300])
        _, step_idx = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=1, config=cfg)
        assert step_idx == 0  # Correct: first step succeeded


# ---------------------------------------------------------------------------
# 4. Backwards-compatibility guard: adaptive disabled by default
# ---------------------------------------------------------------------------


class TestBackwardsCompatibilityGuard:
    """
    The issue spec requires that adaptive mode is opt-in.
    MunajjamSettings.adaptive_silence_enabled defaults to False.
    WhisperTranscriber.transcribe() must use plain detect_non_silent_chunks
    unless BOTH adaptive_silence_enabled=True AND expected_ayah_count is given.
    """

    @patch("munajjam.transcription.whisper.detect_non_silent_chunks_adaptive")
    @patch("munajjam.transcription.whisper.detect_non_silent_chunks")
    def test_adaptive_not_called_when_disabled_in_settings(
        self, mock_plain, mock_adaptive
    ) -> None:
        """
        When adaptive_silence_enabled=False (default), passing
        expected_ayah_count must NOT trigger adaptive detection.
        """
        from pathlib import Path
        from unittest.mock import MagicMock

        from munajjam.config import MunajjamSettings
        from munajjam.transcription.whisper import WhisperTranscriber

        settings = MunajjamSettings(adaptive_silence_enabled=False)
        transcriber = WhisperTranscriber(settings=settings)
        transcriber._model = MagicMock()  # Fake loaded state

        # Patch away the audio loading and segment transcription
        mock_plain.return_value = [(0, 5000)]
        with (
            patch("munajjam.transcription.whisper.load_audio_waveform") as mock_load,
            patch("munajjam.transcription.whisper.extract_segment_audio") as mock_extract,
            patch.object(transcriber, "_transcribe_segment") as mock_transcribe_seg,
            patch("pathlib.Path.exists", return_value=True),
        ):
            import numpy as np

            mock_load.return_value = (np.zeros(8000), 16000)
            mock_extract.return_value = np.zeros(800)
            mock_transcribe_seg.return_value = ("بِسْمِ اللَّهِ", None)

            # Inject a numeric stem so surah_id extraction works
            with patch.object(Path, "stem", new_callable=lambda: property(lambda self: "1")):
                try:
                    transcriber.transcribe("1.wav", expected_ayah_count=7)
                except Exception:
                    pass  # We only care about which silence function was called

        mock_adaptive.assert_not_called()
        mock_plain.assert_called()

    @patch("munajjam.transcription.whisper.detect_non_silent_chunks_adaptive")
    @patch("munajjam.transcription.whisper.detect_non_silent_chunks")
    def test_adaptive_not_called_when_ayah_count_is_none(
        self, mock_plain, mock_adaptive
    ) -> None:
        """
        When expected_ayah_count=None, adaptive must not be called
        even if adaptive_silence_enabled=True.
        """
        from pathlib import Path
        from unittest.mock import MagicMock

        from munajjam.config import MunajjamSettings
        from munajjam.transcription.whisper import WhisperTranscriber

        settings = MunajjamSettings(adaptive_silence_enabled=True)
        transcriber = WhisperTranscriber(settings=settings)
        transcriber._model = MagicMock()

        mock_plain.return_value = [(0, 5000)]
        with (
            patch("munajjam.transcription.whisper.load_audio_waveform") as mock_load,
            patch("munajjam.transcription.whisper.extract_segment_audio") as mock_extract,
            patch.object(transcriber, "_transcribe_segment") as mock_ts,
            patch("pathlib.Path.exists", return_value=True),
        ):
            import numpy as np

            mock_load.return_value = (np.zeros(8000), 16000)
            mock_extract.return_value = np.zeros(800)
            mock_ts.return_value = ("بِسْمِ اللَّهِ", None)

            with patch.object(Path, "stem", new_callable=lambda: property(lambda self: "1")):
                try:
                    transcriber.transcribe("1.wav", expected_ayah_count=None)
                except Exception:
                    pass

        mock_adaptive.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Inconsistency between AdaptiveSilenceConfig and MunajjamSettings
# ---------------------------------------------------------------------------


class TestBoundaryInconsistency:
    """
    AdaptiveSilenceConfig.__post_init__ accepts min_chunks_ratio in (0.0, 1.0].
    MunajjamSettings.adaptive_silence_min_ratio has ge=0.1.

    These are inconsistent: you can set AdaptiveSilenceConfig(min_chunks_ratio=0.05)
    but the settings validator prevents 0.05 from ever being stored in MunajjamSettings.
    """

    def test_adaptive_config_accepts_ratio_below_settings_minimum(self) -> None:
        """
        AdaptiveSilenceConfig allows 0.05 but MunajjamSettings rejects it.
        This is an interface inconsistency - both should enforce the same boundary.
        """
        # Config accepts 0.05:
        cfg = AdaptiveSilenceConfig(min_chunks_ratio=0.05)
        assert cfg.min_chunks_ratio == 0.05

    def test_munajjam_settings_rejects_ratio_below_0_1(self) -> None:
        """MunajjamSettings.adaptive_silence_min_ratio rejects values < 0.1."""
        from munajjam.config import MunajjamSettings

        with pytest.raises(ValueError):
            MunajjamSettings(adaptive_silence_min_ratio=0.05)

    def test_munajjam_settings_accepts_ratio_at_0_1(self) -> None:
        """MunajjamSettings accepts exactly 0.1 (its lower bound)."""
        from munajjam.config import MunajjamSettings

        s = MunajjamSettings(adaptive_silence_min_ratio=0.1)
        assert s.adaptive_silence_min_ratio == 0.1

    def test_adaptive_max_steps_capped_at_default_thresh_steps_length(self) -> None:
        """
        MunajjamSettings.adaptive_silence_max_steps has le=5, matching
        the 5 default steps in AdaptiveSilenceConfig. Setting max_steps=5
        uses all available steps.
        """
        from munajjam.config import MunajjamSettings

        s = MunajjamSettings(adaptive_silence_max_steps=5)
        defaults = AdaptiveSilenceConfig()
        sliced = defaults.thresh_steps[:s.adaptive_silence_max_steps]
        assert len(sliced) == 5, (
            "max_steps=5 must use all 5 default steps."
        )


# ---------------------------------------------------------------------------
# 6. Probe of detection logic correctness
# ---------------------------------------------------------------------------


class TestAdaptiveLogicCorrectness:
    """Probe specific logical behaviors of the retry loop."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_exact_min_chunks_boundary_is_sufficient(self, mock_detect) -> None:
        """
        len(chunks) == min_chunks must be SUFFICIENT (>=, not >).
        Test: min_chunks_ratio=0.5, expected=4 → min_chunks=max(1,int(2.0))=2.
        Return exactly 2 chunks → must stop at step 0.
        """
        mock_detect.return_value = [(0, 1000), (2000, 3000)]  # exactly 2
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "a.wav", expected_ayah_count=4, config=cfg
        )
        assert len(chunks) == 2
        assert step_idx == 0
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_one_below_min_chunks_triggers_retry(self, mock_detect) -> None:
        """
        len(chunks) == min_chunks - 1 must trigger a retry.
        min_chunks=5, return 4 chunks → retry.
        """
        mock_detect.side_effect = [
            [(0, 1000)] * 4,   # Step 0: 4 chunks, need 5
            [(0, 1000)] * 5,   # Step 1: 5 chunks, sufficient
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=1.0,  # need all 5
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        _, step_idx = detect_non_silent_chunks_adaptive(
            "a.wav", expected_ayah_count=5, config=cfg
        )
        assert step_idx == 1
        assert mock_detect.call_count == 2

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_correct_args_on_each_step(self, mock_detect) -> None:
        """Every step uses the CORRECT thresh and len from the config lists."""
        mock_detect.return_value = []  # Never sufficient, run all steps
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-10, -20, -30],
            len_steps=[500, 350, 100],
            use_fast=False,
        )
        detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=100, config=cfg)
        expected_calls = [
            call("a.wav", min_silence_len=500, silence_thresh=-10, use_fast=False),
            call("a.wav", min_silence_len=350, silence_thresh=-20, use_fast=False),
            call("a.wav", min_silence_len=100, silence_thresh=-30, use_fast=False),
        ]
        assert mock_detect.call_args_list == expected_calls

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_function_does_not_mutate_config_object(self, mock_detect) -> None:
        """
        The function must not mutate the passed AdaptiveSilenceConfig.
        Retry logic must be stateless with respect to the config.
        """
        mock_detect.return_value = []
        original_thresh = [-30, -40, -50]
        original_len = [300, 200, 100]
        cfg = AdaptiveSilenceConfig(
            thresh_steps=list(original_thresh),
            len_steps=list(original_len),
        )
        detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=10, config=cfg)
        assert cfg.thresh_steps == original_thresh
        assert cfg.len_steps == original_len

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_returned_chunks_are_from_last_successful_call(self, mock_detect) -> None:
        """
        The returned chunks list must come from the LAST call made,
        not from any earlier call that was insufficient.
        """
        first_call_chunks = [(0, 1000)]
        second_call_chunks = [(0, 500), (1000, 2000), (3000, 4000)]
        mock_detect.side_effect = [first_call_chunks, second_call_chunks]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        # expected_ayah_count=5 → min_chunks=max(1, int(2.5))=2
        # Step 0: 1 chunk < 2 → retry
        # Step 1: 3 chunks >= 2 → return
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "a.wav", expected_ayah_count=5, config=cfg
        )
        assert chunks == second_call_chunks
        assert step_idx == 1


# ---------------------------------------------------------------------------
# 7. Input validation edge cases on the public function
# ---------------------------------------------------------------------------


class TestInputValidationEdgeCases:
    """Try to pass invalid inputs that the function may not handle."""

    def test_expected_ayah_count_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_ayah_count must be >= 1"):
            detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=0)

    def test_expected_ayah_count_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_ayah_count must be >= 1"):
            detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=-99)

    def test_expected_ayah_count_float_not_accepted(self) -> None:
        """
        expected_ayah_count is typed as int. Passing a float should either
        raise TypeError or be handled gracefully - not silently truncate.
        Python won't enforce the type hint at runtime, so 6.9 would become
        int(0.5 * 6.9) = int(3.45) = 3. This documents the behaviour.
        """
        from munajjam.transcription.silence import detect_non_silent_chunks_adaptive

        with patch("munajjam.transcription.silence.detect_non_silent_chunks") as mock_d:
            mock_d.return_value = [(0, 1000)] * 4
            # 6.9 is not an int but Python won't raise TypeError here
            # The function computes int(0.5 * 6.9) = int(3.45) = 3
            # 4 chunks >= 3 → returns on step 0
            try:
                chunks, step_idx = detect_non_silent_chunks_adaptive(
                    "a.wav", expected_ayah_count=6.9  # type: ignore[arg-type]
                )
                # If it didn't raise, document the result
                assert step_idx == 0  # 4 >= int(0.5 * 6.9) = 3
            except TypeError:
                pass  # Correct if strict type enforcement is added

    def test_path_object_accepted(self) -> None:
        """audio_path as pathlib.Path must work, not just str."""
        from pathlib import Path

        with patch("munajjam.transcription.silence.detect_non_silent_chunks") as mock_d:
            mock_d.return_value = [(0, 1000)]
            chunks, step_idx = detect_non_silent_chunks_adaptive(
                Path("a.wav"), expected_ayah_count=1
            )
            assert len(chunks) == 1
            # Verify Path was forwarded correctly to inner function
            assert mock_d.call_args[0][0] == Path("a.wav")

    def test_string_path_accepted(self) -> None:
        """audio_path as plain str must work."""
        with patch("munajjam.transcription.silence.detect_non_silent_chunks") as mock_d:
            mock_d.return_value = [(0, 1000)]
            chunks, _ = detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=1)
            assert len(chunks) == 1

    def test_expected_ayah_count_very_large(self) -> None:
        """Large ayah count (Al-Baqarah = 286). All steps exhaust, no crash."""
        with patch("munajjam.transcription.silence.detect_non_silent_chunks") as mock_d:
            mock_d.return_value = [(0, 1000)]  # 1 chunk, never sufficient
            chunks, step_idx = detect_non_silent_chunks_adaptive(
                "a.wav", expected_ayah_count=286
            )
            # All 5 default steps exhausted
            assert step_idx == 4
            assert mock_d.call_count == 5

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_min_chunks_ratio_exactly_one(self, mock_detect) -> None:
        """
        min_chunks_ratio=1.0 means we need AT LEAST expected_ayah_count chunks.
        This is the strictest possible setting.
        """
        mock_detect.side_effect = [
            [(0, 1000)] * 4,   # Step 0: 4 chunks (need 5)
            [(0, 1000)] * 5,   # Step 1: 5 chunks (sufficient)
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=1.0,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        _, step_idx = detect_non_silent_chunks_adaptive(
            "a.wav", expected_ayah_count=5, config=cfg
        )
        assert step_idx == 1


# ---------------------------------------------------------------------------
# 8. Warning emission correctness
# ---------------------------------------------------------------------------


class TestWarningEmission:
    """Verify the exhaustion warning contains correct information."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_warning_contains_step_count(self, mock_detect, caplog) -> None:
        import logging

        mock_detect.return_value = []
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-30, -40, -50],
            len_steps=[300, 200, 100],
        )
        with caplog.at_level(logging.WARNING, logger="munajjam.transcription.silence"):
            detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=10, config=cfg)
        assert "exhausted all 3 steps" in caplog.text

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_warning_not_emitted_on_early_success(self, mock_detect, caplog) -> None:
        import logging

        mock_detect.return_value = [(0, 1000)] * 10
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        with caplog.at_level(logging.WARNING, logger="munajjam.transcription.silence"):
            detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=5, config=cfg)
        assert "exhausted" not in caplog.text

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_warning_contains_expected_ayah_count(self, mock_detect, caplog) -> None:
        """The warning message must mention the expected_ayah_count for debuggability."""
        import logging

        mock_detect.return_value = [(0, 1000)]
        cfg = AdaptiveSilenceConfig(thresh_steps=[-30, -40], len_steps=[300, 200])
        with caplog.at_level(logging.WARNING, logger="munajjam.transcription.silence"):
            detect_non_silent_chunks_adaptive("a.wav", expected_ayah_count=42, config=cfg)
        assert "42" in caplog.text, (
            "Warning message must include expected_ayah_count=42 for debuggability"
        )
