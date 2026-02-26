"""
Unit tests for adaptive silence detection (Issue #47).

Tests cover:
- AdaptiveSilenceConfig dataclass defaults and custom values
- detect_non_silent_chunks_adaptive() retry logic
- MunajjamSettings adaptive fields
- WhisperTranscriber.transcribe() expected_ayah_count integration
"""

from unittest.mock import patch

import pytest

from munajjam.transcription.silence import (
    AdaptiveSilenceConfig,
    detect_non_silent_chunks_adaptive,
)


class TestAdaptiveSilenceConfig:
    """Test AdaptiveSilenceConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = AdaptiveSilenceConfig()
        assert cfg.min_chunks_ratio == 0.5
        assert len(cfg.thresh_steps) == 5
        assert len(cfg.len_steps) == 5
        assert cfg.use_fast is True

    def test_custom_values(self) -> None:
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.7,
            thresh_steps=[-25, -30],
            len_steps=[400, 200],
            use_fast=False,
        )
        assert cfg.min_chunks_ratio == 0.7
        assert cfg.thresh_steps == [-25, -30]
        assert cfg.len_steps == [400, 200]
        assert cfg.use_fast is False

    def test_default_steps_ordered_strict_to_relaxed(self) -> None:
        """thresh_steps should go more negative (relaxed), len_steps shorter (relaxed)."""
        cfg = AdaptiveSilenceConfig()
        # More negative dB = less audio classified as silent = more chunks = relaxed
        for i in range(len(cfg.thresh_steps) - 1):
            assert cfg.thresh_steps[i] > cfg.thresh_steps[i + 1]
        # Shorter min silence = easier to split = more chunks = relaxed
        for i in range(len(cfg.len_steps) - 1):
            assert cfg.len_steps[i] > cfg.len_steps[i + 1]


class TestDetectNonSilentChunksAdaptive:
    """Test detect_non_silent_chunks_adaptive() retry logic."""

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_returns_on_first_attempt_when_sufficient(self, mock_detect) -> None:
        """No retry needed when first attempt produces enough chunks."""
        mock_detect.return_value = [(0, 1000), (1500, 3000), (3500, 5000)]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40],
            len_steps=[300, 150],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=5, config=cfg,
        )
        assert len(chunks) == 3  # 3 >= max(1, int(0.5*5))=2
        assert step_idx == 0
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_retries_when_first_attempt_insufficient(self, mock_detect) -> None:
        """Retry when first pass produces too few chunks."""
        # First attempt: 1 chunk (insufficient for 10 ayahs needing 5)
        # Second attempt: 6 chunks (sufficient)
        mock_detect.side_effect = [
            [(0, 60000)],
            [(0, 10000), (11000, 20000), (21000, 30000),
             (31000, 40000), (41000, 50000), (51000, 60000)],
        ]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40],
            len_steps=[300, 150],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=10, config=cfg,
        )
        assert len(chunks) == 6
        assert step_idx == 1
        assert mock_detect.call_count == 2

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_returns_last_result_when_all_steps_exhausted(self, mock_detect) -> None:
        """Return last result even if insufficient when all steps exhausted."""
        mock_detect.return_value = [(0, 60000)]  # Always 1 chunk
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40, -50],
            len_steps=[300, 200, 100],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=20, config=cfg,
        )
        assert len(chunks) == 1
        assert step_idx == 2  # Last step index
        assert mock_detect.call_count == 3

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_threshold_progression_passed_to_inner_function(self, mock_detect) -> None:
        """Verify correct args are passed at each retry step."""
        mock_detect.return_value = [(0, 1000)]  # Always insufficient
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30, -40, -50],
            len_steps=[300, 200, 100],
            use_fast=True,
        )
        detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=10, config=cfg,
        )
        assert mock_detect.call_count == 3
        calls = mock_detect.call_args_list
        assert calls[0].kwargs == {"min_silence_len": 300, "silence_thresh": -30, "use_fast": True}
        assert calls[1].kwargs == {"min_silence_len": 200, "silence_thresh": -40, "use_fast": True}
        assert calls[2].kwargs == {"min_silence_len": 100, "silence_thresh": -50, "use_fast": True}

    def test_raises_value_error_for_zero_expected_ayah_count(self) -> None:
        with pytest.raises(ValueError, match="expected_ayah_count must be >= 1"):
            detect_non_silent_chunks_adaptive("audio.wav", expected_ayah_count=0)

    def test_raises_value_error_for_negative_expected_ayah_count(self) -> None:
        with pytest.raises(ValueError, match="expected_ayah_count must be >= 1"):
            detect_non_silent_chunks_adaptive("audio.wav", expected_ayah_count=-5)

    def test_raises_value_error_for_mismatched_step_lists(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            AdaptiveSilenceConfig(
                thresh_steps=[-30, -40, -50],
                len_steps=[300, 200],
            )

    def test_raises_value_error_for_empty_step_lists(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AdaptiveSilenceConfig(thresh_steps=[], len_steps=[])

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_single_step_config(self, mock_detect) -> None:
        """Single-step config works correctly."""
        mock_detect.return_value = [(0, 5000)]
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.5,
            thresh_steps=[-30],
            len_steps=[300],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=1, config=cfg,
        )
        assert len(chunks) == 1
        assert step_idx == 0
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_min_chunks_ratio_boundary(self, mock_detect) -> None:
        """Exact threshold: chunks == min_chunks is sufficient."""
        mock_detect.return_value = [(0, 1000)] * 5  # Exactly 5 chunks
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=1.0,
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=5, config=cfg,
        )
        assert step_idx == 0  # Sufficient on first try
        assert mock_detect.call_count == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_min_chunks_always_at_least_one(self, mock_detect) -> None:
        """min_chunks is clamped to at least 1."""
        mock_detect.return_value = [(0, 5000)]  # 1 chunk
        cfg = AdaptiveSilenceConfig(
            min_chunks_ratio=0.1,  # 0.1 * 1 = 0.1, int → 0, clamped to 1
            thresh_steps=[-30],
            len_steps=[300],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=1, config=cfg,
        )
        assert step_idx == 0  # 1 >= max(1, 0) = 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_use_fast_false_forwarded(self, mock_detect) -> None:
        """use_fast=False is forwarded to inner function."""
        mock_detect.return_value = [(0, 5000)]
        cfg = AdaptiveSilenceConfig(
            use_fast=False,
            thresh_steps=[-30],
            len_steps=[300],
        )
        detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=1, config=cfg,
        )
        assert mock_detect.call_args.kwargs["use_fast"] is False

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_default_config_used_when_none(self, mock_detect) -> None:
        """Default AdaptiveSilenceConfig is used when config=None."""
        mock_detect.return_value = [(0, 5000)] * 10
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=5,
        )
        assert step_idx == 0  # 10 >= max(1, int(0.5*5))=2

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_all_steps_exhausted_with_empty_result(self, mock_detect) -> None:
        """Empty result from all steps returns ([], last_step_idx)."""
        mock_detect.return_value = []
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        chunks, step_idx = detect_non_silent_chunks_adaptive(
            "audio.wav", expected_ayah_count=5, config=cfg,
        )
        assert chunks == []
        assert step_idx == 1

    @patch("munajjam.transcription.silence.detect_non_silent_chunks")
    def test_exhaustion_emits_warning(self, mock_detect, caplog) -> None:
        """Warning logged when all steps exhausted."""
        import logging
        mock_detect.return_value = [(0, 5000)]
        cfg = AdaptiveSilenceConfig(
            thresh_steps=[-30, -40],
            len_steps=[300, 200],
        )
        with caplog.at_level(logging.WARNING, logger="munajjam.transcription.silence"):
            detect_non_silent_chunks_adaptive(
                "audio.wav", expected_ayah_count=10, config=cfg,
            )
        assert "exhausted all 2 steps" in caplog.text

    def test_config_validates_min_chunks_ratio_zero(self) -> None:
        """AdaptiveSilenceConfig rejects min_chunks_ratio=0."""
        with pytest.raises(ValueError, match="min_chunks_ratio"):
            AdaptiveSilenceConfig(min_chunks_ratio=0.0)

    def test_config_validates_min_chunks_ratio_above_one(self) -> None:
        """AdaptiveSilenceConfig rejects min_chunks_ratio > 1."""
        with pytest.raises(ValueError, match="min_chunks_ratio"):
            AdaptiveSilenceConfig(min_chunks_ratio=1.5)


class TestMunajjamSettingsAdaptive:
    """Test adaptive silence fields on MunajjamSettings."""

    def test_adaptive_settings_defaults(self) -> None:
        from munajjam.config import MunajjamSettings
        settings = MunajjamSettings()
        assert settings.adaptive_silence_enabled is False
        assert settings.adaptive_silence_min_ratio == 0.5
        assert settings.adaptive_silence_max_steps == 4

    def test_adaptive_settings_custom(self) -> None:
        from munajjam.config import MunajjamSettings
        settings = MunajjamSettings(
            adaptive_silence_enabled=True,
            adaptive_silence_min_ratio=0.7,
            adaptive_silence_max_steps=5,
        )
        assert settings.adaptive_silence_enabled is True
        assert settings.adaptive_silence_min_ratio == 0.7
        assert settings.adaptive_silence_max_steps == 5

    def test_adaptive_min_ratio_validation(self) -> None:
        from munajjam.config import MunajjamSettings
        with pytest.raises(ValueError):
            MunajjamSettings(adaptive_silence_min_ratio=0.0)
        with pytest.raises(ValueError):
            MunajjamSettings(adaptive_silence_min_ratio=1.5)

    def test_adaptive_max_steps_validation(self) -> None:
        from munajjam.config import MunajjamSettings
        with pytest.raises(ValueError):
            MunajjamSettings(adaptive_silence_max_steps=0)
        with pytest.raises(ValueError):
            MunajjamSettings(adaptive_silence_max_steps=6)


class TestTranscriberAdaptiveIntegration:
    """Test WhisperTranscriber.transcribe() expected_ayah_count parameter."""

    def test_transcribe_signature_accepts_expected_ayah_count(self) -> None:
        """WhisperTranscriber.transcribe() accepts expected_ayah_count param."""
        import inspect

        from munajjam.transcription.whisper import WhisperTranscriber
        sig = inspect.signature(WhisperTranscriber.transcribe)
        assert "expected_ayah_count" in sig.parameters
        param = sig.parameters["expected_ayah_count"]
        assert param.default is None

    def test_transcribe_exports_adaptive_symbols(self) -> None:
        """AdaptiveSilenceConfig and detect_non_silent_chunks_adaptive are exported."""
        from munajjam.transcription import (
            AdaptiveSilenceConfig as ASC,
        )
        from munajjam.transcription import (
            detect_non_silent_chunks_adaptive as dnca,
        )
        assert ASC is not None
        assert dnca is not None
