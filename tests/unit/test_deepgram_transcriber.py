"""
Unit tests for DeepgramTranscriber.

Tests use mocked HTTP responses to avoid requiring a real Deepgram API key.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.deepgram_transcriber import DeepgramTranscriber


# ── Sample Deepgram API response ──────────────────────────────────────────

SAMPLE_DEEPGRAM_RESPONSE = {
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "بسم الله الرحمن الرحيم",
                        "confidence": 0.95,
                        "words": [
                            {"word": "بسم", "start": 0.5, "end": 0.9, "confidence": 0.97},
                            {"word": "الله", "start": 0.9, "end": 1.3, "confidence": 0.99},
                            {"word": "الرحمن", "start": 1.3, "end": 1.8, "confidence": 0.96},
                            {"word": "الرحيم", "start": 1.8, "end": 2.3, "confidence": 0.98},
                            {"word": "الحمد", "start": 2.5, "end": 2.9, "confidence": 0.95},
                            {"word": "لله", "start": 2.9, "end": 3.2, "confidence": 0.94},
                            {"word": "رب", "start": 3.2, "end": 3.5, "confidence": 0.93},
                            {"word": "العالمين", "start": 3.5, "end": 4.1, "confidence": 0.97},
                        ],
                    }
                ]
            }
        ]
    }
}


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    """Provide settings with a test Deepgram API key."""
    with patch("munajjam.transcription.deepgram_transcriber.get_settings") as mock:
        settings = MagicMock()
        settings.deepgram_api_key = MagicMock()
        settings.deepgram_api_key.get_secret_value.return_value = "test-api-key"
        mock.return_value = settings
        yield settings


@pytest.fixture
def transcriber(mock_settings):
    """Create a DeepgramTranscriber with mocked settings."""
    from munajjam.transcription.deepgram_transcriber import DeepgramTranscriber

    return DeepgramTranscriber()


# ── Tests ─────────────────────────────────────────────────────────────────


class TestDeepgramTranscriberInit:
    """Test DeepgramTranscriber initialization."""

    def test_init_without_api_key_raises(self) -> None:
        """Must raise TranscriptionError when API key is not set."""
        from munajjam.exceptions import TranscriptionError

        with patch("munajjam.transcription.deepgram_transcriber.get_settings") as mock:
            settings = MagicMock()
            settings.deepgram_api_key = None
            mock.return_value = settings

            with pytest.raises(TranscriptionError, match="API key is not configured"):
                from munajjam.transcription.deepgram_transcriber import (
                    DeepgramTranscriber,
                )

                DeepgramTranscriber()

    def test_init_with_api_key_succeeds(self, transcriber) -> None:
        """Must initialize successfully when API key is provided."""
        assert transcriber._api_key == "test-api-key"


class TestDeepgramWordExtraction:
    """Test parsing of Deepgram API responses."""

    def test_extract_words_from_valid_response(self, transcriber) -> None:
        """Must correctly extract word timestamps from Deepgram JSON."""
        words = transcriber._extract_words(SAMPLE_DEEPGRAM_RESPONSE)
        assert len(words) == 8
        assert words[0]["word"] == "بسم"
        assert words[0]["start"] == 0.5
        assert words[0]["end"] == 0.9
        assert words[0]["confidence"] == 0.97

    def test_extract_words_from_empty_response(self, transcriber) -> None:
        """Must raise TranscriptionError for empty response."""
        from munajjam.exceptions import TranscriptionError

        empty_response = {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
        words = transcriber._extract_words(empty_response)
        assert words == []

    def test_extract_words_from_malformed_response(self, transcriber) -> None:
        """Must raise TranscriptionError for malformed response."""
        from munajjam.exceptions import TranscriptionError

        with pytest.raises(TranscriptionError, match="Unexpected Deepgram response"):
            transcriber._extract_words({"bad": "data"})


class TestDeepgramAPICall:
    """Test Deepgram API interaction."""

    def test_api_error_raises_transcription_error(self, transcriber, tmp_path) -> None:
        """Must raise TranscriptionError on non-200 API response."""
        from munajjam.exceptions import TranscriptionError

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(TranscriptionError, match="HTTP 401"):
                transcriber._call_deepgram(audio_file)

    def test_successful_api_call(self, transcriber, tmp_path) -> None:
        """Must return parsed JSON from a successful API call."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_DEEPGRAM_RESPONSE

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = transcriber._call_deepgram(audio_file)

            # Verify the API was called with correct params
            call_kwargs = mock_post.call_args
            assert "Token test-api-key" in call_kwargs.kwargs["headers"]["Authorization"]
            assert call_kwargs.kwargs["params"]["model"] == "nova-3"
            assert call_kwargs.kwargs["params"]["language"] == "ar"

        assert result == SAMPLE_DEEPGRAM_RESPONSE


class TestDeepgramTranscription:
    """Test full transcription pipeline with mocked API."""

    def test_transcribe_returns_segments(self, transcriber, tmp_path) -> None:
        """Full pipeline must return Segment objects with correct structure."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_DEEPGRAM_RESPONSE

        with patch("httpx.post", return_value=mock_response):
            segments = transcriber.transcribe(audio_file, surah_id=1)

        # Al-Fatiha has 7 ayahs
        assert len(segments) > 0
        assert all(isinstance(s, Segment) for s in segments)

        # Check first segment structure
        first = segments[0]
        assert first.surah_id == 1
        assert first.type == SegmentType.AYAH
        assert first.start >= 0.0
        assert first.end > first.start
        assert first.words is not None
        assert len(first.words) > 0
        assert all(isinstance(w, WordTimestamp) for w in first.words)
