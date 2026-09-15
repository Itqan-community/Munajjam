"""
Unit tests for ReplicateWhisperXTranscriber.

These tests mock the `replicate` SDK entirely, so they run offline, for free,
and don't depend on Replicate account credit or network access. This is what
verifies the fix for the 404 bug (predictions.create() needs a pinned
`version`, not just a model name) without needing a live API call.
"""

from unittest.mock import MagicMock, patch

import pytest

from munajjam.transcription.replicate_transcriber import (
    ReplicateTranscriptionError,
    ReplicateWhisperXTranscriber,
)


# ---------------------------------------------------------------------------
# _parse_output — pure function, no mocking needed
# ---------------------------------------------------------------------------

def test_parse_output_flattens_words_with_scores():
    fake_output = {
        "segments": [
            {
                "words": [
                    {"word": "بِسْمِ", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "اللَّهِ", "start": 0.5, "end": 1.0, "score": 0.88},
                ]
            }
        ]
    }
    result = ReplicateWhisperXTranscriber._parse_output(fake_output)

    assert len(result) == 2
    assert result[0] == {
        "word": "بِسْمِ",
        "start": 0.0,
        "end": 0.5,
        "confidence": 0.95,
    }
    assert result[1]["word"] == "اللَّهِ"


def test_parse_output_defaults_confidence_when_missing():
    fake_output = {
        "segments": [{"words": [{"word": "test", "start": 0.0, "end": 0.5}]}]
    }
    result = ReplicateWhisperXTranscriber._parse_output(fake_output)

    assert result[0]["confidence"] == 0.9  # documented default fallback


def test_parse_output_skips_words_without_timing():
    fake_output = {
        "segments": [
            {
                "words": [
                    {"word": "no_timing"},  # WhisperX couldn't align this one
                    {"word": "has_timing", "start": 1.0, "end": 1.5},
                ]
            }
        ]
    }
    result = ReplicateWhisperXTranscriber._parse_output(fake_output)

    assert len(result) == 1
    assert result[0]["word"] == "has_timing"


def test_parse_output_handles_empty_or_none_output():
    assert ReplicateWhisperXTranscriber._parse_output(None) == []
    assert ReplicateWhisperXTranscriber._parse_output({}) == []
    assert ReplicateWhisperXTranscriber._parse_output({"segments": []}) == []


def test_parse_output_accepts_bare_list_of_segments():
    # Some Replicate model versions return a bare list instead of {"segments": [...]}
    fake_output = [{"words": [{"word": "x", "start": 0.0, "end": 0.2}]}]
    result = ReplicateWhisperXTranscriber._parse_output(fake_output)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# __init__ — token handling, version resolution
# ---------------------------------------------------------------------------

def test_init_raises_without_api_token(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)

    with pytest.raises(ReplicateTranscriptionError, match="REPLICATE_API_TOKEN"):
        ReplicateWhisperXTranscriber()


@patch("munajjam.transcription.replicate_transcriber.replicate")
def test_init_resolves_and_stores_model_version(mock_replicate_module, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake_token")

    mock_client = MagicMock()
    mock_replicate_module.Client.return_value = mock_client
    mock_client.models.get.return_value.latest_version.id = "abc123version"

    transcriber = ReplicateWhisperXTranscriber()

    mock_client.models.get.assert_called_once_with("victor-upmeet/whisperx")
    assert transcriber._model_version == "abc123version"


# ---------------------------------------------------------------------------
# transcribe() — the core regression test for the 404 fix
# ---------------------------------------------------------------------------

@patch("munajjam.transcription.replicate_transcriber.align_words_to_segments")
@patch("munajjam.transcription.replicate_transcriber.load_surah_ayahs")
@patch("munajjam.transcription.replicate_transcriber.replicate")
def test_transcribe_calls_predictions_create_with_version_not_model(
    mock_replicate_module, mock_load_ayahs, mock_align, monkeypatch, tmp_path
):
    """
    Regression test for the 404 bug: predictions.create() must be called with
    a pinned `version=`, not `model=`, since community-hosted Replicate models
    (like victor-upmeet/whisperx) don't support version-less prediction creation.
    """
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake_token")

    mock_client = MagicMock()
    mock_replicate_module.Client.return_value = mock_client
    mock_client.models.get.return_value.latest_version.id = "resolved_version_hash"

    mock_prediction = MagicMock()
    mock_prediction.status = "succeeded"
    mock_prediction.output = {
        "segments": [{"words": [{"word": "x", "start": 0.0, "end": 0.2, "score": 0.9}]}]
    }
    mock_client.predictions.create.return_value = mock_prediction

    mock_load_ayahs.return_value = [MagicMock(text="بِسْمِ اللَّهِ")]
    mock_align.return_value = ["fake_segment"]

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00\x00")  # dummy content, never actually parsed as audio

    transcriber = ReplicateWhisperXTranscriber()
    result = transcriber.transcribe(audio_file, surah_id=1)

    mock_client.predictions.create.assert_called_once()
    _, kwargs = mock_client.predictions.create.call_args

    assert "version" in kwargs, "predictions.create() must be called with version=, not model="
    assert kwargs["version"] == "resolved_version_hash"
    assert "model" not in kwargs

    assert result == ["fake_segment"]


@patch("munajjam.transcription.replicate_transcriber.load_surah_ayahs")
@patch("munajjam.transcription.replicate_transcriber.replicate")
def test_transcribe_raises_when_prediction_fails(
    mock_replicate_module, mock_load_ayahs, monkeypatch, tmp_path
):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake_token")

    mock_client = MagicMock()
    mock_replicate_module.Client.return_value = mock_client
    mock_client.models.get.return_value.latest_version.id = "v1"

    mock_prediction = MagicMock()
    mock_prediction.status = "failed"
    mock_prediction.error = "insufficient credit"
    mock_prediction.id = "pred_123"
    mock_client.predictions.create.return_value = mock_prediction

    mock_load_ayahs.return_value = [MagicMock(text="test")]

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00\x00")

    transcriber = ReplicateWhisperXTranscriber()

    with pytest.raises(ReplicateTranscriptionError, match="failed"):
        transcriber.transcribe(audio_file, surah_id=1)


@patch("munajjam.transcription.replicate_transcriber.load_surah_ayahs")
def test_transcribe_returns_empty_list_when_no_ayahs(mock_load_ayahs, monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake_token")
    mock_load_ayahs.return_value = []

    with patch("munajjam.transcription.replicate_transcriber.replicate") as mock_replicate_module:
        mock_client = MagicMock()
        mock_replicate_module.Client.return_value = mock_client
        mock_client.models.get.return_value.latest_version.id = "v1"

        transcriber = ReplicateWhisperXTranscriber()
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00\x00")

        result = transcriber.transcribe(audio_file, surah_id=999)

    assert result == []