"""Unit tests for the experimental Chirp 3 transcription backend."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from munajjam.config import MunajjamSettings
from munajjam.exceptions import AudioFileError, ConfigurationError, TranscriptionError
from munajjam.models import Ayah, SegmentType
from munajjam.transcription.chirp import ChirpTranscriber, _duration_to_seconds
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory


def test_whisper_factory_chirp3():
    transcriber = WhisperFactory().create_whisper(
        WhisperBackend.CHIRP3, "ignored", "cpu"
    )
    assert isinstance(transcriber, ChirpTranscriber)


def test_duration_to_seconds():
    duration = MagicMock()
    duration.seconds = 1
    duration.nanos = 500_000_000
    assert _duration_to_seconds(duration) == pytest.approx(1.5)
    assert _duration_to_seconds(None) == 0.0


def test_extract_words_applies_time_offset():
    word = MagicMock()
    word.word = "الحمد"
    word.confidence = 0.8
    word.start_offset.seconds = 1
    word.start_offset.nanos = 500_000_000
    word.end_offset.seconds = 2
    word.end_offset.nanos = 0
    alternative = MagicMock(words=[word])
    result = MagicMock(alternatives=[alternative])
    response = MagicMock(results=[result])

    words = ChirpTranscriber._extract_words(response, time_offset=10.0)
    assert words == [
        {
            "word": "الحمد",
            "start": pytest.approx(11.5),
            "end": pytest.approx(12.0),
            "confidence": 0.8,
        }
    ]


def test_transcribe_missing_audio(tmp_path: Path):
    transcriber = ChirpTranscriber(project_id="demo-project")
    with pytest.raises(AudioFileError):
        transcriber.transcribe(tmp_path / "missing.wav", surah_id=1)


def test_transcribe_requires_project_id(tmp_path: Path):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"fake-audio")
    transcriber = ChirpTranscriber(project_id="demo-project")
    transcriber._project_id = None
    with pytest.raises(ConfigurationError, match="project id"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_missing_credentials_file(tmp_path: Path):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"fake-audio")
    transcriber = ChirpTranscriber(
        project_id="demo-project",
        credentials_path=tmp_path / "no-such-key.json",
    )
    with pytest.raises(ConfigurationError, match="credentials file"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_maps_words_to_ayah_segments(tmp_path: Path):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"fake-audio")
    ayah = Ayah(id=1, surah_id=1, ayah_number=1, text="بسم الله")
    transcriber = ChirpTranscriber(project_id="demo-project")
    transcriber._transcribe_words = MagicMock(
        return_value=[
            {"word": "بسم", "start": 0.0, "end": 0.4, "confidence": 0.9},
            {"word": "الله", "start": 0.4, "end": 0.9, "confidence": 0.95},
        ]
    )

    with patch(
        "munajjam.transcription.chirp.load_surah_ayahs",
        return_value=[ayah],
    ):
        segments = transcriber.transcribe(audio, surah_id=1)

    assert len(segments) == 1
    assert segments[0].id == 1
    assert segments[0].surah_id == 1
    assert segments[0].type == SegmentType.AYAH
    assert segments[0].text == "بسم الله"
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(0.9)
    assert segments[0].words is not None
    assert [w.word for w in segments[0].words] == ["بسم", "الله"]


def test_transcribe_empty_words_raises(tmp_path: Path):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"fake-audio")
    transcriber = ChirpTranscriber(project_id="demo-project")
    transcriber._transcribe_words = MagicMock(return_value=[])
    with patch(
        "munajjam.transcription.chirp.load_surah_ayahs",
        return_value=[Ayah(id=1, surah_id=1, ayah_number=1, text="بسم")],
    ), pytest.raises(TranscriptionError, match="no word-level timestamps"):
        transcriber.transcribe(audio, surah_id=1)


def test_gcp_settings_accept_unprefixed_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-from-env")
    monkeypatch.setenv("GCP_LOCATION", "eu")
    monkeypatch.delenv("MUNAJJAM_GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("MUNAJJAM_GCP_LOCATION", raising=False)
    settings = MunajjamSettings()
    assert settings.gcp_project_id == "proj-from-env"
    assert settings.gcp_location == "eu"
