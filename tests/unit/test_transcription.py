import pytest
from unittest.mock import patch, MagicMock

from munajjam.transcription.whisperFactory import WhisperFactory, WhisperBackend
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperx import Whisperx
from munajjam.models import SegmentType


@pytest.fixture
def factory():
    return WhisperFactory()


def test_whisper_factory_faster_whisper(factory):
    with patch(
        "munajjam.transcription.whisper.WhisperTranscriber.__init__", return_value=None
    ) as mock_init:
        transcriber = factory.create_whisper(
            WhisperBackend.FASTERWHISPER, "base", "cpu"
        )
        assert isinstance(transcriber, WhisperTranscriber)
        mock_init.assert_called_once_with(
            model_id="base", device="cpu", model_type="faster-whisper"
        )


def test_whisper_factory_openai(factory):
    with patch(
        "munajjam.transcription.whisper.WhisperTranscriber.__init__", return_value=None
    ) as mock_init:
        transcriber = factory.create_whisper(
            WhisperBackend.OPENAI, "openai/whisper-large-v3", "cuda"
        )
        assert isinstance(transcriber, WhisperTranscriber)
        mock_init.assert_called_once_with(
            model_id="openai/whisper-large-v3", device="cuda", model_type="transformers"
        )


def test_whisper_factory_whisperx(factory):
    with patch(
        "munajjam.transcription.whisperx.Whisperx.__init__", return_value=None
    ) as mock_init:
        transcriber = factory.create_whisper(WhisperBackend.WHISPERX, "base", "cuda")
        assert isinstance(transcriber, Whisperx)
        mock_init.assert_called_once_with(
            model_name="base", device="cuda", compute_type="float16"
        )


def test_whisper_factory_unsupported(factory):
    with pytest.raises(ValueError, match="Unsupported backend"):
        factory.create_whisper("invalid_backend", "base", "cpu")


@patch("munajjam.transcription.whisperx.whisperx")
def test_whisperx_transcribe(mock_whisperx_module):
    # Mock whisperx load_model and its returned model
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]
    }
    mock_whisperx_module.load_model.return_value = mock_model
    mock_whisperx_module.load_audio.return_value = "mock_audio_data"

    transcriber = Whisperx(model_name="base", device="cpu")

    # Actually call transcribe
    segments = transcriber.transcribe("dummy_audio.wav", batch_size=8, surah_id=1)

    assert len(segments) == 1
    assert segments[0].text == "hello"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5

    mock_whisperx_module.load_audio.assert_called_once_with("dummy_audio.wav")
    mock_model.transcribe.assert_called_once_with("mock_audio_data", batch_size=8)


@patch("munajjam.transcription.whisper.Path.exists", return_value=True)
@patch("munajjam.transcription.whisper.load_audio_waveform")
@patch("munajjam.transcription.whisper.WhisperTranscriber._initialize_model")
def test_whisper_transcriber_transcribe_transformers(
    mock_init_model, mock_load, mock_exists
):
    # Setup standard audio mocking
    mock_load.return_value = ([0.0] * 24000, 16000)

    transcriber = WhisperTranscriber(
        model_id="test", device="cpu", model_type="transformers"
    )

    # Mock settings internally without relying on the actual config singletons entirely
    transcriber._settings = MagicMock()
    transcriber._settings.sample_rate = 16000
    transcriber._resolved_device = "cpu"

    # Mock the transformer processor and model
    mock_processor = MagicMock()
    mock_processor.return_value.to.return_value = {"input_features": MagicMock()}
    mock_processor.batch_decode.return_value = ["اَلْحَمْدُ لِلَّهِ"]
    transcriber._processor = mock_processor

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype="float32")])
    transcriber._model = mock_model

    # Mock librosa get_duration
    with patch("munajjam.transcription.whisper.librosa.get_duration", return_value=1.5):
        # Mock Arabic text detection assuming an ayah mapping function could be invoked
        with patch(
            "munajjam.transcription.whisper.detect_segment_type",
            return_value=(SegmentType.AYAH, 1),
        ):
            segments = transcriber.transcribe("1.wav", surah_id=1)

    assert len(segments) == 1
    assert segments[0].text == "اَلْحَمْدُ لِلَّهِ"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[0].surah_id == 1
    assert segments[0].type == SegmentType.AYAH

    mock_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once()
