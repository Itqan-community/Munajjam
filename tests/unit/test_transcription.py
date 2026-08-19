from unittest.mock import MagicMock, patch

import pytest
from munajjam.models import SegmentType
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory
from munajjam.transcription.whisperx import Whisperx


@pytest.fixture
def factory():
    return WhisperFactory()


def test_whisper_factory_faster_whisper(factory):
    with patch(
        "munajjam.transcription.whisper.WhisperTranscriber.__init__",
        return_value=None,
    ) as mock_init:
        transcriber = factory.create_whisper(
            WhisperBackend.FASTERWHISPER,
            "base",
            "cpu",
        )

        assert isinstance(transcriber, WhisperTranscriber)
        mock_init.assert_called_once_with(
            model_id="base",
            device="cpu",
            model_type="faster-whisper",
        )


def test_whisper_factory_openai(factory):
    with patch(
        "munajjam.transcription.whisper.WhisperTranscriber.__init__",
        return_value=None,
    ) as mock_init:
        transcriber = factory.create_whisper(
            WhisperBackend.OPENAI,
            "openai/whisper-large-v3",
            "cuda",
        )

        assert isinstance(transcriber, WhisperTranscriber)
        mock_init.assert_called_once_with(
            model_id="openai/whisper-large-v3",
            device="cuda",
            model_type="transformers",
        )


def test_whisper_factory_whisperx(factory):
    with patch(
        "munajjam.transcription.whisperx.Whisperx.__init__",
        return_value=None,
    ) as mock_init:
        transcriber = factory.create_whisper(
            WhisperBackend.WHISPERX,
            "base",
            "cuda",
        )

        assert isinstance(transcriber, Whisperx)
        mock_init.assert_called_once_with(
            model_name="base",
            device="cuda",
            compute_type="float16",
        )


def test_whisper_factory_unsupported(factory):
    with pytest.raises(ValueError, match="Unsupported backend"):
        factory.create_whisper("invalid_backend", "base", "cpu")


@patch("munajjam.transcription.whisperx.whisperx")
def test_whisperx_transcribe(mock_whisperx_module):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]
    }

    mock_whisperx_module.load_model.return_value = mock_model
    mock_whisperx_module.load_align_model.return_value = (
        MagicMock(),
        MagicMock(),
    )
    mock_whisperx_module.align.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]
    }
    mock_whisperx_module.load_audio.return_value = "mock_audio_data"

    transcriber = Whisperx(model_name="base", device="cpu")

    segments = transcriber.transcribe(
        "dummy_audio.wav",
        batch_size=8,
        surah_id=1,
    )

    assert len(segments) == 7
    assert segments[0].surah_id == 1
    assert "بِسْمِ" in segments[0].text

    mock_whisperx_module.load_audio.assert_called_once_with("dummy_audio.wav")
    mock_model.transcribe.assert_called_once_with(
        "mock_audio_data",
        batch_size=8,
    )


@patch.dict(
    "sys.modules",
    {
        "torch": MagicMock(),
        "transformers": MagicMock(),
        "transformers.utils": MagicMock(),
    },
)
@patch("munajjam.transcription.whisper.Path.exists", return_value=True)
@patch("munajjam.transcription.whisper.load_audio_waveform")
@patch("munajjam.transcription.whisper.WhisperTranscriber._initialize_model")
def test_whisper_transcriber_transcribe_transformers(
    mock_init_model,
    mock_load,
    mock_exists,
):
    mock_load.return_value = ([0.0] * 24000, 16000)

    transcriber = WhisperTranscriber(
        model_id="test",
        device="cpu",
        model_type="transformers",
    )

    transcriber._settings = MagicMock()
    transcriber._settings.sample_rate = 16000
    transcriber._resolved_device = "cpu"

    mock_processor = MagicMock()
    mock_processor.return_value.to.return_value = {
        "input_features": MagicMock()
    }
    expected_text = 'الحمد لله'
    expected_text = 'الحمد لله'
    mock_processor.batch_decode.return_value = [expected_text]
    transcriber._processor = mock_processor
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter(
        [MagicMock(dtype="float32")]
    )
    transcriber._model = mock_model

    with (
        patch(
            "munajjam.transcription.whisper.librosa.get_duration",
            return_value=1.5,
        ),
        patch(
            "munajjam.transcription.whisper.detect_segment_type",
            return_value=(SegmentType.AYAH, 1),
        ),
    ):
        segments = transcriber.transcribe(
            "1.wav",
            surah_id=1,
        )

    assert len(segments) == 1
    assert segments[0].text == expected_text
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[0].surah_id == 1
    assert segments[0].type == SegmentType.AYAH

    mock_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once()


def test_whisperx_model_size_config(monkeypatch):
    from munajjam.config import MunajjamSettings

    default_settings = MunajjamSettings()
    assert default_settings.whisperx_model_size == "large-v2"

    monkeypatch.setenv("MUNAJJAM_WHISPERX_MODEL_SIZE", "tiny")

    env_settings = MunajjamSettings()
    assert env_settings.whisperx_model_size == "tiny"


def test_whisperx_init_default_and_custom():
    transcriber_default = Whisperx()
    assert transcriber_default.model_name == "large-v2"

    transcriber_custom = Whisperx(model_name="medium")
    assert transcriber_custom.model_name == "medium"


@patch("gc.collect")
@patch("munajjam.transcription.whisperx.torch")
def test_whisperx_unload_model(mock_torch, mock_gc_collect):
    mock_torch.cuda.is_available.return_value = True

    transcriber = Whisperx(
        model_name="base",
        device="cuda",
    )
    transcriber.whisper_model = MagicMock()
    transcriber.align_model = MagicMock()
    transcriber.align_metadata = MagicMock()

    transcriber.unload_model()

    assert transcriber.whisper_model is None
    assert transcriber.align_model is None
    assert transcriber.align_metadata is None

    mock_gc_collect.assert_called_once()
    mock_torch.cuda.empty_cache.assert_called_once()


def test_whisperx_set_model_name_swapping():
    transcriber = Whisperx(
        model_name="small",
        device="cpu",
    )

    mock_model = MagicMock()
    transcriber.whisper_model = mock_model

    transcriber.set_model_name("small")

    assert transcriber.whisper_model == mock_model
    assert transcriber.model_name == "small"

    transcriber.set_model_name("large-v3")

    assert transcriber.whisper_model is None
    assert transcriber.model_name == "large-v3"


@patch("server.get_settings")
@patch("server.global_transcriber")
@patch("server.os.path.exists", return_value=False)
def test_server_run_job_model_size_resolution(
    mock_exists,
    mock_transcriber,
    mock_get_settings,
):
    from server import _run_job, jobs

    mock_get_settings.return_value.whisperx_model_size = "large-v2"
    mock_transcriber.transcribe.return_value = []
    mock_transcriber.model_name = "large-v2"

    jobs["job_1"] = {"status": "queued"}

    _run_job(
        "job_1",
        "dummy.mp3",
        1,
        model_size="tiny",
    )

    mock_transcriber.set_model_name.assert_called_with("tiny")

    jobs["job_2"] = {"status": "queued"}

    _run_job(
        "job_2",
        "dummy.mp3",
        1,
        model_size=None,
    )

    mock_transcriber.set_model_name.assert_called_with("large-v2")


@patch("munajjam.transcription.whisperFactory.FastConformerCTCTranscriber")
def test_whisper_factory_creates_ctc_segmentation_backend(
    mock_ctc_transcriber,
):
    """CTC backend should create the FastConformer transcriber with the requested model."""
    mock_instance = MagicMock()
    mock_ctc_transcriber.return_value = mock_instance

    factory = WhisperFactory()

    result = factory.create_whisper(
        backend=WhisperBackend.CTC_SEGMENTATION,
        model_name="test-fastconformer-model",
        device="cpu",
    )

    mock_ctc_transcriber.assert_called_once_with(
        model_id="test-fastconformer-model",
        device="cpu",
    )

    assert result is mock_instance


def test_whisper_factory_ctc_requires_model_name():
    """CTC backend should fail clearly when no model is configured."""
    factory = WhisperFactory()

    with pytest.raises(
        ValueError,
        match="model_name is required for the CTC segmentation backend",
    ):
        factory.create_whisper(
            backend=WhisperBackend.CTC_SEGMENTATION,
            model_name=None,
            device="cpu",
        )
