from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from munajjam.exceptions import TranscriptionError
from munajjam.models import SegmentType
from munajjam.transcription.ctc_segmentation import (
    FastConformerCTCTranscriber,
    SileroVADChunker,
)
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory
from munajjam.transcription.whisperx import Whisperx


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


@patch("munajjam.transcription.whisperFactory.FastConformerCTCTranscriber")
def test_whisper_factory_ctc_segmentation(mock_cls, factory):
    mock_cls.return_value = MagicMock()
    with patch("munajjam.transcription.whisperFactory.get_settings") as mock_settings:
        s = mock_settings.return_value
        s.fastconformer_model_path = "/models/ctc.onnx"
        s.fastconformer_vocab_path = "/models/vocab.txt"
        s.fastconformer_tokenizer_model_path = "/models/tokenizer.model"
        s.fastconformer_vad_enabled = False
        s.fastconformer_blank_transition_cost_zero = False
        transcriber = factory.create_whisper(WhisperBackend.CTC_SEGMENTATION)
        assert transcriber is mock_cls.return_value
        mock_cls.assert_called_once_with(
            model_path="/models/ctc.onnx",
            vocab_path="/models/vocab.txt",
            tokenizer_model_path="/models/tokenizer.model",
            chunker=None,
            blank_transition_cost_zero=False,
        )


@patch("munajjam.transcription.whisperFactory.FastConformerCTCTranscriber")
def test_whisper_factory_ctc_segmentation_vad_enabled(mock_cls, factory):
    mock_cls.return_value = MagicMock()
    with patch("munajjam.transcription.whisperFactory.get_settings") as mock_settings:
        mock_settings.return_value.fastconformer_vad_enabled = True
        factory.create_whisper(WhisperBackend.CTC_SEGMENTATION)
        _, kwargs = mock_cls.call_args
        assert isinstance(kwargs["chunker"], SileroVADChunker)


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
    mock_whisperx_module.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx_module.align.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]
    }
    mock_whisperx_module.load_audio.return_value = "mock_audio_data"

    transcriber = Whisperx(model_name="base", device="cpu")

    # Actually call transcribe
    segments = transcriber.transcribe("dummy_audio.wav", batch_size=8, surah_id=1)

    assert len(segments) == 7
    assert segments[0].surah_id == 1
    assert "بِسْمِ" in segments[0].text

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
    transcriber = Whisperx(model_name="base", device="cuda")
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
    transcriber = Whisperx(model_name="small", device="cpu")
    mock_model = MagicMock()
    transcriber.whisper_model = mock_model

    # Same model size -> no unload
    transcriber.set_model_name("small")
    assert transcriber.whisper_model == mock_model
    assert transcriber.model_name == "small"

    # Different model size -> unloads and updates model_name
    transcriber.set_model_name("large-v3")
    assert transcriber.whisper_model is None
    assert transcriber.model_name == "large-v3"


@patch("server.get_settings")
@patch("server.global_transcriber")
@patch("server.os.path.exists", return_value=False)
def test_server_run_job_model_size_resolution(
    mock_exists, mock_transcriber, mock_get_settings
):
    from server import _run_job, jobs

    mock_get_settings.return_value.whisperx_model_size = "large-v2"
    mock_transcriber.transcribe.return_value = []
    mock_transcriber.model_name = "large-v2"

    # Explicit model size provided
    jobs["job_1"] = {"status": "queued"}
    _run_job("job_1", "dummy.mp3", 1, model_size="tiny")
    mock_transcriber.set_model_name.assert_called_with("tiny")

    # No model size provided -> falls back to default settings ("large-v2")
    jobs["job_2"] = {"status": "queued"}
    _run_job("job_2", "dummy.mp3", 1, model_size=None)
    mock_transcriber.set_model_name.assert_called_with("large-v2")


def test_server_get_ctc_transcriber_requires_paths(monkeypatch):
    import server

    monkeypatch.setattr(server, "_ctc_transcriber", None)
    mock_settings = MagicMock()
    mock_settings.fastconformer_model_path = None
    mock_settings.fastconformer_tokenizer_model_path = None
    with patch("server.get_settings", return_value=mock_settings):
        with pytest.raises(ValueError, match="MUNAJJAM_FASTCONFORMER_MODEL_PATH"):
            server._get_ctc_transcriber()


def test_server_get_ctc_transcriber_returns_lazy_transcriber(monkeypatch):
    import server

    monkeypatch.setattr(server, "_ctc_transcriber", None)
    mock_settings = MagicMock()
    mock_settings.fastconformer_model_path = "/models/ctc.onnx"
    mock_settings.fastconformer_vocab_path = "/models/vocab.txt"
    mock_settings.fastconformer_tokenizer_model_path = "/models/tokenizer.model"
    mock_settings.fastconformer_vad_enabled = False
    with patch("server.get_settings", return_value=mock_settings):
        transcriber = server._get_ctc_transcriber()
    assert isinstance(transcriber, FastConformerCTCTranscriber)


@patch("server.os.path.exists", return_value=False)
def test_server_ctc_job_success(mock_exists, monkeypatch):
    import server

    monkeypatch.setattr(server, "_ctc_transcriber", None)
    fake = MagicMock()
    fake.transcribe.return_value = []
    with patch("server._get_ctc_transcriber", return_value=fake):
        server.jobs["ctc_job"] = {"status": "queued"}
        server._run_ctc_job("ctc_job", "dummy.mp3", 1)
    assert server.jobs["ctc_job"]["status"] == "success"
    fake.transcribe.assert_called_once_with("dummy.mp3", surah_id=1)


@patch("server.os.path.exists", return_value=False)
def test_server_ctc_job_error_surfaces(mock_exists, monkeypatch):
    import server

    monkeypatch.setattr(server, "_ctc_transcriber", None)
    fake = MagicMock()
    fake.transcribe.side_effect = TranscriptionError("no tokenizer configured")
    with patch("server._get_ctc_transcriber", return_value=fake):
        server.jobs["ctc_job_err"] = {"status": "queued"}
        server._run_ctc_job("ctc_job_err", "dummy.mp3", 1)
    assert server.jobs["ctc_job_err"]["status"] == "error"
    assert "no tokenizer configured" in server.jobs["ctc_job_err"]["error"]


def test_align_audio_invalid_alignment_mode_400(monkeypatch, tmp_path):
    import server

    monkeypatch.chdir(tmp_path)
    with TestClient(server.app) as client:
        resp = client.post(
            "/align/1",
            files={"file": ("a.wav", b"data", "audio/wav")},
            data={"alignment_mode": "bogus"},
        )
    assert resp.status_code == 400
    assert "Invalid alignment_mode" in resp.json()["message"]


def test_align_audio_ctc_mode_queued(monkeypatch, tmp_path):
    import server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_ctc_transcriber", None)
    fake = MagicMock()
    fake.transcribe.return_value = []
    with patch("server._get_ctc_transcriber", return_value=fake), patch(
        "server.os.path.exists", return_value=False
    ):
        with TestClient(server.app) as client:
            resp = client.post(
                "/align/1",
                files={"file": ("a.wav", b"data", "audio/wav")},
                data={"alignment_mode": "ctc_segmentation"},
            )
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    job_id = resp.json()["job_id"]
    assert server.jobs[job_id]["status"] == "success"
    fake.transcribe.assert_called_once()


def test_align_audio_default_mode_stays_whisperx(monkeypatch, tmp_path):
    import server

    monkeypatch.chdir(tmp_path)
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = []
    mock_transcriber.model_name = "large-v2"
    with patch("server.global_transcriber", mock_transcriber), patch(
        "server.os.path.exists", return_value=False
    ):
        with TestClient(server.app) as client:
            # No alignment_mode -> default WhisperX behavior (backward compat)
            resp = client.post(
                "/align/1",
                files={"file": ("a.wav", b"data", "audio/wav")},
            )
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    job_id = resp.json()["job_id"]
    assert server.jobs[job_id]["status"] == "success"
    mock_transcriber.transcribe.assert_called_once()
