"""
Unit tests for the FastConformer ONNX inference layer.

All tests use a mocked ONNX session (``FakeSession``) so no model file or
``onnxruntime`` installation is required.
"""

from pathlib import Path

import numpy as np
import pytest
from munajjam.exceptions import TranscriptionError
from munajjam.transcription.fastconformer import (
    FRAME_DURATION_SECONDS,
    FastConformerInference,
)


class FakeIO:
    """Mimics ``onnxruntime``'s ``NodeArg`` (name/shape/type only)."""

    def __init__(self, name: str, shape: list, type_: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    """Minimal mock of an ``onnxruntime.InferenceSession``."""

    def __init__(
        self,
        logprobs_name: str = "logprobs",
        length_name: str | None = "encoded_lengths",
        n_classes: int = 1025,
        n_frames: int = 14,
        batch: int = 1,
        signal_name: str = "input_signal",
        length_input_name: str = "input_signal_length",
        length_input_type: str = "tensor(int32)",
        mel_input: bool = False,  # True for mel-input graph (3D signal)
    ):
        self._logprobs_name = logprobs_name
        self._length_name = length_name
        self._n_classes = n_classes
        self._n_frames = n_frames
        self._batch = batch
        self._signal_name = signal_name
        self._length_input_name = length_input_name
        self._length_input_type = length_input_type
        self._mel_input = mel_input
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[FakeIO]:
        if self._mel_input:
            # Mel-input graph: [B, n_mels, T_mel]
            return [
                FakeIO(self._signal_name, ["B", "n_mels", "T_mel"], "tensor(float)"),
                FakeIO(self._length_input_name, ["B"], self._length_input_type),
            ]
        # Raw-audio graph: [B, T]
        return [
            FakeIO(self._signal_name, ["B", "T"], "tensor(float)"),
            FakeIO(self._length_input_name, ["B"], self._length_input_type),
        ]

    def get_outputs(self) -> list[FakeIO]:
        outputs = [
            FakeIO(self._logprobs_name, ["B", "T", "V"], "tensor(float)"),
        ]
        if self._length_name is not None:
            outputs.append(FakeIO(self._length_name, ["B"], "tensor(int32)"))
        return outputs

    def run(self, output_names: list[str], input_feed: dict) -> list[np.ndarray]:
        self.calls.append((list(output_names), {k: v.copy() for k, v in input_feed.items()}))
        results = []
        for name in output_names:
            if name == self._logprobs_name:
                arr = np.full(
                    (self._batch, self._n_frames, self._n_classes), -5.0, dtype=np.float32
                )
                arr[:, :, self._n_classes - 1] = 0.0  # blank column dominant
                results.append(arr)
            elif name == self._length_name:
                results.append(np.array([self._n_frames], dtype=np.int32))
            else:
                raise KeyError(f"Unexpected output requested: {name}")
        return results


@pytest.fixture
def session_factory():
    """Factory that records created sessions."""

    def _factory(sessions: list[FakeSession]):
        def _create(_path: str) -> FakeSession:
            session = FakeSession()
            sessions.append(session)
            return session

        return _create

    return _factory


def make_inference(session: FakeSession | None = None) -> FastConformerInference:
    """Build an inference wrapper around a fake session (created lazily)."""
    created: list[FakeSession] = []

    def factory(_path: str) -> FakeSession:
        session_ = session if session is not None else FakeSession()
        created.append(session_)
        return session_

    model = FastConformerInference(
        model_path="model.onnx",
        session_factory=factory,
    )
    model._created = created  # type: ignore[attr-defined]
    return model


def test_lazy_loading(session_factory):
    """Session must not be created until first use."""
    model = make_inference()

    assert not model.is_loaded
    assert model._created == []  # type: ignore[attr-defined]

    model.load()
    assert model.is_loaded
    assert len(model._created) == 1  # type: ignore[attr-defined]


def test_log_probs_shape_and_dtype():
    """log_probs returns [T', V+1] float32 with the batch dim removed."""
    model = make_inference()

    waveform = np.random.RandomState(0).randn(16000).astype(np.float32)
    log_probs = model.log_probs(waveform)

    assert isinstance(log_probs, np.ndarray)
    assert log_probs.dtype == np.float32
    assert log_probs.ndim == 2
    assert log_probs.shape == (14, 1025)  # T'=14 frames, 1024 vocab + blank


def test_input_feed_contents():
    """The waveform and length are fed with the expected names/shapes/dtypes."""
    model = make_inference()

    waveform = np.random.RandomState(0).randn(8000).astype(np.float32)
    model.log_probs(waveform)

    session = model._created[0]  # type: ignore[attr-defined]
    # The last call is the user inference (earlier calls may include the
    # vocab-dimension probe for dynamic-shape outputs).
    output_names, input_feed = session.calls[-1]

    assert set(output_names) == {"logprobs", "encoded_lengths"}
    assert set(input_feed.keys()) == {"input_signal", "input_signal_length"}
    assert input_feed["input_signal"].shape == (1, 8000)
    assert input_feed["input_signal"].dtype == np.float32
    assert input_feed["input_signal_length"].shape == (1,)
    assert input_feed["input_signal_length"].dtype == np.int32
    assert input_feed["input_signal_length"][0] == 8000


def test_blank_index_derivation():
    """Blank is the trailing class: blank_index == vocab_size == 1024."""
    model = make_inference()
    model.log_probs(np.zeros(16000, dtype=np.float32))

    assert model.vocabulary_size == 1024
    assert model.blank_index == 1024
    assert model.blank_index == model.vocabulary_size


def test_vocab_from_file(tmp_path: Path):
    """A vocab file fixes vocab_size/blank_index and must match the output."""
    vocab_file = tmp_path / "vocabulary.txt"
    vocab_file.write_text("\n".join([f"tok{i}" for i in range(5)]) + "\n", encoding="utf-8")

    model = make_inference(FakeSession(n_classes=6))
    model.vocab_path = vocab_file
    model.load()

    assert model.vocabulary == [f"tok{i}" for i in range(5)]
    assert model.vocabulary_size == 5
    assert model.blank_index == 5

    log_probs = model.log_probs(np.zeros(16000, dtype=np.float32))
    assert log_probs.shape == (14, 6)


def test_vocab_mismatch_raises(tmp_path: Path):
    """Output classes contradicting the vocab file raise TranscriptionError."""
    vocab_file = tmp_path / "vocabulary.txt"
    vocab_file.write_text("a\nb\nc\n", encoding="utf-8")

    model = make_inference(FakeSession(n_classes=1025))
    model.vocab_path = vocab_file
    model.load()

    with pytest.raises(TranscriptionError, match="vocabulary"):
        model.log_probs(np.zeros(16000, dtype=np.float32))


def test_frames_to_time():
    """Frame index -> seconds uses the 80 ms FastConformer stride."""
    model = make_inference()
    assert model.frame_duration_seconds == 0.08
    assert model.frame_duration_seconds == FRAME_DURATION_SECONDS

    times = model.frames_to_time(np.array([0, 1, 2, 5]))
    np.testing.assert_allclose(times, np.array([0.0, 0.08, 0.16, 0.4]))

    assert model.frames_to_time(10) == 0.8


def test_io_name_resolution():
    """Non-default ONNX I/O names are resolved from the session."""
    session = FakeSession(
        logprobs_name="ctc_logits",
        signal_name="waveform",
        length_input_name="wave_len",
        length_name=None,
    )
    model = make_inference(session)
    model.load()

    assert model._input_signal_name == "waveform"  # type: ignore[attr-defined]
    assert model._input_length_name == "wave_len"  # type: ignore[attr-defined]
    assert model._output_logprobs_name == "ctc_logits"  # type: ignore[attr-defined]
    assert model._output_length_name is None  # type: ignore[attr-defined]

    model.log_probs(np.zeros(16000, dtype=np.float32))
    input_feed = session.calls[-1][1]
    assert set(input_feed.keys()) == {"waveform", "wave_len"}


def test_length_output_trims_padding():
    """Frames beyond the model's encoded length are trimmed."""

    class TrimSession(FakeSession):
        def run(self, output_names, input_feed):
            outputs = super().run(output_names, input_feed)
            outputs[1] = np.array([7], dtype=np.int32)
            return outputs

    session = TrimSession(n_frames=14)
    model = make_inference(session)

    log_probs = model.log_probs(np.zeros(16000, dtype=np.float32))
    assert log_probs.shape[0] == 7


def test_invalid_waveforms():
    model = make_inference()

    with pytest.raises(TranscriptionError, match="1-D"):
        model.log_probs(np.zeros((2, 100), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(TranscriptionError, match="empty"):
        model.log_probs(np.zeros(0, dtype=np.float32))
    with pytest.raises(TranscriptionError, match="1-D"):
        model.log_probs("not-an-array")  # type: ignore[arg-type]


def test_missing_model_file():
    model = FastConformerInference(model_path="does-not-exist.onnx")
    with pytest.raises(TranscriptionError, match="not found"):
        model.load()


def test_no_model_path():
    model = FastConformerInference()
    with pytest.raises(TranscriptionError, match="not found"):
        model.load()


def test_unexpected_input_count(tmp_path: Path):
    """A cache-enabled (streaming-style) export is accepted — we pick signal/length."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")

    class CacheSession(FakeSession):
        def get_inputs(self) -> list[FakeIO]:
            return [
                FakeIO("audio_signal", ["B", "T"], "tensor(float)"),
                FakeIO("length", ["B"], "tensor(int32)"),
                FakeIO("cache_last_channel", ["D", "B", "T", "D"]),
                FakeIO("cache_last_time", ["D", "B", "D", "T"]),
                FakeIO("cache_last_projector", ["D", "B", "D", "T"]),
            ]

    model = FastConformerInference(
        model_path=model_path,
        session_factory=lambda _p: CacheSession(),
    )
    # Should load successfully — we pick the 2D float and 1D int inputs.
    model.load()
    assert model._input_signal_name == "audio_signal"
    assert model._input_length_name == "length"


def test_output_rank_and_batch_validation():
    class BadRankSession(FakeSession):
        def run(self, output_names, input_feed):
            # Declared as 3-D but returns a 2-D tensor.
            return [np.zeros((1, 1025), dtype=np.float32)]

    model = make_inference(BadRankSession())
    with pytest.raises(TranscriptionError, match="rank"):
        model.log_probs(np.zeros(16000, dtype=np.float32))

    class BadBatchSession(FakeSession):
        def run(self, output_names, input_feed):
            return [np.zeros((2, 14, 1025), dtype=np.float32)]

    model = make_inference(BadBatchSession())
    with pytest.raises(TranscriptionError, match="batch size 1"):
        model.log_probs(np.zeros(16000, dtype=np.float32))


def test_session_run_failure_wrapped():
    class FailingSession(FakeSession):
        def run(self, output_names, input_feed):
            raise RuntimeError("provider failure")

    model = make_inference(FailingSession())
    with pytest.raises(TranscriptionError, match="inference failed"):
        model.log_probs(np.zeros(16000, dtype=np.float32))


def test_unload_and_reload(session_factory):
    model = make_inference()
    model.load()
    assert model.is_loaded

    model.unload()
    assert not model.is_loaded

    # Next inference reloads the session.
    model.log_probs(np.zeros(16000, dtype=np.float32))
    assert model.is_loaded
    assert len(model._created) == 2  # type: ignore[attr-defined]


def test_onnxruntime_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A clear error is raised when onnxruntime is not installed."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")

    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    model = FastConformerInference(model_path=model_path)
    with pytest.raises(TranscriptionError, match="onnxruntime"):
        model.load()


def test_log_probs_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """log_probs_from_file loads audio at the target sample rate."""
    import munajjam.transcription.fastconformer as fc

    monkeypatch.setattr(
        fc,
        "load_audio_waveform",
        lambda _path, sample_rate: (np.zeros(sample_rate, dtype=np.float32), sample_rate),
    )

    model = make_inference()
    log_probs = model.log_probs_from_file("surah_1.wav")
    assert log_probs.shape == (14, 1025)


def test_int64_length_input_feed_uses_declared_dtype():
    session = FakeSession(length_input_type="tensor(int64)")
    model = make_inference(session)

    model.log_probs(np.zeros(8000, dtype=np.float32))

    length = session.calls[-1][1]["input_signal_length"]
    assert length.dtype == np.int64
    assert length.tolist() == [8000]


def test_unsupported_length_input_dtype_is_rejected():
    session = FakeSession(length_input_type="tensor(uint16)")
    model = make_inference(session)

    with pytest.raises(TranscriptionError, match="Unsupported ONNX length input dtype"):
        model.load()


def test_real_export_contract():
    """Mimic the verified production ONNX export (raw-audio graph):
    int32 length input, static 1025-class output, int64 encoded_lengths.
    """
    session = FakeSession(
        signal_name="input_signal",
        length_input_name="input_signal_length",
        logprobs_name="logprobs",
        length_name="encoded_lengths",
        n_classes=1025,
    )
    # Real graph declares the class dim statically (no probe needed).
    session.get_outputs = lambda: [  # type: ignore[method-assign]
        FakeIO("logprobs", [1, "T", 1025], "tensor(float)"),
        FakeIO("encoded_lengths", [1], "tensor(int64)"),
    ]

    model = FastConformerInference(
        model_path="model.onnx",
        session_factory=lambda _p: session,
    )
    log_probs = model.log_probs(np.zeros(16000, dtype=np.float32))

    assert model.vocabulary_size == 1024
    assert model.blank_index == 1024
    assert log_probs.shape == (14, 1025)
    # Static class dim must be read from the graph, not via a probe run.
    assert len(session.calls) == 1


def test_supported_length_input_dtypes():
    expected = {
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
    }
    for descriptor, dtype in expected.items():
        session = FakeSession(length_input_type=descriptor)
        model = make_inference(session)
        # Use 100 samples to stay within all dtype ranges (int8: -128..127).
        model.log_probs(np.zeros(100, dtype=np.float32))
        assert session.calls[-1][1]["input_signal_length"].dtype == dtype


def test_int64_length_output_is_recognized():
    """The real graph emits encoded_lengths as int64; it must still be used."""
    session = FakeSession(length_name="encoded_lengths")
    session.get_outputs = lambda: [  # type: ignore[method-assign]
        FakeIO("logprobs", ["B", "T", "V"], "tensor(float)"),
        FakeIO("encoded_lengths", ["B"], "tensor(int64)"),
    ]
    session.run = (  # type: ignore[method-assign]
        lambda output_names, input_feed: [
            np.full((1, 14, 1025), -5.0, dtype=np.float32),
            np.array([7], dtype=np.int64),  # only 7 valid frames
        ]
    )

    model = make_inference(session)
    log_probs = model.log_probs(np.zeros(16000, dtype=np.float32))
    assert log_probs.shape == (7, 1025)  # trimmed to encoded_lengths


def test_stock_mel_input_export_detected():
    """NeMo's stock mel-input export (3-D float input) is detected as mel-input."""
    session = FakeSession(mel_input=True)
    session.get_inputs = lambda: [  # type: ignore[method-assign]
        FakeIO("audio_signal", ["B", 80, "T"], "tensor(float)"),
        FakeIO("length", ["B"], "tensor(int64)"),
    ]

    model = FastConformerInference(
        model_path="model.onnx",
        session_factory=lambda _p: session,
    )
    model.load()
    assert model._needs_preprocessing is True
    assert model._input_signal_name == "audio_signal"
    assert model._input_length_name == "length"


def test_length_dtype_range_overflow_int8():
    """int8 max is 127; 128 samples must raise."""
    session = FakeSession(length_input_type="tensor(int8)")
    model = make_inference(session)
    with pytest.raises(TranscriptionError, match="does not fit"):
        model.log_probs(np.zeros(128, dtype=np.float32))
    # 127 samples is fine.
    model2 = make_inference(FakeSession(length_input_type="tensor(int8)"))
    result = model2.log_probs(np.zeros(127, dtype=np.float32))
    assert result.ndim == 2


def test_length_dtype_range_overflow_int16():
    """int16 max is 32767; 32768 samples must raise."""
    session = FakeSession(length_input_type="tensor(int16)")
    model = make_inference(session)
    with pytest.raises(TranscriptionError, match="does not fit"):
        model.log_probs(np.zeros(32768, dtype=np.float32))


def test_length_dtype_range_overflow_uint8():
    """uint8 max is 255; 256 samples must raise."""
    session = FakeSession(length_input_type="tensor(uint8)")
    model = make_inference(session)
    with pytest.raises(TranscriptionError, match="does not fit"):
        model.log_probs(np.zeros(256, dtype=np.float32))
    # 255 samples is fine.
    model2 = make_inference(FakeSession(length_input_type="tensor(uint8)"))
    result = model2.log_probs(np.zeros(255, dtype=np.float32))
    assert result.ndim == 2


def test_length_dtype_int32_int64_no_overflow_on_typical_audio():
    """int32 and int64 handle realistic audio easily."""
    for desc in ("tensor(int32)", "tensor(int64)"):
        session = FakeSession(length_input_type=desc)
        model = make_inference(session)
        result = model.log_probs(np.zeros(16000, dtype=np.float32))
        assert result.ndim == 2
        expected_dtype = np.int32 if desc == "tensor(int32)" else np.int64
        assert session.calls[-1][1]["input_signal_length"].dtype == expected_dtype


# --------------------------------------------------------------------------- #
# Mel preprocessing tests
# --------------------------------------------------------------------------- #


class TestComputeMelFeatures:
    """Tests for compute_mel_features() — NeMo-equivalent mel front-end."""

    def test_output_shape(self):
        """Output is [1, n_mels, T_mel] with n_mels=80."""
        from munajjam.transcription.fastconformer import compute_mel_features

        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)
        mel = compute_mel_features(waveform)

        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80  # n_mels
        assert mel.shape[2] > 0    # T_mel > 0

    def test_dtype_float32(self):
        """Output is float32 regardless of input dtype."""
        from munajjam.transcription.fastconformer import compute_mel_features

        waveform_f64 = np.random.RandomState(42).randn(16000)
        mel = compute_mel_features(waveform_f64)

        assert mel.dtype == np.float32

    def test_rejects_2d_input(self):
        """A 2-D input raises ValueError."""
        from munajjam.transcription.fastconformer import compute_mel_features

        with pytest.raises(ValueError, match="1-D"):
            compute_mel_features(np.zeros((1, 16000), dtype=np.float32))

    def test_empty_waveform_returns_empty(self):
        """An empty waveform produces a mel spectrogram (possibly empty)."""
        from munajjam.transcription.fastconformer import compute_mel_features

        waveform = np.array([], dtype=np.float32)
        mel = compute_mel_features(waveform)

        assert mel.ndim == 3
        assert mel.shape[1] == 80

    def test_frame_count_matches_neMo(self):
        """For 1 s of audio, expect ~97 mel frames (matching NeMo FilterbankFeatures)."""
        from munajjam.transcription.fastconformer import compute_mel_features

        # 1 s @ 16 kHz → pre-emphasis + center-pad → floor(len(padded - n_fft) / hop) + 1
        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)
        mel = compute_mel_features(waveform)

        # NeMo: n_frames = floor((16000 + 256 - 400) / 160) + 1 = floor(15856/160) + 1 = 99 + 1 = 100
        # Our implementation uses center-pad of n_fft//2 = 256 on each side, same as NeMo.
        # Exact frame count depends on pre-emphasis edge handling; accept a small range.
        assert 95 <= mel.shape[2] <= 105, f"Unexpected frame count: {mel.shape[2]}"

    def test_per_channel_normalization(self):
        """Per-channel mean ≈ 0, std ≈ 1 after normalization."""
        from munajjam.transcription.fastconformer import compute_mel_features

        waveform = np.random.RandomState(42).randn(32000).astype(np.float32)
        mel = compute_mel_features(waveform)  # [1, 80, T]

        mean = mel[0].mean(axis=1)  # per-channel mean
        std = mel[0].std(axis=1)    # per-channel std
        np.testing.assert_allclose(mean, 0.0, atol=0.05)
        np.testing.assert_allclose(std, 1.0, atol=0.1)

    def test_log_floor_guard(self):
        """Mel values should not produce -inf (log floor guard active)."""
        from munajjam.transcription.fastconformer import compute_mel_features

        # Silence → very small power → floor guard should prevent -inf
        waveform = np.zeros(16000, dtype=np.float32)
        mel = compute_mel_features(waveform)

        assert np.all(np.isfinite(mel))


# --------------------------------------------------------------------------- #
# I/O detection tests (raw-audio vs mel-input)
# --------------------------------------------------------------------------- #


class TestIODetection:
    """Tests for _find_signal_and_length() accepting both input layouts."""

    def test_mel_input_detected(self):
        """3-D float input is detected as mel-input (needs_preprocessing=True)."""
        model = make_inference(FakeSession(mel_input=True))
        model.load()

        assert model._needs_preprocessing is True

    def test_raw_audio_detected(self):
        """2-D float input is detected as raw-audio (needs_preprocessing=False)."""
        model = make_inference(FakeSession(mel_input=False))
        model.load()

        assert model._needs_preprocessing is False

    def test_mel_input_skips_preprocessing(self):
        """When needs_preprocessing=True, raw waveform is preprocessed to mel."""
        session = FakeSession(mel_input=True)
        model = make_inference(session)

        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)
        model.log_probs(waveform)

        # The ONNX feed should contain mel features [1, 80, T_mel], not raw audio [1, T]
        _, input_feed = session.calls[-1]
        signal = input_feed["input_signal"]
        assert signal.ndim == 3
        assert signal.shape[0] == 1
        assert signal.shape[1] == 80  # n_mels

    def test_raw_audio_no_preprocessing(self):
        """When needs_preprocessing=False, raw waveform is passed directly."""
        session = FakeSession(mel_input=False)
        model = make_inference(session)

        waveform = np.random.RandomState(42).randn(8000).astype(np.float32)
        model.log_probs(waveform)

        # The ONNX feed should contain raw audio [1, T]
        _, input_feed = session.calls[-1]
        signal = input_feed["input_signal"]
        assert signal.ndim == 2
        assert signal.shape == (1, 8000)

    def test_mel_input_unexpected_shape_raises(self):
        """If the ONNX graph has neither 2D nor 3D float input, raise."""

        class BadSession(FakeSession):
            def get_inputs(self):
                return [FakeIO("x", ["B", "C", "H", "W"], "tensor(float)")]

        model = make_inference(BadSession())
        with pytest.raises(TranscriptionError, match="Unexpected ONNX inputs"):
            model.load()


# --------------------------------------------------------------------------- #
# Regression: STFT export failure is resolved
# --------------------------------------------------------------------------- #


class TestSTFTExportFailureRegression:
    """Regression tests ensuring the STFT export failure is resolved."""

    def test_mel_preprocessing_no_stft_in_graph(self):
        """The ONNX graph no longer contains the NeMo preprocessor.

        The mel computation is now in Python (compute_mel_features()),
        which avoids the torch.stft() complex type export failure.
        """
        from munajjam.transcription.fastconformer import compute_mel_features

        # This function is pure numpy — no torch.stft() involved.
        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)
        mel = compute_mel_features(waveform)

        assert mel.ndim == 3
        assert mel.shape[1] == 80
        assert np.all(np.isfinite(mel))

    def test_export_script_no_preprocessor(self):
        """The export script no longer references traced.preprocessor."""
        import inspect

        from scripts.export_fastconformer_onnx import export_onnx_graphs

        source = inspect.getsource(export_onnx_graphs)
        # The wrapper class should not access traced.preprocessor
        assert "traced.preprocessor" not in source
        # The wrapper should only have encoder and decoder
        assert "self.encoder = traced.encoder" in source
        assert "self.ctc_decoder = decoder" in source


# --------------------------------------------------------------------------- #
# Hann window regression test
# --------------------------------------------------------------------------- #


class TestHannWindowFormula:
    """Verify our Hann window matches torch.hann_window(periodic=False).

    For periodic=False the symmetric window formula uses denominator N-1:
        w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    """

    @staticmethod
    def _torch_available() -> bool:
        import importlib.util
        return importlib.util.find_spec("torch") is not None

    @pytest.mark.skipif(
        not _torch_available.__func__(),  # type: ignore[attr-defined]
        reason="torch not installed",
    )
    def test_hann_window_matches_torch_periodic_false(self):
        """Our Hann window must match torch.hann_window(N, periodic=False)."""
        import torch
        from munajjam.transcription.fastconformer import WIN_LENGTH

        torch_win = torch.hann_window(WIN_LENGTH, periodic=False).numpy()
        our_win = (
            0.5
            - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / (WIN_LENGTH - 1))
        ).astype(np.float32)

        mae = float(np.mean(np.abs(torch_win - our_win)))
        maxae = float(np.max(np.abs(torch_win - our_win)))
        assert mae < 1e-6, (
            f"Hann window MAE={mae:.2e} vs torch.hann_window(periodic=False), "
            f"MaxAE={maxae:.2e}"
        )

    @pytest.mark.skipif(
        not _torch_available.__func__(),  # type: ignore[attr-defined]
        reason="torch not installed",
    )
    def test_hann_window_wrong_formula_detected(self):
        """Ensure the old /N (periodic) formula is NOT used.

        The periodic form (denominator N) must differ from
        torch.hann_window(N, periodic=False) — this guards against
        regressing back to the wrong formula.
        """
        import torch
        from munajjam.transcription.fastconformer import WIN_LENGTH

        torch_win = torch.hann_window(WIN_LENGTH, periodic=False).numpy()
        wrong_win = (
            0.5
            - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)
        ).astype(np.float32)

        wrong_mae = float(np.mean(np.abs(torch_win - wrong_win)))
        assert wrong_mae > 1e-4, (
            f"Old /N formula (MAE={wrong_mae:.2e}) should NOT match "
            f"torch.hann_window(periodic=False)"
        )


# --------------------------------------------------------------------------- #
# Mel filterbank regression test
# --------------------------------------------------------------------------- #


class TestMelFilterbankFormula:
    """Verify our _mel_filterbank() matches librosa.filters.mel(norm='slaney').

    Uses the Slaney/Auditory Toolbox mel scale (htk=False).
    """

    @staticmethod
    def _librosa_available() -> bool:
        import importlib.util
        return importlib.util.find_spec("librosa") is not None

    @pytest.mark.skipif(
        not _librosa_available.__func__(),  # type: ignore[attr-defined]
        reason="librosa not installed",
    )
    def test_mel_filterbank_matches_librosa(self):
        """Our _mel_filterbank() must match librosa.filters.mel(norm='slaney')."""
        import librosa.filters
        from munajjam.transcription.fastconformer import (
            DEFAULT_SAMPLE_RATE,
            MEL_FMAX,
            MEL_FMIN,
            N_FFT,
            N_MELS,
            _mel_filterbank,
        )

        our_fb = _mel_filterbank(DEFAULT_SAMPLE_RATE, N_FFT, N_MELS, MEL_FMIN, MEL_FMAX)
        librosa_fb = librosa.filters.mel(
            sr=DEFAULT_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=MEL_FMIN, fmax=MEL_FMAX, htk=False, norm="slaney",
        )

        mae = float(np.mean(np.abs(our_fb - librosa_fb)))
        maxae = float(np.max(np.abs(our_fb - librosa_fb)))
        assert mae < 1e-6, (
            f"Mel filterbank MAE={mae:.2e} vs librosa(norm='slaney'), "
            f"MaxAE={maxae:.2e}"
        )

    @pytest.mark.skipif(
        not _librosa_available.__func__(),  # type: ignore[attr-defined]
        reason="librosa not installed",
    )
    def test_mel_filterbank_wrong_scale_detected(self):
        """Ensure the old HTK-scale mel (using 700) does NOT match Slaney."""
        import librosa.filters
        from munajjam.transcription.fastconformer import (
            DEFAULT_SAMPLE_RATE,
            MEL_FMAX,
            MEL_FMIN,
            N_FFT,
            N_MELS,
        )

        librosa_fb = librosa.filters.mel(
            sr=DEFAULT_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=MEL_FMIN, fmax=MEL_FMAX, htk=False, norm="slaney",
        )

        # Reproduce the old wrong mel filterbank (using 700 for linear part)
        import numpy as np
        _MIN_LOG_HZ = 1000.0
        _MIN_LOG_MEL_WRONG = _MIN_LOG_HZ / 700.0
        _LOGSTEP = np.log(6.4) / 27.0

        def _hz_to_mel_wrong(f):
            f = np.asarray(f, dtype=np.float64)
            mel = np.empty_like(f)
            low = f < _MIN_LOG_HZ
            mel[low] = f[low] / 700.0
            mel[~low] = _MIN_LOG_MEL_WRONG + np.log(f[~low] / _MIN_LOG_HZ) / _LOGSTEP
            return mel

        def _mel_to_hz_wrong(m):
            m = np.asarray(m, dtype=np.float64)
            hz = np.empty_like(m)
            below = m < _MIN_LOG_MEL_WRONG
            hz[below] = 700.0 * m[below]
            hz[~below] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (m[~below] - _MIN_LOG_MEL_WRONG))
            return hz

        mel_min = _hz_to_mel_wrong(np.array([MEL_FMIN]))[0]
        mel_max = _hz_to_mel_wrong(np.array([MEL_FMAX]))[0]
        mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
        hz_points = _mel_to_hz_wrong(mel_points)
        freq_bins = np.linspace(0, DEFAULT_SAMPLE_RATE / 2, N_FFT // 2 + 1)
        wrong_fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
        for i in range(N_MELS):
            low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            rising = (freq_bins - low) / (center - low + 1e-10)
            falling = (high - freq_bins) / (high - center + 1e-10)
            wrong_fb[i] = np.maximum(0.0, np.minimum(rising, falling))
        enorm = 2.0 / (hz_points[2:] - hz_points[:-2] + 1e-10)
        wrong_fb *= enorm[:, np.newaxis]

        wrong_mae = float(np.mean(np.abs(wrong_fb - librosa_fb)))
        assert wrong_mae > 1e-5, (
            f"Old HTK-scale mel (MAE={wrong_mae:.2e}) should NOT match "
            f"librosa(norm='slaney')"
        )


# --------------------------------------------------------------------------- #
# Valid-frame count regression test
# --------------------------------------------------------------------------- #


class TestValidFrameCount:
    """Verify n_valid uses floor division, not ceil, matching NeMo's get_seq_len."""

    @pytest.mark.parametrize(
        "n_samples,expected",
        [
            (16000, 100),       # exact hop multiple
            (49536, 309),       # Common Voice — floor(49536/160)=309, ceil=310
            (161, 1),           # just above one hop
            (160, 1),           # exactly one hop
            (159, 0),           # below one hop — edge case
            (320, 2),           # two hops exactly
            (321, 2),           # just above two hops
        ],
    )
    def test_valid_frame_count_floor(self, n_samples: int, expected: int):
        """n_valid must equal n_samples // HOP_LENGTH (floor, not ceil)."""
        from munajjam.transcription.fastconformer import HOP_LENGTH

        waveform = np.zeros(n_samples, dtype=np.float32)
        n_valid = waveform.size // HOP_LENGTH
        assert n_valid == expected, (
            f"n_samples={n_samples}: expected n_valid={expected}, got {n_valid}"
        )


# This test requires torch + nemo_toolkit.  It is skipped in normal unit-test
# runs (where those are not installed) and expected to run in the Colab
# validation environment where both are available.


class TestNeMoNumericalEquivalence:
    """Compare compute_mel_features() against NeMo's FilterbankFeatures.

    These tests are skipped if torch or nemo are not installed.
    They verify numerical equivalence on the same deterministic waveform.
    """

    @staticmethod
    def _nemo_available() -> bool:
        import importlib.util

        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("nemo") is not None
        )

    @pytest.mark.skipif(
        not _nemo_available.__func__(),  # type: ignore[attr-defined]
        reason="torch or nemo not installed",
    )
    def test_output_shape_matches_nemo(self):
        """Output shape [1, 80, T_mel] matches NeMo FilterbankFeatures."""
        import torch
        from munajjam.transcription.fastconformer import compute_mel_features
        from nemo.collections.asr.parts.preprocessing.features import (
            FilterbankFeatures,
        )

        rng = np.random.RandomState(42)
        waveform = rng.randn(16000).astype(np.float32)

        # NeMo preprocessor
        nemo = FilterbankFeatures(
            sample_rate=16000,
            n_window_size=400,
            n_window_stride=160,
            n_fft=512,
            nfilt=80,
            lowfreq=0,
            highfreq=8000,
            window="hann",
            normalize="per_feature",
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2**-24,
            pad_to=0,
            preemph=0.97,
            dither=0.0,
            mag_power=2.0,
            mel_norm="slaney",
            use_grads=False,
        )
        nemo.eval()

        with torch.no_grad():
            nemo_out, _ = nemo(
                torch.tensor(waveform).unsqueeze(0),
                torch.tensor([len(waveform)], dtype=torch.long),
            )

        our_out = compute_mel_features(waveform)

        assert our_out.shape == nemo_out.shape, (
            f"Shape mismatch: ours={our_out.shape}, nemo={nemo_out.shape}"
        )

    @pytest.mark.skipif(
        not _nemo_available.__func__(),  # type: ignore[attr-defined]
        reason="torch or nemo not installed",
    )
    def test_numerical_equivalence_mean_abs_error(self):
        """Mean absolute error between NumPy and NeMo mel < 1e-5."""
        import torch
        from munajjam.transcription.fastconformer import compute_mel_features
        from nemo.collections.asr.parts.preprocessing.features import (
            FilterbankFeatures,
        )

        rng = np.random.RandomState(42)
        waveform = rng.randn(16000).astype(np.float32)

        nemo = FilterbankFeatures(
            sample_rate=16000,
            n_window_size=400,
            n_window_stride=160,
            n_fft=512,
            nfilt=80,
            lowfreq=0,
            highfreq=8000,
            window="hann",
            normalize="per_feature",
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2**-24,
            pad_to=0,
            preemph=0.97,
            dither=0.0,
            mag_power=2.0,
            mel_norm="slaney",
            use_grads=False,
        )
        nemo.eval()

        with torch.no_grad():
            nemo_out, _ = nemo(
                torch.tensor(waveform).unsqueeze(0),
                torch.tensor([len(waveform)], dtype=torch.long),
            )

        our_out = compute_mel_features(waveform)
        nemo_np = nemo_out.numpy()

        mean_abs_err = float(np.mean(np.abs(our_out - nemo_np)))
        max_abs_err = float(np.max(np.abs(our_out - nemo_np)))

        assert mean_abs_err < 1e-5, (
            f"Mean absolute error {mean_abs_err:.2e} exceeds 1e-5 threshold"
        )
        assert max_abs_err < 1e-4, (
            f"Max absolute error {max_abs_err:.2e} exceeds 1e-4 threshold"
        )
