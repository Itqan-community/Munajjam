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
    FastConformerInference,
    FRAME_DURATION_SECONDS,
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
    ):
        self._logprobs_name = logprobs_name
        self._length_name = length_name
        self._n_classes = n_classes
        self._n_frames = n_frames
        self._batch = batch
        self._signal_name = signal_name
        self._length_input_name = length_input_name
        self._length_input_type = length_input_type
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[FakeIO]:
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
    """A cache-enabled (streaming-style) export is rejected with guidance."""
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
    with pytest.raises(TranscriptionError, match="raw-audio"):
        model.load()


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
        model.log_probs(np.zeros(8000, dtype=np.float32))
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


def test_stock_mel_input_export_rejected():
    """NeMo's stock mel-input export (3-D float input) is rejected with a hint."""
    session = FakeSession()
    session.get_inputs = lambda: [  # type: ignore[method-assign]
        FakeIO("audio_signal", ["B", 80, "T"], "tensor(float)"),
        FakeIO("length", ["B"], "tensor(int64)"),
    ]

    model = FastConformerInference(
        model_path="model.onnx",
        session_factory=lambda _p: session,
    )
    with pytest.raises(TranscriptionError, match="raw-audio"):
        model.load()
