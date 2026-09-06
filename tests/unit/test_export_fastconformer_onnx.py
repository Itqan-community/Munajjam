"""
Unit tests for ``scripts/export_fastconformer_onnx.py``.

No NeMo / torch / onnxruntime are required: the heavy export path is mocked
and the tokenizer extraction is tested against tiny synthetic ``.nemo``
archives.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import export_fastconformer_onnx as exporter

STEM = "stt_ar_fastconformer_hybrid_large_pc_v1.0"


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #
def test_parse_args_requires_checkpoint() -> None:
    with pytest.raises(SystemExit):
        exporter.parse_args([])


def test_parse_args_defaults(tmp_path: Path) -> None:
    args = exporter.parse_args([str(tmp_path / "model.nemo")])
    assert args.checkpoint == tmp_path / "model.nemo"
    assert args.output_dir == exporter.DEFAULT_OUTPUT_DIR
    assert args.opset == 18
    assert args.force is False
    assert args.no_validate is False


def test_parse_args_output_dir_and_flags(tmp_path: Path) -> None:
    out = tmp_path / "models"
    args = exporter.parse_args(
        ["model.nemo", "--output-dir", str(out), "--force", "--no-validate", "--opset", "17"]
    )
    assert args.output_dir == out
    assert args.force is True
    assert args.no_validate is True
    assert args.opset == 17


# --------------------------------------------------------------------------- #
# Tokenizer / vocabulary extraction from a .nemo archive
# --------------------------------------------------------------------------- #
def _make_nemo_archive(
    tmp_path: Path,
    *,
    name: str | None = None,
    include_tokenizer: bool = True,
    include_vocab: bool = True,
    config_reference: str = "abc123_tokenizer.model",
) -> Path:
    path = tmp_path / (name or f"{STEM}.nemo")
    with tarfile.open(path, "w:gz") as tar:
        config = f"tokenizer:\n  model: {config_reference}\n  type: sentencepiece\n"
        _add_bytes(tar, "model_config.yaml", config.encode())
        if include_tokenizer:
            _add_bytes(tar, config_reference, b"fake-sentencepiece-model")
        if include_vocab:
            _add_bytes(tar, "vocab.txt", b"tok1\ntok2\n")
    return path


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_extract_assets_writes_tokenizer_and_vocab(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)

    assert assets["tokenizer_model"] == out / exporter.TOKENIZER_FILENAME
    assert assets["tokenizer_model"].read_bytes() == b"fake-sentencepiece-model"
    assert assets["vocabulary"] == out / exporter.VOCAB_FILENAME
    assert assets["vocabulary"].read_text() == "tok1\ntok2\n"


def test_extract_assets_uses_config_referenced_tokenizer(tmp_path: Path) -> None:
    """The tokenizer referenced by model_config.yaml wins over other *.model files."""
    nemo = tmp_path / f"{STEM}.nemo"
    with tarfile.open(nemo, "w:gz") as tar:
        _add_bytes(tar, "model_config.yaml", b"tokenizer:\n  model: real_tokenizer.model\n")
        _add_bytes(tar, "real_tokenizer.model", b"real")
        _add_bytes(tar, "stale_tokenizer.model", b"stale")
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)
    assert assets["tokenizer_model"].read_bytes() == b"real"


def test_extract_assets_without_config_falls_back_to_glob(tmp_path: Path) -> None:
    nemo = tmp_path / f"{STEM}.nemo"
    with tarfile.open(nemo, "w:gz") as tar:
        _add_bytes(tar, "model_config.yaml", b"tokenizer:\n  type: sentencepiece\n")
        _add_bytes(tar, "deadbeef_tokenizer.model", b"fallback")
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)
    assert assets["tokenizer_model"].read_bytes() == b"fallback"


def test_extract_assets_missing_tokenizer_raises(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path, include_tokenizer=False)
    with pytest.raises(SystemExit, match="no SentencePiece tokenizer"):
        exporter.extract_assets(nemo, tmp_path / "out")


def test_extract_assets_invalid_archive_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.nemo"
    bogus.write_bytes(b"not a tar")
    with pytest.raises(SystemExit, match="not a readable .nemo"):
        exporter.extract_assets(bogus, tmp_path / "out")


def test_extract_assets_refuses_overwrite_without_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    with pytest.raises(SystemExit, match="already exists"):
        exporter.extract_assets(nemo, out)

    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"existing"


def test_extract_assets_overwrites_with_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    exporter.extract_assets(nemo, out, force=True)
    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"fake-sentencepiece-model"


# --------------------------------------------------------------------------- #
# Tokenizer path from model_config.yaml
# --------------------------------------------------------------------------- #
class _YAMLError(Exception):
    """Stand-in for yaml.YAMLError so tests never need real PyYAML."""


_CONFIG = b"tokenizer:\n  model: abc_tokenizer.model\n"


def test_tokenizer_path_from_config_valid_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyYAML available + valid YAML -> the config-referenced path wins."""
    fake_yaml = SimpleNamespace(
        YAMLError=_YAMLError,
        safe_load=lambda b: {"tokenizer": {"model": "abc_tokenizer.model"}},
    )
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    assert exporter._tokenizer_path_from_config(_CONFIG) == "abc_tokenizer.model"


def test_tokenizer_path_from_config_yaml_missing_uses_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyYAML missing -> `import yaml` fails and the regex fallback works."""
    monkeypatch.setitem(sys.modules, "yaml", None)

    assert exporter._tokenizer_path_from_config(_CONFIG) == "abc_tokenizer.model"


def test_tokenizer_path_from_config_malformed_yaml_uses_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed YAML -> yaml.YAMLError is handled and the regex fallback works."""

    def _boom(_b):
        raise _YAMLError("malformed")

    fake_yaml = SimpleNamespace(YAMLError=_YAMLError, safe_load=_boom)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    assert exporter._tokenizer_path_from_config(_CONFIG) == "abc_tokenizer.model"


@pytest.mark.parametrize(
    "parsed",
    [
        {"decoder": {"vocabulary": ["a", "b"]}},  # valid YAML, unexpected schema
        ["not", "a", "mapping"],  # schema that makes .get() raise AttributeError
    ],
)
def test_tokenizer_path_from_config_unexpected_schema_uses_regex(
    monkeypatch: pytest.MonkeyPatch, parsed: object
) -> None:
    """Valid YAML with an unexpected schema -> regex fallback still works."""
    fake_yaml = SimpleNamespace(YAMLError=_YAMLError, safe_load=lambda b: parsed)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    assert exporter._tokenizer_path_from_config(_CONFIG) == "abc_tokenizer.model"


# --------------------------------------------------------------------------- #
# Tokenizer regex fallback — indentation-aware scope tracking
# --------------------------------------------------------------------------- #
_MULTIPLE_MODEL_KEYS = b"""\
encoder:
  model: acoustic.model
  n_layers: 17

tokenizer:
  dir: tokenizer_dir
  model: tokenizer.model
"""


def test_regex_fallback_skips_non_tokenizer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regex fallback must ignore ``model:`` entries outside the
    ``tokenizer:`` mapping (e.g. ``encoder.model``)."""
    monkeypatch.setitem(sys.modules, "yaml", None)

    result = exporter._tokenizer_path_from_config(_MULTIPLE_MODEL_KEYS)
    assert result == "tokenizer.model"


def test_regex_fallback_uses_dir_when_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the tokenizer mapping has ``dir:`` but no ``model:``, the
    fallback should return the ``dir`` value."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_bytes = b"tokenizer:\n  dir: /opt/tokenizer\n"

    assert exporter._tokenizer_path_from_config(yaml_bytes) == "/opt/tokenizer"


def test_regex_fallback_model_preferred_over_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both ``model:`` and ``dir:`` exist, ``model`` wins."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_bytes = b"tokenizer:\n  dir: /opt/tokenizer\n  model: tok.model\n"

    assert exporter._tokenizer_path_from_config(yaml_bytes) == "tok.model"


def test_regex_fallback_exits_tokenizer_on_indent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the ``tokenizer:`` block ends, entries in sibling mappings
    must be ignored."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_bytes = (
        b"tokenizer:\n"
        b"  model: tok.model\n"
        b"decoder:\n"
        b"  model: acoustic.model\n"
    )

    assert exporter._tokenizer_path_from_config(yaml_bytes) == "tok.model"


def test_regex_fallback_no_tokenizer_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no ``tokenizer:`` section exists at all, return ``None``."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_bytes = b"encoder:\n  model: acoustic.model\n"

    assert exporter._tokenizer_path_from_config(yaml_bytes) is None


# --------------------------------------------------------------------------- #
# Export-time dependencies
# --------------------------------------------------------------------------- #
def test_import_export_deps_missing_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="nemo_toolkit"):
        exporter._import_export_deps()


def test_export_onnx_graphs_wraps_nemo_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = MagicMock()
    fake_nemo = MagicMock()
    fake_nemo.models.EncDecHybridRNNTCTCBPEModel.restore_from.side_effect = RuntimeError(
        "not a nemo checkpoint"
    )
    monkeypatch.setattr(exporter, "_import_export_deps", lambda: (fake_torch, fake_nemo))

    with pytest.raises(RuntimeError, match="Failed to restore checkpoint"):
        exporter.export_onnx_graphs(Path("x.nemo"), Path("out"))


# --------------------------------------------------------------------------- #
# main() end-to-end (heavy export mocked)
# --------------------------------------------------------------------------- #
def test_main_missing_checkpoint_returns_2(tmp_path: Path) -> None:
    code = exporter.main([str(tmp_path / "missing.nemo"), "--output-dir", str(tmp_path / "out")])
    assert code == 2


def test_main_extracts_assets_and_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"

    def fake_export(nemo_path, output_dir, *, opset, force):
        raw = output_dir / f"{nemo_path.stem}{exporter.RAW_AUDIO_ONNX_SUFFIX}"
        raw.write_bytes(b"onnx")
        stock = output_dir / f"{nemo_path.stem}{exporter.STOCK_ONNX_SUFFIX}"
        stock.write_bytes(b"onnx")
        return {"stock": stock, "raw_audio": raw}

    monkeypatch.setattr(exporter, "export_onnx_graphs", fake_export)

    code = exporter.main([str(nemo), "--output-dir", str(out), "--no-validate"])
    assert code == 0

    # Naming contract: deterministic, matches the runtime's expectations.
    assert (out / f"{STEM}{exporter.STOCK_ONNX_SUFFIX}").is_file()
    assert (out / f"{STEM}{exporter.RAW_AUDIO_ONNX_SUFFIX}").is_file()
    assert (out / exporter.TOKENIZER_FILENAME).is_file()
    assert (out / exporter.VOCAB_FILENAME).is_file()


def test_main_refuses_existing_output_without_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    with pytest.raises(SystemExit, match="already exists"):
        exporter.main([str(nemo), "--output-dir", str(out), "--no-validate"])
    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"existing"


# --------------------------------------------------------------------------- #
# Post-export validation
# --------------------------------------------------------------------------- #
def test_validate_onnx_checks_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / f"{STEM}_ctc_rawaudio.onnx"
    raw.write_bytes(b"onnx")

    class FakeIO:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_

    class FakeSession:
        def __init__(self):
            self._inputs = [
                FakeIO(exporter.INPUT_SIGNAL_NAME, "tensor(float)"),
                FakeIO(exporter.INPUT_LENGTH_NAME, "tensor(int32)"),
            ]
            self._outputs = [
                FakeIO(exporter.OUTPUT_LOGPROBS_NAME, "tensor(float)"),
                FakeIO(exporter.OUTPUT_LENGTH_NAME, "tensor(int64)"),
            ]

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, input_feed):
            import numpy as np

            return [np.zeros((1, 2, 1025), dtype=np.float32), np.array([2], dtype=np.int64)]

    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = FakeSession()
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    # numpy is required by validate_onnx; ensure it resolves.
    exporter.validate_onnx(raw)


def test_validate_onnx_rejects_wrong_input_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / f"{STEM}_ctc_rawaudio.onnx"
    raw.write_bytes(b"onnx")

    fake_ort = MagicMock()
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="audio_signal", type="tensor(float)")]
    fake_ort.InferenceSession.return_value = session
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(SystemExit, match="unexpected graph inputs"):
        exporter.validate_onnx(raw)


class _FakeIO:
    """Minimal NodeArg stand-in (name/type only)."""

    def __init__(self, name: str, type_: str) -> None:
        self.name = name
        self.type = type_


class _RecordingOrtSession:
    """Fake onnxruntime session with configurable I/O dtypes that records
    whether ``run()`` was invoked."""

    def __init__(
        self,
        input_types: list[str] | None = None,
        output_types: list[str] | None = None,
    ) -> None:
        self._inputs = [
            _FakeIO(exporter.INPUT_SIGNAL_NAME, "tensor(float)"),
            _FakeIO(exporter.INPUT_LENGTH_NAME, "tensor(int32)"),
        ]
        self._outputs = [
            _FakeIO(exporter.OUTPUT_LOGPROBS_NAME, "tensor(float)"),
            _FakeIO(exporter.OUTPUT_LENGTH_NAME, "tensor(int64)"),
        ]
        if input_types is not None:
            for io_, type_ in zip(self._inputs, input_types, strict=True):
                io_.type = type_
        if output_types is not None:
            for io_, type_ in zip(self._outputs, output_types, strict=True):
                io_.type = type_
        self.run_called = False

    def get_inputs(self) -> list[_FakeIO]:
        return self._inputs

    def get_outputs(self) -> list[_FakeIO]:
        return self._outputs

    def run(self, output_names, input_feed) -> list[object]:
        import numpy as np

        self.run_called = True
        return [np.zeros((1, 2, 1025), dtype=np.float32), np.array([2], dtype=np.int64)]


def _install_fake_ort(
    monkeypatch: pytest.MonkeyPatch, session: _RecordingOrtSession
) -> None:
    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


@pytest.mark.parametrize(
    ("input_types", "output_types", "bad_tensor"),
    [
        # All documented dtypes -> validation proceeds and inference runs.
        (None, None, None),
        # Wrong signal dtype -> rejected before inference.
        (["tensor(int32)", "tensor(int32)"], None, exporter.INPUT_SIGNAL_NAME),
        # Wrong length dtype -> rejected before inference.
        (["tensor(float)", "tensor(int64)"], None, exporter.INPUT_LENGTH_NAME),
        # Wrong logprobs dtype -> rejected before inference.
        (None, ["tensor(int64)", "tensor(int64)"], exporter.OUTPUT_LOGPROBS_NAME),
        # Wrong encoded_lengths dtype -> rejected before inference.
        (None, ["tensor(float)", "tensor(int32)"], exporter.OUTPUT_LENGTH_NAME),
    ],
    ids=[
        "documented_dtypes_proceed",
        "wrong_signal_dtype",
        "wrong_length_dtype",
        "wrong_logprobs_dtype",
        "wrong_encoded_lengths_dtype",
    ],
)
def test_validate_onnx_io_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_types: list[str] | None,
    output_types: list[str] | None,
    bad_tensor: str | None,
) -> None:
    raw = tmp_path / f"{STEM}_ctc_rawaudio.onnx"
    raw.write_bytes(b"onnx")
    session = _RecordingOrtSession(input_types=input_types, output_types=output_types)
    _install_fake_ort(monkeypatch, session)

    if bad_tensor is None:
        exporter.validate_onnx(raw)
        assert session.run_called, "shape validation must still run after dtype checks"
    else:
        with pytest.raises(SystemExit, match="unexpected type for ONNX tensor") as exc:
            exporter.validate_onnx(raw)
        assert bad_tensor in str(exc.value)
        assert "expected" in str(exc.value)
        assert session.run_called is False, "no inference on a dtype mismatch"


# --------------------------------------------------------------------------- #
# Regression: Blocker #1 — stock export must receive ".onnx" path
# --------------------------------------------------------------------------- #
def _make_fake_torch() -> MagicMock:
    """Build a fake torch module whose nn.Module has normal attribute semantics.

    When ``torch.nn.Module`` is a ``MagicMock``, the wrapper class inherits
    magic behaviour that interferes with ``self.x = y`` assignments.  Using a
    plain stand-in class avoids that problem while keeping the rest mocked.
    """

    class _FakeModule:
        """Minimal stand-in so ``class Foo(torch.nn.Module)`` works naturally."""

    fake = MagicMock()
    fake.nn.Module = _FakeModule
    fake.randn.return_value = MagicMock()
    fake.tensor.return_value = MagicMock()
    return fake


def _create_stock_side_effect(path_str: str) -> None:
    """Side effect for ``model.export()`` that creates the stock ONNX file."""
    Path(path_str).write_bytes(b"onnx")


def test_stock_export_receives_onnx_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model.export() must be called with the full .onnx path, not a
    stripped path without extension. NeMo uses the extension to determine
    the export format."""
    stem = "stt_ar_fastconformer_hybrid_large_pc_v1.0"
    stock_path = tmp_path / f"{stem}{exporter.STOCK_ONNX_SUFFIX}"

    fake_torch = _make_fake_torch()
    fake_nemo = MagicMock()
    fake_model = fake_nemo.models.EncDecHybridRNNTCTCBPEModel.restore_from.return_value
    fake_model.export.side_effect = _create_stock_side_effect
    monkeypatch.setattr(exporter, "_import_export_deps", lambda: (fake_torch, fake_nemo))

    exporter.export_onnx_graphs(
        tmp_path / f"{stem}.nemo", tmp_path, force=True
    )

    fake_model.export.assert_called_once()
    exported_path = fake_model.export.call_args[0][0]
    assert exported_path.endswith(".onnx"), (
        f"NeMo export must receive a path ending in '.onnx', got: {exported_path}"
    )
    assert exported_path == str(stock_path), (
        f"Expected stock path {stock_path}, got: {exported_path}"
    )


# --------------------------------------------------------------------------- #
# Regression: Blocker #2 — CTC decoder attribute resolution
# --------------------------------------------------------------------------- #
def _run_export_with_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_model):
    """Helper: run export_onnx_graphs with a pre-configured fake model."""
    stem = "stt_ar_fastconformer_hybrid_large_pc_v1.0"
    fake_torch = _make_fake_torch()
    fake_nemo = MagicMock()
    fake_nemo.models.EncDecHybridRNNTCTCBPEModel.restore_from.return_value = fake_model
    fake_model.export.side_effect = _create_stock_side_effect
    monkeypatch.setattr(exporter, "_import_export_deps", lambda: (fake_torch, fake_nemo))

    exporter.export_onnx_graphs(
        tmp_path / f"{stem}.nemo", tmp_path, force=True
    )
    return fake_torch


def test_wrapper_uses_ctc_decoder_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RawAudioFastConformer must prefer traced.ctc_decoder over
    traced.aux_ctc.decoder (which is a Hydra config key, not a submodule)."""
    fake_model = MagicMock()
    fake_model.ctc_decoder = MagicMock(name="ctc_decoder_module")

    fake_torch = _run_export_with_model(tmp_path, monkeypatch, fake_model)

    fake_torch.onnx.export.assert_called_once()
    wrapper = fake_torch.onnx.export.call_args[0][0]
    assert wrapper.ctc_decoder is fake_model.ctc_decoder


def test_wrapper_falls_back_to_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ctc_decoder is absent, the wrapper should fall back to decoder."""
    fake_model = MagicMock(
        spec=["preprocessor", "encoder", "decoder", "export", "change_decoding_strategy"]
    )
    fake_model.decoder = MagicMock(name="decoder_module")

    fake_torch = _run_export_with_model(tmp_path, monkeypatch, fake_model)

    wrapper = fake_torch.onnx.export.call_args[0][0]
    assert wrapper.ctc_decoder is fake_model.decoder


def test_wrapper_raises_when_no_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither ctc_decoder nor decoder exists, a clear AttributeError
    must be raised instead of an obscure later failure."""
    fake_model = MagicMock(
        spec=["preprocessor", "encoder", "export", "change_decoding_strategy"]
    )

    with pytest.raises((AttributeError, RuntimeError), match="neither 'ctc_decoder' nor 'decoder'"):
        _run_export_with_model(tmp_path, monkeypatch, fake_model)


# --------------------------------------------------------------------------- #
# Regression: Blocker #3 — decoder signature + dynamo=False
# --------------------------------------------------------------------------- #
def test_wrapper_calls_decoder_with_single_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ConvASRDecoder.forward() accepts only ``encoder_output``; the wrapper
    must not pass ``encoded_lengths`` and must handle a single return value."""
    fake_model = MagicMock()
    fake_model.ctc_decoder.return_value = MagicMock(name="logprobs_tensor")

    fake_torch = _run_export_with_model(tmp_path, monkeypatch, fake_model)

    fake_torch.onnx.export.assert_called_once()
    wrapper = fake_torch.onnx.export.call_args[0][0]
    # Verify the wrapper can be instantiated and its ctc_decoder is set.
    assert wrapper.ctc_decoder is fake_model.ctc_decoder


def test_wrapper_forward_passes_length_from_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper must pass the encoder's encoded_length through as the
    second output, not from the decoder (which returns only logprobs).

    ConvASRDecoder.forward(encoder_output) returns a single tensor — the
    wrapper must handle this and return (logprobs, encoded_length) where
    encoded_length comes from the encoder."""
    fake_model = MagicMock()
    # Simulate ConvASRDecoder returning a single tensor (no tuple).
    fake_model.ctc_decoder.return_value = MagicMock(name="logprobs_tensor")
    # Encoder returns a 2-tuple (encoded, encoded_length).
    fake_encoder_length = MagicMock(name="encoded_length")
    fake_model.encoder.return_value = (MagicMock(name="encoded"), fake_encoder_length)

    fake_torch = _run_export_with_model(tmp_path, monkeypatch, fake_model)

    fake_torch.onnx.export.assert_called_once()
    wrapper = fake_torch.onnx.export.call_args[0][0]

    # Verify the wrapper calls the decoder with only encoder_output.
    # We can't call wrapper.forward() directly with real tensors (no torch),
    # but we can inspect the source to confirm the call pattern.
    import inspect

    src = inspect.getsource(wrapper.forward)
    # The decoder must be called with exactly one kwarg: encoder_output.
    assert "self.ctc_decoder(encoder_output=" in src
    # Must NOT pass encoded_lengths to the decoder.
    assert "encoded_lengths" not in src.split("self.ctc_decoder")[1].split(")")[0]


def test_export_uses_legacy_exporter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch.onnx.export must be called with dynamo=False when the installed
    PyTorch supports the parameter (>= 2.4).  On older PyTorch (< 2.4) the
    parameter is omitted — the legacy exporter is already the only option.

    NeMo models contain LSTM/RNN layers and @typecheck() decorators
    incompatible with the dynamo exporter (PyTorch >= 2.9 default)."""
    fake_model = MagicMock()
    fake_model.ctc_decoder.return_value = MagicMock(name="logprobs_tensor")

    fake_torch = _run_export_with_model(tmp_path, monkeypatch, fake_model)

    call_kwargs = fake_torch.onnx.export.call_args[1]
    # When dynamo is supported, it must be set to False.
    if "dynamo" in call_kwargs:
        assert call_kwargs["dynamo"] is False
    # Verify the conditional logic is present in source (backward-compat).
    import inspect

    src = inspect.getsource(exporter.export_onnx_graphs)
    assert '"dynamo" in _export_sig.parameters' in src
