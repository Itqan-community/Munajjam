"""
Unit tests for FastConformer model provisioning
(``munajjam.transcription.fastconformer_models``).

All tests are offline: downloads are mocked, cache directories are tmp_paths.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
from munajjam.exceptions import ConfigurationError
from munajjam.transcription.fastconformer_models import (
    FASTCONFORMER_MODEL_FILENAME,
    FASTCONFORMER_TOKENIZER_FILENAME,
    FASTCONFORMER_VOCAB_FILENAME,
    FastConformerAssets,
    fastconformer_cache_dir,
    resolve_fastconformer_assets,
)

MODEL_STEM = "stt_ar_fastconformer_hybrid_large_pc_v1.0"


class FakeSettings:
    """Minimal settings stand-in exposing only the provisioning knobs."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        vocab_path: str | None = None,
        cache_dir: str | None = None,
        hf_repo_id: str | None = None,
        hf_revision: str | None = None,
    ) -> None:
        self.fastconformer_model_path = model_path
        self.fastconformer_tokenizer_model_path = tokenizer_path
        self.fastconformer_vocab_path = vocab_path
        self.fastconformer_cache_dir = cache_dir
        self.fastconformer_hf_repo_id = hf_repo_id
        self.fastconformer_hf_revision = hf_revision


def _write_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / f"{MODEL_STEM}_ctc_rawaudio.onnx"
    model.write_bytes(b"onnx")
    tokenizer = tmp_path / FASTCONFORMER_TOKENIZER_FILENAME
    tokenizer.write_bytes(b"sp")
    vocab = tmp_path / FASTCONFORMER_VOCAB_FILENAME
    vocab.write_bytes(b"tokens")
    return model, tokenizer, vocab


# --------------------------------------------------------------------------- #
# Explicit env paths
# --------------------------------------------------------------------------- #
def test_explicit_paths_win(tmp_path: Path) -> None:
    model, tokenizer, vocab = _write_files(tmp_path)
    settings = FakeSettings(
        model_path=str(model),
        tokenizer_path=str(tokenizer),
        vocab_path=str(vocab),
        cache_dir=str(tmp_path / "other_cache"),
    )
    assets = resolve_fastconformer_assets(settings)
    assert assets == FastConformerAssets(
        model_path=model, tokenizer_model_path=tokenizer, vocab_path=vocab
    )


def test_explicit_paths_without_vocab(tmp_path: Path) -> None:
    model, tokenizer, _ = _write_files(tmp_path)
    settings = FakeSettings(model_path=str(model), tokenizer_path=str(tokenizer))
    assets = resolve_fastconformer_assets(settings)
    assert assets.model_path == model
    assert assets.tokenizer_model_path == tokenizer
    assert assets.vocab_path is None


def test_explicit_invalid_model_path_raises(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"sp")
    settings = FakeSettings(
        model_path=str(tmp_path / "missing.onnx"), tokenizer_path=str(tokenizer)
    )
    with pytest.raises(ConfigurationError, match="ONNX model not found"):
        resolve_fastconformer_assets(settings)


def test_explicit_invalid_tokenizer_path_raises(tmp_path: Path) -> None:
    model, _, _ = _write_files(tmp_path)
    settings = FakeSettings(
        model_path=str(model), tokenizer_path=str(tmp_path / "missing.model")
    )
    with pytest.raises(ConfigurationError, match="tokenizer not found"):
        resolve_fastconformer_assets(settings)


def test_explicit_invalid_vocab_raises(tmp_path: Path) -> None:
    model, tokenizer, _ = _write_files(tmp_path)
    settings = FakeSettings(
        model_path=str(model),
        tokenizer_path=str(tokenizer),
        vocab_path=str(tmp_path / "missing.txt"),
    )
    with pytest.raises(ConfigurationError, match="vocabulary file not found"):
        resolve_fastconformer_assets(settings)


def test_partial_explicit_config_raises(tmp_path: Path) -> None:
    model, _, _ = _write_files(tmp_path)
    # Only the model is set -> must fail, not silently fall back to the cache.
    settings = FakeSettings(model_path=str(model))
    with pytest.raises(ConfigurationError, match="TOKENIZER_MODEL_PATH is not set"):
        resolve_fastconformer_assets(settings)

    settings = FakeSettings(tokenizer_path=str(tmp_path / "tokenizer.model"))
    with pytest.raises(ConfigurationError, match="MODEL_PATH is not set"):
        resolve_fastconformer_assets(settings)


# --------------------------------------------------------------------------- #
# Cache reuse
# --------------------------------------------------------------------------- #
def test_cache_reused_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    model, tokenizer, vocab = _write_files(cache)

    def _fail(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("cache hit must not trigger a download")

    monkeypatch.setattr(
        "munajjam.transcription.fastconformer_models._provision_from_hf", _fail
    )
    assets = resolve_fastconformer_assets(FakeSettings(cache_dir=str(cache)))
    assert assets == FastConformerAssets(
        model_path=model, tokenizer_model_path=tokenizer, vocab_path=vocab
    )


def test_partial_cache_falls_through_to_error(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / FASTCONFORMER_MODEL_FILENAME).write_bytes(b"onnx")  # no tokenizer
    with pytest.raises(ConfigurationError, match="not available"):
        resolve_fastconformer_assets(FakeSettings(cache_dir=str(cache)))


def test_empty_cache_file_is_not_usable(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / FASTCONFORMER_MODEL_FILENAME).write_bytes(b"")
    (cache / FASTCONFORMER_TOKENIZER_FILENAME).write_bytes(b"")
    with pytest.raises(ConfigurationError, match="not available"):
        resolve_fastconformer_assets(FakeSettings(cache_dir=str(cache)))


# --------------------------------------------------------------------------- #
# Missing everything -> actionable error
# --------------------------------------------------------------------------- #
def test_missing_everything_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_fastconformer_assets(FakeSettings(cache_dir=str(tmp_path / "empty")))
    message = str(excinfo.value)
    assert "MUNAJJAM_FASTCONFORMER_MODEL_PATH" in message
    assert "export_fastconformer_onnx.py" in message
    # No fake download URLs are invented.
    assert "huggingface.co" not in message


def test_default_cache_dir_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert fastconformer_cache_dir(FakeSettings()) == (
        tmp_path / ".cache" / "munajjam" / "fastconformer"
    )


def test_cache_dir_override_wins(tmp_path: Path) -> None:
    assert fastconformer_cache_dir(FakeSettings(cache_dir=str(tmp_path))) == tmp_path


# --------------------------------------------------------------------------- #
# Hugging Face download (opt-in)
# --------------------------------------------------------------------------- #
def _install_fake_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fail: bool = False) -> None:
    fake = ModuleType("huggingface_hub")
    calls: list[tuple[str, str]] = []

    def hf_hub_download(
        repo_id: str, filename: str, revision: str | None = None, local_dir=None
    ) -> str:
        calls.append((repo_id, filename))
        if fail:
            raise ConnectionError("boom: no network")
        dest = Path(local_dir) / filename
        dest.write_bytes(b"downloaded")
        return str(dest)

    fake.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    fake._calls = calls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


def test_hf_repo_downloads_canonical_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    _install_fake_hf(monkeypatch, cache)
    settings = FakeSettings(cache_dir=str(cache), hf_repo_id="some/org-repo")
    assets = resolve_fastconformer_assets(settings)

    assert assets.model_path.name == FASTCONFORMER_MODEL_FILENAME
    assert assets.tokenizer_model_path.name == FASTCONFORMER_TOKENIZER_FILENAME
    assert assets.model_path.is_file()
    assert assets.tokenizer_model_path.is_file()


def test_hf_download_failure_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    _install_fake_hf(monkeypatch, cache, fail=True)
    settings = FakeSettings(cache_dir=str(cache), hf_repo_id="some/org-repo")
    with pytest.raises(ConfigurationError, match="Failed to download"):
        resolve_fastconformer_assets(settings)


def test_hf_not_configured_no_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    _install_fake_hf(monkeypatch, cache)
    with pytest.raises(ConfigurationError, match="not available"):
        resolve_fastconformer_assets(FakeSettings(cache_dir=str(cache)))
    fake = sys.modules["huggingface_hub"]
    assert fake._calls == []  # type: ignore[attr-defined]


def test_hf_missing_package_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    settings = FakeSettings(cache_dir=str(tmp_path), hf_repo_id="some/org-repo")
    with pytest.raises(ConfigurationError, match="huggingface_hub"):
        resolve_fastconformer_assets(settings)


# --------------------------------------------------------------------------- #
# Concurrency: single provisioning under parallel first access
# --------------------------------------------------------------------------- #
def test_concurrent_first_resolution_downloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two threads racing a cold cache must trigger exactly one download: the
    loser re-checks the cache under the provisioning lock and reuses it."""
    cache = tmp_path / "cache"
    cache.mkdir()
    start = threading.Barrier(2)  # both threads reach resolve() together
    call_count = 0
    count_lock = threading.Lock()

    def slow_provision(_settings, cache_dir: Path) -> FastConformerAssets | None:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.2)  # keep the lock held so the second thread must wait
        model = cache_dir / FASTCONFORMER_MODEL_FILENAME
        tokenizer = cache_dir / FASTCONFORMER_TOKENIZER_FILENAME
        model.write_bytes(b"onnx")
        tokenizer.write_bytes(b"sp")
        return FastConformerAssets(model_path=model, tokenizer_model_path=tokenizer)

    monkeypatch.setattr(
        "munajjam.transcription.fastconformer_models._provision_from_hf", slow_provision
    )

    results: list[FastConformerAssets] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            start.wait(timeout=10)
            results.append(resolve_fastconformer_assets(FakeSettings(cache_dir=str(cache))))
        except Exception as e:  # noqa: BLE001 - collect for assertion
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors
    assert len(results) == 2
    assert call_count == 1, "the second thread must reuse the cache, not re-download"
