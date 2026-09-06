"""
FastConformer model provisioning (ONNX graph + SentencePiece tokenizer).

Resolves the assets needed by the CTC segmentation backend
(:class:`~munajjam.transcription.ctc_segmentation.FastConformerCTCTranscriber`)
in one place, so download/cache logic is not scattered across ``server.py``,
``whisperFactory.py`` or the inference layer.

Resolution order (first match wins):

1. **Explicit paths** — ``MUNAJJAM_FASTCONFORMER_MODEL_PATH`` (+
   ``MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH``, optional
   ``MUNAJJAM_FASTCONFORMER_VOCAB_PATH``). Explicit configuration always wins
   and is validated: a half-configured or invalid setup raises
   :class:`~munajjam.exceptions.ConfigurationError` instead of silently
   falling back.
2. **Provisioning cache** — a deterministic cache directory
   (``MUNAJJAM_FASTCONFORMER_CACHE_DIR`` or ``~/.cache/munajjam/fastconformer``)
   containing the canonical filenames. Cached assets are reused as-is; no
   re-download.
3. **Hugging Face** — only when ``MUNAJJAM_FASTCONFORMER_HF_REPO_ID`` is set
   to a repo that actually hosts the pre-exported files (none is assumed by
   default; see docs/fastconformer-onnx-validation.md). Uses ``huggingface_hub``
   (concurrency-safe, cache-aware) pinned to a revision when configured.
4. **Actionable error** — a :class:`~munajjam.exceptions.ConfigurationError`
   explaining how to run ``scripts/export_fastconformer_onnx.py`` (no fake
   URLs are invented).

Concurrency: resolution is guarded by a module-level lock so concurrent first
requests (e.g. two parallel ``/align`` jobs) never double-download or
double-initialize the model assets.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from munajjam.config import MunajjamSettings, get_settings
from munajjam.exceptions import ConfigurationError
from munajjam.transcription.fastconformer import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)

# Canonical (deterministic) model stem derived from the reference checkpoint,
# e.g. "stt_ar_fastconformer_hybrid_large_pc_v1.0". The export script writes
# "<stem>_ctc_rawaudio.onnx" for this checkpoint; the cache looks for the same
# name so a manual export can be dropped straight into the cache directory.
FASTCONFORMER_MODEL_STEM = DEFAULT_MODEL_ID.split("/")[-1]

# Canonical filenames expected inside the provisioning cache directory and in
# a configured Hugging Face repo (see docs/fastconformer-onnx-validation.md).
FASTCONFORMER_MODEL_FILENAME = f"{FASTCONFORMER_MODEL_STEM}_ctc_rawaudio.onnx"
FASTCONFORMER_TOKENIZER_FILENAME = "tokenizer.model"
FASTCONFORMER_VOCAB_FILENAME = "vocabulary.txt"

EXPORT_SCRIPT_HINT = (
    "Export it yourself with:\n"
    "  python scripts/export_fastconformer_onnx.py <checkpoint.nemo> "
    "--output-dir <cache-or-any-directory>\n"
    "then either set MUNAJJAM_FASTCONFORMER_MODEL_PATH and "
    "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH to the produced files, or "
    "write them into the provisioning cache directory with the canonical "
    f"filenames {FASTCONFORMER_MODEL_FILENAME!r} and "
    f"{FASTCONFORMER_TOKENIZER_FILENAME!r}."
)

_provision_lock = threading.Lock()


@dataclass(frozen=True)
class FastConformerAssets:
    """Resolved paths for the FastConformer CTC backend."""

    model_path: Path
    tokenizer_model_path: Path
    vocab_path: Path | None = None


def fastconformer_cache_dir(settings: MunajjamSettings | None = None) -> Path:
    """Return the provisioning cache directory (deterministic)."""
    settings = settings or get_settings()
    if settings.fastconformer_cache_dir:
        return Path(settings.fastconformer_cache_dir)
    return Path.home() / ".cache" / "munajjam" / "fastconformer"


def _expected_cache_files(
    cache_dir: Path,
) -> tuple[Path, Path, Path | None]:
    """Canonical cache filenames (model, tokenizer, optional vocabulary)."""
    model = cache_dir / FASTCONFORMER_MODEL_FILENAME
    tokenizer = cache_dir / FASTCONFORMER_TOKENIZER_FILENAME
    vocab = cache_dir / FASTCONFORMER_VOCAB_FILENAME
    return model, tokenizer, vocab if vocab.is_file() else None


def resolve_fastconformer_assets(
    settings: MunajjamSettings | None = None,
) -> FastConformerAssets:
    """
    Resolve (and, when needed, provision) the FastConformer CTC assets.

    Precedence: explicit env paths -> cache -> configured Hugging Face repo ->
    actionable :class:`~munajjam.exceptions.ConfigurationError`.

    Raises:
        ConfigurationError: When no asset source is available, explicit paths
            are invalid, or an automatic download fails. The message is safe
            to surface to API clients (no stack traces, no secrets).
    """
    settings = settings or get_settings()

    explicit = _resolve_explicit(settings)
    if explicit is not None:
        return explicit

    cache_dir = fastconformer_cache_dir(settings)
    cached = _resolve_from_cache(cache_dir)
    if cached is not None:
        return cached

    with _provision_lock:
        # Another thread may have provisioned the cache while we waited.
        cached = _resolve_from_cache(cache_dir)
        if cached is not None:
            return cached
        downloaded = _provision_from_hf(settings, cache_dir)
        if downloaded is not None:
            return downloaded

    raise _provisioning_error(cache_dir)


# --------------------------------------------------------------------------- #
# 1. Explicit paths
# --------------------------------------------------------------------------- #
def _resolve_explicit(settings: MunajjamSettings) -> FastConformerAssets | None:
    """Validate and return explicitly configured paths, if any are set.

    Explicit configuration wins, but it must be *complete* and *valid*:
    partial configuration (one path set, the other not) or missing files raise
    :class:`~munajjam.exceptions.ConfigurationError` rather than silently
    falling back to the cache.
    """
    model = settings.fastconformer_model_path
    tokenizer = settings.fastconformer_tokenizer_model_path
    if not model and not tokenizer:
        return None

    if not model:
        raise ConfigurationError(
            "MUNAJJAM_FASTCONFORMER_MODEL_PATH is not set but "
            "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH is. Set both to the "
            "exported ONNX graph and the SentencePiece tokenizer.model.",
            setting_name="fastconformer_model_path",
        )
    if not tokenizer:
        raise ConfigurationError(
            "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH is not set but "
            "MUNAJJAM_FASTCONFORMER_MODEL_PATH is. Set both to the exported "
            "ONNX graph and the SentencePiece tokenizer.model.",
            setting_name="fastconformer_tokenizer_model_path",
        )

    model_path = Path(model)
    if not model_path.is_file():
        raise ConfigurationError(
            f"FastConformer ONNX model not found: {model_path}",
            setting_name="fastconformer_model_path",
        )
    tokenizer_path = Path(tokenizer)
    if not tokenizer_path.is_file():
        raise ConfigurationError(
            f"FastConformer SentencePiece tokenizer not found: {tokenizer_path}",
            setting_name="fastconformer_tokenizer_model_path",
        )

    vocab_path: Path | None = None
    if settings.fastconformer_vocab_path:
        vocab_path = Path(settings.fastconformer_vocab_path)
        if not vocab_path.is_file():
            raise ConfigurationError(
                f"FastConformer vocabulary file not found: {vocab_path}",
                setting_name="fastconformer_vocab_path",
            )

    logger.info(
        "Using explicit FastConformer assets: model=%s tokenizer=%s",
        model_path,
        tokenizer_path,
    )
    return FastConformerAssets(
        model_path=model_path,
        tokenizer_model_path=tokenizer_path,
        vocab_path=vocab_path,
    )


# --------------------------------------------------------------------------- #
# 2. Provisioning cache
# --------------------------------------------------------------------------- #
def _resolve_from_cache(cache_dir: Path) -> FastConformerAssets | None:
    """Return cached assets when the canonical files are present and non-empty."""
    model, tokenizer, vocab = _expected_cache_files(cache_dir)
    if (
        model.is_file()
        and model.stat().st_size > 0
        and tokenizer.is_file()
        and tokenizer.stat().st_size > 0
    ):
        logger.info("Using cached FastConformer assets from %s", cache_dir)
        return FastConformerAssets(
            model_path=model,
            tokenizer_model_path=tokenizer,
            vocab_path=vocab,
        )
    return None


# --------------------------------------------------------------------------- #
# 3. Hugging Face download (opt-in, only for real hosted assets)
# --------------------------------------------------------------------------- #
def _provision_from_hf(
    settings: MunajjamSettings,
    cache_dir: Path,
) -> FastConformerAssets | None:
    """Download canonical filenames from a configured Hugging Face repo.

    Returns ``None`` when no repo is configured (the caller then raises the
    actionable fallback error). Raises :class:`~munajjam.exceptions
    .ConfigurationError` when a configured download fails.

    No Hugging Face repo is assumed to exist by default — this only runs when
    ``fastconformer_hf_repo_id`` is explicitly set to a repo that really
    hosts the pre-exported files (see docs/fastconformer-onnx-validation.md).
    """
    repo_id = settings.fastconformer_hf_repo_id
    if not repo_id:
        return None

    revision = settings.fastconformer_hf_revision
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ConfigurationError(
            "Automatic FastConformer download requires the huggingface_hub "
            "package. Install with: pip install huggingface_hub",
            setting_name="fastconformer_hf_repo_id",
        ) from e

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading FastConformer assets from %s (revision=%s) into %s",
            repo_id,
            revision or "main",
            cache_dir,
        )
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=FASTCONFORMER_MODEL_FILENAME,
            revision=revision,
            local_dir=cache_dir,
        )
        tokenizer_file = hf_hub_download(
            repo_id=repo_id,
            filename=FASTCONFORMER_TOKENIZER_FILENAME,
            revision=revision,
            local_dir=cache_dir,
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to download FastConformer assets from Hugging Face repo {repo_id!r}: {e}",
            setting_name="fastconformer_hf_repo_id",
        ) from e

    return FastConformerAssets(
        model_path=Path(model_file),
        tokenizer_model_path=Path(tokenizer_file),
        vocab_path=_expected_cache_files(cache_dir)[2],
    )


# --------------------------------------------------------------------------- #
# 4. Actionable fallback error
# --------------------------------------------------------------------------- #
def _provisioning_error(cache_dir: Path) -> ConfigurationError:
    """Controlled error explaining exactly how to make the assets available.

    The cache path is kept out of the client-facing message (it may contain
    the user's home directory) but is logged server-side.
    """
    logger.error(
        "FastConformer CTC assets are not provisioned. Checked cache dir: %s",
        cache_dir,
    )
    return ConfigurationError(
        "FastConformer CTC model assets are not available and no automatic "
        "download source is configured. Either set "
        "MUNAJJAM_FASTCONFORMER_MODEL_PATH and "
        "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH, or export the model. " + EXPORT_SCRIPT_HINT,
    )
