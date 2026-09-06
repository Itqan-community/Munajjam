"""
FastConformer inference layer (experimental).

A thin, lazy ONNX Runtime wrapper around NVIDIA FastConformer hybrid
(RNNT + CTC) ASR models, e.g.
``nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`` (115M params, Arabic).

This module only covers *acoustic inference*: it loads an exported ONNX
graph and turns a raw waveform into CTC log-probabilities of shape ``[T, V]``.
The public API (``log_probs(waveform)``) accepts raw waveforms; mel
preprocessing is performed internally via ``compute_mel_features()`` before
calling ONNX inference.
Higher-level concerns of the Global CTC Segmentation pipeline (silero-vad
chunking, quranic-phonemizer G2P, ``ctc-segmentation``, blank reward, dynamic
silence trimming) are intentionally out of scope here and will live in
``transcription/ctc_segmentation.py``.

ONNX model contract
-------------------
All values below were verified empirically against a real export of
``nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`` (see
``scripts/export_fastconformer_onnx.py``, whose built-in ``--validate`` runs
the contract check).

The checkpoint is a NeMo ``EncDecHybridRNNTCTCBPEModel`` (hybrid RNNT/CTC).
The ONNX graph must expose the **CTC head**, i.e. it must be exported after
``model.change_decoding_strategy(decoder_type="ctc")``, otherwise the RNNT
decoder would be exported instead. Two export shapes exist:

1. NeMo's stock ``model.export()`` (``cur_decoder="ctc"``) — the audio
   preprocessor is **not** part of the graph:

       input  audio_signal  float32 [B, 80, T_mel]   (log-mel features)
       input  length        int64   [B]              (mel frame counts)
       output logprobs      float32 [B, T', V+1]

2. Munajjam's production export (``scripts/export_fastconformer_onnx.py``,
   ``*_ctc_rawaudio.onnx``) — encoder + CTC decoder only.  The NeMo
   preprocessor is excluded because ``torch.stft()`` produces complex types
   that the legacy TorchScript ONNX exporter cannot handle.  Mel features
   are computed in Python by ``compute_mel_features()`` before ONNX inference:

       input  input_signal        float32 [B, 80, T_mel] (log-mel features)
       input  input_signal_length int32   [B]            (valid frame count)
       output logprobs            float32 [B, T', V+1]
       output encoded_lengths     int64   [B]            (true frame count)

This loader targets contract (2).  The public API still accepts raw
waveforms — ``log_probs(waveform)`` performs mel preprocessing internally
via ``compute_mel_features()``.

Tensor names are resolved from the session at load time (by role/shape), so
exports that rename the I/O keep working; the names above are the defaults
used by the production export.

Vocabulary & blank index
------------------------
* Vocabulary source: the SentencePiece unigram tokenizer baked into the
  ``.nemo`` checkpoint (``tokenizer.model`` / ``model_config.yaml``'s
  ``labels``). Vocabulary size is 1024 for this checkpoint (verified: the
  ONNX output declares 1025 classes).
* The hybrid model's CTC head is a ``ConvASRDecoder`` with ``add_blank=True``,
  so the blank is the **last** class: ``blank_index == vocab_size`` (1024),
  and the log-prob tensor has ``V+1 = 1025`` columns. NeMo's ``CTCLoss`` is
  constructed with ``blank=num_classes``, confirming the last-column blank.
* Output is already log-softmax normalized by the graph (verified: rows of
  ``exp(log_probs)`` sum to 1).

Frame-to-time mapping
---------------------
FastConformer subsamples the log-mel frames 8x:
``time_s = frame_index * window_stride * subsampling_factor``
= ``frame_index * 0.01 s * 8`` = ``frame_index * 0.08 s`` (80 ms per frame).
Verified empirically on the exported graph: ``T' = time // 1280 + 1`` for
``time`` audio samples at 16 kHz (1280 samples = 80 ms).

Mel preprocessing
-----------------
``compute_mel_features(waveform)`` implements NeMo's ``FilterbankFeatures``
exactly: pre-emphasis (0.97) → center-padded STFT (n_fft=512, win=400,
hop=160, Hann window) → power spectrum → mel filterbank (80 mels,
fmin=0, fmax=8000, Slaney norm) → log (floor 2⁻²⁴) → per-channel
mean/variance normalisation (online, per utterance).  No librosa dependency
at runtime.

Notes
-----
* Batch size 1 only — each call processes a single utterance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from munajjam.exceptions import TranscriptionError
from munajjam.transcription.silence import load_audio_waveform

logger = logging.getLogger(__name__)

# Reference model (maintainer guidance).
DEFAULT_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"

# Expected audio format for the exported ONNX graph.
DEFAULT_SAMPLE_RATE = 16000

# FastConformer subsampling + NeMo mel preprocessor hop (window_stride).
SUBSAMPLING_FACTOR = 8
WINDOW_STRIDE_SECONDS = 0.01
FRAME_DURATION_SECONDS = WINDOW_STRIDE_SECONDS * SUBSAMPLING_FACTOR

# NeMo export defaults for a full-model (CTC head) export.
DEFAULT_INPUT_SIGNAL_NAME = "input_signal"
DEFAULT_INPUT_LENGTH_NAME = "input_signal_length"
DEFAULT_OUTPUT_LOGPROBS_NAME = "logprobs"
DEFAULT_OUTPUT_LENGTH_NAMES = ("encoded_lengths", "encoder_output_length")

# --------------------------------------------------------------------------- #
# Mel preprocessing parameters — must match NeMo FilterbankFeatures exactly.
# These are the effective values for the FastConformer hybrid checkpoint after
# config overrides (window_size=0.025, features=80, pad_to=0).
# --------------------------------------------------------------------------- #
N_FFT = 512
WIN_LENGTH = 400           # 0.025 s × 16 000 Hz
HOP_LENGTH = 160           # 0.010 s × 16 000 Hz
N_MELS = 80
PREEMPH = 0.97
LOG_FLOOR = 2**-24         # ≈ 5.96e-8, added (not clamped)
MEL_FMIN = 0
MEL_FMAX = 8000            # sample_rate / 2


def compute_mel_features(waveform: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Compute log-mel features matching NeMo's ``FilterbankFeatures`` exactly.

    Implements: pre-emphasis → STFT → power spectrum → mel filterbank →
    log → per-channel mean/variance normalisation.  No dither (inference mode)
    and no padding (``pad_to=0`` in the FastConformer config).

    Args:
        waveform: 1-D float32 array at *sample_rate* Hz.
        sample_rate: Sample rate (default 16 000).

    Returns:
        float32 array of shape ``[1, n_mels, T_mel]`` ready for the ONNX
        encoder graph.
    """
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1-D")
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    if waveform.size == 0:
        return np.zeros((1, N_MELS, 0), dtype=np.float32)

    # 1. Pre-emphasis
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - PREEMPH * waveform[:-1]

    # 2. Hann window (non-periodic, matching torch.hann_window(periodic=False))
    #    For periodic=False: N = window_length - 1, so denominator is WIN_LENGTH - 1.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / (WIN_LENGTH - 1))
    # PyTorch pads win_length window on BOTH sides to n_fft before applying:
    #   "If win_length < n_fft, window will be padded on both sides to
    #   length n_fft before being applied."
    window_full = np.zeros(N_FFT, dtype=np.float32)
    _win_offset = (N_FFT - WIN_LENGTH) // 2
    window_full[_win_offset : _win_offset + WIN_LENGTH] = window

    # 3. Pad for center STFT (pad n_fft//2 on each side, matching torch.stft
    #    center=True, pad_mode="constant" as used by NeMo's FilterbankFeatures)
    pad_width = N_FFT // 2
    padded = np.pad(emphasized, pad_width, mode="constant")

    # 4. STFT frame extraction — torch.stft(center=True) extracts n_fft-sized
    #    frames at positions 0, hop, 2*hop, ... from the center-padded signal.
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    indices = np.arange(N_FFT)[np.newaxis, :] + np.arange(n_frames)[:, np.newaxis] * HOP_LENGTH
    frames = padded[indices] * window_full  # [n_frames, N_FFT]

    # 5. Real DFT via rfft
    fft_result = np.fft.rfft(frames, n=N_FFT)  # [n_frames, n_fft//2+1]
    power_spec = np.abs(fft_result) ** 2        # power spectrum

    # 6. Mel filterbank (pre-computed via librosa convention, norm="slaney")
    mel_fb = _mel_filterbank(sample_rate, N_FFT, N_MELS, MEL_FMIN, MEL_FMAX)
    mel_spec = power_spec @ mel_fb.T            # [n_frames, n_mels]

    # 7. Log with floor guard
    log_mel = np.log(mel_spec + LOG_FLOOR)      # [n_frames, n_mels]

    # 8. Per-channel (per-feature) mean/variance normalisation.
    #    NeMo's ``get_seq_len`` computes valid frame count as
    #    ``floor(n_samples / hop_length)`` (for exact_pad=False).
    #    Centre-padded STFT produces one extra frame; NeMo masks it out.
    n_valid = waveform.size // HOP_LENGTH
    if n_valid > 0:
        valid = log_mel[:n_valid]                   # [n_valid, n_mels]
        mean = valid.mean(axis=0, keepdims=True)
        variance = np.sum((valid - mean) ** 2, axis=0, keepdims=True) / max(n_valid - 1, 1)
        std = np.sqrt(variance)
        std = np.where(np.isnan(std), 0.0, std)  # single-frame edge case
        std = std + 1e-5
        log_mel = (log_mel - mean) / std
    # Mask frames beyond n_valid to 0 (matching NeMo's seq_len masking)
    if log_mel.shape[0] > n_valid:
        log_mel[n_valid:] = 0.0

    # 9. Transpose to [n_mels, n_frames] and add batch dim
    return log_mel.T[np.newaxis, ...].astype(np.float32)


def _mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Build a mel filterbank matrix (``[n_mels, n_fft//2+1]``).

    Reproduces ``librosa.filters.mel(..., norm="slaney")`` with pure numpy
    so the runtime has no librosa dependency for preprocessing.

    Uses the Slaney/Auditory Toolbox mel scale (``librosa``'s default when
    ``htk=False``), **not** the HTK formula.  Below 1 kHz the scale is
    linear; above 1 kHz it is logarithmic.
    """
    # --- Slaney mel-scale conversion (librosa htk=False) ---
    _MIN_LOG_HZ = 1000.0
    _F_SP = 200.0 / 3.0                          # ≈ 66.667  (linear spacing)
    _MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP           # = 15.0
    _LOGSTEP = np.log(6.4) / 27.0               # ≈ 0.06875

    def _hz_to_mel(f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        mel = np.empty_like(f)
        low = f < _MIN_LOG_HZ
        mel[low] = f[low] / _F_SP
        mel[~low] = _MIN_LOG_MEL + np.log(f[~low] / _MIN_LOG_HZ) / _LOGSTEP
        return mel

    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        mel = np.asarray(mel, dtype=np.float64)
        hz = np.empty_like(mel)
        below = mel < _MIN_LOG_MEL
        hz[below] = _F_SP * mel[below]
        hz[~below] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel[~below] - _MIN_LOG_MEL))
        return hz

    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # Frequency bins for rfft
    freq_bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # Rising slope
        rising = (freq_bins - low) / (center - low + 1e-10)
        # Falling slope
        falling = (high - freq_bins) / (high - center + 1e-10)
        filterbank[i] = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney-style area normalization
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2] + 1e-10)
    filterbank *= enorm[:, np.newaxis]

    return filterbank


class FastConformerInference:
    """
    Lazy ONNX Runtime wrapper for FastConformer hybrid models.

    The ONNX session is created on first use (``load()`` or the first
    inference call), so importing/instantiating this class is cheap.

    Example:
        model = FastConformerInference(model_path="stt_ar_fastconformer.onnx")
        waveform, sr = load_audio_waveform("surah_1.wav", sample_rate=16000)
        log_probs = model.log_probs(waveform)   # [T, V+1]
        print(model.blank_index)                # 1024

    Args:
        model_path: Path to the exported ``.onnx`` file. If ``None``, the
            default model ID is logged but no download is attempted here;
            ``load()`` will fail with a clear error.
        vocab_path: Optional path to a vocabulary text file (one token per
            line, e.g. NeMo's ``vocabulary.txt`` or ``labels`` dump). When
            omitted, the vocabulary size is derived from the ONNX output
            dimension.
        session_factory: Optional callable ``(path: str) -> session`` used
            instead of ``onnxruntime.InferenceSession`` (primarily for tests).
            The returned object must expose ``get_inputs()``,
            ``get_outputs()`` and ``run(output_names, input_feed)`` like the
            ONNX Runtime session.
        sample_rate: Sample rate the waveform must be provided at.
        input_signal_name / input_length_name: Explicit ONNX input names.
            When ``None``, they are resolved from the session by shape.
        output_logprobs_name: Explicit ONNX output name for the log-probs
            tensor. When ``None``, resolved from the session (3-D output).
        output_length_name: Explicit ONNX output name for the per-utterance
            frame count. When ``None``, the loader looks for a 1-D int
            output named like NeMo's ``encoded_lengths``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        session_factory: Callable[[str], Any] | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        input_signal_name: str | None = None,
        input_length_name: str | None = None,
        output_logprobs_name: str | None = None,
        output_length_name: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.sample_rate = sample_rate
        self._session_factory = session_factory

        # Explicit names win; otherwise resolved lazily from the session.
        self._input_signal_name = input_signal_name
        self._input_length_name = input_length_name
        self._output_logprobs_name = output_logprobs_name
        self._output_length_name = output_length_name

        self._session: Any = None
        self._vocab: list[str] | None = None
        self._derived_vocab_size: int | None = None
        self._input_length_dtype: np.dtype[Any] | None = None
        self._needs_preprocessing: bool = False  # True when ONNX graph expects mel input

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        """Whether the ONNX session has been created."""
        return self._session is not None

    def load(self) -> None:
        """
        Create the ONNX Runtime session (no-op if already loaded).

        Raises:
            TranscriptionError: If the model file is missing, onnxruntime is
                not installed, or the graph I/O does not match the expected
                FastConformer hybrid CTC contract.
        """
        if self._session is not None:
            return

        # With a custom session_factory the path is an identifier only; the
        # file-existence check applies to the real onnxruntime path.
        if self.model_path is None or (
            self._session_factory is None and not self.model_path.exists()
        ):
            raise TranscriptionError(
                "FastConformer ONNX model not found",
                context={
                    "model_path": str(self.model_path) if self.model_path else None,
                    "hint": (
                        f"Export the model with NeMo (cur_decoder='ctc') or download "
                        f"'{DEFAULT_MODEL_ID}' as ONNX."
                    ),
                },
            )

        try:
            import onnxruntime
        except ImportError as e:
            raise TranscriptionError(
                "onnxruntime is required for FastConformer inference. "
                "Install with: pip install onnxruntime"
            ) from e

        if self._session_factory is not None:
            self._session = self._session_factory(str(self.model_path))
        else:
            self._session = onnxruntime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )

        self._load_vocabulary()
        self._resolve_io_names()
        logger.info(
            "Loaded FastConformer ONNX session: %s (vocab=%s, blank=%s)",
            self.model_path,
            self.vocabulary_size,
            self.blank_index,
        )

    def unload(self) -> None:
        """Drop the ONNX session reference to free memory."""
        self._session = None

    def _get_session(self) -> Any:
        """Return the loaded session, loading it lazily on first use."""
        if self._session is None:
            self.load()
        assert self._session is not None
        return self._session

    # ------------------------------------------------------------------ #
    # Vocabulary / blank
    # ------------------------------------------------------------------ #
    def _load_vocabulary(self) -> None:
        """Load the optional vocabulary file (one token per line)."""
        if self.vocab_path is None:
            return
        if not self.vocab_path.exists():
            raise TranscriptionError(
                "Vocabulary file not found",
                context={"vocab_path": str(self.vocab_path)},
            )
        tokens = self.vocab_path.read_text(encoding="utf-8").splitlines()
        # A final newline is a line terminator, not a vocabulary token. Keep
        # empty entries inside the file, but discard only trailing blank lines.
        while tokens and tokens[-1] == "":
            tokens.pop()
        if not tokens:
            raise TranscriptionError(
                "Vocabulary file is empty",
                context={"vocab_path": str(self.vocab_path)},
            )
        self._vocab = tokens

    @property
    def vocabulary(self) -> list[str] | None:
        """The loaded vocabulary tokens, if a vocab file was provided."""
        return self._vocab

    @property
    def vocabulary_size(self) -> int:
        """
        Number of non-blank classes.

        Uses the loaded vocabulary file, otherwise derives it from the ONNX
        output dimension (``V_out - 1``).

        Raises:
            TranscriptionError: If no vocab file is available and the model
                cannot be loaded.
        """
        if self._vocab is not None:
            return len(self._vocab)
        if self._derived_vocab_size is not None:
            return self._derived_vocab_size
        session = self._get_session()
        dim = self._logprobs_dim(session)
        self._derived_vocab_size = dim - 1
        return self._derived_vocab_size

    @property
    def blank_index(self) -> int:
        """
        Index of the CTC blank class.

        For NeMo hybrid models the blank is appended as the *last* class, so
        ``blank_index == vocabulary_size`` (1024 for the reference model).
        """
        return self.vocabulary_size

    # ------------------------------------------------------------------ #
    # Frame-to-time mapping
    # ------------------------------------------------------------------ #
    @property
    def frame_duration_seconds(self) -> float:
        """Seconds covered by one CTC frame (80 ms for FastConformer)."""
        return FRAME_DURATION_SECONDS

    def frames_to_time(self, frames: int | np.ndarray) -> float | np.ndarray:
        """
        Convert CTC frame indices to seconds.

        Uses NeMo's convention: ``time = frame * window_stride * subsampling``,
        i.e. ``frame * 0.08 s`` for FastConformer at 16 kHz.
        """
        if isinstance(frames, (int, np.integer)):
            return float(frames) * self.frame_duration_seconds
        return np.asarray(frames, dtype=np.float64) * self.frame_duration_seconds

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        """
        Run CTC inference on a waveform.

        Args:
            waveform: Mono audio as a 1-D ``float32``/``float64`` array at
                ``self.sample_rate`` (16 kHz).

        Returns:
            CTC log-probabilities of shape ``[T, V+1]`` (batch dim removed),
            where ``V+1`` includes the trailing blank class. The tensor is
            ``log_softmax``-normalized by the model's CTC head.

        Raises:
            TranscriptionError: If the waveform is invalid, the session fails,
                or the output dimension contradicts the expected vocabulary.
        """
        log_probs = self._raw_log_probs(waveform)

        # Blank is the trailing class: V_out == vocab_size + 1.
        if log_probs.shape[1] != self.vocabulary_size + 1:
            raise TranscriptionError(
                "ONNX output vocabulary does not match expected vocab size",
                context={
                    "output_classes": int(log_probs.shape[1]),
                    "expected_classes": self.vocabulary_size + 1,
                    "blank_index": self.blank_index,
                },
            )

        return log_probs

    def _raw_log_probs(self, waveform: np.ndarray) -> np.ndarray:
        """Run the ONNX session and return ``[T', V+1]`` log-probs.

        No vocabulary validation is performed here (used by the public
        ``log_probs()`` and by the output-dimension probe).
        """
        session = self._get_session()

        if not isinstance(waveform, np.ndarray) or waveform.ndim != 1:
            raise TranscriptionError(
                "waveform must be a 1-D numpy array",
                context={"ndim": getattr(waveform, "ndim", None)},
            )
        if waveform.size == 0:
            raise TranscriptionError("waveform must not be empty")

        length_dtype = self._input_length_dtype or np.dtype(np.int32)

        if self._needs_preprocessing:
            # Graph expects mel features [B, n_mels, T_mel].
            mel = compute_mel_features(waveform, self.sample_rate)  # [1, n_mels, T_mel]
            signal = mel
            n_frames = mel.shape[2]
            if n_frames < 1:
                raise TranscriptionError("Mel spectrogram is empty after preprocessing")
        else:
            # Graph expects raw waveform [B, T].
            signal = np.ascontiguousarray(waveform, dtype=np.float32)[np.newaxis, :]
            n_frames = signal.shape[1]

        # Validate that the frame count fits in the ONNX length dtype.
        try:
            iinfo = np.iinfo(length_dtype)
        except ValueError:
            pass  # not an integer dtype (shouldn't happen for length inputs)
        else:
            if n_frames < iinfo.min or n_frames > iinfo.max:
                raise TranscriptionError(
                    "Frame count does not fit in the ONNX length input dtype",
                    context={
                        "frames": n_frames,
                        "length_dtype": str(length_dtype),
                        "range": (int(iinfo.min), int(iinfo.max)),
                    },
                )
        length = np.array([n_frames], dtype=length_dtype)  # [1]

        input_feed = {
            self._input_signal_name: signal,
            self._input_length_name: length,
        }
        output_names = [self._output_logprobs_name]
        if self._output_length_name is not None:
            output_names.append(self._output_length_name)

        try:
            outputs = session.run(output_names, input_feed)
        except Exception as e:  # noqa: BLE001 - surface any runtime failure
            raise TranscriptionError(
                "FastConformer ONNX inference failed",
                context={"error": str(e)},
            ) from e

        log_probs: np.ndarray = np.asarray(outputs[0], dtype=np.float32)
        if log_probs.ndim != 3:
            raise TranscriptionError(
                "Unexpected log-probs rank from ONNX model",
                context={"shape": log_probs.shape},
            )
        if log_probs.shape[0] != 1:
            raise TranscriptionError(
                "FastConformer inference supports batch size 1",
                context={"batch": log_probs.shape[0]},
            )
        log_probs = log_probs[0]  # [T', V+1]

        # Trim padding using the model's own frame count when available.
        if self._output_length_name is not None and len(outputs) > 1:
            n_frames = int(np.asarray(outputs[1]).reshape(-1)[0])
            log_probs = log_probs[:n_frames]

        return log_probs

    def log_probs_from_file(self, audio_path: str | Path) -> np.ndarray:
        """
        Load an audio file at ``self.sample_rate`` and return its CTC
        log-probabilities (``[T, V+1]``).
        """
        waveform, _ = load_audio_waveform(audio_path, sample_rate=self.sample_rate)
        return self.log_probs(waveform)

    # ------------------------------------------------------------------ #
    # I/O resolution helpers
    # ------------------------------------------------------------------ #
    def _resolve_io_names(self) -> None:
        """Resolve any unset ONNX input/output names from the session."""
        session = self._get_session()

        inputs = list(session.get_inputs())
        signal_name, length_name, length_dtype, needs_preproc = (
            self._find_signal_and_length(inputs)
        )
        self._input_length_dtype = length_dtype
        self._needs_preprocessing = needs_preproc
        if self._input_signal_name is None:
            self._input_signal_name = signal_name
        if self._input_length_name is None:
            self._input_length_name = length_name

        outputs = list(session.get_outputs())
        if self._output_logprobs_name is None:
            self._output_logprobs_name = self._find_logprobs_output(outputs)
        if self._output_length_name is None:
            self._output_length_name = self._find_length_output(outputs)

        logger.debug(
            "Resolved ONNX I/O: inputs=%s outputs=%s (logprobs=%s, length=%s)",
            [i.name for i in inputs],
            [o.name for o in outputs],
            self._output_logprobs_name,
            self._output_length_name,
        )

    @staticmethod
    def _find_signal_and_length(
        inputs: list[Any],
    ) -> tuple[str, str, np.dtype[Any], bool]:
        """
        Identify the waveform and length inputs by shape/type.

        Supports two graph layouts:

        * **Raw-audio** (Munajjam export): 2-D float ``[B, T]`` waveform +
          1-D int ``[B]`` length → returns ``needs_preprocessing=False``.
        * **Mel-input** (NeMo stock / encoder+decoder-only export):
          3-D float ``[B, n_mels, T_mel]`` mel features + 1-D int ``[B]``
          length → returns ``needs_preprocessing=True`` (caller must convert
          raw waveform → mel before calling ONNX).
        """
        one_d_int = [i for i in inputs if _ndim(i) == 1 and _is_int(i)]
        two_d_float = [i for i in inputs if _ndim(i) == 2 and _is_float(i)]
        three_d_float = [i for i in inputs if _ndim(i) == 3 and _is_float(i)]

        if not one_d_int:
            raise TranscriptionError(
                "Unexpected ONNX inputs: no 1-D integer length input found.",
                context={"found": [getattr(i, "name", None) for i in inputs]},
            )

        length_dtype = _numpy_length_dtype(one_d_int[0])
        length_name = str(one_d_int[0].name)

        # Prefer 3-D mel input (encoder+decoder-only graph).
        if three_d_float:
            return str(three_d_float[0].name), length_name, length_dtype, True

        # Backward-compatible path for older raw-audio exports (the
        # production export is now mel-input, but legacy 2-D graphs
        # are still accepted without preprocessing).
        if two_d_float:
            return str(two_d_float[0].name), length_name, length_dtype, False

        raise TranscriptionError(
            "Unexpected ONNX inputs: expected a 2-D float waveform "
            "[B, T] or 3-D float mel features [B, n_mels, T_mel].",
            context={"found": [getattr(i, "name", None) for i in inputs]},
        )

    @staticmethod
    def _find_logprobs_output(outputs: list[Any]) -> str:
        """Pick the 3-D output (``[B, T, V+1]`` log-probs)."""
        candidates = [o for o in outputs if _ndim(o) == 3]
        if len(candidates) != 1:
            raise TranscriptionError(
                "Could not identify the CTC log-probs output (expected a single 3-D tensor)",
                context={"found": [getattr(o, "name", None) for o in outputs]},
            )
        return str(candidates[0].name)

    def _find_length_output(self, outputs: list[Any]) -> str | None:
        """Find the optional 1-D int length output (NeMo naming)."""
        for name in DEFAULT_OUTPUT_LENGTH_NAMES:
            if any(getattr(o, "name", None) == name for o in outputs):
                return name
        one_d_int = [o for o in outputs if _ndim(o) == 1 and _is_int(o)]
        if len(one_d_int) == 1:
            return str(one_d_int[0].name)
        return None

    def _logprobs_dim(self, session: Any) -> int:
        """Read the declared class dimension of the log-probs output."""
        outputs = list(session.get_outputs())
        name = self._output_logprobs_name or self._find_logprobs_output(outputs)
        for out in outputs:
            if getattr(out, "name", None) == name:
                shape = list(out.shape)
                if len(shape) == 3 and isinstance(shape[2], int):
                    return shape[2]
                break
        # Fall back to a dry-run inference on a tiny input that fits any
        # supported length dtype (100 samples < int8_min).
        self._output_logprobs_name = name
        probe = np.zeros(min(100, self.sample_rate), dtype=np.float32)
        try:
            return int(self._raw_log_probs(probe).shape[1])
        except Exception as e:  # noqa: BLE001 - wrap probe failures
            raise TranscriptionError(
                "Could not determine log-probs output dimension",
                context={"error": str(e)},
            ) from e


def _ndim(io: Any) -> int | None:
    """Number of dimensions declared by an ONNX I/O descriptor."""
    shape = getattr(io, "shape", None)
    if shape is None:
        return None
    return len(list(shape))


def _is_float(io: Any) -> bool:
    return "float" in str(getattr(io, "type", "")).lower()


_ONNX_INTEGER_TYPES = {
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
}


def _is_int(io: Any) -> bool:
    return str(getattr(io, "type", "")).lower() in _ONNX_INTEGER_TYPES


def _numpy_length_dtype(io: Any) -> np.dtype[Any]:
    """Map a supported ONNX integer tensor descriptor to an exact NumPy dtype."""
    type_name = str(getattr(io, "type", "")).lower()
    dtypes: dict[str, np.dtype[Any]] = {
        "tensor(int8)": np.dtype(np.int8),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(uint8)": np.dtype(np.uint8),
    }
    try:
        return dtypes[type_name]
    except KeyError as e:
        raise TranscriptionError(
            "Unsupported ONNX length input dtype",
            context={
                "input": getattr(io, "name", None),
                "dtype": getattr(io, "type", None),
                "supported": sorted(dtypes),
            },
        ) from e
