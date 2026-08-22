"""
Global Quran CTC segmentation (issue #104).

Aligns a Quran recitation to the canonical reference text using CTC forced
alignment on top of the validated FastConformer acoustic layer
(:class:`~munajjam.transcription.fastconformer.FastConformerInference`).

Alignment reference path
------------------------
The reference model ``nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`` uses
an **Arabic-orthography SentencePiece vocabulary** (1024 tokens, no
diacritics). The CTC classes are therefore Arabic subword tokens, and the
alignment reference must be the *tokenized canonical text*, not phonemes:

    Quran reference -> canonical Uthmani text -> normalize (strip diacritics
    and Quranic marks, normalize wasla) -> SentencePiece tokenization ->
    CTC segmentation against ``log_probs [T, 1025]`` -> word timestamps.

The ``Hetchy/Quranic-Phonemizer`` (IPA) output **cannot** be used as CTC
alignment targets: its symbols are Latin/IPA phonemes that are 100%
out-of-vocabulary for the Arabic-orthography model. It is kept here only as a
*validation-only* adapter (:class:`QuranicPhonemizerAdapter`) for cross
checking canonical/reference text consistency; it never feeds the alignment.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from munajjam.data.quran import load_surah_ayahs
from munajjam.exceptions import TranscriptionError
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.fastconformer import (
    DEFAULT_SAMPLE_RATE,
    FRAME_DURATION_SECONDS,
    FastConformerInference,
)
from munajjam.transcription.silence import load_audio_waveform

logger = logging.getLogger(__name__)

# Chunks shorter than this (0.1 s at 16 kHz) are treated as silence and
# skipped; they carry no alignable speech.
_MIN_CHUNK_SAMPLES = DEFAULT_SAMPLE_RATE // 10

# --------------------------------------------------------------------------- #
# Quranic text normalization (model-compatible)
# --------------------------------------------------------------------------- #

# Everything below is stripped before tokenization because the model's
# SentencePiece vocabulary contains none of it (verified empirically against
# the real tokenizer: 0/172759 <unk> across all 6236 bundled ayahs):
#   U+0610..U+061A  Arabic sign ranges (e.g. honourifics, takhallus)
#   U+064B..U+065F  Arabic diacritics (tashkeel: fatha, kasra, shadda, ...)
#   U+0670           superscript alef (seen in "ٱلرَّحْمَٰنِ")
#   U+06D6..U+06ED  Quranic annotation marks (stop signs, rub el hizb, ...)
#   U+0640           tatweel (kashida)
_DIACRITICS_AND_MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")

# Alef wasla U+0671 is not in the vocabulary; normalize it to plain alef.
_ALEF_WASLA = "\u0671"
_ALEF = "\u0627"


def normalize_quran_text(text: str) -> str:
    """
    Normalize canonical Quran text for CTC alignment with FastConformer.

    The model's SentencePiece vocabulary contains only undiacritized Arabic
    letters (U+0621..U+064A) plus subword pieces, so the normalization:

    * strips Arabic diacritics and Quranic annotation marks,
    * normalizes alef wasla (``ٱ``) to plain alef (``ا``),
    * collapses whitespace.

    It deliberately **does not** apply the comparison-oriented rewrites from
    ``munajjam.core.arabic.normalize_arabic`` (``ة``->``ه``, ``ى``->``ي``,
    ``ؤ``->``و``, ``ئ``->``ي``): those letters exist in the model vocabulary
    as-is and rewriting them would change the token sequence the acoustic
    model was trained on.

    Args:
        text: Canonical Uthmani Quran text (may include diacritics and
            Quranic annotation marks).

    Returns:
        Normalized text safe to feed the SentencePiece tokenizer.
    """
    if not text:
        return ""
    stripped = _DIACRITICS_AND_MARKS.sub("", text)
    stripped = stripped.replace(_ALEF_WASLA, _ALEF)
    return " ".join(stripped.split())


# --------------------------------------------------------------------------- #
# Reference tokenization
# --------------------------------------------------------------------------- #


class CTCReferenceTokenizer(Protocol):
    """Tokenizes normalized reference text into CTC class indices.

    Implementations must produce token ids in the range ``[0, vocab_size)``
    that map 1:1 to the acoustic model's CTC classes (for the reference model
    the SentencePiece piece id *is* the CTC class index, and the blank is the
    trailing class ``vocab_size``).
    """

    @property
    def vocab_size(self) -> int:
        """Number of non-blank CTC classes."""
        ...

    @property
    def unk_id(self) -> int:
        """Token id used for out-of-vocabulary input."""
        ...

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` into a list of token ids."""
        ...


class SentencePieceTokenizer:
    """CTC reference tokenizer backed by the model's SentencePiece model.

    The reference checkpoint ships a SentencePiece unigram model
    (``tokenizer.model`` inside the ``.nemo``). Its piece ids map directly to
    the FastConformer CTC class indices (verified against the exported graph:
    ``vocab_size == 1024``, blank == ``1024``, ``vocab.txt[i] == piece i+1``,
    i.e. the labels file merely omits ``<unk>`` at piece 0).
    """

    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece  # lazy: only needed for actual alignment runs

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self._model_path = str(model_path)

    @property
    def vocab_size(self) -> int:
        return int(self._sp.get_piece_size())

    @property
    def unk_id(self) -> int:
        return int(self._sp.unk_id())

    def encode(self, text: str) -> list[int]:
        ids = self._sp.encode(text, out_type=int)
        return [int(i) for i in ids]


# --------------------------------------------------------------------------- #
# CTC forced alignment helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlignedWord:
    """A reference word aligned to a time span in the audio."""

    text: str
    start: float
    end: float
    probability: float


def frames_to_time(
    frames: int | np.ndarray,
    frame_duration: float = FRAME_DURATION_SECONDS,
) -> float | np.ndarray:
    """
    Convert CTC frame indices to seconds.

    FastConformer subsamples the log-mel frames 8x, so each CTC frame covers
    ``window_stride * subsampling = 0.01 * 8 = 0.08`` seconds at 16 kHz
    (verified empirically on the exported graph: ``T' = samples // 1280 + 1``).

    Note: the leading-frame offset question (whether NeMo's preprocessor pads
    the start) remains an open validation item; no undocumented correction is
    applied here.
    """
    if isinstance(frames, (int, np.integer)):
        return float(frames) * frame_duration
    return np.asarray(frames, dtype=np.float64) * frame_duration


def _ctc_parameters(
    *,
    blank_index: int,
    frame_duration: float,
    n_classes: int,
) -> Any:
    """Build ``ctc_segmentation`` parameters for this model.

    ``char_list`` must be indexable by every CTC class (including blank), as
    the library uses it during backtracking to render states; plain strings
    of the class indices are enough.
    """
    try:
        from ctc_segmentation import CtcSegmentationParameters
    except ImportError as e:  # pragma: no cover - depends on environment
        raise TranscriptionError(
            "ctc-segmentation is required for CTC alignment. "
            "Install with: pip install ctc-segmentation"
        ) from e

    return CtcSegmentationParameters(
        blank=blank_index,
        index_duration=frame_duration,
        char_list=[str(i) for i in range(n_classes)],
    )


def align_words_to_log_probs(
    log_probs: np.ndarray,
    words: Sequence[str],
    word_token_ids: Sequence[Sequence[int]],
    *,
    blank_index: int,
    frame_duration: float = FRAME_DURATION_SECONDS,
    min_confidence: float | None = None,
) -> list[AlignedWord]:
    """
    Forced-align reference words to CTC log-probabilities.

    Runs ``ctc-segmentation``'s monotonic CTC alignment (blank-aware,
    repeated-token aware) over ``log_probs [T, C]`` with each word as an
    utterance, returning one :class:`AlignedWord` per input word.

    Args:
        log_probs: CTC log-probabilities ``[T, C]`` (log-softmax applied).
        words: Reference words in recitation order.
        word_token_ids: Token id sequence for each word (as produced by the
            model's own tokenizer).
        blank_index: CTC blank class index (the trailing class, 1024 for the
            reference model).
        frame_duration: Seconds per CTC frame (0.08 for FastConformer).
        min_confidence: When set, raise if any aligned word's confidence
            falls below it (each confidence is the exponential of the mean
            log-probability over the word's frames, in ``[0, 1]``).

    Returns:
        Aligned words with monotonic timestamps in seconds.

    Raises:
        TranscriptionError: If inputs are invalid, the reference cannot be
            aligned (e.g. audio shorter than the token sequence), timestamps
            are degenerate, or confidence drops below ``min_confidence``.
    """
    lpz = np.asarray(log_probs, dtype=np.float32)
    if lpz.ndim != 2 or lpz.shape[0] == 0 or lpz.shape[1] == 0:
        raise TranscriptionError(
            "log_probs must be a non-empty [T, C] matrix",
            context={"shape": lpz.shape},
        )
    n_classes = int(lpz.shape[1])
    if not 0 <= blank_index < n_classes:
        raise TranscriptionError(
            "blank_index out of range for log-prob classes",
            context={"blank_index": blank_index, "n_classes": n_classes},
        )
    if frame_duration <= 0:
        raise TranscriptionError("frame_duration must be positive")

    if len(words) != len(word_token_ids) or not words:
        raise TranscriptionError(
            "words and word_token_ids must be non-empty and parallel",
            context={"n_words": len(words), "n_token_lists": len(word_token_ids)},
        )

    token_lists: list[np.ndarray] = []
    for word, tokens in zip(words, word_token_ids, strict=True):
        ids = [int(t) for t in tokens]
        if not ids:
            raise TranscriptionError(
                "reference word tokenized to an empty sequence",
                context={"word": word},
            )
        if any(i < 0 or i >= n_classes for i in ids):
            raise TranscriptionError(
                "reference token id outside CTC class range",
                context={"word": word, "n_classes": n_classes, "token_ids": ids},
            )
        if blank_index in ids:
            # The blank class must never be a reference token: ctc-segmentation
            # would treat it as a real emission and corrupt the alignment.
            raise TranscriptionError(
                "reference token id collides with the CTC blank class",
                context={"word": word, "blank_index": blank_index, "token_ids": ids},
            )
        token_lists.append(np.asarray(ids, dtype=np.int64))

    config = _ctc_parameters(
        blank_index=blank_index,
        frame_duration=frame_duration,
        n_classes=n_classes,
    )

    try:
        from ctc_segmentation import (
            ctc_segmentation,
            determine_utterance_segments,
            prepare_token_list,
        )
    except ImportError as e:  # pragma: no cover - depends on environment
        raise TranscriptionError(
            "ctc-segmentation is required for CTC alignment. "
            "Install with: pip install ctc-segmentation"
        ) from e

    try:
        ground_truth, utt_begin_indices = prepare_token_list(config, token_lists)
        timings, char_probs, _ = ctc_segmentation(config, lpz, ground_truth)
        segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, list(words)
        )
    except AssertionError as e:
        # ctc_segmentation raises when the token sequence is longer than the
        # number of frames ("Audio is shorter than text!").
        raise TranscriptionError(
            "Reference token sequence is longer than the audio can represent; "
            "alignment impossible in a single pass",
            context={
                "n_tokens": sum(len(t) for t in token_lists),
                "n_frames": int(lpz.shape[0]),
                "hint": "Split the recitation into per-ayah chunks (VAD) or "
                "provide longer audio before aligning.",
            },
        ) from e

    aligned: list[AlignedWord] = []
    previous_end = 0.0
    for word, (start, end, score) in zip(words, segments, strict=True):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise TranscriptionError(
                "CTC alignment produced a degenerate word span",
                context={"word": word, "start": float(start), "end": float(end)},
            )
        if start < previous_end - 1e-6:
            raise TranscriptionError(
                "CTC alignment produced non-monotonic word timestamps",
                context={"word": word, "start": float(start), "previous_end": previous_end},
            )
        previous_end = end
        probability = float(np.clip(np.exp(score), 0.0, 1.0))
        aligned.append(
            AlignedWord(text=word, start=float(start), end=float(end), probability=probability)
        )

    if min_confidence is not None:
        low = [(a.text, a.probability) for a in aligned if a.probability < min_confidence]
        if low:
            raise TranscriptionError(
                "CTC alignment confidence below threshold",
                context={"min_confidence": min_confidence, "low_words": low},
            )

    return aligned


# --------------------------------------------------------------------------- #
# Chunking (VAD preprocessing for long audio)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AudioChunk:
    """A slice of the waveform with its offset in the original audio."""

    waveform: np.ndarray
    start_seconds: float


class AudioChunker(Protocol):
    """Splits a waveform into chunks for incremental alignment."""

    def chunk(self, waveform: np.ndarray, sample_rate: int) -> Iterable[AudioChunk]:
        """Yield waveform chunks in recitation order with global offsets."""
        ...


class SinglePassChunker:
    """Align the whole waveform in one pass (no VAD)."""

    def chunk(self, waveform: np.ndarray, sample_rate: int) -> Iterable[AudioChunk]:
        yield AudioChunk(waveform=waveform, start_seconds=0.0)


class SileroVADChunker:
    """VAD-based chunker backed by the optional ``silero-vad`` package.

    Splits the waveform at detected speech/silence boundaries so long
    recitations can be aligned chunk-by-chunk (issue #104 requires VAD
    preprocessing for long audio). The dependency (and its ONNX model, which
    ``torch.hub`` may download on first use) is loaded lazily on first
    ``chunk()`` call, so importing/instantiating this class is cheap and unit
    tests never need it.

    Each yielded :class:`AudioChunk` keeps ``start_seconds`` as the chunk's
    offset in the original audio; the transcriber adds that offset to the
    chunk-local CTC timestamps to produce global wall-clock times.
    """

    def __init__(
        self,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as e:
                raise TranscriptionError(
                    "silero-vad is required for VAD chunking. Install with: pip install silero-vad"
                ) from e
            self._model = load_silero_vad()
        return self._model

    def chunk(self, waveform: np.ndarray, sample_rate: int) -> Iterable[AudioChunk]:
        try:
            import torch
            from silero_vad import get_speech_timestamps
        except ImportError as e:
            raise TranscriptionError(
                "silero-vad (and torch) are required for VAD chunking. "
                "Install with: pip install silero-vad"
            ) from e

        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        stamps = get_speech_timestamps(
            tensor,
            self._get_model(),
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        for stamp in stamps:
            start = int(stamp["start"])
            end = int(stamp["end"])
            yield AudioChunk(
                waveform=waveform[start:end],
                start_seconds=start / sample_rate,
            )


def chunk_local_to_global(
    chunk_start_seconds: float,
    frame_index: int,
    frame_duration: float = FRAME_DURATION_SECONDS,
) -> float:
    """Map a chunk-local CTC frame index to global audio seconds."""
    return chunk_start_seconds + float(frame_index) * frame_duration


def _fit_reference_prefix(
    words: Sequence[str],
    word_token_ids: Sequence[Sequence[int]],
    n_frames: int,
) -> int:
    """
    Largest prefix of the remaining reference that fits in ``n_frames``.

    ``ctc_segmentation`` requires at least one frame per token plus one blank
    separator per word (its ``ground_truth`` length is
    ``2 + n_words + sum(len(tokens))``); this mirrors that constraint so a
    chunk is never handed a reference longer than its frames.

    Returns:
        Number of leading words that fit; ``0`` if even the first word does
        not fit.
    """
    length = 1  # leading start marker in the library's ground truth
    fitted = 0
    for _, tokens in zip(words, word_token_ids, strict=True):
        cost = 1 + len(tokens)  # leading blank + this word's tokens
        if length + cost + 1 > n_frames:  # +1 trailing blank
            break
        length += cost
        fitted += 1
    return fitted


# --------------------------------------------------------------------------- #
# Validation-only Quranic-Phonemizer adapter
# --------------------------------------------------------------------------- #


class QuranicPhonemizerAdapter:
    """
    Validation-only adapter around ``Hetchy/Quranic-Phonemizer``.

    The phonemizer resolves **Quran corpus references** (e.g. ``"1:1"`` or
    ``"1:1-1:7"``) to its internal canonical text and produces IPA phonemes.
    Those phonemes are Latin/IPA symbols that are out-of-vocabulary for the
    FastConformer Arabic-orthography CTC model, so they are **never** used as
    CTC alignment targets. This adapter exists solely to validate references
    and cross-check canonical text consistency. Import of the library is lazy.
    """

    def __init__(self, phonemizer: Any | None = None) -> None:
        self._phonemizer = phonemizer

    def _get_phonemizer(self) -> Any:
        if self._phonemizer is None:
            try:
                from quranic_phonemizer import Phonemizer
            except ImportError as e:
                raise TranscriptionError(
                    "quranic-phonemizer is not installed; it is optional and "
                    "only used for reference validation, never for alignment"
                ) from e
            self._phonemizer = Phonemizer()
        return self._phonemizer

    def validate_reference(self, reference: str) -> bool:
        """
        Return whether ``reference`` (e.g. ``"1:1"``) resolves in the
        phonemizer's Quran corpus.
        """
        try:
            self._get_phonemizer().phonemize(reference, stop_signs=())
            return True
        except Exception:  # noqa: BLE001 - any lookup failure => invalid ref
            return False

    def canonical_text(self, reference: str) -> str | None:
        """Canonical Uthmani text for a reference, or ``None`` if invalid."""
        try:
            return str(self._get_phonemizer().phonemize(reference, stop_signs=()).text())
        except Exception:  # noqa: BLE001 - invalid reference
            return None

    def phonemes(self, reference: str) -> tuple[str, ...] | None:
        """IPA phoneme sequence for a reference, or ``None`` if invalid."""
        try:
            return tuple(self._get_phonemizer().phonemize(reference, stop_signs=()).phonemes())
        except Exception:  # noqa: BLE001 - invalid reference
            return None


# --------------------------------------------------------------------------- #
# Transcriber
# --------------------------------------------------------------------------- #


class FastConformerCTCTranscriber(BaseTranscriber):
    """
    Quran CTC transcriber: aligns a recitation to canonical ayah text.

    Pipeline: load waveform -> VAD chunking (optional) -> FastConformer CTC
    log-probs per chunk -> normalize + tokenize canonical reference with the
    model's own SentencePiece tokenizer -> CTC forced alignment per chunk
    with a progressing reference cursor -> global timestamps -> per-ayah
    :class:`~munajjam.models.Segment` with word timestamps.

    The acoustic model and tokenizer are injected (or built lazily from
    paths); neither is loaded during construction, so unit tests can mock
    both without downloading the NVIDIA model.

    Args:
        inference: Acoustic layer. When ``None``, a
            :class:`~munajjam.transcription.fastconformer.FastConformerInference`
            is created lazily from ``model_path`` / ``vocab_path`` (the
            ONNX session itself stays lazy until first inference).
        tokenizer: Reference tokenizer. When ``None`` and
            ``tokenizer_model_path`` is given, a
            :class:`SentencePieceTokenizer` is built lazily. When neither is
            provided, ``transcribe()`` raises a clear error.
        chunker: Optional waveform chunker (VAD). Defaults to single-pass.
        phonemizer_adapter: Optional validation-only phonemizer adapter.
        min_confidence: Optional minimum per-word alignment confidence.
    """

    def __init__(
        self,
        inference: FastConformerInference | None = None,
        tokenizer: CTCReferenceTokenizer | None = None,
        chunker: AudioChunker | None = None,
        phonemizer_adapter: QuranicPhonemizerAdapter | None = None,
        model_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        tokenizer_model_path: str | Path | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._inference = inference
        self._tokenizer = tokenizer
        self._chunker = chunker or SinglePassChunker()
        self._phonemizer_adapter = phonemizer_adapter
        self._model_path = Path(model_path) if model_path else None
        self._vocab_path = Path(vocab_path) if vocab_path else None
        self._tokenizer_model_path = Path(tokenizer_model_path) if tokenizer_model_path else None
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    # Lazy component access
    # ------------------------------------------------------------------ #
    @property
    def inference(self) -> FastConformerInference:
        """The acoustic layer, created lazily on first use."""
        if self._inference is None:
            self._inference = FastConformerInference(
                model_path=self._model_path,
                vocab_path=self._vocab_path,
            )
        return self._inference

    @property
    def tokenizer(self) -> CTCReferenceTokenizer:
        """The reference tokenizer, created lazily on first use."""
        if self._tokenizer is None:
            if self._tokenizer_model_path is None:
                raise TranscriptionError(
                    "No reference tokenizer configured. Provide a tokenizer "
                    "(the model's SentencePiece 'tokenizer.model' from the "
                    ".nemo checkpoint) via tokenizer_model_path or the "
                    "tokenizer argument.",
                )
            self._tokenizer = SentencePieceTokenizer(self._tokenizer_model_path)
        return self._tokenizer

    def unload(self) -> None:
        """Release the acoustic layer (if loaded) to free memory."""
        if self._inference is not None:
            self._inference.unload()

    # ------------------------------------------------------------------ #
    # Reference preparation
    # ------------------------------------------------------------------ #
    def _build_reference(self, surah_id: int) -> tuple[list[str], list[list[int]]]:
        """Tokenize the surah's canonical words into CTC token ids.

        Raises:
            TranscriptionError: If any word is empty, tokenizes to nothing,
                or contains out-of-vocabulary (``<unk>``) tokens — the
                normalized reference must be fully representable.
        """
        ayahs = load_surah_ayahs(surah_id)
        tokenizer = self.tokenizer
        words: list[str] = []
        word_token_ids: list[list[int]] = []
        for ayah in ayahs:
            for raw_word in ayah.text.split():
                normalized = normalize_quran_text(raw_word)
                if not normalized:
                    raise TranscriptionError(
                        "reference word normalized to empty text",
                        context={"surah_id": surah_id, "word": raw_word},
                    )
                ids = tokenizer.encode(normalized)
                if not ids:
                    raise TranscriptionError(
                        "reference word tokenized to an empty sequence",
                        context={"surah_id": surah_id, "word": raw_word},
                    )
                if tokenizer.unk_id in ids:
                    raise TranscriptionError(
                        "reference word contains out-of-vocabulary tokens; "
                        "normalized reference cannot be represented by the "
                        "model vocabulary",
                        context={
                            "surah_id": surah_id,
                            "word": raw_word,
                            "normalized": normalized,
                            "token_ids": ids,
                        },
                    )
                words.append(raw_word)
                word_token_ids.append(ids)
        return words, word_token_ids

    # ------------------------------------------------------------------ #
    # BaseTranscriber contract
    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        """
        Align a recitation to the surah's canonical ayahs.

        Returns one :class:`~munajjam.models.Segment` per ayah with word
        timestamps, mirroring :class:`~munajjam.transcription.whisperx.Whisperx`.

        Raises:
            TranscriptionError: If the reference cannot be tokenized, the
                audio cannot represent the reference (too short), or the
                alignment is degenerate.
        """
        del batch_size  # alignment is batch-independent
        words, word_token_ids = self._build_reference(surah_id)
        if not words:
            return []

        inference = self.inference
        waveform, sample_rate = load_audio_waveform(audio_path, sample_rate=inference.sample_rate)
        chunks = list(self._chunker.chunk(waveform, sample_rate))
        if not chunks:
            raise TranscriptionError(
                "No speech detected in the audio; nothing to align",
                context={"hint": "Check the VAD chunker and the audio content."},
            )

        aligned_words = self._align_chunks(words, word_token_ids, chunks, inference)
        return self._build_segments(surah_id, aligned_words)

    def _align_chunks(
        self,
        words: list[str],
        word_token_ids: list[list[int]],
        chunks: Sequence[AudioChunk],
        inference: FastConformerInference,
    ) -> list[AlignedWord]:
        """
        Align the reference across chunks with a progressing word cursor.

        Every chunk is run through the acoustic layer independently, then the
        next unaligned prefix of the reference that fits the chunk's frames is
        aligned against it. Chunk-local CTC timestamps are shifted by the
        chunk's global audio offset, so the returned timestamps are in global
        wall-clock seconds. Words are never aligned twice (the cursor only
        advances) and no word is dropped silently.
        """
        aligned_all: list[AlignedWord] = []
        cursor = 0
        last_end = 0.0
        for chunk in chunks:
            # Treat near-silent fragments (e.g. VAD edge padding) as empty.
            if chunk.waveform.size < _MIN_CHUNK_SAMPLES:
                continue

            log_probs = inference.log_probs(chunk.waveform)
            n_frames = int(log_probs.shape[0])
            if n_frames == 0:
                continue

            remaining_words = words[cursor:]
            remaining_tokens = word_token_ids[cursor:]
            n = _fit_reference_prefix(remaining_words, remaining_tokens, n_frames)
            if n == 0:
                raise TranscriptionError(
                    "audio chunk too short to represent even one reference word",
                    context={
                        "chunk_start": chunk.start_seconds,
                        "n_frames": n_frames,
                    },
                )

            chunk_aligned = align_words_to_log_probs(
                log_probs,
                remaining_words[:n],
                remaining_tokens[:n],
                blank_index=inference.blank_index,
                frame_duration=inference.frame_duration_seconds,
                min_confidence=self._min_confidence,
            )

            for word in chunk_aligned:
                start = word.start + chunk.start_seconds
                end = word.end + chunk.start_seconds
                if start < last_end - 1e-6:
                    raise TranscriptionError(
                        "CTC alignment produced non-monotonic timestamps across chunks",
                        context={
                            "word": word.text,
                            "start": start,
                            "previous_end": last_end,
                        },
                    )
                aligned_all.append(AlignedWord(word.text, start, end, word.probability))
                last_end = max(last_end, end)

            cursor += n
            if cursor >= len(words):
                break

        if cursor < len(words):
            raise TranscriptionError(
                "not all reference words could be aligned to the audio "
                "(reference too long or audio too short)",
                context={"aligned_words": cursor, "total_words": len(words)},
            )
        return aligned_all

    def _build_segments(self, surah_id: int, aligned_words: list[AlignedWord]) -> list[Segment]:
        """Group aligned words into per-ayah Segments (canonical text)."""
        ayahs = load_surah_ayahs(surah_id)
        segments: list[Segment] = []
        word_index = 0
        for ayah in ayahs:
            n_words = len(ayah.text.split())
            ayah_words = aligned_words[word_index : word_index + n_words]
            word_index += n_words
            if not ayah_words:
                continue
            word_timestamps = [
                WordTimestamp(
                    word=w.text,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in ayah_words
            ]
            confidence = float(sum(w.probability for w in ayah_words) / len(ayah_words))
            segments.append(
                Segment(
                    id=ayah.ayah_number,
                    surah_id=surah_id,
                    start=ayah_words[0].start,
                    end=ayah_words[-1].end,
                    text=ayah.text,
                    type=SegmentType.AYAH,
                    words=word_timestamps,
                    confidence=confidence,
                )
            )
        return segments
