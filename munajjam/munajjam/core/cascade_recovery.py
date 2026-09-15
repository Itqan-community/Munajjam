"""
Cascade recovery for word-level alignment drift.

This module detects sequences of consecutive words that WhisperX failed to
align (produced with placeholder timestamps and ``confidence == 0.0``) and
recovers their true timestamps through a bounded acoustic re-alignment pass
followed by a deterministic interpolation fallback.

Why unaligned words occur
-------------------------
WhisperX's wav2vec2 alignment stage sometimes fails to emit a time for a word
because:

* the word is short and acoustically ambiguous,
* a reciter breath or pause interrupts the token,
* the reference text carries Quranic diacritics that the acoustic tokenizer
  cannot represent.

When that happens, the initial pipeline assigns a placeholder timestamp
(e.g. ``start == end`` or a hard-coded ``0.1s`` duration) with a ``0.0``
confidence.

Coordinate systems
------------------
There are three distinct time coordinate systems that must never be mixed:

* **Global audio time** — timestamps relative to the full recording
  (``word.start`` / ``word.end`` in the final output).
* **Local sliced-audio time** — timestamps relative to the start of an
  extracted audio slice (what ``whisperx.align`` returns).
* **WhisperX/model output time** — timestamps returned by the alignment model
  for the audio buffer it was handed.

The only legal conversion from local to global time is::

    global_start = slice_start + local_start
    global_end   = slice_start + local_end

This is implemented exactly once in :func:`local_to_global_timestamp`.

Recovery flow
-------------
1. Detect unaligned words (:func:`detect_unaligned_word_gaps`).
2. For each gap, derive the recovery window from the surrounding valid
   anchors (never from placeholder timestamps).
3. Slice the exact audio interval covering the gap.
4. Attempt acoustic re-alignment (:func:`realign_unaligned_gap_acoustic`).
5. Map acoustic output back to canonical words with DP, preserving
   canonical text and monotonic ordering.
6. Validate recovered timestamps against the gap and its anchors.
7. Fall back to character-weighted interpolation only when acoustic recovery
   genuinely fails, recording an explicit failure reason.

Design invariants
-----------------
* Canonical Quranic word text is preserved verbatim — the acoustic stage only
  supplies timing evidence.
* Every recovered word satisfies ``gap_start <= start <= end <= gap_end``.
* Anchors (the valid words surrounding a gap) are never modified.
* The final word count equals the original word count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from ..models import AlignmentResult, Ayah, Segment
from .dp_core import compute_alignment_cost
from .matcher import similarity

logger = logging.getLogger("munajjam")

# Single authoritative floating-point tolerance for timestamp comparisons.
TIMESTAMP_TOLERANCE = 1e-3

# Default context padding applied around a gap before acoustic alignment.
DEFAULT_CONTEXT_PAD_SEC = 0.25


def timestamps_equal(a: float, b: float) -> bool:
    """Return True if two timestamps are equal within :data:`TIMESTAMP_TOLERANCE`."""
    return abs(a - b) <= TIMESTAMP_TOLERANCE


def local_to_global_timestamp(
    local_start: float,
    local_end: float,
    slice_start: float,
) -> tuple[float, float]:
    """Convert local sliced-audio timestamps to global audio time.

    Args:
        local_start: Word start time relative to the audio slice.
        local_end: Word end time relative to the audio slice.
        slice_start: Global time at which the audio slice begins.

    Returns:
        ``(global_start, global_end)``.
    """
    return (slice_start + local_start, slice_start + local_end)


class RecoveryFailureReason(str, Enum):
    """Explicit diagnostic category explaining why acoustic recovery failed."""

    AUDIO_LOAD_FAILED = "AUDIO_LOAD_FAILED"
    INVALID_AUDIO_WINDOW = "INVALID_AUDIO_WINDOW"
    ACOUSTIC_ALIGNMENT_FAILED = "ACOUSTIC_ALIGNMENT_FAILED"
    EMPTY_ALIGNMENT = "EMPTY_ALIGNMENT"
    WORD_COUNT_MISMATCH = "WORD_COUNT_MISMATCH"
    TEXT_MISMATCH = "TEXT_MISMATCH"
    TIMESTAMP_OUT_OF_BOUNDS = "TIMESTAMP_OUT_OF_BOUNDS"
    NON_MONOTONIC_TIMESTAMPS = "NON_MONOTONIC_TIMESTAMPS"
    INVALID_DURATION = "INVALID_DURATION"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"


@dataclass
class RecoveredWord:
    """A single recovered word with canonical text and validated timing."""

    word: str
    start: float
    end: float
    confidence: float
    confidence_source: Literal["acoustic_recovery", "interpolation"]


@dataclass
class GapRecoveryResult:
    """Result of recovering a single unaligned-word gap."""

    words: list[RecoveredWord]
    method: Literal["acoustic", "interpolation"]
    success: bool
    failure_reason: RecoveryFailureReason | None
    gap_start: float
    gap_end: float


def is_placeholder_word(word: dict[str, Any]) -> bool:
    """Return True if a word dictionary is a placeholder/unaligned fallback.

    A word is considered a placeholder when it is explicitly flagged, when its
    confidence is at or below the threshold, or when its duration is
    non-positive. A short word with genuine positive confidence and a positive
    duration is intentionally NOT classified as a placeholder.

    Args:
        word: Word dictionary with ``word``, ``start``, ``end`` and optional
            ``confidence`` / ``is_placeholder`` / ``fallback`` keys.

    Returns:
        True if the word is unaligned and needs recovery.
    """
    if bool(word.get("is_placeholder", False)) or bool(word.get("fallback", False)):
        return True
    conf = float(word.get("confidence", 0.0))
    if conf <= 0.0:
        return True
    dur = float(word.get("end", 0.0)) - float(word.get("start", 0.0))
    if dur <= 0.0:
        return True
    return False


# ---------------------------------------------------------------------------
# Cascade (ayah-level) recovery — retained for dp_core compatibility
# ---------------------------------------------------------------------------


def find_cascade_sequences(
    results: list[AlignmentResult],
    threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[tuple[int, int]]:
    """Find sequences of consecutive low-scoring ayahs (cascades).

    Args:
        results: List of alignment results.
        threshold: Similarity threshold below which a result is low-scoring.
        min_cascade_length: Minimum run length to qualify as a cascade.

    Returns:
        List of ``(start_idx, end_idx)`` tuples (end exclusive).
    """
    cascades: list[tuple[int, int]] = []
    i = 0

    while i < len(results):
        if results[i].similarity_score < threshold:
            start = i
            while i < len(results) and results[i].similarity_score < threshold:
                i += 1
            end = i

            if end - start >= min_cascade_length:
                cascades.append((start, end))
        else:
            i += 1

    return cascades


def _recover_cascade_with_resync(
    segments: list[Segment],
    ayahs: list[Ayah],
    results: list[AlignmentResult],
    cascade_start: int,
    cascade_end: int,
    silences_sec: list[tuple[float, float]],
    context_ayahs: int = 1,
) -> list[AlignmentResult] | None:
    """Attempt to recover a cascade by re-aligning using silence boundaries.

    Strategy:
    1. Extend the cascade range by one ayah on each side for context.
    2. Find the segment range covered by these ayahs.
    3. Re-run DP alignment on this portion with a silence-boundary bonus.

    Returns:
        New alignment results for the cascade region, or None if recovery failed.
    """
    extended_start = max(0, cascade_start - context_ayahs)
    extended_end = min(len(results), cascade_end + context_ayahs)

    seg_start_time = results[extended_start].start_time
    seg_end_time = results[extended_end - 1].end_time

    seg_indices = []
    for idx, seg in enumerate(segments):
        if seg.start >= seg_start_time - 0.5 and seg.end <= seg_end_time + 0.5:
            seg_indices.append(idx)

    if not seg_indices:
        return None

    seg_range_start = min(seg_indices)
    seg_range_end = max(seg_indices) + 1

    sub_segments = segments[seg_range_start:seg_range_end]
    sub_ayahs = [results[i].ayah for i in range(extended_start, extended_end)]

    if len(sub_segments) < len(sub_ayahs):
        return None

    relevant_silences = [
        (sil_start, sil_end)
        for sil_start, sil_end in silences_sec
        if seg_start_time <= sil_start <= seg_end_time
    ]

    n_sub_seg = len(sub_segments)
    n_sub_ayah = len(sub_ayahs)

    INF = float("inf")
    dp: dict[tuple[int, int], tuple[float, str, int, tuple[int, int] | None]] = {}
    dp[(0, 0)] = (0.0, "", 0, None)

    silence_aligned_ends = set()
    for idx, seg in enumerate(sub_segments):
        for sil_start, _sil_end in relevant_silences:
            if abs(seg.end - sil_start) < 0.3:
                silence_aligned_ends.add(idx + 1)

    max_segs = min(6, n_sub_seg)

    for j in range(1, n_sub_ayah + 1):
        for i in range(j, n_sub_seg + 1):
            best = None
            best_cost = INF

            for k in range(1, min(max_segs, i) + 1):
                prev_i = i - k
                prev_j = j - 1

                if (prev_i, prev_j) not in dp:
                    continue

                prev_cost, _, _, _ = dp[(prev_i, prev_j)]

                merged_text = " ".join(seg.text for seg in sub_segments[prev_i:i])
                cost = compute_alignment_cost(merged_text, sub_ayahs[j - 1].text)

                if i in silence_aligned_ends:
                    cost -= 0.15

                total_cost = prev_cost + cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best = (total_cost, merged_text, prev_i, (prev_i, prev_j))

            if best is not None:
                dp[(i, j)] = best

    best_end = None
    best_end_cost = INF

    for i in range(n_sub_ayah, n_sub_seg + 1):
        if (i, n_sub_ayah) in dp:
            if dp[(i, n_sub_ayah)][0] < best_end_cost:
                best_end_cost = dp[(i, n_sub_ayah)][0]
                best_end = (i, n_sub_ayah)

    if best_end is None:
        return None

    path: list[tuple[int, int, int, str]] = []
    current: tuple[int, int] | None = best_end

    while current and current in dp:
        _cost, merged_text, seg_start, parent = dp[current]
        i, j = current

        if parent is not None:
            path.append((seg_start, i, j, merged_text))

        current = parent

    path.reverse()

    new_results = []
    for seg_start_idx, seg_end_idx, ayah_idx, merged_text in path:
        if seg_start_idx >= len(sub_segments) or seg_end_idx > len(sub_segments):
            continue

        ayah = sub_ayahs[ayah_idx - 1]
        start_time = sub_segments[seg_start_idx].start
        end_time = sub_segments[seg_end_idx - 1].end

        sim_score = similarity(merged_text, ayah.text)

        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        new_results.append(result)

    if len(new_results) != extended_end - extended_start:
        return None

    old_results_range = results[extended_start:extended_end]

    for old, new in zip(old_results_range, new_results, strict=False):
        drop = old.similarity_score - new.similarity_score

        if old.similarity_score >= 0.75 and drop > 0.08:
            return None

        if old.similarity_score >= 0.5 and drop > 0.12:
            return None

        if old.similarity_score >= 0.75 and new.similarity_score < 0.70:
            return None

    context = 1
    cascade_old_start = max(0, context)
    cascade_old_end = (
        min(len(old_results_range), len(old_results_range) - context)
        if len(old_results_range) > 2
        else len(old_results_range)
    )

    old_cascade_sim = sum(
        r.similarity_score for r in old_results_range[cascade_old_start:cascade_old_end]
    )
    new_cascade_sim = sum(
        r.similarity_score for r in new_results[cascade_old_start:cascade_old_end]
    )

    cascade_len = cascade_old_end - cascade_old_start
    if cascade_len == 0:
        return None

    if new_cascade_sim / cascade_len > old_cascade_sim / cascade_len + 0.08:
        return new_results

    return None


def apply_cascade_recovery(
    segments: list[Segment],
    ayahs: list[Ayah],
    results: list[AlignmentResult],
    silences_ms: list[tuple[int, int]] | None = None,
    cascade_threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[AlignmentResult]:
    """Post-process alignment results to recover cascaded ayah-level failures.

    Args:
        segments: Original segments.
        ayahs: Original ayahs.
        results: Initial alignment results.
        silences_ms: Silence periods in milliseconds.
        cascade_threshold: Similarity below which triggers cascade detection.
        min_cascade_length: Minimum consecutive failures to form a cascade.

    Returns:
        Improved alignment results.
    """
    if not results:
        return results

    silences_sec = []
    if silences_ms:
        for start_ms, end_ms in silences_ms:
            silences_sec.append((start_ms / 1000.0, end_ms / 1000.0))

    cascades = find_cascade_sequences(results, cascade_threshold, min_cascade_length)

    if not cascades:
        return results

    improved_results = list(results)

    for cascade_start, cascade_end in reversed(cascades):
        recovery = _recover_cascade_with_resync(
            segments,
            ayahs,
            improved_results,
            cascade_start,
            cascade_end,
            silences_sec,
        )

        if recovery:
            context = 1
            ext_start = max(0, cascade_start - context)
            ext_end = min(len(improved_results), cascade_end + context)

            improved_results = improved_results[:ext_start] + recovery + improved_results[ext_end:]

    return improved_results


# ---------------------------------------------------------------------------
# Word-level unaligned gap recovery
# ---------------------------------------------------------------------------


@dataclass
class UnalignedWordGap:
    """Represents a gap of unaligned fallback words needing recovery."""

    start_word_idx: int
    end_word_idx: int
    words: list[str]
    gap_start_time: float
    gap_end_time: float


@dataclass
class _GapAnchors:
    """The valid aligned words immediately surrounding a gap."""

    previous_end: float | None
    next_start: float | None


def _find_gap_anchors(words: list[dict], gap: UnalignedWordGap) -> _GapAnchors:
    """Resolve the valid anchors bordering a gap from non-placeholder words.

    Anchors are derived solely from the valid neighboring words, never from the
    placeholder timestamps inside the gap itself.
    """
    previous_end: float | None = None
    for k in range(gap.start_word_idx - 1, -1, -1):
        if not is_placeholder_word(words[k]):
            previous_end = float(words[k]["end"])
            break

    next_start: float | None = None
    for k in range(gap.end_word_idx, len(words)):
        if not is_placeholder_word(words[k]):
            next_start = float(words[k]["start"])
            break

    return _GapAnchors(previous_end=previous_end, next_start=next_start)


def calculate_gap_bounds(
    gap: UnalignedWordGap,
    anchors: _GapAnchors,
    audio_duration: float,
) -> tuple[float, float]:
    """Compute the ``(gap_start, gap_end)`` recovery window from valid anchors.

    Window rules (spec §8):
    * both anchors present -> ``[previous_end, next_start]``
    * no previous anchor     -> ``[0.0, next_start]``
    * no next anchor         -> ``[previous_end, audio_duration]``
    * neither anchor         -> ``[0.0, audio_duration]``

    Args:
        gap: The unaligned gap.
        anchors: The resolved surrounding anchors.
        audio_duration: Total audio duration in seconds.

    Returns:
        ``(gap_start, gap_end)`` normalized to a non-decreasing interval.
    """
    if anchors.previous_end is not None and anchors.next_start is not None:
        gap_start = anchors.previous_end
        gap_end = max(anchors.next_start, gap_start)
    elif anchors.previous_end is not None:
        gap_start = anchors.previous_end
        gap_end = max(audio_duration, gap_start)
    elif anchors.next_start is not None:
        gap_start = 0.0
        gap_end = max(anchors.next_start, gap_start)
    else:
        gap_start = 0.0
        gap_end = max(audio_duration, 0.0)

    return gap_start, gap_end


def detect_unaligned_word_gaps(
    words: list[dict],
    min_confidence_thresh: float = 0.1,
    max_placeholder_duration: float = 0.15,
) -> list[UnalignedWordGap]:
    """Detect sequences of consecutive unaligned fallback words.

    A word is detected as unaligned by :func:`is_placeholder_word`. The
    ``min_confidence_thresh`` and ``max_placeholder_duration`` arguments are
    retained for backwards compatibility; the actual detection predicate does
    not use a fragile duration-only cutoff on valid words.

    Args:
        words: List of word dictionaries containing ``word``, ``start``, ``end``
            and ``confidence``.
        min_confidence_thresh: Confidence threshold for unaligned detection.
        max_placeholder_duration: Maximum duration threshold for placeholder
            validation (retained for API compatibility).

    Returns:
        List of :class:`UnalignedWordGap` objects.
    """
    gaps: list[UnalignedWordGap] = []
    if not words:
        return gaps

    n = len(words)
    i = 0
    while i < n:
        if is_placeholder_word(words[i]):
            start_idx = i
            while i < n and is_placeholder_word(words[i]):
                i += 1
            end_idx = i

            gap_words = [str(words[k]["word"]) for k in range(start_idx, end_idx)]

            prev_end = (
                float(words[start_idx - 1]["end"])
                if start_idx > 0 and not is_placeholder_word(words[start_idx - 1])
                else 0.0
            )
            next_start = (
                float(words[end_idx]["start"])
                if end_idx < n and not is_placeholder_word(words[end_idx])
                else 0.0
            )

            # Use anchor-derived bounds when anchors exist, otherwise fall back
            # to the placeholder-adjacent timestamps with a non-negative interval.
            gap_start = prev_end
            gap_end = next_start if next_start >= prev_end else prev_end

            if gap_end <= gap_start and start_idx > 0:
                gap_start = float(words[start_idx - 1]["end"])
            if gap_end <= gap_start and end_idx < n:
                gap_end = float(words[end_idx]["start"])
            if gap_end < gap_start:
                gap_end = gap_start

            gaps.append(
                UnalignedWordGap(
                    start_word_idx=start_idx,
                    end_word_idx=end_idx,
                    words=gap_words,
                    gap_start_time=gap_start,
                    gap_end_time=gap_end,
                )
            )
        else:
            i += 1

    return gaps


def slice_audio_array(
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, float, float]:
    """Extract a slice of the audio array corresponding to ``[start_sec, end_sec]``.

    Args:
        audio: 1D numpy array of audio samples.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        ``(sliced_audio, actual_start_sec, actual_end_sec)``.
    """
    total_samples = len(audio)
    start_sample = min(total_samples, max(0, int(start_sec * sample_rate)))
    end_sample = min(total_samples, int(end_sec * sample_rate))

    if end_sample <= start_sample:
        end_sample = min(total_samples, start_sample + int(0.1 * sample_rate))

    audio_slice = audio[start_sample:end_sample]
    actual_start_sec = start_sample / float(sample_rate)
    actual_end_sec = end_sample / float(sample_rate)
    return audio_slice, actual_start_sec, actual_end_sec


def _map_acoustic_to_canonical_dp(
    canonical_words: list[str],
    normalized_canonical: list[str],
    acoustic_words: list[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    """Map acoustic words to canonical words using a deterministic DP alignment.

    Returns a list of length ``len(canonical_words)`` where each entry is either
    the matched acoustic word (with a ``word`` key added) or ``None`` when the
    canonical word has no acoustic counterpart (a deletion).
    """
    from rapidfuzz import fuzz

    from .arabic import normalize_arabic

    n_c = len(canonical_words)
    n_a = len(acoustic_words)

    if n_a == 0:
        return [None] * n_c

    norm_acoustic = [normalize_arabic(str(w["word"])) for w in acoustic_words]

    # DP with explicit insertion/deletion/substitution costs.
    # Rows = canonical, columns = acoustic.
    INF_NEG = float("-inf")
    gap_penalty = -0.5

    dp = [[INF_NEG] * (n_a + 1) for _ in range(n_c + 1)]
    dp[0][0] = 0.0
    for i in range(1, n_c + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, n_a + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    for i in range(1, n_c + 1):
        rw = normalized_canonical[i - 1]
        for j in range(1, n_a + 1):
            ew = norm_acoustic[j - 1]
            match_score = fuzz.ratio(rw, ew) / 100.0
            if match_score < 0.5:
                match_score = -1.0
            diag = dp[i - 1][j - 1] + match_score
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(diag, delete, insert)

    mapped: list[dict[str, Any] | None] = [None] * n_c
    i, j = n_c, n_a
    while i > 0 and j > 0:
        rw = normalized_canonical[i - 1]
        ew = norm_acoustic[j - 1]
        match_score = fuzz.ratio(rw, ew) / 100.0
        if match_score >= 0.5 and abs(dp[i][j] - (dp[i - 1][j - 1] + match_score)) < 1e-6:
            mapped[i - 1] = dict(acoustic_words[j - 1])
            i -= 1
            j -= 1
        elif abs(dp[i][j] - (dp[i - 1][j] + gap_penalty)) < 1e-6:
            mapped[i - 1] = None
            i -= 1
        else:
            j -= 1

    return mapped


def validate_recovered_gap(
    recovered_words: list[RecoveredWord],
    gap_start: float,
    gap_end: float,
    previous_anchor_end: float | None,
    next_anchor_start: float | None,
) -> tuple[bool, RecoveryFailureReason | None]:
    """Validate a recovered word sequence against its gap and anchors.

    Validates finiteness, duration, monotonicity, containment within the gap,
    and non-overlap with the surrounding anchors.

    Args:
        recovered_words: Recovered words in canonical order.
        gap_start: Recovery window start.
        gap_end: Recovery window end.
        previous_anchor_end: End of the preceding valid anchor (or None).
        next_anchor_start: Start of the succeeding valid anchor (or None).

    Returns:
        ``(valid, failure_reason)``.
    """
    if not recovered_words:
        return False, RecoveryFailureReason.EMPTY_ALIGNMENT

    tol = TIMESTAMP_TOLERANCE
    prev_end = gap_start

    for w in recovered_words:
        if not (np.isfinite(w.start) and np.isfinite(w.end)):
            return False, RecoveryFailureReason.MODEL_OUTPUT_INVALID
        if w.end <= w.start:
            return False, RecoveryFailureReason.INVALID_DURATION
        if w.start < gap_start - tol or w.end > gap_end + tol:
            return False, RecoveryFailureReason.TIMESTAMP_OUT_OF_BOUNDS
        if w.start < prev_end - tol:
            return False, RecoveryFailureReason.NON_MONOTONIC_TIMESTAMPS
        prev_end = w.end

    if previous_anchor_end is not None:
        if recovered_words[0].start < previous_anchor_end - tol:
            return False, RecoveryFailureReason.TIMESTAMP_OUT_OF_BOUNDS

    if next_anchor_start is not None:
        if recovered_words[-1].end > next_anchor_start + tol:
            return False, RecoveryFailureReason.TIMESTAMP_OUT_OF_BOUNDS

    return True, None


def _interpolate_gap(
    gap: UnalignedWordGap,
    gap_start: float,
    gap_end: float,
) -> list[RecoveredWord]:
    """Deterministic character-weighted interpolation within the gap bounds."""
    total_duration = gap_end - gap_start
    n = len(gap.words)

    if total_duration <= 0 or n == 0:
        total_duration = 0.1 * n
        if total_duration <= 0:
            total_duration = 0.1

    char_lens = [max(1, len(w)) for w in gap.words]
    total_chars = sum(char_lens)

    words: list[RecoveredWord] = []
    curr_t = gap_start
    for idx, w in enumerate(gap.words):
        w_dur = (char_lens[idx] / total_chars) * total_duration
        w_end = min(gap_end, curr_t + w_dur)
        words.append(
            RecoveredWord(
                word=w,
                start=round(curr_t, 3),
                end=round(w_end, 3),
                confidence=0.60,
                confidence_source="interpolation",
            )
        )
        curr_t = w_end

    return words


def realign_unaligned_gap_acoustic(
    gap: UnalignedWordGap,
    audio: np.ndarray,
    align_model: Any,
    align_metadata: Any,
    device: str = "cpu",
    sample_rate: int = 16000,
    context_pad_sec: float = DEFAULT_CONTEXT_PAD_SEC,
    is_trailing: bool = False,
    is_leading: bool = False,
    audio_duration: float | None = None,
) -> tuple[list[RecoveredWord] | None, RecoveryFailureReason | None]:
    """Perform acoustic realignment on a dynamically sliced audio window.

    The sliced audio is aligned with ``whisperx.align``; the returned local
    word timestamps are converted to global time via
    :func:`local_to_global_timestamp` and mapped back to the canonical words
    using DP. Canonical text is always preserved.

    Args:
        gap: The unaligned gap.
        audio: Full audio numpy array.
        align_model: WhisperX / wav2vec2 alignment model.
        align_metadata: WhisperX alignment metadata dictionary.
        device: Torch compute device.
        sample_rate: Audio sample rate.
        context_pad_sec: Context padding in seconds around the gap.
        is_trailing: Whether the gap is at the end of the transcription.
        is_leading: Whether the gap is at the beginning of the transcription.
        audio_duration: Total audio duration (derived from ``audio`` if None).

    Returns:
        ``(recovered_words, failure_reason)``. ``recovered_words`` is None on
        failure, with a non-None ``failure_reason`` explaining why.
    """
    if audio is None or align_model is None or align_metadata is None:
        return None, RecoveryFailureReason.AUDIO_LOAD_FAILED

    try:
        import whisperx
    except ImportError:
        return None, RecoveryFailureReason.MODEL_OUTPUT_INVALID

    from .arabic import normalize_arabic

    total_audio_sec = len(audio) / float(sample_rate) if audio_duration is None else audio_duration

    # Resolve the recovery window from real audio bounds.
    recovery_start = gap.gap_start_time
    recovery_end = gap.gap_end_time
    if is_leading:
        recovery_start = 0.0
    if is_trailing:
        recovery_end = max(total_audio_sec, recovery_end)

    if recovery_end <= recovery_start:
        return None, RecoveryFailureReason.INVALID_AUDIO_WINDOW

    # Optional padding — clamped to available audio. Zero padding is preferred
    # unless the caller explicitly requests it (spec §9).
    slice_start = max(0.0, recovery_start - context_pad_sec)
    slice_end = min(total_audio_sec, recovery_end + context_pad_sec)

    audio_slice, actual_slice_start, actual_slice_end = slice_audio_array(
        audio, slice_start, slice_end, sample_rate=sample_rate
    )
    slice_duration = actual_slice_end - actual_slice_start

    if slice_duration <= 0.05 or len(audio_slice) == 0:
        return None, RecoveryFailureReason.INVALID_AUDIO_WINDOW

    normalized_gap_words = [normalize_arabic(w) for w in gap.words]
    gap_text = " ".join([w for w in normalized_gap_words if w])
    if not gap_text:
        gap_text = " ".join(gap.words)

    segments_to_align = [
        {
            "text": gap_text,
            "start": 0.0,
            "end": slice_duration,
        }
    ]

    try:
        align_result = whisperx.align(
            segments_to_align,
            align_model,
            align_metadata,
            audio_slice,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:  # noqa: BLE001 - alignment model failure
        logger.warning("acoustic alignment raised: %s", exc)
        return None, RecoveryFailureReason.ACOUSTIC_ALIGNMENT_FAILED

    extracted_words: list[dict[str, Any]] = []
    if isinstance(align_result, dict) and "segments" in align_result:
        for seg in align_result["segments"]:
            if isinstance(seg, dict) and "words" in seg:
                for w in seg["words"]:
                    if isinstance(w, dict) and "start" in w and "end" in w:
                        extracted_words.append(
                            {
                                "word": str(w["word"]),
                                "start": float(w["start"]),
                                "end": float(w["end"]),
                                "confidence": float(w.get("score", 0.85)),
                            }
                        )

    if not extracted_words:
        return None, RecoveryFailureReason.EMPTY_ALIGNMENT

    n_gap = len(gap.words)
    if len(extracted_words) != n_gap:
        mapped = _map_acoustic_to_canonical_dp(gap.words, normalized_gap_words, extracted_words)
    else:
        mapped = list(extracted_words)

    # Build recovered words in canonical order, in global time.
    recovered: list[RecoveredWord] = []
    for k in range(n_gap):
        item = mapped[k]
        if item is None:
            return None, RecoveryFailureReason.WORD_COUNT_MISMATCH

        local_start = float(item["start"])
        local_end = float(item["end"])
        global_start, global_end = local_to_global_timestamp(
            local_start, local_end, actual_slice_start
        )

        # Reject major violations rather than silently clamping them (spec §16).
        # Allow a small tolerance for floating-point rounding only.
        if global_start < recovery_start - TIMESTAMP_TOLERANCE:
            return None, RecoveryFailureReason.TIMESTAMP_OUT_OF_BOUNDS
        if global_end > recovery_end + TIMESTAMP_TOLERANCE:
            return None, RecoveryFailureReason.TIMESTAMP_OUT_OF_BOUNDS
        if global_end <= global_start:
            return None, RecoveryFailureReason.INVALID_DURATION

        # Clamp only tiny floating-point deviations into the recovery window.
        global_start = max(recovery_start, global_start)
        global_end = min(recovery_end, global_end)

        recovered.append(
            RecoveredWord(
                word=gap.words[k],
                start=round(global_start, 3),
                end=round(global_end, 3),
                confidence=float(item.get("confidence", 0.85)),
                confidence_source="acoustic_recovery",
            )
        )

    return recovered, None


def recover_unaligned_word_gaps(
    words: list[dict],
    min_confidence_thresh: float = 0.1,
    max_placeholder_duration: float = 0.15,
    audio: np.ndarray | None = None,
    align_model: Any | None = None,
    align_metadata: Any | None = None,
    device: str = "cpu",
    sample_rate: int = 16000,
    audio_duration: float | None = None,
) -> list[dict]:
    """Recover timestamps for unaligned fallback words.

    For every detected gap this function derives the recovery window from the
    surrounding valid anchors, attempts acoustic re-alignment, and falls back
    to character-weighted interpolation only when acoustic recovery fails. The
    recovered words are merged back into the timeline preserving canonical text,
    ordering, and (where applicable) the original word count.

    Args:
        words: List of word timestamp dictionaries.
        min_confidence_thresh: Confidence threshold for unaligned detection.
        max_placeholder_duration: Maximum duration threshold (compat).
        audio: Full audio waveform array.
        align_model: Acoustic alignment model.
        align_metadata: Alignment model metadata.
        device: Torch compute device.
        sample_rate: Audio sample rate.
        audio_duration: Total audio duration (derived from audio if None).

    Returns:
        Updated list of word dictionaries with recovered start/end/confidence.
    """
    gaps = detect_unaligned_word_gaps(
        words,
        min_confidence_thresh=min_confidence_thresh,
        max_placeholder_duration=max_placeholder_duration,
    )
    if not gaps:
        return words

    if audio_duration is None:
        if audio is not None and sample_rate > 0:
            audio_duration = len(audio) / float(sample_rate)
        else:
            audio_duration = float(words[-1]["end"]) if words else 0.0

    recovered_words = list(words)
    n_words = len(words)

    # Resolve anchors and bounds for all gaps up-front using the ORIGINAL
    # word list, so one recovery never corrupts another's boundaries (§18/§31).
    resolve_plan: list[tuple[UnalignedWordGap, bool, bool, float, float]] = []
    for gap in gaps:
        is_leading = gap.start_word_idx == 0
        is_trailing = gap.end_word_idx >= n_words
        anchors = _find_gap_anchors(words, gap)
        gap_start, gap_end = calculate_gap_bounds(gap, anchors, audio_duration)

        # Override the gap's original (stale) bounds with anchor-derived bounds.
        gap.gap_start_time = gap_start
        gap.gap_end_time = gap_end

        resolve_plan.append((gap, is_leading, is_trailing, gap_start, gap_end))

    for gap, is_leading, is_trailing, gap_start, gap_end in resolve_plan:
        failure_reason: RecoveryFailureReason | None = RecoveryFailureReason.AUDIO_LOAD_FAILED
        recovered: list[RecoveredWord] | None = None

        if audio is not None and align_model is not None and align_metadata is not None:
            recovered, failure_reason = realign_unaligned_gap_acoustic(
                gap=gap,
                audio=audio,
                align_model=align_model,
                align_metadata=align_metadata,
                device=device,
                sample_rate=sample_rate,
                is_trailing=is_trailing,
                is_leading=is_leading,
                audio_duration=audio_duration,
            )

        if recovered is None or len(recovered) != (gap.end_word_idx - gap.start_word_idx):
            # Fall back to interpolation with an explicit reason.
            recorded_reason = (
                failure_reason if recovered is None else RecoveryFailureReason.WORD_COUNT_MISMATCH
            )
            logger.info(
                "gap [%s, %s] acoustic recovery failed (%s); using interpolation",
                gap_start,
                gap_end,
                recorded_reason.value if recorded_reason else "unknown",
            )
            recovered = _interpolate_gap(gap, gap_start, gap_end)

        # Validate before merging.
        anchors = _find_gap_anchors(words, gap)
        valid, reason = validate_recovered_gap(
            recovered,
            gap_start,
            gap_end,
            anchors.previous_end,
            anchors.next_start,
        )
        if not valid:
            logger.warning(
                "recovered gap [%s, %s] failed validation (%s); skipping",
                gap_start,
                gap_end,
                reason.value if reason else "unknown",
            )
            continue

        for idx_offset, w_idx in enumerate(range(gap.start_word_idx, gap.end_word_idx)):
            rw = recovered[idx_offset]
            recovered_words[w_idx] = {
                "word": gap.words[idx_offset],
                "start": rw.start,
                "end": rw.end,
                "confidence": rw.confidence,
            }

    return recovered_words


__all__ = [
    "RecoveryFailureReason",
    "RecoveredWord",
    "GapRecoveryResult",
    "UnalignedWordGap",
    "TIMESTAMP_TOLERANCE",
    "timestamps_equal",
    "local_to_global_timestamp",
    "is_placeholder_word",
    "calculate_gap_bounds",
    "validate_recovered_gap",
    "detect_unaligned_word_gaps",
    "slice_audio_array",
    "realign_unaligned_gap_acoustic",
    "recover_unaligned_word_gaps",
    "find_cascade_sequences",
    "apply_cascade_recovery",
]
