"""
Cascade recovery for alignment drift.

Detects sequences of consecutive low-scoring ayahs and attempts
to re-align them using silence boundaries for better sync.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models import AlignmentResult, Ayah, Segment
from .dp_core import compute_alignment_cost
from .matcher import similarity


def find_cascade_sequences(
    results: list[AlignmentResult],
    threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[tuple[int, int]]:
    """
    Find sequences of consecutive low-scoring ayahs (cascades).

    Args:
        results: List of alignment results
        threshold: Similarity threshold below which is considered low
        min_cascade_length: Minimum length to be considered a cascade

    Returns:
        List of (start_idx, end_idx) tuples for each cascade
    """
    cascades = []
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
    """
    Attempt to recover a cascade by re-aligning using silence boundaries.

    Strategy:
    1. Extend the cascade range by 1 ayah on each side for context
    2. Find the segment range covered by these ayahs
    3. Find silence boundaries within this range
    4. Re-run DP alignment on just this portion with emphasis on silence breaks

    Returns:
        New alignment results for the cascade region, or None if recovery failed
    """
    # Extend range for context
    extended_start = max(0, cascade_start - context_ayahs)
    extended_end = min(len(results), cascade_end + context_ayahs)

    # Get segment range for the extended ayah range
    seg_start_time = results[extended_start].start_time
    seg_end_time = results[extended_end - 1].end_time

    # Find segments in this time range
    seg_indices = []
    for idx, seg in enumerate(segments):
        if seg.start >= seg_start_time - 0.5 and seg.end <= seg_end_time + 0.5:
            seg_indices.append(idx)

    if not seg_indices:
        return None

    seg_range_start = min(seg_indices)
    seg_range_end = max(seg_indices) + 1

    # Extract the segments and ayahs for re-alignment
    sub_segments = segments[seg_range_start:seg_range_end]
    sub_ayahs = [results[i].ayah for i in range(extended_start, extended_end)]

    if len(sub_segments) < len(sub_ayahs):
        return None

    # Find silences in this time range
    relevant_silences = []
    for sil_start, sil_end in silences_sec:
        if seg_start_time <= sil_start <= seg_end_time:
            relevant_silences.append((sil_start, sil_end))

    n_sub_seg = len(sub_segments)
    n_sub_ayah = len(sub_ayahs)

    INF = float("inf")
    dp: dict[tuple[int, int], tuple[float, str, int, tuple[int, int] | None]] = {}
    dp[(0, 0)] = (0.0, "", 0, None)

    # Build set of segment indices that align with silences
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

                # Bonus for ending at silence boundary
                if i in silence_aligned_ends:
                    cost -= 0.15

                total_cost = prev_cost + cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best = (total_cost, merged_text, prev_i, (prev_i, prev_j))

            if best is not None:
                dp[(i, j)] = best

    # Find best ending
    best_end = None
    best_end_cost = INF

    for i in range(n_sub_ayah, n_sub_seg + 1):
        if (i, n_sub_ayah) in dp:
            if dp[(i, n_sub_ayah)][0] < best_end_cost:
                best_end_cost = dp[(i, n_sub_ayah)][0]
                best_end = (i, n_sub_ayah)

    if best_end is None:
        return None

    # Backtrack and build new results
    path: list[tuple[int, int, int, str]] = []
    current: tuple[int, int] | None = best_end

    while current and current in dp:
        _cost, merged_text, seg_start, parent = dp[current]
        i, j = current

        if parent is not None:
            path.append((seg_start, i, j, merged_text))

        current = parent

    path.reverse()

    # Convert to results
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

    # Check if recovery improved the results
    if len(new_results) != extended_end - extended_start:
        return None

    old_results_range = results[extended_start:extended_end]

    # Conservative check: Don't accept recovery if ANY ayah degrades significantly
    for old, new in zip(old_results_range, new_results, strict=False):
        drop = old.similarity_score - new.similarity_score

        # Strict protection for good ayahs (>= 75%): max 8% drop
        if old.similarity_score >= 0.75 and drop > 0.08:
            return None

        # Protect mediocre ayahs (50-75%): max 12% drop
        if old.similarity_score >= 0.5 and drop > 0.12:
            return None

        # Never let a good ayah (>=75%) drop below 70%
        if old.similarity_score >= 0.75 and new.similarity_score < 0.70:
            return None

    # Check overall improvement in the cascade region
    context = 1
    cascade_old_start = max(0, context)
    cascade_old_end = (
        min(len(old_results_range), len(old_results_range) - context)
        if len(old_results_range) > 2
        else len(old_results_range)
    )

    cascade_new_start = cascade_old_start
    cascade_new_end = cascade_old_end

    old_cascade_sim = sum(
        r.similarity_score for r in old_results_range[cascade_old_start:cascade_old_end]
    )
    new_cascade_sim = sum(
        r.similarity_score for r in new_results[cascade_new_start:cascade_new_end]
    )

    cascade_len = cascade_old_end - cascade_old_start
    if cascade_len == 0:
        return None

    old_avg = old_cascade_sim / cascade_len
    new_avg = new_cascade_sim / cascade_len

    # Require significant improvement in the cascade region
    if new_avg > old_avg + 0.08:
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
    """
    Post-process alignment results to recover cascaded failures.

    Detects sequences of consecutive low-scoring ayahs and attempts
    to re-align them using silence boundaries for better sync.

    Args:
        segments: Original segments
        ayahs: Original ayahs
        results: Initial alignment results
        silences_ms: Silence periods in milliseconds
        cascade_threshold: Similarity below which triggers cascade detection
        min_cascade_length: Minimum consecutive failures to be a cascade

    Returns:
        Improved alignment results
    """
    if not results:
        return results

    # Convert silences to seconds
    silences_sec = []
    if silences_ms:
        for start_ms, end_ms in silences_ms:
            silences_sec.append((start_ms / 1000.0, end_ms / 1000.0))

    # Find cascades
    cascades = find_cascade_sequences(results, cascade_threshold, min_cascade_length)

    if not cascades:
        return results

    # Process each cascade (in reverse order to maintain indices)
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


@dataclass
class UnalignedWordGap:
    """Represents a gap of unaligned fallback words needing recovery."""

    start_word_idx: int
    end_word_idx: int
    words: list[str]
    gap_start_time: float
    gap_end_time: float


def detect_unaligned_word_gaps(
    words: list[dict],
    min_confidence_thresh: float = 0.1,
    max_placeholder_duration: float = 0.15,
) -> list[UnalignedWordGap]:
    """
    Detect sequences of unaligned fallback words.

    A word is detected as unaligned if:
    - It is explicitly marked with 'is_placeholder': True or 'fallback': True.
    - Its confidence is <= min_confidence_thresh (e.g. 0.0 fallback from ASR).
    - Its duration is non-positive (<= 0.0).

    Args:
        words: List of word timestamp dictionaries containing 'word', 'start', 'end', and 'confidence'.
        min_confidence_thresh: Confidence threshold below which a word is considered unaligned.
        max_placeholder_duration: Maximum duration threshold for placeholder validation.

    Returns:
        List of UnalignedWordGap objects.
    """
    gaps: list[UnalignedWordGap] = []
    if not words:
        return gaps

    def _is_unaligned(w: dict) -> bool:
        if bool(w.get("is_placeholder", False)) or bool(w.get("fallback", False)):
            return True
        conf = float(w.get("confidence", 0.0))
        if conf <= min_confidence_thresh:
            return True
        dur = float(w.get("end", 0.0)) - float(w.get("start", 0.0))
        if dur <= 0.0:
            return True
        return False

    n = len(words)
    i = 0
    while i < n:
        if _is_unaligned(words[i]):
            start_idx = i
            while i < n and _is_unaligned(words[i]):
                i += 1
            end_idx = i

            prev_end = (
                float(words[start_idx - 1]["end"])
                if start_idx > 0
                else float(words[start_idx]["start"])
            )
            next_start = (
                float(words[end_idx]["start"]) if end_idx < n else float(words[end_idx - 1]["end"])
            )
            if next_start < prev_end:
                next_start = prev_end

            gap_words = [str(words[k]["word"]) for k in range(start_idx, end_idx)]
            gaps.append(
                UnalignedWordGap(
                    start_word_idx=start_idx,
                    end_word_idx=end_idx,
                    words=gap_words,
                    gap_start_time=prev_end,
                    gap_end_time=next_start,
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
    """
    Extract a slice of the audio array corresponding to [start_sec, end_sec].

    Args:
        audio: 1D numpy array of audio samples.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sample_rate: Audio sample rate in Hz (default: 16000).

    Returns:
        Tuple of (sliced_audio_array, actual_start_sec, actual_end_sec).
    """
    total_samples = len(audio)
    start_sample = min(total_samples, max(0, int(start_sec * sample_rate)))
    end_sample = min(total_samples, int(end_sec * sample_rate))

    if end_sample <= start_sample:
        # Guarantee non-empty slice of at least 100ms
        end_sample = min(total_samples, start_sample + int(0.1 * sample_rate))

    audio_slice = audio[start_sample:end_sample]
    actual_start_sec = start_sample / float(sample_rate)
    actual_end_sec = end_sample / float(sample_rate)
    return audio_slice, actual_start_sec, actual_end_sec


def realign_unaligned_gap_acoustic(
    gap: UnalignedWordGap,
    audio: np.ndarray,
    align_model: Any,
    align_metadata: Any,
    device: str = "cpu",
    sample_rate: int = 16000,
    context_pad_sec: float = 0.25,
) -> list[dict] | None:
    """
    Perform acoustic realignment on a dynamically sliced audio window for unaligned words.

    Args:
        gap: UnalignedWordGap object.
        audio: Full audio numpy array.
        align_model: WhisperX / wav2vec2 alignment model.
        align_metadata: WhisperX alignment metadata dictionary.
        device: Torch compute device ("cpu" or "cuda").
        sample_rate: Audio sample rate.
        context_pad_sec: Context padding in seconds around the gap.

    Returns:
        List of aligned word dictionaries with true acoustic timestamps and scores, or None if realignment failed.
    """
    if audio is None or align_model is None or align_metadata is None:
        return None

    try:
        import whisperx
    except ImportError:
        return None

    slice_start = max(0.0, gap.gap_start_time - context_pad_sec)
    slice_end = gap.gap_end_time + context_pad_sec
    audio_slice, actual_slice_start, actual_slice_end = slice_audio_array(
        audio, slice_start, slice_end, sample_rate=sample_rate
    )
    slice_duration = actual_slice_end - actual_slice_start

    if slice_duration <= 0.05 or len(audio_slice) == 0:
        return None

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
    except Exception:
        return None

    extracted_words = []
    if "segments" in align_result:
        for seg in align_result["segments"]:
            if isinstance(seg, dict) and "words" in seg:
                for w in seg["words"]:
                    if isinstance(w, dict) and "start" in w and "end" in w:
                        extracted_words.append(
                            {
                                "word": str(w["word"]),
                                "start": round(actual_slice_start + float(w["start"]), 3),
                                "end": round(actual_slice_start + float(w["end"]), 3),
                                "confidence": round(float(w.get("score", 0.85)), 3),
                            }
                        )

    if len(extracted_words) == len(gap.words):
        # All words successfully realigned acoustically
        return extracted_words

    return None


def recover_unaligned_word_gaps(
    words: list[dict],
    min_confidence_thresh: float = 0.1,
    max_placeholder_duration: float = 0.15,
    audio: np.ndarray | None = None,
    align_model: Any | None = None,
    align_metadata: Any | None = None,
    device: str = "cpu",
    sample_rate: int = 16000,
) -> list[dict]:
    """
    Recover timestamps for unaligned fallback words by running acoustic realignment on sliced audio gaps.

    Args:
        words: List of word timestamp dictionaries.
        min_confidence_thresh: Confidence threshold for unaligned detection.
        max_placeholder_duration: Maximum duration threshold for placeholder validation.
        audio: Full audio waveform array (optional for acoustic pass).
        align_model: Acoustic alignment model (optional).
        align_metadata: Alignment model metadata (optional).
        device: Torch compute device.
        sample_rate: Audio sample rate.

    Returns:
        Updated list of word timestamp dictionaries with recovered start, end, and confidence scores.
    """
    gaps = detect_unaligned_word_gaps(
        words,
        min_confidence_thresh=min_confidence_thresh,
        max_placeholder_duration=max_placeholder_duration,
    )
    if not gaps:
        return words

    recovered_words = list(words)
    for gap in gaps:
        # 1. Attempt true acoustic realignment on dynamically sliced audio
        if audio is not None and align_model is not None and align_metadata is not None:
            acoustic_words = realign_unaligned_gap_acoustic(
                gap=gap,
                audio=audio,
                align_model=align_model,
                align_metadata=align_metadata,
                device=device,
                sample_rate=sample_rate,
            )
            if acoustic_words and len(acoustic_words) == (gap.end_word_idx - gap.start_word_idx):
                for idx_offset, w_idx in enumerate(range(gap.start_word_idx, gap.end_word_idx)):
                    recovered_words[w_idx] = acoustic_words[idx_offset]
                continue

        # 2. Robust fallback interpolation within surrounding anchor bounds
        total_duration = gap.gap_end_time - gap.gap_start_time
        if total_duration <= 0:
            continue

        char_lens = [max(1, len(w)) for w in gap.words]
        total_chars = sum(char_lens)

        curr_t = gap.gap_start_time
        for idx_offset, w_idx in enumerate(range(gap.start_word_idx, gap.end_word_idx)):
            w_dur = (char_lens[idx_offset] / total_chars) * total_duration
            w_end = curr_t + w_dur
            recovered_words[w_idx] = {
                "word": gap.words[idx_offset],
                "start": curr_t,
                "end": w_end,
                "confidence": 0.60,
            }
            curr_t = w_end

    return recovered_words
