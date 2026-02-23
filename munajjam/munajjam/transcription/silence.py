"""
Silence detection utilities for audio processing.

Provides both pydub (accurate) and librosa (fast) implementations.
Use the fast implementation for long files (>5 minutes).
"""

from pathlib import Path
from typing import Any


def detect_silences(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
) -> list[tuple[int, int]]:
    """
    Detect silent portions in an audio file.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)

    Returns:
        List of (start_ms, end_ms) tuples for silent portions
    """
    if use_fast:
        try:
            return _detect_silences_fast(audio_path, min_silence_len, silence_thresh)
        except:  # noqa: E722 - Bare except to catch numpy/scipy C-extension errors
            pass  # Fallback to pydub

    return _detect_silences_pydub(audio_path, min_silence_len, silence_thresh)


def _detect_silences_pydub(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """Pydub-based silence detection (slower but reliable)."""
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_wav(str(audio_path))
    silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return [(s[0], s[1]) for s in silences]


def _detect_silences_fast(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """
    Fast silence detection using librosa + numpy.

    ~10-50x faster than pydub for long files.
    """
    # Import inside try block to catch numpy/scipy version conflicts
    try:
        import librosa
        import numpy as np
    except (ImportError, AttributeError) as e:
        raise ImportError(f"librosa not available: {e}") from e

    # Load audio at native sample rate for accuracy
    y, sr = librosa.load(str(audio_path), sr=None)

    # Convert dB threshold to amplitude ratio
    # pydub uses dBFS relative to max, we'll use RMS-based detection
    # -30 dB ≈ 0.0316 amplitude ratio
    amplitude_thresh = 10 ** (silence_thresh / 20)

    # Calculate frame-based RMS energy
    # Use ~10ms frames for good resolution
    frame_length = int(sr * 0.01)  # 10ms frames
    hop_length = frame_length // 2  # 50% overlap

    # Calculate RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize RMS to 0-1 range
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_normalized = rms / rms_max

    # Find frames below threshold (silent)
    is_silent = rms_normalized < amplitude_thresh

    # Convert to time-based regions
    frame_times_ms = (
        librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
    )

    # Find contiguous silent regions
    silences = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = frame_times_ms[i]
        elif not silent and in_silence:
            # End of silence
            in_silence = False
            silence_end = frame_times_ms[i]
            duration = silence_end - silence_start
            if duration >= min_silence_len:
                silences.append((int(silence_start), int(silence_end)))

    # Handle case where audio ends in silence
    if in_silence:
        silence_end = frame_times_ms[-1]
        duration = silence_end - silence_start
        if duration >= min_silence_len:
            silences.append((int(silence_start), int(silence_end)))

    return silences


def detect_non_silent_chunks(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
    adaptive: bool = False,
    expected_chunks: int | None = None,
    min_chunks_ratio: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Detect non-silent (speech) portions in an audio file.

    Supports adaptive retry mode: when ``adaptive=True`` and the number of
    detected chunks is fewer than ``min_chunks_ratio * expected_chunks``,
    the function automatically retries with progressively relaxed thresholds
    (higher ``silence_thresh`` to detect more silence, and shorter
    ``min_silence_len``) until enough chunks are found or all retry levels
    are exhausted.  The best result (most chunks) across all attempts is
    returned.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)
        adaptive: Enable adaptive retry when too few chunks are detected.
            Existing callers are unaffected because this defaults to ``False``.
        expected_chunks: Expected number of non-silent chunks (e.g. ayah count).
            Required when ``adaptive=True``; ignored otherwise.
        min_chunks_ratio: Fraction of ``expected_chunks`` that must be found
            before the adaptive retry stops (default 0.5 → at least 50 %).

    Returns:
        List of (start_ms, end_ms) tuples for non-silent portions
    """
    if adaptive and expected_chunks is not None and expected_chunks > 0 and min_chunks_ratio <= 0:
        raise ValueError(
            "detect_non_silent_chunks: min_chunks_ratio must be > 0 when "
            f"adaptive=True (got {min_chunks_ratio}). Provide a positive "
            "ratio relative to expected_chunks."
        )

    chunks = _detect_non_silent_chunks_raw(audio_path, min_silence_len, silence_thresh, use_fast)

    if not adaptive or expected_chunks is None or expected_chunks <= 0:
        return chunks

    # Adaptive retry: relax thresholds progressively until we have enough chunks
    # Each level relaxes the dB threshold (allow quieter sounds) and shortens
    # the minimum silence length (detect shorter pauses).
    retry_levels = [
        # (silence_thresh_delta, min_silence_len_factor)
        (+5, 0.75),  # level 1: slightly more sensitive
        (+10, 0.5),  # level 2: moderately more sensitive
        (+15, 0.35),  # level 3: quite sensitive
        (+20, 0.25),  # level 4: very sensitive (last resort)
    ]

    min_required = max(1, int(min_chunks_ratio * expected_chunks))

    best_chunks = chunks

    for thresh_delta, len_factor in retry_levels:
        if len(best_chunks) >= min_required:
            break

        relaxed_thresh = min(
            silence_thresh + thresh_delta, -10
        )  # e.g. -30 + 5 = -25, capped at -10
        relaxed_len = max(50, int(min_silence_len * len_factor))  # never below 50 ms

        retried = _detect_non_silent_chunks_raw(audio_path, relaxed_len, relaxed_thresh, use_fast)
        if len(retried) > len(best_chunks):
            best_chunks = retried

    return best_chunks


def _detect_non_silent_chunks_raw(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
) -> list[tuple[int, int]]:
    """
    Internal helper: detect non-silent chunks with fixed thresholds.

    Delegates to the fast (librosa) or accurate (pydub) backend.
    """
    if use_fast:
        try:
            return _detect_non_silent_fast(audio_path, min_silence_len, silence_thresh)
        except:  # noqa: E722 - Bare except to catch numpy/scipy C-extension errors
            pass  # Fallback to pydub

    return _detect_non_silent_pydub(audio_path, min_silence_len, silence_thresh)


def _detect_non_silent_pydub(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """Pydub-based non-silent detection (slower but reliable)."""
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_wav(str(audio_path))
    chunks = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return [(c[0], c[1]) for c in chunks]


def _detect_non_silent_fast(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """
    Fast non-silent chunk detection using librosa + numpy.

    Returns the inverse of silence detection.
    """
    # Import inside try block to catch numpy/scipy version conflicts
    try:
        import librosa
        import numpy as np
    except (ImportError, AttributeError) as e:
        raise ImportError(f"librosa not available: {e}") from e

    # Load audio once
    y, sr = librosa.load(str(audio_path), sr=None)
    duration_ms = int(len(y) / sr * 1000)

    # Convert dB threshold to amplitude ratio
    amplitude_thresh = 10 ** (silence_thresh / 20)

    # Calculate frame-based RMS energy
    frame_length = int(sr * 0.01)  # 10ms frames
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_normalized = rms / rms_max

    # Find frames above threshold (non-silent)
    is_speech = rms_normalized >= amplitude_thresh

    frame_times_ms = (
        librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
    )

    # Find contiguous speech regions
    chunks = []
    in_speech = False
    speech_start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech = True
            speech_start = frame_times_ms[i]
        elif not speech and in_speech:
            in_speech = False
            speech_end = frame_times_ms[i]
            chunks.append((int(speech_start), int(speech_end)))

    # Handle case where audio ends in speech
    if in_speech:
        chunks.append((int(speech_start), duration_ms))

    # Merge chunks that are separated by less than min_silence_len
    if len(chunks) > 1:
        merged = [chunks[0]]
        for start, end in chunks[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_silence_len:
                # Merge with previous chunk
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        chunks = merged

    return chunks if chunks else [(0, duration_ms)]


def compute_energy_envelope(
    audio_path: str | Path,
    window_ms: int = 50,
) -> list[tuple[float, float]]:
    """
    Compute RMS energy envelope of an audio file.

    Returns a list of (time_seconds, rms_energy) tuples at the given
    window resolution. Useful for detecting local energy minima as
    potential ayah boundary points.

    Args:
        audio_path: Path to audio file
        window_ms: Window size in milliseconds (default 50ms)

    Returns:
        List of (time_sec, rms) tuples
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None)

    frame_length = int(sr * window_ms / 1000)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return [(float(t), float(r)) for t, r in zip(times, rms, strict=True)]


def find_energy_minima(
    envelope: list[tuple[float, float]],
    search_start: float,
    search_end: float,
    top_n: int = 3,
) -> list[float]:
    """
    Find local energy minima within a time range.

    Used to find the best boundary point near an estimated ayah boundary.

    Args:
        envelope: Energy envelope from compute_energy_envelope()
        search_start: Start of search window (seconds)
        search_end: End of search window (seconds)
        top_n: Number of top minima to return

    Returns:
        List of times (seconds) at local energy minima, sorted by energy (lowest first)
    """
    candidates = [(t, e) for t, e in envelope if search_start <= t <= search_end]

    if not candidates:
        return []

    # Sort by energy ascending (lowest energy = best boundary)
    candidates.sort(key=lambda x: x[1])
    return [t for t, _ in candidates[:top_n]]


def load_audio_waveform(
    audio_path: str | Path,
    sample_rate: int = 16000,
) -> tuple:
    """
    Load audio waveform for processing.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Tuple of (waveform_array, sample_rate)
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sample_rate)
    return y, sr


def extract_segment_audio(
    waveform: Any,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> Any:
    """
    Extract a segment from a waveform.

    Args:
        waveform: Audio waveform array
        sample_rate: Sample rate of the waveform
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds

    Returns:
        Segment waveform array
    """
    start_sample = int((start_ms / 1000) * sample_rate)
    end_sample = int((end_ms / 1000) * sample_rate)
    return waveform[start_sample:end_sample]
