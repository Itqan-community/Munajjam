"""
Silence detection utilities for audio processing.

Provides both pydub (accurate) and librosa (fast) implementations.
Use the fast implementation for long files (>5 minutes).
"""

from pathlib import Path


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
) -> list[tuple[int, int]]:
    """
    Detect non-silent (speech) portions in an audio file.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)

    Returns:
        List of (start_ms, end_ms) tuples for non-silent portions
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


def detect_non_silent_chunks_adaptive(
    audio_path: str | Path,
    expected_chunks: int,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    max_retries: int = 3,
    min_chunk_ratio: float = 0.5,
    silence_len_step: int = 50,
    silence_thresh_step: int = 5,
    use_fast: bool = True,
) -> tuple[list[tuple[int, int]], dict]:
    """
    Adaptive non-silent chunk detection with progressive threshold relaxation.

    When initial detection produces too few chunks relative to ``expected_chunks``,
    automatically retries with relaxed thresholds until enough chunks are found
    or ``max_retries`` is exhausted.

    Args:
        audio_path: Path to audio file.
        expected_chunks: Expected number of non-silent chunks (e.g. ayah count).
        min_silence_len: Initial minimum silence duration in ms.
        silence_thresh: Initial silence threshold in dB.
        max_retries: Maximum number of retry attempts with relaxed thresholds.
        min_chunk_ratio: Accept result when chunks >= expected_chunks * min_chunk_ratio.
        silence_len_step: Amount to decrease min_silence_len per retry (ms).
        silence_thresh_step: Amount to increase silence_thresh per retry (dB).
        use_fast: Use fast librosa-based detection.

    Returns:
        Tuple of (chunks_list, metadata_dict).
        metadata contains: retries_used, final_silence_thresh, final_min_silence_len,
        chunk_count.
    """
    current_thresh = silence_thresh
    current_min_len = min_silence_len
    target = max(1, int(expected_chunks * min_chunk_ratio))
    best_chunks: list[tuple[int, int]] = []
    best_count = 0

    for attempt in range(1 + max_retries):
        chunks = detect_non_silent_chunks(
            audio_path,
            min_silence_len=current_min_len,
            silence_thresh=current_thresh,
            use_fast=use_fast,
        )

        if len(chunks) > best_count:
            best_chunks = chunks
            best_count = len(chunks)

        if len(chunks) >= target:
            return chunks, {
                "retries_used": attempt,
                "final_silence_thresh": current_thresh,
                "final_min_silence_len": current_min_len,
                "chunk_count": len(chunks),
                "adapted": attempt > 0,
            }

        # Relax thresholds for next attempt
        current_thresh = min(current_thresh + silence_thresh_step, -5)
        current_min_len = max(current_min_len - silence_len_step, 100)

    return best_chunks, {
        "retries_used": max_retries,
        "final_silence_thresh": current_thresh - silence_thresh_step,
        "final_min_silence_len": current_min_len + silence_len_step,
        "chunk_count": best_count,
        "adapted": True,
    }


def detect_silences_adaptive(
    audio_path: str | Path,
    expected_silences: int,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    max_retries: int = 3,
    min_silence_ratio: float = 0.5,
    silence_len_step: int = 50,
    silence_thresh_step: int = 5,
    use_fast: bool = True,
) -> tuple[list[tuple[int, int]], dict]:
    """
    Adaptive silence detection with progressive threshold relaxation.

    When initial detection produces too few silences relative to
    ``expected_silences``, automatically retries with relaxed thresholds.

    Args:
        audio_path: Path to audio file.
        expected_silences: Expected number of silence regions.
        min_silence_len: Initial minimum silence duration in ms.
        silence_thresh: Initial silence threshold in dB.
        max_retries: Maximum number of retry attempts.
        min_silence_ratio: Accept when silences >= expected * ratio.
        silence_len_step: Amount to decrease min_silence_len per retry (ms).
        silence_thresh_step: Amount to increase silence_thresh per retry (dB).
        use_fast: Use fast librosa-based detection.

    Returns:
        Tuple of (silences_list, metadata_dict).
    """
    current_thresh = silence_thresh
    current_min_len = min_silence_len
    target = max(1, int(expected_silences * min_silence_ratio))
    best_silences: list[tuple[int, int]] = []
    best_count = 0

    for attempt in range(1 + max_retries):
        silences = detect_silences(
            audio_path,
            min_silence_len=current_min_len,
            silence_thresh=current_thresh,
            use_fast=use_fast,
        )

        if len(silences) > best_count:
            best_silences = silences
            best_count = len(silences)

        if len(silences) >= target:
            return silences, {
                "retries_used": attempt,
                "final_silence_thresh": current_thresh,
                "final_min_silence_len": current_min_len,
                "silence_count": len(silences),
                "adapted": attempt > 0,
            }

        current_thresh = min(current_thresh + silence_thresh_step, -5)
        current_min_len = max(current_min_len - silence_len_step, 100)

    return best_silences, {
        "retries_used": max_retries,
        "final_silence_thresh": current_thresh - silence_thresh_step,
        "final_min_silence_len": current_min_len + silence_len_step,
        "silence_count": best_count,
        "adapted": True,
    }


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

    return [(float(t), float(r)) for t, r in zip(times, rms, strict=False)]


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
    waveform,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
):
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
