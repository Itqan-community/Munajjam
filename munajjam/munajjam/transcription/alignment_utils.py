"""
Shared word-to-ayah alignment utilities.

This module holds the fuzzy-matching / boundary-refinement logic that maps a flat
list of timestamped words (from *any* transcription engine) onto the reference
Quran text for a surah, producing the project's standard `Segment` objects.

It was extracted from `Whisperx.transcribe` so that alternative backends (e.g.
`ReplicateWhisperXTranscriber`) can reuse the exact same alignment behavior
instead of re-implementing it, and so the two stay in sync if the algorithm is
tuned in the future.
"""

import re
from pathlib import Path
from typing import Any

import soundfile as sf
from rapidfuzz import fuzz

from munajjam.models import Segment, SegmentType, WordTimestamp

# Loose alias: an ayah-like object exposing `.ayah_number: int` and `.text: str`,
# as returned by `munajjam.data.load_surah_ayahs`.
AyahLike = Any


def normalize_arabic(text: str) -> str:
    """Strip diacritics/non-Arabic chars and normalize alef variants for fuzzy matching."""
    text = re.sub(r"[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    return text.strip()


def align_words_to_segments(
    *,
    ref_words: list[str],
    extracted_words: list[dict[str, Any]],
    ayahs: list[AyahLike],
    surah_id: int,
    audio_path: str | Path,
) -> list[Segment]:
    """
    Map timestamped words from a transcription engine onto reference ayah text.

    Args:
        ref_words: Flattened reference words for the surah (in ayah order).
        extracted_words: Timestamped words from the engine. Each dict must have
            "word", "start", "end", and "confidence" keys.
        ayahs: The surah's Ayah objects (same order used to build ref_words).
        surah_id: Surah number.
        audio_path: Path to the source audio file, used to determine total
            duration for the final segment's end time.

    Returns:
        List of Segment objects, one per ayah, with per-word timestamps.
    """
    n = len(ref_words)
    m = len(extracted_words)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        rw = normalize_arabic(ref_words[i - 1])
        for j in range(1, m + 1):
            ew = normalize_arabic(extracted_words[j - 1]["word"])
            match_score = fuzz.ratio(rw, ew) / 100.0
            if match_score < 0.6:
                match_score = -1.0
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + match_score)

    mapped_alignments: list[dict[str, Any] | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        rw = normalize_arabic(ref_words[i - 1])
        ew = normalize_arabic(extracted_words[j - 1]["word"])
        match_score = fuzz.ratio(rw, ew) / 100.0

        if match_score >= 0.6 and dp[i][j] == dp[i - 1][j - 1] + match_score:
            mapped_alignments[i - 1] = extracted_words[j - 1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1

    w_alignments: list[dict[str, Any]] = []
    for k in range(n):
        if mapped_alignments[k]:
            w_alignments.append(
                {
                    "word": ref_words[k],
                    "start": mapped_alignments[k]["start"],
                    "end": mapped_alignments[k]["end"],
                    "confidence": mapped_alignments[k]["confidence"],
                }
            )
        else:
            prev_end = w_alignments[-1]["end"] if w_alignments else 0
            w_alignments.append(
                {
                    "word": ref_words[k],
                    "start": prev_end,
                    "end": prev_end + 0.1,
                    "confidence": 0.0,
                }
            )

    final_alignments = w_alignments

    try:
        total_duration = sf.info(str(audio_path)).duration
    except Exception:
        total_duration = final_alignments[-1]["end"] + 2.0 if final_alignments else 2.0

    ayah_boundary_indices = set()
    w_idx = 0
    for ayah in ayahs:
        w_idx += len(ayah.text.split())
        ayah_boundary_indices.add(w_idx - 1)

    for k in range(len(final_alignments)):
        if k > 0:
            if final_alignments[k]["start"] < final_alignments[k - 1]["end"]:
                final_alignments[k]["start"] = final_alignments[k - 1]["end"]

        if k < len(final_alignments) - 1:
            next_start = final_alignments[k + 1]["start"]
            current_end = final_alignments[k]["end"]
            gap = next_start - current_end

            if gap > 0:
                if k in ayah_boundary_indices:
                    if gap <= 0.3:
                        start_buffer = min(gap, 0.1)
                        final_alignments[k + 1]["start"] = round(next_start - start_buffer, 3)
                        final_alignments[k]["end"] = round(next_start - start_buffer, 3)
                    elif gap >= 0.4:
                        final_alignments[k + 1]["start"] = round(next_start - 0.2, 3)
                        final_alignments[k]["end"] = round(next_start - 0.2, 3)
                    else:
                        mid = gap / 2.0
                        final_alignments[k + 1]["start"] = round(next_start - mid, 3)
                        final_alignments[k]["end"] = round(next_start - mid, 3)
                else:
                    if gap > 0.1:
                        final_alignments[k]["end"] = round(next_start - 0.1, 3)
        else:
            final_alignments[k]["end"] = round(total_duration, 3)

        if final_alignments[k]["end"] <= final_alignments[k]["start"]:
            final_alignments[k]["end"] = round(final_alignments[k]["start"] + 0.1, 3)

    word_idx = 0
    segments = []

    for ayah in ayahs:
        ayah_words_count = len(ayah.text.split())
        ayah_alignments = final_alignments[word_idx : word_idx + ayah_words_count]
        word_idx += ayah_words_count

        if not ayah_alignments:
            continue

        words = []
        avg_conf = 0.0
        for wa in ayah_alignments:
            words.append(
                WordTimestamp(
                    word=wa["word"],
                    start=wa["start"],
                    end=wa["end"],
                    probability=wa["confidence"],
                )
            )
            avg_conf += wa["confidence"]

        if words:
            avg_conf /= len(words)
            segments.append(
                Segment(
                    id=ayah.ayah_number,
                    surah_id=surah_id,
                    start=words[0].start,
                    end=words[-1].end,
                    text=ayah.text,
                    type=SegmentType.AYAH,
                    words=words,
                    confidence=avg_conf,
                )
            )

    return segments