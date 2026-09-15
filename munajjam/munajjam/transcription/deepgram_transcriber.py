"""
Experimental cloud-based transcription via Deepgram API.

This module provides a lightweight, GPU-free transcriber that sends audio
to the Deepgram cloud API and maps the returned word-level timestamps to
Quran ayah boundaries using fuzzy DP alignment.

No PyTorch, no whisperx, no CUDA — only httpx for HTTP calls.

Requires:
    pip install httpx   (or:  pip install munajjam[deepgram])

Environment:
    MUNAJJAM_DEEPGRAM_API_KEY=<your-key>
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from munajjam.config import get_settings
from munajjam.data import load_surah_ayahs
from munajjam.exceptions import TranscriptionError
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber

# Deepgram API endpoint for pre-recorded audio
_DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy matching (strip diacritics, normalize alef)."""
    text = re.sub(r"[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    return text.strip()


class DeepgramTranscriber(BaseTranscriber):
    """
    Experimental cloud transcriber using the Deepgram REST API.

    Sends audio to Deepgram, retrieves word-level timestamps, and aligns
    them to Quran ayah boundaries using fuzzy DP matching.

    This transcriber does NOT require PyTorch, CUDA, or any local GPU.
    """

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.deepgram_api_key
        if api_key is None:
            raise TranscriptionError(
                "Deepgram API key is not configured. "
                "Set MUNAJJAM_DEEPGRAM_API_KEY environment variable."
            )
        self._api_key = api_key.get_secret_value()

    def _call_deepgram(self, audio_path: str | Path) -> dict[str, Any]:
        """Send audio to Deepgram API and return the JSON response."""
        try:
            import httpx
        except ImportError:
            raise TranscriptionError(
                "httpx is required for Deepgram mode. "
                "Install with: pip install httpx  (or: pip install munajjam[deepgram])"
            ) from None

        audio_bytes = Path(audio_path).read_bytes()

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/mpeg",
        }
        params = {
            "model": "nova-3",
            "language": "ar",
            "utterances": "true",
            "punctuate": "true",
        }

        response = httpx.post(
            _DEEPGRAM_API_URL,
            headers=headers,
            params=params,
            content=audio_bytes,
            timeout=300.0,
        )

        if response.status_code != 200:
            raise TranscriptionError(
                f"Deepgram API error (HTTP {response.status_code}): {response.text}"
            )

        return response.json()

    def _extract_words(
        self, deepgram_response: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract word-level timestamps from Deepgram JSON response."""
        try:
            channels = deepgram_response["results"]["channels"]
            alternatives = channels[0]["alternatives"]
            words = alternatives[0].get("words", [])
        except (KeyError, IndexError) as e:
            raise TranscriptionError(
                f"Unexpected Deepgram response structure: {e}"
            ) from e

        return [
            {
                "word": str(w["word"]),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "confidence": float(w.get("confidence", 0.9)),
            }
            for w in words
            if "word" in w and "start" in w and "end" in w
        ]

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        """
        Transcribe audio via Deepgram API and align to Quran ayahs.

        Args:
            audio_path: Path to the audio file.
            surah_id: Surah number (1-114).
            batch_size: Ignored (kept for interface compatibility).

        Returns:
            List of Segment objects aligned to ayah boundaries.
        """
        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []

        # Build reference word list from canonical Quran text
        ref_words: list[str] = []
        for ayah in ayahs:
            for w in ayah.text.split():
                ref_words.append(w)

        # Call Deepgram API
        print("[Deepgram] Sending audio to Deepgram API (nova-3, ar)...")
        deepgram_response = self._call_deepgram(audio_path)
        extracted_words = self._extract_words(deepgram_response)

        if not extracted_words:
            raise TranscriptionError(
                "Deepgram returned no words. Check audio quality and language."
            )

        print(
            f"[Deepgram] Received {len(extracted_words)} words, "
            f"aligning to {len(ayahs)} ayahs..."
        )

        # --- Word-level DP alignment (same approach as whisperx.py) ---
        n = len(ref_words)
        m = len(extracted_words)
        dp = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            rw = _normalize_arabic(ref_words[i - 1])
            for j in range(1, m + 1):
                ew = _normalize_arabic(str(extracted_words[j - 1]["word"]))
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.6:
                    match_score = -1.0
                dp[i][j] = max(
                    dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + match_score
                )

        # Backtrack to find optimal alignment
        mapped_alignments: list[dict[str, Any] | None] = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = _normalize_arabic(ref_words[i - 1])
            ew = _normalize_arabic(str(extracted_words[j - 1]["word"]))
            match_score = fuzz.ratio(rw, ew) / 100.0

            if match_score >= 0.6 and dp[i][j] == dp[i - 1][j - 1] + match_score:
                mapped_alignments[i - 1] = extracted_words[j - 1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j]:
                i -= 1
            else:
                j -= 1

        # Build final word alignments (fill gaps with estimated times)
        w_alignments: list[dict[str, Any]] = []
        for k in range(n):
            align_item = mapped_alignments[k]
            if align_item is not None:
                w_alignments.append(
                    {
                        "word": ref_words[k],
                        "start": align_item["start"],
                        "end": align_item["end"],
                        "confidence": align_item["confidence"],
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

        gc.collect()

        # --- Build Segment objects per ayah ---
        word_idx = 0
        segments: list[Segment] = []

        for ayah in ayahs:
            ayah_words_count = len(ayah.text.split())
            ayah_alignments = w_alignments[word_idx : word_idx + ayah_words_count]
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

        print(f"[Deepgram] Alignment complete: {len(segments)} ayah segments")
        return segments
