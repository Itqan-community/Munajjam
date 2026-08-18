"""
Experimental Google Cloud Speech-to-Text Chirp 3 transcriber.

This backend is for research/evaluation only. It calls the Speech-to-Text V2
API (model ``chirp_3``, language ``ar-SA``) and maps word-level timestamps
onto Munajjam ``Segment`` objects so results can be compared with WhisperX.
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from munajjam.config import MunajjamSettings, get_settings
from munajjam.data import load_surah_ayahs
from munajjam.exceptions import AudioFileError, ConfigurationError, TranscriptionError
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)

CHIRP3_MODEL = "chirp_3"
CHIRP3_LANGUAGE = "ar-SA"
# Speech.Recognize is intended for audio shorter than ~1 minute.
MAX_CHUNK_SECONDS = 50.0
_MIN_MATCH_RATIO = 0.6


def _duration_to_seconds(duration: Any) -> float:
    """Convert a protobuf Duration (or duck-typed mock) to seconds."""
    if duration is None:
        return 0.0
    seconds = float(getattr(duration, "seconds", 0) or 0)
    nanos = float(getattr(duration, "nanos", 0) or 0)
    return seconds + nanos / 1e9


def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    return text.strip()


def _map_words_to_ayah_segments(
    *,
    surah_id: int,
    ref_ayahs: list[Any],
    extracted_words: list[dict[str, Any]],
) -> list[Segment]:
    """Align ASR words onto canonical ayah text via fuzzy DP matching."""
    ref_words: list[str] = []
    for ayah in ref_ayahs:
        ref_words.extend(ayah.text.split())

    if not ref_words:
        return []

    n = len(ref_words)
    m = len(extracted_words)
    dp = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        rw = _normalize_arabic(ref_words[i - 1])
        for j in range(1, m + 1):
            ew = _normalize_arabic(str(extracted_words[j - 1]["word"]))
            match_score = fuzz.ratio(rw, ew) / 100.0
            if match_score < _MIN_MATCH_RATIO:
                match_score = -1.0
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + match_score)

    mapped: list[dict[str, Any] | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        rw = _normalize_arabic(ref_words[i - 1])
        ew = _normalize_arabic(str(extracted_words[j - 1]["word"]))
        match_score = fuzz.ratio(rw, ew) / 100.0
        if match_score >= _MIN_MATCH_RATIO and math.isclose(
            dp[i][j], dp[i - 1][j - 1] + match_score, rel_tol=1e-9, abs_tol=1e-9
        ):
            mapped[i - 1] = extracted_words[j - 1]
            i -= 1
            j -= 1
        elif math.isclose(dp[i][j], dp[i - 1][j], rel_tol=1e-9, abs_tol=1e-9):
            i -= 1
        else:
            j -= 1

    alignments: list[dict[str, Any]] = []
    for k in range(n):
        hit = mapped[k]
        if hit:
            alignments.append(
                {
                    "word": ref_words[k],
                    "start": float(hit["start"]),
                    "end": float(hit["end"]),
                    "confidence": float(hit.get("confidence") or 0.0),
                }
            )
        else:
            prev_end = alignments[-1]["end"] if alignments else 0.0
            alignments.append(
                {
                    "word": ref_words[k],
                    "start": prev_end,
                    "end": prev_end + 0.1,
                    "confidence": 0.0,
                }
            )

    for k in range(len(alignments)):
        if k > 0 and alignments[k]["start"] < alignments[k - 1]["end"]:
            alignments[k]["start"] = alignments[k - 1]["end"]
        if alignments[k]["end"] <= alignments[k]["start"]:
            alignments[k]["end"] = round(alignments[k]["start"] + 0.1, 3)

    segments: list[Segment] = []
    word_idx = 0
    for ayah in ref_ayahs:
        ayah_word_count = len(ayah.text.split())
        ayah_alignments = alignments[word_idx : word_idx + ayah_word_count]
        word_idx += ayah_word_count
        if not ayah_alignments:
            continue

        words = [
            WordTimestamp(
                word=item["word"],
                start=item["start"],
                end=item["end"],
                probability=min(max(item["confidence"], 0.0), 1.0),
            )
            for item in ayah_alignments
        ]
        avg_conf = sum(w.probability for w in words) / len(words)
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


class ChirpTranscriber(BaseTranscriber):
    """
    Experimental Chirp 3 cloud transcriber.

    Credentials are read from ``MunajjamSettings`` (``MUNAJJAM_GCP_*`` /
    ``GCP_*`` env vars). The Google client library is imported lazily so the
    rest of Munajjam does not require ``google-cloud-speech``.
    """

    def __init__(
        self,
        project_id: str | None = None,
        credentials_path: str | Path | None = None,
        location: str | None = None,
        settings: MunajjamSettings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._project_id = project_id or self._settings.gcp_project_id
        creds = (
            credentials_path
            if credentials_path is not None
            else self._settings.gcp_credentials_path
        )
        self._credentials_path = Path(creds) if creds else None
        self._location = location or self._settings.gcp_location
        self._client = client

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        del batch_size  # Cloud Recognize does not use local batching.
        path = Path(audio_path)
        if not path.is_file():
            raise AudioFileError(str(path), reason="file does not exist")

        self._ensure_configured()

        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []

        try:
            extracted_words = self._transcribe_words(path)
        except (ConfigurationError, AudioFileError, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Chirp 3 transcription failed: {exc}",
                audio_path=str(path),
            ) from exc

        if not extracted_words:
            raise TranscriptionError(
                "Chirp 3 returned no word-level timestamps",
                audio_path=str(path),
            )

        return _map_words_to_ayah_segments(
            surah_id=surah_id,
            ref_ayahs=ayahs,
            extracted_words=extracted_words,
        )

    def _ensure_configured(self) -> None:
        if not self._project_id:
            raise ConfigurationError(
                "Chirp 3 requires a Google Cloud project id. "
                "Set MUNAJJAM_GCP_PROJECT_ID or GCP_PROJECT_ID.",
                setting_name="gcp_project_id",
            )
        if self._credentials_path and not self._credentials_path.is_file():
            raise ConfigurationError(
                f"GCP credentials file not found: {self._credentials_path}",
                setting_name="gcp_credentials_path",
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud.speech_v2 import SpeechClient
        except ImportError as exc:
            raise ConfigurationError(
                "google-cloud-speech is required for the Chirp 3 backend. "
                'Install it with: pip install "munajjam[gcp]" '
                'or pip install "google-cloud-speech>=2.27.0".',
                setting_name="gcp",
            ) from exc

        endpoint = f"{self._location}-speech.googleapis.com"
        client_options = ClientOptions(api_endpoint=endpoint)

        if self._credentials_path:
            try:
                from google.oauth2 import service_account
            except ImportError as exc:
                raise ConfigurationError(
                    "google-auth is required to load a service-account JSON file.",
                    setting_name="gcp_credentials_path",
                ) from exc
            credentials = service_account.Credentials.from_service_account_file(
                str(self._credentials_path)
            )
            self._client = SpeechClient(credentials=credentials, client_options=client_options)
        else:
            # Fall back to Application Default Credentials / GOOGLE_APPLICATION_CREDENTIALS.
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.debug("Using GOOGLE_APPLICATION_CREDENTIALS for Chirp 3")
            self._client = SpeechClient(client_options=client_options)

        return self._client

    def _transcribe_words(self, audio_path: Path) -> list[dict[str, Any]]:
        chunks = self._split_audio(audio_path)
        words: list[dict[str, Any]] = []
        for offset_seconds, chunk_bytes in chunks:
            words.extend(self._recognize_chunk(chunk_bytes, time_offset=offset_seconds))
        return words

    def _split_audio(self, audio_path: Path) -> list[tuple[float, bytes]]:
        """Split long recitations so each Recognize call stays under ~1 minute."""
        try:
            from pydub import AudioSegment
        except ImportError:
            return [(0.0, audio_path.read_bytes())]

        audio = AudioSegment.from_file(str(audio_path))
        duration_s = len(audio) / 1000.0
        if duration_s <= MAX_CHUNK_SECONDS:
            return [(0.0, audio_path.read_bytes())]

        logger.info(
            "Audio is %.1fs; splitting into %.0fs chunks for Chirp 3 Recognize",
            duration_s,
            MAX_CHUNK_SECONDS,
        )
        chunk_ms = int(MAX_CHUNK_SECONDS * 1000)
        chunks: list[tuple[float, bytes]] = []
        for start_ms in range(0, len(audio), chunk_ms):
            piece = audio[start_ms : start_ms + chunk_ms]
            buffer = io.BytesIO()
            piece.export(buffer, format="wav")
            chunks.append((start_ms / 1000.0, buffer.getvalue()))
        return chunks

    def _recognize_chunk(self, audio_bytes: bytes, *, time_offset: float) -> list[dict[str, Any]]:
        try:
            from google.cloud.speech_v2.types import cloud_speech
        except ImportError as exc:
            raise ConfigurationError(
                "google-cloud-speech is required for the Chirp 3 backend.",
                setting_name="gcp",
            ) from exc

        client = self._get_client()
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[CHIRP3_LANGUAGE],
            model=CHIRP3_MODEL,
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
            ),
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=(f"projects/{self._project_id}/locations/{self._location}/recognizers/_"),
            config=config,
            content=audio_bytes,
        )

        try:
            response = client.recognize(request=request)
        except Exception as exc:
            raise TranscriptionError(f"Google Cloud Speech-to-Text request failed: {exc}") from exc

        return self._extract_words(response, time_offset=time_offset)

    @staticmethod
    def _extract_words(response: Any, *, time_offset: float) -> list[dict[str, Any]]:
        words: list[dict[str, Any]] = []
        for result in getattr(response, "results", []) or []:
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue
            for word_info in getattr(alternatives[0], "words", []) or []:
                start = time_offset + _duration_to_seconds(getattr(word_info, "start_offset", None))
                end = time_offset + _duration_to_seconds(getattr(word_info, "end_offset", None))
                if end < start:
                    end = start
                confidence = getattr(word_info, "confidence", 0.0) or 0.0
                words.append(
                    {
                        "word": getattr(word_info, "word", "") or "",
                        "start": start,
                        "end": end,
                        "confidence": float(confidence),
                    }
                )
        return words
