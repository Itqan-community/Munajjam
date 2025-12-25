"""
HuggingFace Inference Endpoint transcription implementation.

Uses a deployed Whisper model on HuggingFace Inference Endpoints for Quran recitation.
"""

import asyncio
import io
import re
import time
from pathlib import Path

from munajjam.config import MunajjamSettings, get_settings
from munajjam.exceptions import TranscriptionError, AudioFileError, ConfigurationError
from munajjam.models import Segment, SegmentType
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    detect_non_silent_chunks,
    load_audio_waveform,
    extract_segment_audio,
)


# Regex patterns for special segment detection (reused from whisper.py)
ISTI3AZA_PATTERN = re.compile(r"[اأإآٱو]?عوذ\s*بالله\s*من\s*الشيطان\s*الرجيم")
BASMALA_PATTERN = re.compile(r"(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم")


def _normalize_for_detection(text: str) -> str:
    """Normalize text for pattern detection."""
    text = re.sub(r"[أإآاٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_segment_type(text: str) -> tuple[SegmentType, int]:
    """
    Detect segment type from transcribed text.

    Returns:
        Tuple of (segment_type, segment_id)
        segment_id is 0 for special segments, positive otherwise
    """
    normalized = _normalize_for_detection(text)

    if ISTI3AZA_PATTERN.search(normalized):
        return SegmentType.ISTI3AZA, 0

    if BASMALA_PATTERN.search(normalized):
        return SegmentType.BASMALA, 0

    return SegmentType.AYAH, 1  # Will be renumbered later


class HFInferenceTranscriber(BaseTranscriber):
    """
    HuggingFace Inference Endpoint transcriber for Quran audio.

    Uses a deployed Whisper model on HuggingFace Inference Endpoints.
    This is a lightweight alternative that offloads computation to the cloud.

    Example:
        transcriber = HFInferenceTranscriber(
            endpoint_url="https://your-endpoint.aws.endpoints.huggingface.cloud",
            api_token="hf_your_token",
        )

        segments = transcriber.transcribe("surah_1.wav")

    Or using environment variables:
        export MUNAJJAM_HF_INFERENCE_ENDPOINT="https://your-endpoint.aws.endpoints.huggingface.cloud"
        export MUNAJJAM_HF_API_TOKEN="hf_your_token"

        transcriber = HFInferenceTranscriber()
        segments = transcriber.transcribe("surah_1.wav")
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_token: str | None = None,
        settings: MunajjamSettings | None = None,
    ):
        """
        Initialize the HF Inference transcriber.

        Args:
            endpoint_url: HuggingFace Inference Endpoint URL (overrides settings)
            api_token: HuggingFace API token (overrides settings)
            settings: Settings instance to use
        """
        self._settings = settings or get_settings()

        self._endpoint_url = endpoint_url or self._settings.hf_inference_endpoint
        self._api_token = api_token or self._settings.hf_api_token

        # Validate configuration
        if not self._endpoint_url:
            raise ConfigurationError(
                "HuggingFace Inference Endpoint URL is required. "
                "Set via endpoint_url parameter or MUNAJJAM_HF_INFERENCE_ENDPOINT env var."
            )
        if not self._api_token:
            raise ConfigurationError(
                "HuggingFace API token is required. "
                "Set via api_token parameter or MUNAJJAM_HF_API_TOKEN env var."
            )

        # HTTP client (lazy initialization)
        self._client = None

    @property
    def is_loaded(self) -> bool:
        """Whether the client is ready (always True for remote endpoint)."""
        return True

    @property
    def endpoint_url(self) -> str:
        """Current endpoint URL."""
        return self._endpoint_url

    def load(self) -> None:
        """
        Initialize the HTTP client.

        For remote endpoints, this is lightweight - just creates the client.
        """
        if self._client is not None:
            return  # Already initialized

        try:
            import httpx
        except ImportError:
            raise TranscriptionError(
                "httpx not installed. "
                "Install with: pip install munajjam[hf-inference]"
            )

        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self._api_token}",
            },
        )
        print(f"🌐 HF Inference Endpoint ready: {self._endpoint_url[:50]}...")

    def unload(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        """
        Transcribe an audio file to segments using HF Inference Endpoint.

        Args:
            audio_path: Path to the audio file (WAV)

        Returns:
            List of transcribed Segment objects
        """
        # Auto-load if not already loaded
        if self._client is None:
            self.load()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioFileError(str(audio_path), "File not found")

        # Extract surah ID from filename
        surah_id = int(audio_path.stem)

        # Detect non-silent chunks
        chunks = detect_non_silent_chunks(
            audio_path,
            min_silence_len=self._settings.min_silence_ms,
            silence_thresh=self._settings.silence_threshold_db,
        )

        # Load audio waveform
        waveform, sr = load_audio_waveform(
            audio_path,
            sample_rate=self._settings.sample_rate,
        )

        segments = []
        segment_idx = 1

        for start_ms, end_ms in chunks:
            # Extract segment audio
            segment_audio = extract_segment_audio(waveform, sr, start_ms, end_ms)

            if len(segment_audio) == 0:
                continue

            # Transcribe segment via API
            try:
                text = self._transcribe_segment(segment_audio, sr)
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to transcribe segment at {start_ms}ms-{end_ms}ms: {e}",
                    audio_path=str(audio_path),
                )

            # Detect segment type
            seg_type, seg_id = _detect_segment_type(text)
            if seg_type == SegmentType.AYAH:
                seg_id = segment_idx
                segment_idx += 1

            segment = Segment(
                id=seg_id,
                surah_id=surah_id,
                start=round(start_ms / 1000, 2),
                end=round(end_ms / 1000, 2),
                text=text.strip(),
                type=seg_type,
            )

            segments.append(segment)

        return segments

    def _transcribe_segment(
        self,
        segment_audio,
        sample_rate: int,
        max_retries: int = 10,
        initial_delay: float = 2.0,
    ) -> str:
        """
        Transcribe a single audio segment via the HF Inference Endpoint.

        Includes retry logic with exponential backoff for cold start handling.
        HF endpoints may return 503 when scaling from zero.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise TranscriptionError(
                "soundfile not installed. "
                "Install with: pip install soundfile"
            )

        # Convert numpy array to WAV bytes
        audio_bytes_data = io.BytesIO()
        sf.write(audio_bytes_data, segment_audio, sample_rate, format="WAV")
        audio_bytes_data.seek(0)
        audio_content = audio_bytes_data.read()

        last_error = None
        delay = initial_delay

        for attempt in range(max_retries):
            # Send to HF Inference Endpoint
            response = self._client.post(
                self._endpoint_url,
                content=audio_content,
                headers={"Content-Type": "audio/wav"},
            )

            if response.status_code == 200:
                # Success - parse response
                result = response.json()

                # Handle different response formats
                if isinstance(result, dict):
                    # Standard ASR response format
                    text = result.get("text", "")
                elif isinstance(result, list) and len(result) > 0:
                    # Some models return a list
                    text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
                else:
                    text = str(result)

                return text

            elif response.status_code == 503:
                # Service unavailable - endpoint is warming up (cold start)
                last_error = f"503 Service Unavailable - endpoint warming up"
                if attempt == 0:
                    print(f"⏳ Endpoint warming up (cold start), waiting...")
                else:
                    print(f"   Retry {attempt + 1}/{max_retries}, waiting {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)  # Exponential backoff, max 30s

            else:
                # Other error - don't retry
                raise TranscriptionError(
                    f"HF Inference API error: {response.status_code} - {response.text}"
                )

        # All retries exhausted
        raise TranscriptionError(
            f"HF Inference API failed after {max_retries} retries: {last_error}"
        )

    async def transcribe_async(self, audio_path: str | Path) -> list[Segment]:
        """
        Asynchronously transcribe an audio file.

        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio_path)

