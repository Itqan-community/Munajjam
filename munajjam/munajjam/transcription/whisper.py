"""
Whisper-based transcription implementation.


Uses Tarteel AI's Whisper models fine-tuned for Quran recitation.
"""

from typing import Any, Literal

from munajjam.config import MunajjamSettings, get_settings
from munajjam.transcription.base import BaseTranscriber


class WhisperTranscriber(BaseTranscriber):
    """
    Whisper-based transcriber for Quran audio.


    Uses Tarteel AI's Whisper models fine-tuned for Quran recitation.
    Supports both standard Transformers and Faster Whisper backends.


    Example:
        transcriber = WhisperTranscriber()
        transcriber.load()


        segments = transcriber.transcribe("surah_1.wav")


        transcriber.unload()


    Or using context manager:
        with WhisperTranscriber() as transcriber:
            segments = transcriber.transcribe("surah_1.wav")
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] | None = None,
        model_type: Literal["transformers", "faster-whisper"] | None = None,
        settings: MunajjamSettings | None = None,
    ):
        """
        Initialize the Whisper transcriber.


        Args:
            model_id: HuggingFace model ID (overrides settings)
            device: Device for inference (overrides settings)
            model_type: Model backend type (overrides settings)
            settings: Settings instance to use
        """
        self._settings = settings or get_settings()

        self._model_id = model_id or self._settings.model_id
        self._device = device or self._settings.device
        self._model_type = model_type or self._settings.model_type

        # Model state
        self._model: Any = None
        self._processor: Any = None
