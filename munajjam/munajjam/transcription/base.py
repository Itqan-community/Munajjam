"""
Abstract base class for audio transcription.

This module defines the interface that all transcriber implementations must follow.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from munajjam.models import Segment


class BaseTranscriber(ABC):
    """
    Abstract interface for audio transcription.

    All transcriber implementations (Whisper, custom models, etc.)
    must implement this interface.

        with WhisperTranscriber() as transcriber:
            segments = transcriber.transcribe("surah_1.wav", surah_id=1)
    """

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        ...
        """
        Transcribe an audio file to segments.

        Args:
            audio_path: Path to the audio file (WAV recommended)
            batch_size: Batch size for transcribing


        Returns:
            List of transcribed Segment objects

        Raises:
            TranscriptionError: If transcription fails
            AudioFileError: If audio file cannot be read
        """
