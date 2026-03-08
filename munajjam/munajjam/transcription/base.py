"""
Abstract base class for audio transcription.

This module defines the interface that all transcriber implementations must follow.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from munajjam.models import Segment


class BaseTranscriber(ABC):
    """
    Abstract interface for audio transcription.

    All transcriber implementations (Whisper, custom models, etc.)
    must implement this interface.

    Example:
        class MyTranscriber(BaseTranscriber):
            def transcribe(self, audio_path: str) -> list[Segment]:
                # Custom implementation
                ...
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path, batch_size: int = 16) -> list[Segment]:
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
        pass




