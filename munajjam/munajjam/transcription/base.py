"""
Abstract base class for audio transcription.

This module defines the interface that all transcriber implementations must follow.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Iterator

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
    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        """
        Transcribe an audio file to segments.

        Args:
            audio_path: Path to the audio file (WAV recommended)

        Returns:
            List of transcribed Segment objects

        Raises:
            TranscriptionError: If transcription fails
            AudioFileError: If audio file cannot be read
        """
        pass

    @abstractmethod
    async def transcribe_async(self, audio_path: str | Path) -> list[Segment]:
        """
        Asynchronously transcribe an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of transcribed Segment objects
        """
        pass

    def transcribe_stream(self, audio_path: str | Path) -> Iterator[Segment]:
        """
        Transcribe audio and yield segments as they are processed.

        Default implementation transcribes all at once and yields.
        Override for true streaming support.

        Args:
            audio_path: Path to the audio file

        Yields:
            Segment objects as they are transcribed
        """
        segments = self.transcribe(audio_path)
        yield from segments

    async def transcribe_stream_async(
        self, audio_path: str | Path
    ) -> AsyncIterator[Segment]:
        """
        Asynchronously transcribe and yield segments.

        Default implementation transcribes all at once and yields.
        Override for true streaming support.

        Args:
            audio_path: Path to the audio file

        Yields:
            Segment objects as they are transcribed
        """
        segments = await self.transcribe_async(audio_path)
        for segment in segments:
            yield segment

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        Call this before transcription to pre-load the model.
        Useful for avoiding cold start latency.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory.

        Call this to free up memory when done transcribing.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        pass

    def __enter__(self) -> "BaseTranscriber":
        """Context manager entry - loads the model."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - unloads the model."""
        self.unload()

