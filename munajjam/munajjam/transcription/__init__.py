"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from typing import Any

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import detect_non_silent_chunks, detect_silences

__all__ = [
    "BaseTranscriber",
    "WhisperTranscriber",
    "DeepgramTranscriber",
    "detect_silences",
    "detect_non_silent_chunks",
]


def __getattr__(name: str) -> Any:
    """Lazy-load transcriber classes on attribute access."""
    if name == "WhisperTranscriber":
        from munajjam.transcription.whisper import WhisperTranscriber

        return WhisperTranscriber
    elif name == "DeepgramTranscriber":
        from munajjam.transcription.deepgram_transcriber import DeepgramTranscriber

        return DeepgramTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


