"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import detect_non_silent_chunks, detect_silences
from munajjam.transcription.whisper import WhisperTranscriber

__all__ = [
    "BaseTranscriber",
    "WhisperTranscriber",
    "detect_silences",
    "detect_non_silent_chunks",
]
