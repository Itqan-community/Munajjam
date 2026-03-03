"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    detect_non_silent_chunks,
    detect_non_silent_chunks_adaptive,
    detect_silences,
    detect_silences_adaptive,
)
from munajjam.transcription.whisper import WhisperTranscriber

__all__ = [
    "BaseTranscriber",
    "WhisperTranscriber",
    "detect_silences",
    "detect_silences_adaptive",
    "detect_non_silent_chunks",
    "detect_non_silent_chunks_adaptive",
]
