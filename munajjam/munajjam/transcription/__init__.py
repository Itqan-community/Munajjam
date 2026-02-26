"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    AdaptiveSilenceConfig,
    detect_non_silent_chunks,
    detect_non_silent_chunks_adaptive,
    detect_silences,
)
from munajjam.transcription.whisper import WhisperTranscriber

__all__ = [
    "AdaptiveSilenceConfig",
    "BaseTranscriber",
    "WhisperTranscriber",
    "detect_non_silent_chunks",
    "detect_non_silent_chunks_adaptive",
    "detect_silences",
]

