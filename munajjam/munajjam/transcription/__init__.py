"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.hf_inference import HFInferenceTranscriber
from munajjam.transcription.silence import detect_silences, detect_non_silent_chunks

__all__ = [
    "BaseTranscriber",
    "WhisperTranscriber",
    "HFInferenceTranscriber",
    "detect_silences",
    "detect_non_silent_chunks",
]

