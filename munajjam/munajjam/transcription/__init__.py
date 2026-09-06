"""
Transcription module for Munajjam library.

Provides abstract interface and implementations for audio transcription.
"""

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.ctc_segmentation import (
    FastConformerCTCTranscriber,
    SentencePieceTokenizer,
    align_words_to_log_probs,
    frames_to_time,
    normalize_quran_text,
)
from munajjam.transcription.fastconformer import FastConformerInference
from munajjam.transcription.silence import detect_non_silent_chunks, detect_silences
from munajjam.transcription.whisper import WhisperTranscriber

__all__ = [
    "BaseTranscriber",
    "FastConformerCTCTranscriber",
    "FastConformerInference",
    "SentencePieceTokenizer",
    "WhisperTranscriber",
    "align_words_to_log_probs",
    "detect_silences",
    "detect_non_silent_chunks",
    "frames_to_time",
    "normalize_quran_text",
]
