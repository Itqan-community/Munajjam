from enum import Enum
from typing import Literal

from munajjam.transcription.base import BaseTranscriber


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"
    DEEPGRAM = "deepgram"


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float16",
    ) -> BaseTranscriber:
        if backend == WhisperBackend.FASTERWHISPER:
            from munajjam.transcription.whisper import WhisperTranscriber

            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            from munajjam.transcription.whisper import WhisperTranscriber

            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="transformers"
            )
        elif backend == WhisperBackend.WHISPERX:
            from munajjam.transcription.whisperx import Whisperx

            return Whisperx(
                model_name=model_name, device=device, compute_type=compute_type
            )
        elif backend == WhisperBackend.DEEPGRAM:
            from munajjam.transcription.deepgram_transcriber import (
                DeepgramTranscriber,
            )

            return DeepgramTranscriber()

        raise ValueError(f"Unsupported backend: {backend}")
