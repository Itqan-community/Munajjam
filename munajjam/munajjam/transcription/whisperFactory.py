from enum import Enum
from typing import Literal

from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperx import Whisperx


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"
    CHIRP3 = "chirp3"


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float16",
    ) -> BaseTranscriber:
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            return WhisperTranscriber(model_id=model_name, device=device, model_type="transformers")
        elif backend == WhisperBackend.WHISPERX:
            return Whisperx(model_name=model_name, device=device, compute_type=compute_type)
        elif backend == WhisperBackend.CHIRP3:
            from munajjam.transcription.chirp import ChirpTranscriber

            return ChirpTranscriber()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
