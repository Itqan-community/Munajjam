from enum import Enum
from typing import Literal

from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperx import Whisperx


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float16",
    ) -> WhisperTranscriber | Whisperx:
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            return WhisperTranscriber(model_id=model_name, device=device, model_type="transformers")
        elif backend == WhisperBackend.WHISPERX:
            return Whisperx(model_name=model_name, device=device, compute_type=compute_type)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
