from enum import Enum
from typing import Literal

from munajjam.transcription.replicate_transcriber import (
    DEFAULT_REPLICATE_MODEL,
    ReplicateWhisperXTranscriber,
)
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperx import Whisperx


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"
    REPLICATE_API = "replicate_api"  # experimental: cloud WhisperX via Replicate


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float16",
    ) -> WhisperTranscriber | Whisperx | ReplicateWhisperXTranscriber:
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            return WhisperTranscriber(model_id=model_name, device=device, model_type="transformers")
        elif backend == WhisperBackend.WHISPERX:
            return Whisperx(model_name=model_name, device=device, compute_type=compute_type)
        elif backend == WhisperBackend.REPLICATE_API:
            # `model_name` doubles as the Replicate model slug here (e.g.
            # "victor-upmeet/whisperx"); fall back to the default public model
            # if the caller passes an empty/placeholder value.
            replicate_model = model_name or DEFAULT_REPLICATE_MODEL
            return ReplicateWhisperXTranscriber(model=replicate_model)
        else:
            raise ValueError(f"Unsupported backend: {backend}")