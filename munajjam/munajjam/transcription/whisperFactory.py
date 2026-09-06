from enum import Enum
from typing import Literal

from munajjam.config import get_settings
from munajjam.transcription.ctc_segmentation import (
    FastConformerCTCTranscriber,
    SileroVADChunker,
)
from munajjam.transcription.fastconformer_models import resolve_fastconformer_assets
from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.whisperx import Whisperx


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"
    CTC_SEGMENTATION = "ctc_segmentation"


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float16",
    ) -> WhisperTranscriber | Whisperx | FastConformerCTCTranscriber:
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            return WhisperTranscriber(model_id=model_name, device=device, model_type="transformers")
        elif backend == WhisperBackend.WHISPERX:
            return Whisperx(model_name=model_name, device=device, compute_type=compute_type)
        elif backend == WhisperBackend.CTC_SEGMENTATION:
            # Provision the ONNX graph + tokenizer up front (explicit env
            # paths win, then the cache, then a configured HF repo; a missing
            # source raises ConfigurationError). The transcriber itself stays
            # lazy: no ONNX session / tokenizer is loaded here.
            settings = get_settings()
            assets = resolve_fastconformer_assets(settings)
            chunker = SileroVADChunker() if settings.fastconformer_vad_enabled else None
            return FastConformerCTCTranscriber(
                model_path=assets.model_path,
                vocab_path=assets.vocab_path,
                tokenizer_model_path=assets.tokenizer_model_path,
                chunker=chunker,
                blank_transition_cost_zero=settings.fastconformer_blank_transition_cost_zero,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
