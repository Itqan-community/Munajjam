"""
Whisper-based transcription implementation.

Uses Tarteel AI's Whisper models fine-tuned for Quran recitation.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import librosa

from munajjam.config import MunajjamSettings, get_settings
from munajjam.core.arabic import detect_segment_type
from munajjam.exceptions import ModelNotLoadedError, TranscriptionError
from munajjam.models import Segment
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    load_audio_waveform,
)

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
    """
    Whisper-based transcriber for Quran audio.

    Uses Tarteel AI's Whisper models fine-tuned for Quran recitation.
    Supports both standard Transformers and Faster Whisper backends.

    Example:
        transcriber = WhisperTranscriber()
        transcriber.load()

        segments = transcriber.transcribe("surah_1.wav")

        transcriber.unload()

    Or using context manager:
        with WhisperTranscriber() as transcriber:
            segments = transcriber.transcribe("surah_1.wav")
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: Literal["auto", "cpu", "cuda", "mps"] | None = None,
        model_type: Literal["transformers", "faster-whisper"] | None = None,
        settings: MunajjamSettings | None = None,
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model_id: HuggingFace model ID (overrides settings)
            device: Device for inference (overrides settings)
            model_type: Model backend type (overrides settings)
            settings: Settings instance to use
        """
        self._settings = settings or get_settings()

        self._model_id = model_id or self._settings.model_id
        self._device = device or self._settings.device
        self._model_type = model_type or self._settings.model_type

        # Model state
        self._model: Any = None
        self._processor: Any = None
        self._resolved_device: str | None = None

        # Load the model directly upon initialization
        self._initialize_model()

    @property
    def model_id(self) -> str:
        """Current model ID."""
        return self._model_id

    @property
    def device(self) -> str:
        """Resolved device."""
        if self._resolved_device:
            return self._resolved_device
        return self._settings.get_resolved_device()

    def _initialize_model(self) -> None:
        """Load the Whisper model into memory."""

        import torch

        # Resolve device
        if self._device == "auto":
            if torch.cuda.is_available():
                self._resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                self._resolved_device = "mps"
            else:
                self._resolved_device = "cpu"
        else:
            self._resolved_device = self._device

        logger.info(f"Loading model: {self._model_id}")
        logger.info(f"   Backend: {self._model_type}")
        logger.info(f"   Device: {self._resolved_device}")

        if self._model_type == "faster-whisper":
            self._load_faster_whisper()
        else:
            self._load_transformers()

        logger.info("Model loaded successfully")

    def _load_transformers(self) -> None:
        """Load Transformers-based Whisper model."""
        import warnings

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from transformers.utils import logging as transformers_logging

        # Temporarily suppress warnings during model loading
        transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            logger.info("   Loading processor...")
            self._processor = AutoProcessor.from_pretrained(self._model_id)

            # Determine dtype
            if self._resolved_device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            logger.info(f"   Loading model weights (dtype: {torch_dtype})...")
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_id,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,  # Use safetensors to avoid PyTorch 2.5.1 security restrictions
            ).to(self._resolved_device)

        # Restore verbosity after loading
        transformers_logging.set_verbosity_warning()

        self._model.eval()

    def _load_faster_whisper(self) -> None:
        """Load Faster Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriptionError(
                "faster-whisper not installed. Install with: pip install munajjam[faster-whisper]"
            ) from e

        device = self._resolved_device
        if device == "mps":
            device = "cpu"  # Faster Whisper doesn't support MPS
            logger.info("   Note: Faster Whisper doesn't support MPS, using CPU instead")

        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"   Loading model (compute_type: {compute_type})...")

        try:
            self._model = WhisperModel(
                self._model_id,
                device=device,
                compute_type=compute_type,
            )
        except ValueError as e:
            if "float16" in str(e):
                logger.warning(
                    "   Fallback: Device does not support float16 effectively, trying float32..."
                )
                self._model = WhisperModel(
                    self._model_id,
                    device=device,
                    compute_type="float32",
                )
            else:
                raise
        self._processor = None  # Faster Whisper doesn't use processor

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        """
        Transcribe an audio file to segments.

        Args:
            audio_path: Path to the audio file (WAV)
            batch_size: Batch size for transcribing
            surah_id: surah number

        Returns:
            List of transcribed Segment objects
        """

        audio_path = Path(audio_path)

        if self._model_type == "faster-whisper":
            segments = self._transcribe_faster_whisper(
                audio_path, surah_id=surah_id, batch_size=batch_size
            )
        else:
            segments = self._transcribe_transformers(
                audio_path, surah_id=surah_id, batch_size=batch_size
            )

        return segments

    def _transcribe_transformers(
        self, audio_path: Path, *, surah_id: int, batch_size: int = 16
    ) -> list[Segment]:
        """Transcribe using Transformers."""
        import warnings

        import torch
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Load full audio
            waveform, sr = load_audio_waveform(
                audio_path,
                sample_rate=self._settings.sample_rate,
            )

            inputs = self._processor(
                waveform,
                sampling_rate=sr,
                return_tensors="pt",
            ).to(self._resolved_device)

            input_features = inputs["input_features"]
            model_dtype = next(self._model.parameters()).dtype
            input_features = input_features.to(dtype=model_dtype)

            with torch.no_grad():
                # NOTE: Transformers backend doesn't output word timestamps natively without complex
                # logits/attention extraction. We output a single segment.
                ids = self._model.generate(input_features)

            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]

            # Since Transformers doesn't easily give chunk timestamps out-of-the-box like faster-whisper,
            # we simply return a single segment representing the whole audio.
            # (Note: In production for large files, pipeline() is preferred)
            duration = librosa.get_duration(y=waveform, sr=sr)
            seg_type, seg_id = detect_segment_type(text)

            return [
                Segment(
                    id=seg_id,
                    surah_id=surah_id,
                    start=0.0,
                    end=round(duration, 2),
                    text=text.strip(),
                    type=seg_type,
                )
            ]

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        """Transcribe an audio file using Faster Whisper (whisper.cpp)."""

        if self._model is None:
            raise ModelNotLoadedError("Faster Whisper model not loaded.")

        # We pass string to faster-whisper directly
        if batch_size > 1:
            try:
                from faster_whisper import BatchedInferencePipeline

                pipeline = BatchedInferencePipeline(self._model)
                segments_result, _ = pipeline.transcribe(
                    str(audio_path),
                    batch_size=batch_size,
                    beam_size=5,
                    language="ar",
                    word_timestamps=True,
                )
            except (ImportError, AttributeError):
                # Fallback to standard transcribe if BatchedInferencePipeline not available
                segments_result, _ = self._model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    language="ar",
                    word_timestamps=True,
                )
        else:
            segments_result, _ = self._model.transcribe(
                str(audio_path),
                beam_size=5,
                language="ar",
                word_timestamps=True,
            )

        segments = []
        for s in segments_result:
            text = s.text.strip()
            seg_type, seg_id = detect_segment_type(text)
            segments.append(
                Segment(
                    id=seg_id,
                    surah_id=surah_id,
                    start=round(s.start, 2),
                    end=round(s.end, 2),
                    text=text,
                    type=seg_type,
                )
            )

        return segments
