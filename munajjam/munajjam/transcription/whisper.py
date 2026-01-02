"""
Whisper-based transcription implementation.

Uses Tarteel AI's Whisper models fine-tuned for Quran recitation.
"""

import asyncio
import re
from pathlib import Path
from typing import Callable, Literal

from munajjam.config import MunajjamSettings, get_settings
from munajjam.exceptions import TranscriptionError, ModelNotLoadedError, AudioFileError
from munajjam.models import Segment, SegmentType
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    detect_non_silent_chunks,
    load_audio_waveform,
    extract_segment_audio,
)


# Regex patterns for special segment detection
ISTI3AZA_PATTERN = re.compile(r"[اأإآٱو]?عوذ\s*بالله\s*من\s*الشيطان\s*الرجيم")
BASMALA_PATTERN = re.compile(r"(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم")


def _normalize_for_detection(text: str) -> str:
    """Normalize text for pattern detection."""
    text = re.sub(r"[أإآاٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_segment_type(text: str) -> tuple[SegmentType, int]:
    """
    Detect segment type from transcribed text.

    Returns:
        Tuple of (segment_type, segment_id)
        segment_id is 0 for special segments, positive otherwise
    """
    normalized = _normalize_for_detection(text)

    if ISTI3AZA_PATTERN.search(normalized):
        return SegmentType.ISTI3AZA, 0

    if BASMALA_PATTERN.search(normalized):
        return SegmentType.BASMALA, 0

    return SegmentType.AYAH, 1  # Will be renumbered later


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
        self._model = None
        self._processor = None
        self._resolved_device: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._model is not None

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

    def load(self) -> None:
        """Load the Whisper model into memory."""
        if self._model is not None:
            return  # Already loaded

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

        print(f"📦 Loading model: {self._model_id}")
        print(f"   Backend: {self._model_type}")
        print(f"   Device: {self._resolved_device}")

        if self._model_type == "faster-whisper":
            self._load_faster_whisper()
        else:
            self._load_transformers()
        
        print(f"✅ Model loaded successfully")

    def _load_transformers(self) -> None:
        """Load Transformers-based Whisper model."""
        import torch
        import warnings
        import logging
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        from transformers.utils import logging as transformers_logging

        # Temporarily suppress warnings during model loading
        transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            print(f"   Loading processor...")
            self._processor = AutoProcessor.from_pretrained(self._model_id)

            # Determine dtype
            if self._resolved_device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            print(f"   Loading model weights (dtype: {torch_dtype})...")
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
        except ImportError:
            raise TranscriptionError(
                "faster-whisper not installed. "
                "Install with: pip install munajjam[faster-whisper]"
            )

        device = self._resolved_device
        if device == "mps":
            device = "cpu"  # Faster Whisper doesn't support MPS
            print(f"   Note: Faster Whisper doesn't support MPS, using CPU instead")

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"   Loading model (compute_type: {compute_type})...")

        self._model = WhisperModel(
            self._model_id,
            device=device,
            compute_type=compute_type,
        )
        self._processor = None  # Faster Whisper doesn't use processor

    def unload(self) -> None:
        """Unload the model from memory."""
        self._model = None
        self._processor = None
        self._resolved_device = None

        # Force garbage collection
        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(
        self,
        audio_path: str | Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Segment]:
        """
        Transcribe an audio file to segments.

        Args:
            audio_path: Path to the audio file (WAV)
            progress_callback: Optional callback function(current, total, text) for progress updates

        Returns:
            List of transcribed Segment objects
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioFileError(str(audio_path), "File not found")

        # Extract surah ID from filename
        surah_id = int(audio_path.stem)

        # Detect non-silent chunks
        chunks = detect_non_silent_chunks(
            audio_path,
            min_silence_len=self._settings.min_silence_ms,
            silence_thresh=self._settings.silence_threshold_db,
        )

        # Load audio waveform
        waveform, sr = load_audio_waveform(
            audio_path,
            sample_rate=self._settings.sample_rate,
        )

        segments = []
        segment_idx = 1
        total_chunks = len(chunks)

        for i, (start_ms, end_ms) in enumerate(chunks):
            # Extract segment audio
            segment_audio = extract_segment_audio(waveform, sr, start_ms, end_ms)

            if len(segment_audio) == 0:
                continue

            # Transcribe segment
            try:
                text = self._transcribe_segment(segment_audio, sr)
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to transcribe segment at {start_ms}ms-{end_ms}ms: {e}",
                    audio_path=str(audio_path),
                )

            # Detect segment type
            seg_type, seg_id = _detect_segment_type(text)
            if seg_type == SegmentType.AYAH:
                seg_id = segment_idx
                segment_idx += 1

            segment = Segment(
                id=seg_id,
                surah_id=surah_id,
                start=round(start_ms / 1000, 2),
                end=round(end_ms / 1000, 2),
                text=text.strip(),
                type=seg_type,
            )

            segments.append(segment)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_chunks, text.strip()[:50])

        return segments

    def _transcribe_segment(self, segment_audio, sample_rate: int) -> str:
        """Transcribe a single audio segment."""
        if self._model_type == "faster-whisper":
            return self._transcribe_faster_whisper(segment_audio, sample_rate)
        else:
            return self._transcribe_transformers(segment_audio, sample_rate)

    def _transcribe_transformers(self, segment_audio, sample_rate: int) -> str:
        """Transcribe using Transformers."""
        import torch
        import warnings
        import logging
        import sys
        import io
        from contextlib import redirect_stderr
        from transformers import GenerationConfig
        from transformers.utils import logging as transformers_logging

        # Suppress transformers warnings comprehensively
        # These warnings are informational and don't affect functionality
        transformers_loggers = [
            logging.getLogger("transformers"),
            logging.getLogger("transformers.generation"),
            logging.getLogger("transformers.generation.utils"),
        ]
        original_levels = [logger.level for logger in transformers_loggers]
        for logger in transformers_loggers:
            logger.setLevel(logging.ERROR)

        # Set transformers verbosity to error
        transformers_logging.set_verbosity_error()

        # Suppress Python warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            # Create a null device to suppress stderr output for warnings
            # (Some transformers warnings are printed directly via print/warning)
            null_stream = io.StringIO()
            
            try:
                inputs = self._processor(
                    segment_audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                ).to(self._resolved_device)

                # Extract input features and attention mask
                input_features = inputs["input_features"]
                attention_mask = inputs.get("attention_mask")
                
                # Convert input features to model's dtype (float16 on CUDA, float32 on CPU)
                model_dtype = next(self._model.parameters()).dtype
                input_features = input_features.to(dtype=model_dtype)
                
                # Create attention mask if it doesn't exist
                # Whisper models need attention_mask because pad_token == eos_token
                if attention_mask is None:
                    # For Whisper, input_features shape is [batch, n_mels, time_frames]
                    # Attention mask should be [batch, time_frames]
                    batch_size = input_features.shape[0]
                    time_frames = input_features.shape[2]
                    attention_mask = torch.ones(
                        (batch_size, time_frames),
                        dtype=torch.long,
                        device=self._resolved_device
                    )

                # Use model's generation config and explicitly set parameters
                # Get the model's default generation config and copy it
                generation_config = GenerationConfig.from_dict(self._model.generation_config.to_dict())
                generation_config.max_new_tokens = 128
                generation_config.num_beams = 1

                # Redirect stderr during generate to suppress warnings
                with redirect_stderr(null_stream):
                    with torch.no_grad():
                        ids = self._model.generate(
                            input_features=input_features,
                            attention_mask=attention_mask,
                            generation_config=generation_config,
                        )

                text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
                return text
            finally:
                # Restore original logging levels
                for logger, original_level in zip(transformers_loggers, original_levels):
                    logger.setLevel(original_level)
                # Restore transformers verbosity
                transformers_logging.set_verbosity_warning()

    def _transcribe_faster_whisper(self, segment_audio, sample_rate: int) -> str:
        """Transcribe using Faster Whisper."""
        import tempfile
        import os

        try:
            import soundfile as sf
        except ImportError:
            raise TranscriptionError(
                "soundfile not installed. "
                "Install with: pip install munajjam[faster-whisper]"
            )

        # Save to temp file (Faster Whisper needs file path)
        # On Windows, we need to close the file before another process can read it
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Write audio to the temp file (file is closed now)
            sf.write(tmp_path, segment_audio, sample_rate)

            # Transcribe
            segments_result, _ = self._model.transcribe(
                tmp_path,
                beam_size=1,
                language="ar",
            )

            text = ""
            for seg in segments_result:
                text = seg.text.strip()
                break

            return text
            
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except (PermissionError, OSError):
                    pass  # Ignore cleanup errors on Windows

    async def transcribe_async(self, audio_path: str | Path) -> list[Segment]:
        """
        Asynchronously transcribe an audio file.

        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio_path)

