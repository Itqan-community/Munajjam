import gc
from pathlib import Path
from typing import Any

try:
    import torch

    # Workaround for PyTorch 2.6+ weights_only=True default which breaks pyannote/lightning
    _original_torch_load = getattr(torch, "load", None)
    if callable(_original_torch_load):

        def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            assert callable(_original_torch_load)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

    import whisperx
except ImportError:
    torch = None  # type: ignore[assignment]
    whisperx = None  # type: ignore[assignment]

import numpy as np
import soundfile as sf
from rapidfuzz import fuzz

from munajjam.config import get_settings
from munajjam.data import load_surah_ayahs
from munajjam.exceptions import TranscriptionError
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.silence import (
    annotate_segments_with_breaths,
    detect_reciter_breaths,
)


class Whisperx(BaseTranscriber):
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        settings = get_settings()
        self.model_name = model_name or settings.whisperx_model_size
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = settings.get_resolved_device()
        self.device = resolved_device
        if self.device == "cpu" and compute_type == "float16":
            self.compute_type = "int8"
        else:
            self.compute_type = compute_type
        self.wav2vec2_model_id = settings.wav2vec2_model_id
        self.min_pause_duration_ms = settings.min_pause_duration_ms

        self.whisper_model: Any = None
        self.align_model: Any = None
        self.align_metadata: Any = None

    def unload_model(self) -> None:
        """Safely unload active WhisperX and alignment models from VRAM/RAM."""
        if getattr(self, "whisper_model", None) is not None:
            del self.whisper_model
            self.whisper_model = None
        if getattr(self, "align_model", None) is not None:
            del self.align_model
            self.align_model = None
        if getattr(self, "align_metadata", None) is not None:
            del self.align_metadata
            self.align_metadata = None

        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_model_name(self, model_name: str) -> None:
        """Update model name and safely unload existing model if size changes."""
        if self.model_name != model_name:
            self.unload_model()
            self.model_name = model_name

    def _normalize_arabic(self, text: str) -> str:
        from ..core.arabic import normalize_arabic

        return normalize_arabic(text)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        ayahs = load_surah_ayahs(surah_id)

        def _get_ayah_words(text: str) -> list[str]:
            return [w for w in text.split() if self._normalize_arabic(w).strip()]

        ref_words = []
        for ayah in ayahs:
            for w in _get_ayah_words(ayah.text):
                ref_words.append(w)

        if whisperx is None:
            raise TranscriptionError(
                "whisperx not installed. Install with: pip install git+https://github.com/m-bain/whisperx.git"
            )

        if not self.whisper_model:
            print(f"Loading WhisperX model {self.model_name}...")
            self.whisper_model = whisperx.load_model(
                self.model_name, self.device, compute_type=self.compute_type, language="ar"
            )

        assert self.whisper_model is not None
        audio = whisperx.load_audio(str(audio_path))

        breaths = (
            detect_reciter_breaths(audio_path, min_pause_duration_ms=self.min_pause_duration_ms)
            or []
        )
        result = self.whisper_model.transcribe(audio, batch_size=batch_size)

        # Filter and normalize valid non-empty segments for whisperx.align
        valid_segments: list[dict[str, Any]] = []
        for segment in result.get("segments", []):
            if isinstance(segment, dict):
                norm_text = self._normalize_arabic(str(segment.get("text", "")))
                if norm_text.strip():
                    segment["text"] = norm_text.strip()
                    valid_segments.append(segment)
        result["segments"] = valid_segments

        if getattr(self, "align_model", None) is None:
            print("Loading WhisperX alignment model...")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="ar", device=self.device
            )

        assert self.align_model is not None
        assert self.align_metadata is not None
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        extracted_words: list[dict[str, Any]] = []
        for segment in result["segments"]:
            if isinstance(segment, dict) and "words" in segment:
                for w in segment["words"]:
                    if isinstance(w, dict) and "start" in w and "end" in w:
                        extracted_words.append(
                            {
                                "word": str(w["word"]),
                                "start": float(w["start"]),
                                "end": float(w["end"]),
                                "confidence": float(w.get("score", 0.9)),
                            }
                        )

        n = len(ref_words)
        m = len(extracted_words)
        dp = np.zeros((n + 1, m + 1))

        # Base case penalty for deleting reference words
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] - 0.5

        for i in range(1, n + 1):
            rw = self._normalize_arabic(ref_words[i - 1])
            for j in range(1, m + 1):
                ew = self._normalize_arabic(str(extracted_words[j - 1]["word"]))
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.5:
                    match_score = -1.0
                dp[i][j] = max(dp[i - 1][j] - 0.5, dp[i][j - 1], dp[i - 1][j - 1] + match_score)

        mapped_alignments: list[dict[str, Any] | None] = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = self._normalize_arabic(ref_words[i - 1])
            ew = self._normalize_arabic(str(extracted_words[j - 1]["word"]))
            match_score = fuzz.ratio(rw, ew) / 100.0

            if match_score >= 0.5 and abs(dp[i][j] - (dp[i - 1][j - 1] + match_score)) < 1e-5:
                mapped_alignments[i - 1] = extracted_words[j - 1]
                i -= 1
                j -= 1
            elif abs(dp[i][j] - dp[i][j - 1]) < 1e-5:
                # Extra audio word (e.g. repetition or cough) -> advance audio
                j -= 1
            else:
                # Reference word truly missing from audio
                i -= 1

        w_alignments: list[dict[str, Any]] = []
        for k in range(n):
            align_item = mapped_alignments[k]
            if align_item is not None:
                w_alignments.append(
                    {
                        "word": ref_words[k],
                        "start": align_item["start"],
                        "end": align_item["end"],
                        "confidence": align_item["confidence"],
                    }
                )
            else:
                prev_end = w_alignments[-1]["end"] if w_alignments else 0
                w_alignments.append(
                    {
                        "word": ref_words[k],
                        "start": prev_end,
                        "end": prev_end + 0.1,
                        "confidence": 0.0,
                    }
                )

        # Memory cleanup for temp variables, but keep models loaded
        gc.collect()

        from ..core.cascade_recovery import recover_unaligned_word_gaps

        try:
            total_duration = sf.info(str(audio_path)).duration
        except Exception:
            total_duration = len(audio) / float(getattr(audio, "sampling_rate", 16000))

        w_alignments = recover_unaligned_word_gaps(
            w_alignments,
            audio=audio,
            align_model=self.align_model,
            align_metadata=self.align_metadata,
            device=self.device,
            audio_duration=total_duration,
        )
        final_alignments = w_alignments

        ayah_boundary_indices = set()
        w_idx = 0
        for ayah in ayahs:
            w_idx += len(_get_ayah_words(ayah.text))
            ayah_boundary_indices.add(w_idx - 1)

        for k in range(len(final_alignments)):
            if k > 0:
                if final_alignments[k]["start"] < final_alignments[k - 1]["end"]:
                    final_alignments[k]["start"] = final_alignments[k - 1]["end"]

            if k < len(final_alignments) - 1:
                next_start = final_alignments[k + 1]["start"]
                current_end = final_alignments[k]["end"]
                gap = next_start - current_end

                if gap > 0:
                    if k in ayah_boundary_indices:
                        # Clean ayah boundary transition: preserve natural end of ayah without stretching into silence
                        if gap <= 0.3:
                            start_buffer = min(gap, 0.1)
                            final_alignments[k + 1]["start"] = round(next_start - start_buffer, 3)
                            final_alignments[k]["end"] = round(next_start - start_buffer, 3)
                        else:
                            final_alignments[k]["end"] = round(current_end + min(gap * 0.1, 0.2), 3)
                            final_alignments[k + 1]["start"] = round(
                                max(current_end, next_start - min(gap * 0.1, 0.15)), 3
                            )
                    else:
                        # Intra-ayah word gap: bridge small continuous speech gaps,
                        # but preserve natural breath pauses and reciter repetition gaps without stretching
                        if gap <= 0.25:
                            final_alignments[k]["end"] = round(next_start, 3)
                        else:
                            final_alignments[k]["end"] = round(
                                current_end + min(gap * 0.15, 0.15), 3
                            )
            else:
                # Clean end for the very last word of the surah
                final_alignments[k]["end"] = round(min(total_duration, current_end + 0.3), 3)

            if final_alignments[k]["end"] <= final_alignments[k]["start"]:
                final_alignments[k]["end"] = round(final_alignments[k]["start"] + 0.1, 3)

        word_idx = 0
        segments = []

        for ayah in ayahs:
            ayah_words_count = len(_get_ayah_words(ayah.text))
            ayah_alignments = final_alignments[word_idx : word_idx + ayah_words_count]
            word_idx += ayah_words_count

            if not ayah_alignments:
                continue

            words = []
            avg_conf = 0.0
            for wa in ayah_alignments:
                words.append(
                    WordTimestamp(
                        word=wa["word"],
                        start=wa["start"],
                        end=wa["end"],
                        probability=wa["confidence"],
                    )
                )
                avg_conf += wa["confidence"]

            if words:
                avg_conf /= len(words)
                segments.append(
                    Segment(
                        id=ayah.ayah_number,
                        surah_id=surah_id,
                        start=words[0].start,
                        end=words[-1].end,
                        text=ayah.text,
                        type=SegmentType.AYAH,
                        words=words,
                        confidence=avg_conf,
                    )
                )

        return annotate_segments_with_breaths(segments, breaths=breaths)
