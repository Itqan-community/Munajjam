"""
EXPERIMENTAL: Cloud-based WhisperX transcription via the Replicate API.

This backend sends audio to a publicly hosted WhisperX model (VAD + Wav2Vec2 CTC
alignment) running on Replicate's GPUs instead of running the pipeline locally.
It exists to evaluate whether serverless inference can match the accuracy of the
local `Whisperx` pipeline without the burden of local GPU hosting. It is not
intended to be the default backend.

Requires the `REPLICATE_API_TOKEN` environment variable to be set. Get a token at
https://replicate.com/account/api-tokens.
"""

import os
import time
from pathlib import Path
from typing import Any

try:
    import replicate
except ImportError:
    replicate = None  # type: ignore[assignment]

from munajjam.data import load_surah_ayahs
from munajjam.models import Segment
from munajjam.transcription.alignment_utils import align_words_to_segments
from munajjam.transcription.base import BaseTranscriber

# Public WhisperX model hosted on Replicate: https://replicate.com/victor-upmeet/whisperx
DEFAULT_REPLICATE_MODEL = "victor-upmeet/whisperx"

_TERMINAL_STATES = {"succeeded", "failed", "canceled"}


class ReplicateTranscriptionError(RuntimeError):
    """Raised when a Replicate WhisperX prediction fails or returns no usable output."""


class ReplicateWhisperXTranscriber(BaseTranscriber):
    """
    EXPERIMENTAL: Cloud-based WhisperX transcriber backed by the Replicate API.

    Acts as a lightweight proxy: it uploads audio to Replicate, polls the
    prediction until it completes, and maps the returned word-level timestamps
    onto the surah's reference ayah text using the same alignment logic as the
    local `Whisperx` backend (see `alignment_utils.align_words_to_segments`).

        transcriber = ReplicateWhisperXTranscriber()
        segments = transcriber.transcribe("surah_1.wav", surah_id=1)
    """

    def __init__(
        self,
        model: str = DEFAULT_REPLICATE_MODEL,
        *,
        align_output: bool = True,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> None:
        if replicate is None:
            raise ImportError(
                "The 'replicate' package is required for ReplicateWhisperXTranscriber. "
                "Install it with: pip install replicate"
            )

        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise ReplicateTranscriptionError(
                "REPLICATE_API_TOKEN environment variable is not set. Get a token at "
                "https://replicate.com/account/api-tokens and set it before using the "
                "experimental REPLICATE_API backend."
            )

        self.model = model
        self.align_output = align_output
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._client = replicate.Client(api_token=api_token)
        # Community-hosted models require a pinned version for predictions.create();
        # resolve it dynamically so this doesn't silently 404 again if the model updates.
        self._model_version = self._client.models.get(self.model).latest_version.id

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []

        audio_path = Path(audio_path)

        print(f"[Replicate] Submitting {audio_path.name} to {self.model} ...")
        with open(audio_path, "rb") as audio_file:
            prediction = self._client.predictions.create(
                version=self._model_version,
                input={
                    "audio_file": audio_file,
                    "align_output": self.align_output,
                    "language": "ar",
                    "batch_size": batch_size,
                },
            )

        prediction = self._poll_until_done(prediction)

        if prediction.status != "succeeded":
            raise ReplicateTranscriptionError(
                f"Replicate prediction {prediction.id} finished with status "
                f"'{prediction.status}': {prediction.error or 'unknown error'}"
            )

        extracted_words = self._parse_output(prediction.output)
        if not extracted_words:
            raise ReplicateTranscriptionError(
                f"Replicate prediction {prediction.id} succeeded but returned no "
                "word-level timestamps."
            )

        ref_words = [w for ayah in ayahs for w in ayah.text.split()]

        return align_words_to_segments(
            ref_words=ref_words,
            extracted_words=extracted_words,
            ayahs=ayahs,
            surah_id=surah_id,
            audio_path=audio_path,
        )

    def _poll_until_done(self, prediction: Any) -> Any:
        """Poll the prediction until it reaches a terminal state or times out."""
        elapsed = 0.0
        while prediction.status not in _TERMINAL_STATES:
            if elapsed >= self.timeout:
                raise ReplicateTranscriptionError(
                    f"Replicate prediction {prediction.id} timed out after "
                    f"{self.timeout}s (last status: {prediction.status})"
                )
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval
            prediction.reload()
        return prediction

    @staticmethod
    def _parse_output(output: Any) -> list[dict[str, Any]]:
        """
        Flatten Replicate's WhisperX output JSON into the flat word-list shape
        expected by `alignment_utils.align_words_to_segments`
        (a list of {"word", "start", "end", "confidence"} dicts).
        """
        if not output:
            return []

        segments = output.get("segments") if isinstance(output, dict) else output
        if not segments:
            return []

        extracted_words: list[dict[str, Any]] = []
        for segment in segments:
            for w in segment.get("words", []):
                if "start" not in w or "end" not in w:
                    # WhisperX drops timing for words it couldn't confidently align
                    continue
                extracted_words.append(
                    {
                        "word": (w.get("word") or "").strip(),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "confidence": float(w.get("score", w.get("confidence", 0.9))),
                    }
                )
        return extracted_words