"""FastConformer CTC segmentation transcriber."""

import re
from pathlib import Path

from munajjam.models.segment import Segment
from munajjam.transcription.base import BaseTranscriber


class FastConformerCTCTranscriber(BaseTranscriber):
    """Experimental FastConformer-based global CTC alignment backend."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        blank_reward: float = 0.0,
        riwaya: str = "hafs",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.blank_reward = blank_reward
        self.riwaya = riwaya

    def _load_vocab(self, vocab_path: str | Path) -> tuple[list[str], int]:
        """Load a CTC vocabulary and return its tokens and blank-token index."""
        vocab_path = Path(vocab_path)

        if not vocab_path.is_file():
            raise FileNotFoundError(f"CTC vocabulary not found: {vocab_path}")

        vocab = [
            line.strip()
            for line in vocab_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not vocab:
            raise ValueError("CTC vocabulary is empty")

        if "<blk>" not in vocab:
            raise ValueError("CTC vocabulary does not contain <blk>")

        blank_id = vocab.index("<blk>")
        return vocab, blank_id
    def _normalize_for_ctc(self, text: str) -> str:
        """Normalize Quranic Arabic for the FastConformer CTC vocabulary."""
        text = re.sub(
            r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]",
            "",
            text,
        )
        text = re.sub(r"[\u0640\u06E5\u06E6]", "", text)
        text = re.sub(r"[إأآٱ]", "ا", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        """Transcribe and align a recitation using global CTC segmentation."""
        raise NotImplementedError(
            "Global CTC segmentation is not implemented yet."
        )
