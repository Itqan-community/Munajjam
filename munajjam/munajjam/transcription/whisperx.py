from pathlib import Path

import whisperx

from munajjam.core.arabic import detect_segment_type
from munajjam.models import Segment
from munajjam.transcription.base import BaseTranscriber


class Whisperx(BaseTranscriber):
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.model = whisperx.load_model(model_name, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=batch_size)

        segments = []
        for s in result["segments"]:
            text = s["text"].strip()
            seg_type, seg_id = detect_segment_type(text)
            segments.append(
                Segment(
                    id=seg_id,
                    surah_id=surah_id,
                    start=round(s["start"], 2),
                    end=round(s["end"], 2),
                    text=text,
                    type=seg_type,
                )
            )
        return segments
