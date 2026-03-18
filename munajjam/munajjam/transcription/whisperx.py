from munajjam.models import Segment
import os

from munajjam.transcription.base import BaseTranscriber
import whisperx
from whisperx.schema import TranscriptionResult 
from munajjam.core.arabic import detect_segment_type, infer_surah_number
from typing import List
class Whisperx(BaseTranscriber):
    def __init__(self, model_name:str, device:str="cuda",compute_type:str="float16"):
        self.model_name = model_name
        self.device = device
        self.model = whisperx.load_model(model_name, device=device,compute_type=compute_type)

    def transcribe(self, audio_path: str, batch_size: int = 16) -> List[Segment]: 
        
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=batch_size)
        
        surah_id = infer_surah_number(audio_path)
        segments = []
        for i, s in enumerate(result["segments"]):
            text = s["text"].strip()
            seg_type, _ = detect_segment_type(text)
            segments.append(
                Segment(
                    id=i,
                    surah_id=surah_id,
                    start=round(s["start"], 2),
                    end=round(s["end"], 2),
                    text=text,
                    type=seg_type,
                )
            )
        return segments
      