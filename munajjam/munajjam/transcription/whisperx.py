from munajjam.models import Segment

from munajjam.transcription.base import BaseTranscriber
import whisperx
from typing import List
class Whisperx(BaseTranscriber):
    def __init__(self, model_name:str, device:str="cuda",compute_type:str="float32"):
        self.model_name = model_name
        self.device = device
        self.model = whisperx.load_model(model_name, device=device,compute_type=compute_type)

    def transcribe(self, audio_path: str , batch_size:int=16) -> List[Segment]:
        audio = whisperx.load_audio(audio_path)
        segments = self.model.transcribe(audio,batch_size=batch_size)
        return [Segment(id=i+1, surah_id=1, start=seg['start'], end=seg['end'], text=seg['text']) for i, seg in enumerate(segments["segments"])]