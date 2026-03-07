from munajjam.munajjam.models.segment import Segment

from .base import BaseTranscriber
import whisperx
from typing import List
class Whisperx(BaseTranscriber):
    def __init__(self, model_name:str, device:str="cuda",batch_size:int=16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = whisperx.load_model(model_name, device=device)
    def trasncribe(self, audio_path: str , patch_Size:int=16) -> List[Segment]:
        audio = whisperx.load_audio(audio_path)
        segments = self.model.transcribe(audio,patch_Size=patch_Size)
        return [Segment(start=seg['start'], end=seg['end'], text=seg['text']) for seg in segments]