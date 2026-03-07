from .whisper import WhisperTranscriber
from .whisperx import Whisperx
from enum import Enum
class WhisperBackend(Enum):
    FASTERWHISPER = "fasterwhisper"
    WHISPERX = "whisperx"

class WhisperFactory:
    def create_whisper(self, backend:WhisperBackend,model_name:str=None,device:str="cuda"):
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(model_size="base")  
        elif backend == WhisperBackend.OPENAI:
             #for now, also it needs to be decoupled from faster whisper
             return Whisper(model_name=model_name, device=device)
        elif backend == WhisperBackend.WHISPERX:
             return Whisperx(model_name=model_name, device=device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")