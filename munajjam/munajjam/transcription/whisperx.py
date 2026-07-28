import gc
import re
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch

# Workaround for PyTorch 2.6+ weights_only=True default which breaks pyannote/lightning
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from rapidfuzz import fuzz
from pathlib import Path

from munajjam.core.arabic import detect_segment_type
from munajjam.models import Segment, WordTimestamp, SegmentType
from munajjam.transcription.base import BaseTranscriber
from munajjam.data import load_surah_ayahs
from munajjam.config import get_settings


class Whisperx(BaseTranscriber):
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        
        settings = get_settings()
        self.wav2vec2_model_id = settings.wav2vec2_model_id

        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None

    def _normalize_arabic(self, text: str) -> str:
        text = re.sub(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
        text = re.sub(r'[أإآٱ]', 'ا', text)
        text = re.sub(r'[^\u0621-\u064A\s]', '', text)
        return text.strip()

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
            
        ref_words = []
        for ayah in ayahs:
            for w in ayah.text.split():
                ref_words.append(w)
        
        if not self.whisper_model:
            print(f"Loading WhisperX model {self.model_name}...")
            self.whisper_model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type, language="ar")
        
        audio = whisperx.load_audio(str(audio_path))
        result = self.whisper_model.transcribe(audio, batch_size=batch_size)
        
        # --- Reference Text Injection ---
        if result["segments"] and ref_words:
            transcribed_words = []
            for seg_idx, segment in enumerate(result["segments"]):
                for w in segment["text"].split():
                    transcribed_words.append({"word": w, "seg_idx": seg_idx})
                    
            n_ref = len(ref_words)
            m_tr = len(transcribed_words)
            if m_tr > 0:
                dp_inj = np.zeros((n_ref + 1, m_tr + 1))
                
                for i in range(1, n_ref + 1):
                    rw = self._normalize_arabic(ref_words[i-1])
                    for j in range(1, m_tr + 1):
                        ew = self._normalize_arabic(transcribed_words[j-1]["word"])
                        match_score = fuzz.ratio(rw, ew) / 100.0
                        if match_score < 0.6: match_score = -1.0
                        dp_inj[i][j] = max(
                            dp_inj[i-1][j],
                            dp_inj[i][j-1],
                            dp_inj[i-1][j-1] + match_score
                        )
                        
                mapped_seg_indices = [None] * n_ref
                i, j = n_ref, m_tr
                while i > 0 and j > 0:
                    rw = self._normalize_arabic(ref_words[i-1])
                    ew = self._normalize_arabic(transcribed_words[j-1]["word"])
                    match_score = fuzz.ratio(rw, ew) / 100.0
                    
                    if match_score >= 0.6 and dp_inj[i][j] == dp_inj[i-1][j-1] + match_score:
                        mapped_seg_indices[i-1] = transcribed_words[j-1]["seg_idx"]
                        i -= 1
                        j -= 1
                    elif dp_inj[i][j] == dp_inj[i-1][j]:
                        i -= 1
                    else:
                        j -= 1
                        
                seg_ref_texts = {idx: [] for idx in range(len(result["segments"]))}
                last_seg_idx = 0
                for k in range(n_ref):
                    if mapped_seg_indices[k] is not None:
                        seg_idx = mapped_seg_indices[k]
                        seg_ref_texts[seg_idx].append(ref_words[k])
                        last_seg_idx = seg_idx
                    else:
                        seg_ref_texts[last_seg_idx].append(ref_words[k])
                        
                for idx, segment in enumerate(result["segments"]):
                    segment["text"] = " ".join(seg_ref_texts[idx])
        # --- End Injection ---
        
        if getattr(self, "align_model", None) is None:
            print(f"Loading WhisperX alignment model...")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code="ar", device=self.device)
            
        result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)
        
        extracted_words = []
        for segment in result["segments"]:
            if "words" in segment:
                for w in segment["words"]:
                    if "start" in w and "end" in w:
                        extracted_words.append({
                            "word": w["word"],
                            "start": w["start"],
                            "end": w["end"],
                            "confidence": w.get("score", 0.9)
                        })

        n = len(ref_words)
        m = len(extracted_words)
        dp = np.zeros((n + 1, m + 1))
        
        for i in range(1, n + 1):
            rw = self._normalize_arabic(ref_words[i-1])
            for j in range(1, m + 1):
                ew = self._normalize_arabic(extracted_words[j-1]["word"])
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.6: match_score = -1.0
                dp[i][j] = max(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1] + match_score
                )
        
        mapped_alignments = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = self._normalize_arabic(ref_words[i-1])
            ew = self._normalize_arabic(extracted_words[j-1]["word"])
            match_score = fuzz.ratio(rw, ew) / 100.0
            
            if match_score >= 0.6 and dp[i][j] == dp[i-1][j-1] + match_score:
                mapped_alignments[i-1] = extracted_words[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j]:
                i -= 1
            else:
                j -= 1
        
        w_alignments = []
        for k in range(n):
            if mapped_alignments[k]:
                w_alignments.append({
                    "word": ref_words[k],
                    "start": mapped_alignments[k]["start"],
                    "end": mapped_alignments[k]["end"],
                    "confidence": mapped_alignments[k]["confidence"]
                })
            else:
                prev_end = w_alignments[-1]["end"] if w_alignments else 0
                w_alignments.append({
                    "word": ref_words[k],
                    "start": prev_end,
                    "end": prev_end + 0.1,
                    "confidence": 0.0
                })

        # Memory cleanup for temp variables, but keep models loaded
        gc.collect()
            
        final_alignments = w_alignments

        try:
            total_duration = sf.info(str(audio_path)).duration
        except Exception:
            total_duration = final_alignments[-1]["end"] + 2.0

        ayah_boundary_indices = set()
        w_idx = 0
        for ayah in ayahs:
            w_idx += len(ayah.text.split())
            ayah_boundary_indices.add(w_idx - 1)

        for k in range(len(final_alignments)):
            if k > 0:
                if final_alignments[k]["start"] < final_alignments[k-1]["end"]:
                    final_alignments[k]["start"] = final_alignments[k-1]["end"]
            
            if k < len(final_alignments) - 1:
                next_start = final_alignments[k+1]["start"]
                current_end = final_alignments[k]["end"]
                gap = next_start - current_end
                
                if gap > 0:
                    if k in ayah_boundary_indices:
                        if gap <= 0.3:
                            start_buffer = min(gap, 0.1)
                            final_alignments[k+1]["start"] = round(next_start - start_buffer, 3)
                            final_alignments[k]["end"] = round(next_start - start_buffer, 3)
                        elif gap >= 0.4:
                            final_alignments[k+1]["start"] = round(next_start - 0.2, 3)
                            final_alignments[k]["end"] = round(next_start - 0.2, 3)
                        else:
                            mid = gap / 2.0
                            final_alignments[k+1]["start"] = round(next_start - mid, 3)
                            final_alignments[k]["end"] = round(next_start - mid, 3)
                    else:
                        if gap > 0.1:
                            final_alignments[k]["end"] = round(next_start - 0.1, 3)
            else:
                final_alignments[k]["end"] = round(total_duration, 3)

            if final_alignments[k]["end"] <= final_alignments[k]["start"]:
                final_alignments[k]["end"] = round(final_alignments[k]["start"] + 0.1, 3)

        word_idx = 0
        segments = []
        
        for ayah in ayahs:
            ayah_words_count = len(ayah.text.split())
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
                        probability=wa["confidence"]
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
                        confidence=avg_conf
                    )
                )
            
        return segments
