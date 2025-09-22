import json
import re
from pydub import AudioSegment, silence
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def normalize_text(text: str) -> str:
    # 1. Remove diacritics
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # 2. Remove numbers (Arabic + English)
    text = re.sub(r"[0-9٠-٩]+", "", text)
    # 3. Remove isti'aza
    text = re.sub(r"(اعوذ بالله من الشيطان الرجيم)", "", text)
    text = re.sub(r"(أعوذ بالله من الشيطان الرجيم)", "", text)
    # 4. Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # 5. Lowercase (for safety with English chars)
    text = text.lower()
    # 6. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Step 0: Load model 
# -----------------------------
def load_model():
    print("Loading Tarteel Whisper model...")
    model_id = "tarteel-ai/whisper-base-ar-quran"
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    return processor, model, device


# -----------------------------
# Step 1: Transcribe using preloaded model
# -----------------------------
def transcribe(audio_path, processor=None, model=None, device=None):
    if processor is None or model is None or device is None:
        processor, model, device = load_model()  # fallback لو اتنادت من غير موديل

    # load audio
    audio = AudioSegment.from_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=16000)

    # detect silence & non-silence
    silences = silence.detect_silence(audio, min_silence_len=300, silence_thresh=-30)
    chunks = silence.detect_nonsilent(audio, min_silence_len=300, silence_thresh=-30)

    segments = []
    for idx, (start_ms, end_ms) in enumerate(chunks, 1):
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)

        with torch.no_grad():
            ids = model.generate(**inputs)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        print(f"Segment {idx}: {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s {text.strip()}")

        segments.append({
            "id": idx,
            "start": round(start_ms/1000, 2),
            "end": round(end_ms/1000, 2),
            "text": text.strip()
        })

    # save json files
    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    with open("silences.json", "w", encoding="utf-8") as f:
        json.dump(silences, f, ensure_ascii=False, indent=2)

    return segments, silences
