import json
import re
from pydub import AudioSegment, silence
import librosa
import torch
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -----------------------------
# Regex patterns for filtering specific phrases
# -----------------------------
# Pattern to detect "I seek refuge in Allah from the accursed Satan"
isti3aza_pattern = re.compile(r"أ?عوذ بالله من الشيطان الرجيم", re.IGNORECASE)
# Pattern to detect "Bismillah al-Rahman al-Rahim"
basmala_pattern = re.compile(r"بسم الله الرحمن الرحيم", re.IGNORECASE)

# -----------------------------
# Helper functions
# -----------------------------
def normalize_arabic(text):
    """
    Normalize Arabic text:
    - Replace different forms of alef with 'ا'
    - Replace 'ى' with 'ي'
    - Replace 'ة' with 'ه'
    - Remove punctuation
    - Remove extra spaces
    """
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_config(config_file=os.path.join("data", "current_config.json")):
    """
    Load recitation configuration (UUID and reciter name) from JSON.
    Returns:
        surah_uuid, reciter_name
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("SURAH_UUID"), config.get("RECITER_NAME")
    except FileNotFoundError:
        print(f"Error: {config_file} not found.")
        return None, None

# -----------------------------
# Step 0: Load Whisper model
# -----------------------------
def load_model():
    """
    Load Tarteel Whisper model for Arabic Quran transcription.
    Returns:
        processor, model, device
    """
    print("Loading Tarteel Whisper model...")
    model_id = "tarteel-ai/whisper-base-ar-quran"
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    return processor, model, device

# -----------------------------
# Step 1: Transcribe audio
# -----------------------------
def transcribe(audio_path, processor=None, model=None, device=None):
    """
    Transcribe a Quran audio file into text segments.
    Steps:
    1. Load model if not provided.
    2. Load surah UUID from config.
    3. Detect silent and non-silent chunks in audio.
    4. Transcribe each chunk using the Whisper model.
    5. Skip Isti'aza segments and Basmala for sura_id != 1.
    6. Save segments and silences to JSON files.
    Returns:
        segments, silences
    """
    if processor is None or model is None or device is None:
        processor, model, device = load_model()

    surah_uuid, _ = load_config()
    if not surah_uuid:
        print("Error: Missing SURAH_UUID in config.")
        return [], []

    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=16000) # y -> the actual sound wave represented as numpy array, sr-> sample rate

    # Extract sura ID from filename
    sura_id_str = os.path.splitext(os.path.basename(audio_path))[0]
    sura_id = int(sura_id_str)

    # Detect silent and non-silent parts of the audio
    silences = silence.detect_silence(audio, min_silence_len=300, silence_thresh=-30)
    chunks = silence.detect_nonsilent(audio, min_silence_len=300, silence_thresh=-30)

    segments = []
    # Process each non-silent chunk
    for idx, (start_ms, end_ms) in enumerate(chunks, 1):
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        # Prepare input for model
        inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs)

        # Decode transcription
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        # Skip Isti'aza segments
        if isti3aza_pattern.search(normalize_arabic(text)):
            print("Skipping Isti'aza segment.")
            continue

        # Skip Basmala if sura_id is not 1
        if basmala_pattern.search(normalize_arabic(text)) and sura_id != 1:
            print(f"Skipping Basmala because sura_id={sura_id} is not 1.")
            continue

        # Print segment info
        print(f"Segment {idx}: {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s {text.strip()}")

        # Append segment info
        segments.append({
            "id": idx,
            "sura_id": sura_id,
            "surah_uuid": surah_uuid,
            "start": round(start_ms/1000, 2),
            "end": round(end_ms/1000, 2),
            "text": text.strip()
        })

    # -----------------------------
    # Save results per sura_id
    # -----------------------------
    os.makedirs("data/segments", exist_ok=True)
    os.makedirs("data/silences", exist_ok=True)

    segments_path = os.path.join("data/segments", f"{sura_id}_segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    silences_path = os.path.join("data/silences", f"{sura_id}_silences.json")
    with open(silences_path, "w", encoding="utf-8") as f:
        json.dump(silences, f, ensure_ascii=False, indent=2)

    return segments, silences