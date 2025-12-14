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
isti3aza_pattern = re.compile(r"اعوذ بالله من الشيطان الرجيم")
# Pattern to detect "Bismillah al-Rahman al-Rahim" (allow slight ASR variations)
basmala_pattern = re.compile(r"(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم")

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
# Cache for loaded model to avoid reloading
_model_cache = {
    "model_key": None,
    "processor": None,
    "model": None,
    "device": None
}

def clear_model_cache():
    """
    Clear the cached model. Useful if you want to force reload or switch models.
    """
    global _model_cache
    _model_cache = {
        "model_key": None,
        "processor": None,
        "model": None,
        "device": None
    }
    print("Model cache cleared.")

def load_model_config(config_file="model_config.json"):
    """
    Load model configuration from JSON file.
    Returns:
        selected_model_key, model_config dict
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            selected = config.get("selected_model", "whisper-base-ar-quran")
            model_info = config.get("models", {}).get(selected)
            if not model_info:
                print(f"Warning: Model '{selected}' not found in config. Using default.")
                selected = "whisper-base-ar-quran"
                model_info = config.get("models", {}).get(selected)
            return selected, model_info
    except FileNotFoundError:
        print(f"Warning: {config_file} not found. Using default model.")
        return "whisper-base-ar-quran", {
            "model_id": "tarteel-ai/whisper-base-ar-quran",
            "type": "transformers",
            "description": "Default Tarteel Whisper Base model"
        }

def load_model():
    """
    Load Tarteel Whisper model for Arabic Quran transcription.
    Reads model selection from model_config.json.
    Optimized for macOS with MPS (Metal Performance Shaders) support.
    Uses caching to avoid reloading the same model.
    Returns:
        processor, model, device (or model, device for faster-whisper)
    """
    # Load model configuration
    model_key, model_config = load_model_config()
    model_id = model_config.get("model_id")
    model_type = model_config.get("type", "transformers")
    description = model_config.get("description", "")
    
    # Check if model is already cached
    if _model_cache["model_key"] == model_key and _model_cache["model"] is not None:
        print(f"\n{'='*60}")
        print(f"Using cached model: {model_key}")
        print(f"Model ID: {model_id}")
        print(f"Type: {model_type}")
        print(f"{'='*60}\n")
        
        # Return cached model
        if _model_cache["processor"] is None:
            # faster-whisper returns (model, device)
            return _model_cache["model"], _model_cache["device"]
        else:
            # transformers returns (processor, model, device)
            return _model_cache["processor"], _model_cache["model"], _model_cache["device"]
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_key}")
    print(f"Model ID: {model_id}")
    print(f"Type: {model_type}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}\n")
    
    # Handle faster-whisper type
    if model_type == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("ERROR: 'faster-whisper' package not installed.")
            print("Please install it with: pip install faster-whisper")
            print("Falling back to transformers model...")
            # Fallback to default transformers model
            model_key, model_config = "whisper-base-ar-quran", {
                "model_id": "tarteel-ai/whisper-base-ar-quran",
                "type": "transformers"
            }
            model_id = model_config["model_id"]
            model_type = "transformers"
        
        if model_type == "faster-whisper":
            # Determine device for faster-whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            print(f"Loading Faster Whisper model on device: {device}")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            print(f"Model loaded successfully on {device}")
            
            # Cache the model
            _model_cache["model_key"] = model_key
            _model_cache["model"] = model
            _model_cache["device"] = device
            _model_cache["processor"] = None  # faster-whisper doesn't use processor
            
            return model, device
    
    # Standard transformers implementation
    if model_type != "transformers":
        print(f"Warning: Unknown model type '{model_type}'. Using transformers.")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Determine best device for macOS
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16  # Use float16 for CUDA
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU acceleration
        torch_dtype = torch.float32  # MPS works best with float32
        print("Using MPS (Metal Performance Shaders) for Apple Silicon acceleration")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("Using CPU (consider upgrading to Apple Silicon for better performance)")
    
    # Load model with optimizations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Enable eval mode for faster inference
    model.eval()
    
    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device != "mps":  # torch.compile not fully supported on MPS yet
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled for faster inference")
        except Exception as e:
            print(f"Model compilation skipped: {e}")
    
    print(f"Model loaded on device: {device}")
    
    # Cache the model
    _model_cache["model_key"] = model_key
    _model_cache["processor"] = processor
    _model_cache["model"] = model
    _model_cache["device"] = device
    
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
    # Load model if not provided
    if processor is None or model is None or device is None:
        result = load_model()
        # Check if it's faster-whisper (returns model, device) or transformers (returns processor, model, device)
        if len(result) == 2:
            model, device = result
            processor = None  # faster-whisper doesn't use processor
        else:
            processor, model, device = result

    surah_uuid, _ = load_config()
    if not surah_uuid:
        print("Error: Missing SURAH_UUID in config.")
        return [], []

    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=16000) # y -> the actual sound wave represented as numpy array, sr-> sample rate

    # Extract sura ID from filename
    sura_id_str = os.path.splitext(os.path.basename(audio_path))[0]
    sura_id = int(sura_id_str)  # Integer for data storage

    # Detect silent and non-silent parts of the audio
    silences = silence.detect_silence(audio, min_silence_len=300, silence_thresh=-30)
    chunks = silence.detect_nonsilent(audio, min_silence_len=300, silence_thresh=-30)

    segments = []
    # Check if using faster-whisper (processor will be None)
    is_faster_whisper = processor is None
    
    # Process each non-silent chunk
    for idx, (start_ms, end_ms) in enumerate(chunks, 1):
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        if is_faster_whisper:
            # Faster Whisper API
            import tempfile
            try:
                import soundfile as sf
            except ImportError:
                print("ERROR: 'soundfile' package required for faster-whisper.")
                print("Please install it with: pip install soundfile")
                raise
            
            # Save segment to temporary file for faster-whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, segment, sr)
                segments_result, info = model.transcribe(
                    tmp_file.name,
                    beam_size=1,  # Greedy decoding for speed
                    language="ar"
                )
                # Get the first (and only) segment
                text = ""
                for segment_result in segments_result:
                    text = segment_result.text.strip()
                    break
                os.unlink(tmp_file.name)  # Clean up temp file
        else:
            # Standard Transformers API
            # Prepare input for model
            inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                # Optimize generation parameters for faster inference
                ids = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Limit tokens for faster generation
                    num_beams=1,  # Use greedy decoding (faster than beam search)
                    attention_mask=inputs.get("attention_mask"),  # Explicitly pass attention mask
                )

            # Decode transcription
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        # Check for Isti'aza or Basmala segments
        is_isti3aza = isti3aza_pattern.search(normalize_arabic(text))
        is_basmala = basmala_pattern.search(normalize_arabic(text))
        
        # Determine segment type and ID
        if is_isti3aza:
            segment_type = "isti3aza"
            segment_id = 0  # Special ID for Isti'aza
            print(f"Segment {idx} (Isti'aza): {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s {text.strip()}")
        elif is_basmala:
            segment_type = "basmala"
            segment_id = 0  # Special ID for Basmala
            print(f"Segment {idx} (Basmala): {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s {text.strip()}")
        else:
            segment_type = "ayah"
            segment_id = idx
            print(f"Segment {idx}: {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s {text.strip()}")

        # Append segment info (including Isti'aza and Basmala)
        segments.append({
            "id": segment_id,
            "sura_id": sura_id,
            "surah_uuid": surah_uuid,
            "start": round(start_ms/1000, 2),
            "end": round(end_ms/1000, 2),
            "text": text.strip(),
            "type": segment_type  # Add type field to distinguish
        })

    # -----------------------------
    # Save results per sura_id
    # -----------------------------
    os.makedirs("data/segments", exist_ok=True)
    os.makedirs("data/silences", exist_ok=True)

    segments_path = os.path.join("data/segments", f"{sura_id_str}_segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    silences_path = os.path.join("data/silences", f"{sura_id_str}_silences.json")
    with open(silences_path, "w", encoding="utf-8") as f:
        json.dump(silences, f, ensure_ascii=False, indent=2)

    return segments, silences