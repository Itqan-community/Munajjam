from munajjam.transcription import WhisperTranscriber
from munajjam.transcription import HFInferenceTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import align_segments
from munajjam.data import load_surah_ayahs

# Config
audio_path = "Quran/badr_alturki_audio/001.wav"
surah_id = 1

# 1. Detect silences
silences = detect_silences(audio_path)

# 2. Transcribe (using HF Inference Endpoint)
# Old local transcription:
transcriber = WhisperTranscriber()
transcriber.load()
segments = transcriber.transcribe(audio_path)
transcriber.unload()

# transcriber = HFInferenceTranscriber(
#     endpoint_url="https://abt2snizmdjfr2rf.us-east-1.aws.endpoints.huggingface.cloud",
#     api_token="YOUR_HF_TOKEN_HERE",
# )
# segments = transcriber.transcribe(audio_path)

# 3. Align
ayahs = load_surah_ayahs(surah_id)
results = align_segments(segments, ayahs, silences_ms=silences)

# 4. Output
print(f"✅ Aligned {len(results)}/{len(ayahs)} ayahs")
for r in results:
    print(
        f"   Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s ({r.similarity_score:.0%})")
