"""Debug script to investigate alignment issues."""
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from munajjam.core.arabic import normalize_arabic
from munajjam.core.matcher import similarity
from munajjam.data import load_surah_ayahs
from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences

print("=== Debugging Surah 107 Alignment ===\n")

# Load ayahs
ayahs = load_surah_ayahs(107)
print(f"Reference Ayahs ({len(ayahs)}):")
for a in ayahs:
    norm = normalize_arabic(a.text)
    print(f"  {a.ayah_number}: {a.text[:40]}...")
    print(f"      normalized: {norm[:40]}...")

# Transcribe
print("\nTranscribing...")
t = WhisperTranscriber(model_id='OdyAsh/faster-whisper-base-ar-quran', model_type='faster-whisper')
t.load()
segments = t.transcribe('Quran/badr_alturki_audio/107.wav')
t.unload()

print(f"\nTranscribed Segments ({len(segments)}):")
for s in segments:
    norm = normalize_arabic(s.text)
    print(f"  {s.type.value}: {s.text[:40]}...")
    print(f"      normalized: {norm[:40]}...")

# Test similarity between first ayah segment and first reference ayah
print("\n=== Similarity Test ===")
# Find first ayah segment
ayah_segments = [s for s in segments if s.type.value == 'ayah']
if ayah_segments and ayahs:
    seg_text = ayah_segments[0].text
    ref_text = ayahs[0].text
    
    print(f"Segment: {seg_text}")
    print(f"Reference: {ref_text}")
    
    score = similarity(seg_text, ref_text)
    print(f"Similarity: {score:.1%}")
    
    # Also test normalized
    norm_seg = normalize_arabic(seg_text)
    norm_ref = normalize_arabic(ref_text)
    print(f"\nNormalized segment: {norm_seg}")
    print(f"Normalized reference: {norm_ref}")
    
    score2 = similarity(norm_seg, norm_ref)
    print(f"Normalized similarity: {score2:.1%}")

# Now test full alignment
print("\n=== Full Alignment Test ===")
from munajjam.core.aligner_dp import align_segments_dp_with_constraints
from munajjam.transcription.silence import detect_silences

silences = detect_silences('Quran/badr_alturki_audio/107.wav')
results = align_segments_dp_with_constraints(segments, ayahs, silences_ms=silences)

print(f"Aligned {len(results)} / {len(ayahs)} ayahs:")
for r in results:
    print(f"  Ayah {r.ayah.ayah_number}: {r.start_time:.1f}s - {r.end_time:.1f}s ({r.similarity_score:.1%})")

