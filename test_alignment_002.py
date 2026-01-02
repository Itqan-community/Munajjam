#!/usr/bin/env python3
"""
Test script for iterating on alignment algorithm for Surah 002.
Transcribes once, then allows quick re-alignment.

Usage:
    python test_alignment_002.py --transcribe  # Transcribe and save (run once)
    python test_alignment_002.py               # Align using cached transcription
"""

import json
import sys
import time
from pathlib import Path

# Add munajjam to path
sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import align_segments
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType

SURAH_ID = 2
AUDIO_PATH = "Quran/badr_alturki_audio/002.wav"
CACHE_DIR = Path("cache")
SEGMENTS_FILE = CACHE_DIR / "surah_002_segments.json"
SILENCES_FILE = CACHE_DIR / "surah_002_silences.json"


def transcribe_and_save():
    """Transcribe audio and save segments to cache."""
    print("=" * 60)
    print("🎤 TRANSCRIBING SURAH 002 (this will take a while)")
    print("=" * 60)
    
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Detect silences
    print("\n🔇 Detecting silences...")
    start = time.time()
    silences = detect_silences(AUDIO_PATH)
    print(f"   Found {len(silences)} silence periods in {time.time()-start:.2f}s")
    
    # Save silences
    with open(SILENCES_FILE, "w") as f:
        json.dump(silences, f)
    print(f"   Saved to {SILENCES_FILE}")
    
    # Transcribe
    print("\n🎤 Transcribing...")
    start = time.time()
    
    transcriber = WhisperTranscriber()
    transcriber.load()
    segments = transcriber.transcribe(AUDIO_PATH)
    transcriber.unload()
    
    print(f"   Transcribed {len(segments)} segments in {time.time()-start:.2f}s")
    
    # Save segments
    segments_data = [
        {
            "id": seg.id,
            "surah_id": seg.surah_id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "type": seg.type.value,
        }
        for seg in segments
    ]
    
    with open(SEGMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)
    print(f"   Saved to {SEGMENTS_FILE}")
    
    print("\n✅ Transcription cached! Now run without --transcribe to test alignment.")


def load_cached():
    """Load cached segments and silences."""
    if not SEGMENTS_FILE.exists():
        print(f"❌ No cached segments found at {SEGMENTS_FILE}")
        print("   Run with --transcribe first")
        sys.exit(1)
    
    # Load segments
    with open(SEGMENTS_FILE, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)
    
    segments = []
    for seg in raw_segments:
        seg_type = SegmentType(seg.get("type", "ayah"))
        segments.append(Segment(
            id=seg["id"],
            surah_id=seg["surah_id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            type=seg_type,
        ))
    
    # Load silences
    silences = []
    if SILENCES_FILE.exists():
        with open(SILENCES_FILE, "r") as f:
            silences = json.load(f)
    
    return segments, silences


def test_alignment():
    """Test alignment using cached transcription."""
    print("=" * 60)
    print("🔗 TESTING ALIGNMENT FOR SURAH 002")
    print("=" * 60)
    
    # Load cached data
    print("\n📂 Loading cached transcription...")
    segments, silences = load_cached()
    print(f"   Loaded {len(segments)} segments, {len(silences)} silences")
    
    # Load ayahs
    ayahs = load_surah_ayahs(SURAH_ID)
    print(f"   Loaded {len(ayahs)} reference ayahs")
    
    # Align
    print("\n🔗 Aligning...")
    start = time.time()
    results = align_segments(segments, ayahs, silences_ms=silences)
    align_time = time.time() - start
    print(f"   Aligned {len(results)}/{len(ayahs)} ayahs in {align_time:.2f}s")
    
    # Calculate stats
    if results:
        similarities = [r.similarity_score for r in results]
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        low_sim_count = sum(1 for s in similarities if s < 0.7)
    else:
        avg_sim = min_sim = 0
        low_sim_count = 0
    
    print(f"\n📊 STATS:")
    print(f"   Aligned: {len(results)}/{len(ayahs)} ({len(results)/len(ayahs)*100:.1f}%)")
    print(f"   Avg similarity: {avg_sim:.3f}")
    print(f"   Min similarity: {min_sim:.3f}")
    print(f"   Low similarity (<0.7): {low_sim_count}")
    
    # Find missing ayahs
    aligned_nums = {r.ayah.ayah_number for r in results}
    missing = [i for i in range(1, len(ayahs) + 1) if i not in aligned_nums]
    
    if missing:
        # Group consecutive
        ranges = []
        start_n = missing[0]
        prev = missing[0]
        for n in missing[1:]:
            if n == prev + 1:
                prev = n
            else:
                ranges.append((start_n, prev))
                start_n = n
                prev = n
        ranges.append((start_n, prev))
        
        print(f"\n❌ MISSING AYAHS ({len(missing)} total):")
        for s, e in ranges:
            if s == e:
                print(f"   Ayah {s}")
            else:
                print(f"   Ayahs {s}-{e} ({e-s+1} ayahs)")
    
    # Show problem areas (low similarity)
    print(f"\n⚠️  LOW SIMILARITY AYAHS (< 0.7):")
    for r in results:
        if r.similarity_score < 0.7:
            duration = r.end_time - r.start_time
            print(f"   Ayah {r.ayah.ayah_number}: sim={r.similarity_score:.3f}, duration={duration:.1f}s")
    
    # Show last few aligned
    print(f"\n📝 LAST 5 ALIGNED AYAHS:")
    for r in results[-5:]:
        duration = r.end_time - r.start_time
        print(f"   Ayah {r.ayah.ayah_number}: sim={r.similarity_score:.3f}, duration={duration:.1f}s")
        print(f"      Time: {r.start_time:.1f}s - {r.end_time:.1f}s")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    if "--transcribe" in sys.argv:
        transcribe_and_save()
    else:
        test_alignment()
