#!/usr/bin/env python3
"""
Test DP alignment for Surah 002 using cached transcription.

Usage:
    python test_dp_surah002.py --transcribe  # Transcribe and cache (run once)
    python test_dp_surah002.py               # Compare aligners using cache
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import align_segments
from munajjam.core.aligner_dp import align_segments_dp_with_constraints
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType, Ayah

CACHE_DIR = Path("cache")
SEGMENTS_FILE = CACHE_DIR / "surah_002_segments.json"
SILENCES_FILE = CACHE_DIR / "surah_002_silences.json"
AUDIO_PATH = "Quran/badr_alturki_audio/002.wav"


def transcribe_and_cache():
    """Transcribe Surah 002 and cache results using faster-whisper."""
    print("=" * 60)
    print("🎤 TRANSCRIBING SURAH 002 (Al-Baqarah)")
    print("   Using faster-whisper (Quran-tuned)")
    print("=" * 60)
    
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Silences
    print("\n🔇 Detecting silences...")
    start = time.time()
    silences = detect_silences(AUDIO_PATH)
    print(f"   ✓ Found {len(silences)} silence periods in {time.time()-start:.2f}s")
    
    with open(SILENCES_FILE, "w") as f:
        json.dump(silences, f)
    
    # Transcribe with faster-whisper
    print("\n🎤 Transcribing segments...")
    start = time.time()
    
    # Use faster-whisper with Quran-tuned model
    transcriber = WhisperTranscriber(
        model_id="OdyAsh/faster-whisper-base-ar-quran",
        model_type="faster-whisper"
    )
    transcriber.load()
    
    # Progress callback like batch_process.py
    def progress_callback(current: int, total: int, text: str):
        percent = (current / total) * 100
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        display_text = text[:35] + "..." if len(text) > 35 else text
        print(f"\r   [{bar}] {current}/{total} ({percent:.0f}%) {display_text}", end="", flush=True)
    
    segments = transcriber.transcribe(AUDIO_PATH, progress_callback=progress_callback)
    print()  # New line after progress
    transcriber.unload()
    
    elapsed = time.time() - start
    print(f"   ✓ Transcribed {len(segments)} segments in {elapsed:.1f}s")
    
    # Save
    data = [
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
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Cached to {SEGMENTS_FILE}")
    print("   Now run without --transcribe to compare aligners")


def load_cache():
    """Load cached segments and silences."""
    if not SEGMENTS_FILE.exists():
        print(f"❌ No cache found. Run with --transcribe first.")
        sys.exit(1)
    
    with open(SEGMENTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    segments = [
        Segment(
            id=s["id"],
            surah_id=s["surah_id"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            type=SegmentType(s.get("type", "ayah")),
        )
        for s in raw
    ]
    
    silences = []
    if SILENCES_FILE.exists():
        with open(SILENCES_FILE, "r") as f:
            silences = json.load(f)
    
    return segments, silences


def compare_aligners():
    """Compare greedy vs DP alignment for Surah 002."""
    print("=" * 60)
    print("🧪 COMPARING ALIGNERS FOR SURAH 002 (AL-BAQARAH)")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading cached transcription...")
    segments, silences = load_cache()
    print(f"   {len(segments)} segments, {len(silences)} silences")
    
    ayahs = load_surah_ayahs(2)
    print(f"   {len(ayahs)} reference ayahs")
    
    # Greedy
    print("\n" + "=" * 60)
    print("🔗 GREEDY ALIGNMENT")
    print("=" * 60)
    
    start = time.time()
    greedy_results = align_segments(segments, ayahs, silences_ms=silences)
    greedy_time = time.time() - start
    
    print(f"   Aligned: {len(greedy_results)}/{len(ayahs)} ayahs")
    if greedy_results:
        avg_sim = sum(r.similarity_score for r in greedy_results) / len(greedy_results)
        min_sim = min(r.similarity_score for r in greedy_results)
        print(f"   Avg similarity: {avg_sim:.3f}")
        print(f"   Min similarity: {min_sim:.3f}")
    print(f"   Time: {greedy_time:.2f}s")
    
    # DP
    print("\n" + "=" * 60)
    print("🧠 DP ALIGNMENT")
    print("=" * 60)
    
    start = time.time()
    
    def on_progress(current, total):
        pct = current / total * 100
        print(f"   Processing: {current}/{total} ({pct:.0f}%)", end='\r')
    
    dp_results = align_segments_dp_with_constraints(
        segments, ayahs,
        silences_ms=silences,
        max_segments_per_ayah=15,
        on_progress=on_progress,
    )
    dp_time = time.time() - start
    print()
    
    print(f"   Aligned: {len(dp_results)}/{len(ayahs)} ayahs")
    if dp_results:
        avg_sim = sum(r.similarity_score for r in dp_results) / len(dp_results)
        min_sim = min(r.similarity_score for r in dp_results)
        print(f"   Avg similarity: {avg_sim:.3f}")
        print(f"   Min similarity: {min_sim:.3f}")
    print(f"   Time: {dp_time:.2f}s")
    
    # Comparison
    print("\n" + "=" * 60)
    print("📊 COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Greedy':<15} {'DP':<15}")
    print("-" * 55)
    print(f"{'Ayahs aligned':<25} {len(greedy_results):<15} {len(dp_results):<15}")
    
    if greedy_results and dp_results:
        g_avg = sum(r.similarity_score for r in greedy_results) / len(greedy_results)
        d_avg = sum(r.similarity_score for r in dp_results) / len(dp_results)
        print(f"{'Avg similarity':<25} {g_avg:<15.3f} {d_avg:<15.3f}")
        
        g_min = min(r.similarity_score for r in greedy_results)
        d_min = min(r.similarity_score for r in dp_results)
        print(f"{'Min similarity':<25} {g_min:<15.3f} {d_min:<15.3f}")
        
        g_low = sum(1 for r in greedy_results if r.similarity_score < 0.7)
        d_low = sum(1 for r in dp_results if r.similarity_score < 0.7)
        print(f"{'Low sim (<0.7) count':<25} {g_low:<15} {d_low:<15}")
    
    print(f"{'Time (seconds)':<25} {greedy_time:<15.2f} {dp_time:<15.2f}")
    
    # Find biggest improvements
    print("\n" + "=" * 60)
    print("🎯 BIGGEST DIFFERENCES (Top 10)")
    print("=" * 60)
    
    greedy_dict = {r.ayah.ayah_number: r for r in greedy_results}
    dp_dict = {r.ayah.ayah_number: r for r in dp_results}
    
    differences = []
    for num in set(greedy_dict.keys()) | set(dp_dict.keys()):
        g = greedy_dict.get(num)
        d = dp_dict.get(num)
        
        if g and d:
            diff = d.similarity_score - g.similarity_score
            differences.append((num, g.similarity_score, d.similarity_score, diff))
        elif g:
            differences.append((num, g.similarity_score, None, -1))
        elif d:
            differences.append((num, None, d.similarity_score, 1))
    
    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x[3] if x[3] != -1 and x[3] != 1 else 0), reverse=True)
    
    print(f"\n{'Ayah':<6} {'Greedy':<10} {'DP':<10} {'Diff':<10} {'Note'}")
    print("-" * 50)
    
    for num, g_sim, d_sim, diff in differences[:10]:
        g_str = f"{g_sim:.3f}" if g_sim else "---"
        d_str = f"{d_sim:.3f}" if d_sim else "---"
        diff_str = f"{diff:+.3f}" if diff not in (-1, 1) else "---"
        
        note = ""
        if diff > 0.1:
            note = "✅ DP better"
        elif diff < -0.1:
            note = "⚠️ Greedy better"
        elif g_sim is None:
            note = "✨ New in DP"
        elif d_sim is None:
            note = "❌ Missing in DP"
        
        print(f"{num:<6} {g_str:<10} {d_str:<10} {diff_str:<10} {note}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if "--transcribe" in sys.argv:
        transcribe_and_cache()
    else:
        compare_aligners()
