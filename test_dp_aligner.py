#!/usr/bin/env python3
"""
Test script to compare Greedy vs DP alignment algorithms.

Usage:
    python test_dp_aligner.py <surah_id>
    python test_dp_aligner.py 114  # Quick test with short surah
    python test_dp_aligner.py 1    # Al-Fatiha
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import align_segments
from munajjam.core.aligner_dp import align_segments_dp, align_segments_dp_with_constraints
from munajjam.data import load_surah_ayahs, get_ayah_count
from munajjam.models import Segment, SegmentType, Ayah


def transcribe_surah(surah_id: int) -> tuple[list[Segment], list[tuple[int, int]]]:
    """Transcribe a surah using faster-whisper and return segments + silences."""
    audio_path = f"Quran/badr_alturki_audio/{surah_id:03d}.wav"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"🔇 Detecting silences...")
    silences = detect_silences(audio_path)
    print(f"   Found {len(silences)} silence periods")
    
    print(f"🎤 Transcribing (faster-whisper)...")
    
    # Use faster-whisper with Quran-tuned model
    transcriber = WhisperTranscriber(
        model_id="OdyAsh/faster-whisper-base-ar-quran",
        model_type="faster-whisper"
    )
    transcriber.load()
    
    # Progress callback
    def progress_callback(current: int, total: int, text: str):
        percent = (current / total) * 100
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        display_text = text[:30] + "..." if len(text) > 30 else text
        print(f"\r   [{bar}] {current}/{total} ({percent:.0f}%) {display_text}", end="", flush=True)
    
    segments = transcriber.transcribe(audio_path, progress_callback=progress_callback)
    print()  # New line after progress
    transcriber.unload()
    print(f"   Transcribed {len(segments)} segments")
    
    return segments, silences


def compare_aligners(segments: list[Segment], ayahs: list[Ayah], silences: list[tuple[int, int]]):
    """Compare greedy vs DP alignment."""
    
    print("\n" + "=" * 60)
    print("🔗 GREEDY ALIGNMENT (current)")
    print("=" * 60)
    
    start = time.time()
    greedy_results = align_segments(segments, ayahs, silences_ms=silences)
    greedy_time = time.time() - start
    
    print(f"   Aligned: {len(greedy_results)}/{len(ayahs)} ayahs")
    if greedy_results:
        avg_sim = sum(r.similarity_score for r in greedy_results) / len(greedy_results)
        print(f"   Avg similarity: {avg_sim:.3f}")
    print(f"   Time: {greedy_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("🧠 DP ALIGNMENT (new)")
    print("=" * 60)
    
    start = time.time()
    
    def on_progress(current, total):
        if current % 10 == 0 or current == total:
            print(f"   Processing ayah {current}/{total}...", end='\r')
    
    dp_results = align_segments_dp_with_constraints(
        segments, ayahs, 
        silences_ms=silences,
        max_segments_per_ayah=15,
        on_progress=on_progress
    )
    dp_time = time.time() - start
    print()  # New line after progress
    
    print(f"   Aligned: {len(dp_results)}/{len(ayahs)} ayahs")
    if dp_results:
        avg_sim = sum(r.similarity_score for r in dp_results) / len(dp_results)
        print(f"   Avg similarity: {avg_sim:.3f}")
    print(f"   Time: {dp_time:.2f}s")
    
    # Detailed comparison
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
    
    # Show ayah-by-ayah comparison
    print("\n" + "=" * 60)
    print("📝 AYAH-BY-AYAH COMPARISON (showing differences)")
    print("=" * 60)
    
    greedy_dict = {r.ayah.ayah_number: r for r in greedy_results}
    dp_dict = {r.ayah.ayah_number: r for r in dp_results}
    
    all_ayah_nums = sorted(set(greedy_dict.keys()) | set(dp_dict.keys()))
    
    differences = []
    for num in all_ayah_nums:
        g = greedy_dict.get(num)
        d = dp_dict.get(num)
        
        if g and d:
            sim_diff = d.similarity_score - g.similarity_score
            time_diff = abs(d.start_time - g.start_time)
            
            if abs(sim_diff) > 0.05 or time_diff > 1.0:
                differences.append((num, g, d, sim_diff, time_diff))
        elif g and not d:
            differences.append((num, g, None, None, None))
        elif d and not g:
            differences.append((num, None, d, None, None))
    
    if differences:
        print(f"\n{'Ayah':<6} {'Greedy Sim':<12} {'DP Sim':<12} {'Diff':<10} {'Note'}")
        print("-" * 55)
        
        for num, g, d, sim_diff, time_diff in differences[:20]:  # Show first 20
            g_sim = f"{g.similarity_score:.3f}" if g else "---"
            d_sim = f"{d.similarity_score:.3f}" if d else "---"
            diff_str = f"{sim_diff:+.3f}" if sim_diff is not None else "---"
            
            note = ""
            if sim_diff is not None:
                if sim_diff > 0.1:
                    note = "✅ DP better"
                elif sim_diff < -0.1:
                    note = "⚠️ Greedy better"
            elif g and not d:
                note = "❌ Missing in DP"
            elif d and not g:
                note = "✨ New in DP"
            
            print(f"{num:<6} {g_sim:<12} {d_sim:<12} {diff_str:<10} {note}")
        
        if len(differences) > 20:
            print(f"... and {len(differences) - 20} more differences")
    else:
        print("\n✅ No significant differences!")
    
    return greedy_results, dp_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_dp_aligner.py <surah_id>")
        print("\nRecommended for testing:")
        print("  python test_dp_aligner.py 114  # An-Nas (6 ayahs, ~30s)")
        print("  python test_dp_aligner.py 1    # Al-Fatiha (7 ayahs, ~40s)")
        print("  python test_dp_aligner.py 112  # Al-Ikhlas (4 ayahs, ~15s)")
        sys.exit(0)
    
    surah_id = int(sys.argv[1])
    
    print("=" * 60)
    print(f"🧪 COMPARING ALIGNERS FOR SURAH {surah_id}")
    print("=" * 60)
    
    print(f"\n📖 Surah {surah_id}: {get_ayah_count(surah_id)} ayahs")
    
    # Transcribe
    segments, silences = transcribe_surah(surah_id)
    
    # Load ayahs
    ayahs = load_surah_ayahs(surah_id)
    print(f"📚 Loaded {len(ayahs)} reference ayahs")
    
    # Compare
    compare_aligners(segments, ayahs, silences)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
