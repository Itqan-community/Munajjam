#!/usr/bin/env python3
"""
Reprocess alignment from cached transcriptions.

This script runs the alignment algorithm on cached segments and silences,
allowing fast iteration on the algorithm without re-running Whisper.

Usage:
    # Reprocess a single surah
    python reprocess_from_cache.py --surah 9
    
    # Reprocess and save to output folder
    python reprocess_from_cache.py --surah 9 --save
    
    # Compare with original output
    python reprocess_from_cache.py --surah 9 --compare
"""

import json
import sys
import argparse
from pathlib import Path

# Add munajjam to path
sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.core.aligner_dp import align_segments_dp_with_constraints, _find_cascade_sequences, apply_hybrid_fallback
from munajjam.core.matcher import similarity
from munajjam.data import load_surah_ayahs, get_surah_name
from munajjam.models import Segment, SegmentType


CACHE_FOLDER = Path("cache")
OUTPUT_FOLDER = Path("output")


def load_cached_segments(surah_id: int) -> list[Segment]:
    """Load cached segments."""
    segments_file = CACHE_FOLDER / f"surah_{surah_id:03d}_segments.json"
    if not segments_file.exists():
        raise FileNotFoundError(f"No cached segments for surah {surah_id}")
    
    with open(segments_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [
        Segment(
            id=item.get("id", i),
            surah_id=item.get("surah_id", surah_id),
            start=item["start"],
            end=item["end"],
            text=item["text"],
            type=SegmentType(item.get("type", "ayah")),
            confidence=item.get("confidence"),
        )
        for i, item in enumerate(data)
    ]


def load_cached_silences(surah_id: int) -> list[tuple[int, int]]:
    """Load cached silences."""
    silences_file = CACHE_FOLDER / f"surah_{surah_id:03d}_silences.json"
    if not silences_file.exists():
        return []
    
    with open(silences_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_original_output(surah_id: int) -> dict:
    """Load the original output JSON."""
    output_file = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    if not output_file.exists():
        return {}
    
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(surah_id: int, surah_name: str, results: list, total_ayahs: int):
    """Save alignment results to JSON file."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    similarities = [r.similarity_score for r in results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    output_data = {
        "surah_id": surah_id,
        "surah_name": surah_name,
        "reciter": "Badr Al-Turki",
        "total_ayahs": total_ayahs,
        "aligned_ayahs": len(results),
        "avg_similarity": round(avg_similarity, 3),
        "ayahs": [
            {
                "ayah_number": r.ayah.ayah_number,
                "start": r.start_time,
                "end": r.end_time,
                "text": r.ayah.text,
                "similarity": round(r.similarity_score, 3),
            }
            for r in results
        ],
    }
    
    output_path = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 Saved to {output_path}")


def compare_results(original: dict, new_results: list, threshold: float = 0.8):
    """Compare original and new alignment results."""
    if not original or "ayahs" not in original:
        print("\n  ⚠️  No original results to compare.")
        return
    
    orig_ayahs = original["ayahs"]
    
    if len(orig_ayahs) != len(new_results):
        print(f"\n  ⚠️  Length mismatch: original={len(orig_ayahs)}, new={len(new_results)}")
        return
    
    # Overall stats
    orig_avg = sum(a["similarity"] for a in orig_ayahs) / len(orig_ayahs)
    new_avg = sum(r.similarity_score for r in new_results) / len(new_results)
    
    orig_below = sum(1 for a in orig_ayahs if a["similarity"] < threshold)
    new_below = sum(1 for r in new_results if r.similarity_score < threshold)
    
    # Count cascades
    orig_cascades = []
    i = 0
    while i < len(orig_ayahs):
        if orig_ayahs[i]["similarity"] < 0.7:
            start = i
            while i < len(orig_ayahs) and orig_ayahs[i]["similarity"] < 0.7:
                i += 1
            if i - start >= 2:
                orig_cascades.append((start, i))
        else:
            i += 1
    
    new_cascades = _find_cascade_sequences(new_results, threshold=0.7, min_cascade_length=2)
    
    print(f"\n  {'='*60}")
    print(f"  COMPARISON: Original vs New")
    print(f"  {'='*60}")
    print(f"\n  {'Metric':<35} {'Original':>12} {'New':>12} {'Change':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Average Similarity':<35} {orig_avg*100:>11.2f}% {new_avg*100:>11.2f}% {(new_avg-orig_avg)*100:>+11.2f}%")
    print(f"  {'Ayahs Below ' + str(int(threshold*100)) + '%':<35} {orig_below:>12} {new_below:>12} {new_below-orig_below:>+12}")
    print(f"  {'Cascade Sequences (2+ consec)':<35} {len(orig_cascades):>12} {len(new_cascades):>12} {len(new_cascades)-len(orig_cascades):>+12}")
    
    # Show biggest improvements
    improvements = []
    degradations = []
    
    for i, (orig, new) in enumerate(zip(orig_ayahs, new_results)):
        change = new.similarity_score - orig["similarity"]
        if change > 0.03:
            improvements.append((i + 1, orig["similarity"], new.similarity_score, change))
        elif change < -0.03:
            degradations.append((i + 1, orig["similarity"], new.similarity_score, change))
    
    if improvements:
        print(f"\n  📈 IMPROVED AYAHS (top 10):")
        print(f"  {'Ayah':>8} {'Original':>12} {'New':>12} {'Change':>12}")
        print(f"  {'-'*50}")
        for ayah_num, orig_sim, new_sim, change in sorted(improvements, key=lambda x: -x[3])[:10]:
            print(f"  {ayah_num:>8} {orig_sim*100:>11.1f}% {new_sim*100:>11.1f}% {change*100:>+11.1f}%")
    
    if degradations:
        print(f"\n  📉 DEGRADED AYAHS (top 10):")
        print(f"  {'Ayah':>8} {'Original':>12} {'New':>12} {'Change':>12}")
        print(f"  {'-'*50}")
        for ayah_num, orig_sim, new_sim, change in sorted(degradations, key=lambda x: x[3])[:10]:
            print(f"  {ayah_num:>8} {orig_sim*100:>11.1f}% {new_sim*100:>11.1f}% {change*100:>+11.1f}%")
    
    print(f"\n  {'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Reprocess alignment from cache")
    parser.add_argument("--surah", "-s", type=int, required=True, help="Surah ID to process")
    parser.add_argument("--save", action="store_true", help="Save results to output folder")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare with original output")
    args = parser.parse_args()
    
    surah_id = args.surah
    surah_name = get_surah_name(surah_id)
    
    print(f"\n{'='*60}")
    print(f"  REPROCESSING: Surah {surah_id} - {surah_name}")
    print(f"{'='*60}")
    
    # Load cached data
    print(f"\n  📂 Loading cached data...")
    
    try:
        segments = load_cached_segments(surah_id)
        print(f"     Segments: {len(segments)}")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print(f"     Run: python cache_transcriptions.py --surah {surah_id}")
        return 1
    
    silences = load_cached_silences(surah_id)
    print(f"     Silences: {len(silences)}")
    
    # Load ayahs
    ayahs = load_surah_ayahs(surah_id)
    print(f"     Ayahs: {len(ayahs)}")
    
    # Run alignment
    print(f"\n  🔄 Running DP alignment with improvements...")
    
    def progress(current, total):
        if current % 20 == 0 or current == total:
            print(f"     Progress: {current}/{total}")
    
    results = align_segments_dp_with_constraints(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences,
        max_segments_per_ayah=8,
        on_progress=progress,
    )
    
    # Apply hybrid fallback: use original results for low-scoring ayahs
    original = load_original_output(surah_id)
    if original and "ayahs" in original:
        before_fallback = sum(1 for r in results if r.similarity_score < 0.8)
        results = apply_hybrid_fallback(results, original["ayahs"], threshold=0.8)
        after_fallback = sum(1 for r in results if r.similarity_score < 0.8)
        if before_fallback != after_fallback:
            print(f"\n  🔄 Hybrid fallback: improved {before_fallback - after_fallback} ayahs")
    
    print(f"\n  ✅ Aligned {len(results)}/{len(ayahs)} ayahs")
    
    # Calculate stats
    avg_sim = sum(r.similarity_score for r in results) / len(results) if results else 0
    below_80 = sum(1 for r in results if r.similarity_score < 0.8)
    below_70 = sum(1 for r in results if r.similarity_score < 0.7)
    
    cascades = _find_cascade_sequences(results, threshold=0.7, min_cascade_length=2)
    
    print(f"\n  📊 RESULTS:")
    print(f"     Average Similarity: {avg_sim*100:.2f}%")
    print(f"     Below 80%: {below_80} ayahs ({below_80/len(results)*100:.1f}%)")
    print(f"     Below 70%: {below_70} ayahs ({below_70/len(results)*100:.1f}%)")
    print(f"     Cascade Sequences: {len(cascades)}")
    
    # Compare with original
    if args.compare:
        original = load_original_output(surah_id)
        compare_results(original, results)
    
    # Save
    if args.save:
        save_output(surah_id, surah_name, results, len(ayahs))
    
    return 0


if __name__ == "__main__":
    exit(main())
