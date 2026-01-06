#!/usr/bin/env python3
"""
Deep Investigation Script for Low-Scoring Ayahs

Analyzes a specific surah to understand why certain ayahs have low similarity scores.
Identifies patterns and potential causes for algorithm improvement.

Usage:
    python investigate_surah.py --surah 9
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse

# Add munajjam to path
sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.core.arabic import normalize_arabic
from munajjam.core.matcher import similarity


@dataclass
class AyahAnalysis:
    """Detailed analysis of a single ayah."""
    ayah_number: int
    similarity: float
    start: float
    end: float
    duration: float
    text: str
    word_count: int
    words_per_second: float
    gap_before: float  # Gap from previous ayah
    gap_after: float   # Gap to next ayah
    prev_similarity: Optional[float]
    next_similarity: Optional[float]
    issues: list[str]


def load_surah(output_dir: Path, surah_id: int) -> dict:
    """Load a surah JSON file."""
    file_path = output_dir / f"surah_{surah_id:03d}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Surah file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_ayah(
    ayah: dict,
    prev_ayah: Optional[dict],
    next_ayah: Optional[dict]
) -> AyahAnalysis:
    """Perform deep analysis on a single ayah."""
    start = ayah["start"]
    end = ayah["end"]
    duration = end - start
    text = ayah["text"]
    
    # Count words (normalized)
    word_count = len(normalize_arabic(text).split())
    words_per_second = word_count / duration if duration > 0 else 0
    
    # Calculate gaps
    gap_before = start - prev_ayah["end"] if prev_ayah else 0
    gap_after = next_ayah["start"] - end if next_ayah else 0
    
    # Collect issues
    issues = []
    
    # Issue 1: Very short duration for long text
    if word_count > 10 and duration < 5:
        issues.append(f"SHORT_DURATION: {word_count} words in {duration:.1f}s")
    
    # Issue 2: Very long duration for short text
    if word_count < 5 and duration > 20:
        issues.append(f"LONG_DURATION: {word_count} words in {duration:.1f}s")
    
    # Issue 3: Abnormal reading speed
    # Normal Quran recitation: ~2-4 words/second
    if words_per_second < 1.0:
        issues.append(f"SLOW_PACE: {words_per_second:.2f} words/sec")
    elif words_per_second > 6.0:
        issues.append(f"FAST_PACE: {words_per_second:.2f} words/sec")
    
    # Issue 4: Large gap before (might indicate missed content)
    if gap_before > 3.0:
        issues.append(f"GAP_BEFORE: {gap_before:.1f}s gap")
    
    # Issue 5: Negative gap (overlap - shouldn't happen)
    if gap_before < -0.5:
        issues.append(f"OVERLAP_BEFORE: {abs(gap_before):.1f}s overlap")
    
    # Issue 6: Cascade pattern - previous and current both low
    prev_sim = prev_ayah["similarity"] if prev_ayah else None
    if prev_sim and prev_sim < 0.8 and ayah["similarity"] < 0.8:
        issues.append("CASCADE: Previous ayah also low")
    
    # Issue 7: Very low similarity might indicate wrong alignment
    if ayah["similarity"] < 0.5:
        issues.append("CRITICAL: Likely misaligned")
    
    return AyahAnalysis(
        ayah_number=ayah["ayah_number"],
        similarity=ayah["similarity"],
        start=start,
        end=end,
        duration=duration,
        text=text[:80] + "..." if len(text) > 80 else text,
        word_count=word_count,
        words_per_second=words_per_second,
        gap_before=gap_before,
        gap_after=gap_after,
        prev_similarity=prev_sim,
        next_similarity=next_ayah["similarity"] if next_ayah else None,
        issues=issues,
    )


def identify_patterns(analyses: list[AyahAnalysis]) -> dict[str, list[int]]:
    """Identify common patterns across low-scoring ayahs."""
    patterns = {
        "cascade_sequences": [],  # Consecutive low scores
        "isolated_failures": [],  # Single low score surrounded by good
        "gap_issues": [],         # Large gaps before
        "duration_mismatch": [],  # Duration doesn't match word count
        "critical_failures": [],  # Below 50%
    }
    
    # Find cascade sequences
    i = 0
    while i < len(analyses):
        if analyses[i].similarity < 0.8:
            # Find sequence length
            j = i
            while j < len(analyses) and analyses[j].similarity < 0.8:
                j += 1
            seq_len = j - i
            
            if seq_len >= 2:
                patterns["cascade_sequences"].extend([a.ayah_number for a in analyses[i:j]])
            elif seq_len == 1:
                patterns["isolated_failures"].append(analyses[i].ayah_number)
            
            i = j
        else:
            i += 1
    
    # Other patterns
    for a in analyses:
        if a.similarity < 0.5:
            patterns["critical_failures"].append(a.ayah_number)
        if a.gap_before > 3.0 and a.similarity < 0.8:
            patterns["gap_issues"].append(a.ayah_number)
        if ("SHORT_DURATION" in " ".join(a.issues) or "LONG_DURATION" in " ".join(a.issues)):
            patterns["duration_mismatch"].append(a.ayah_number)
    
    return patterns


def print_detailed_analysis(surah_data: dict, threshold: float = 0.8):
    """Print detailed analysis of low-scoring ayahs."""
    ayahs = surah_data["ayahs"]
    surah_name = surah_data["surah_name"]
    surah_id = surah_data["surah_id"]
    
    print("\n" + "=" * 80)
    print(f"  DEEP INVESTIGATION: Surah {surah_id} ({surah_name})")
    print("=" * 80)
    
    # Analyze all ayahs
    analyses = []
    for i, ayah in enumerate(ayahs):
        prev_ayah = ayahs[i - 1] if i > 0 else None
        next_ayah = ayahs[i + 1] if i < len(ayahs) - 1 else None
        analysis = analyze_ayah(ayah, prev_ayah, next_ayah)
        analyses.append(analysis)
    
    # Filter to low-scoring
    low_scoring = [a for a in analyses if a.similarity < threshold]
    
    print(f"\n  Total Ayahs: {len(ayahs)}")
    print(f"  Low Scoring (<{threshold*100:.0f}%): {len(low_scoring)} ({len(low_scoring)/len(ayahs)*100:.1f}%)")
    
    # Identify patterns
    patterns = identify_patterns(analyses)
    
    print("\n" + "-" * 80)
    print("  PATTERN ANALYSIS")
    print("-" * 80)
    
    print(f"\n  Cascade Sequences (consecutive failures): {len(patterns['cascade_sequences'])} ayahs")
    if patterns['cascade_sequences']:
        # Group into sequences
        seqs = []
        current_seq = [patterns['cascade_sequences'][0]]
        for i in range(1, len(patterns['cascade_sequences'])):
            if patterns['cascade_sequences'][i] == patterns['cascade_sequences'][i-1] + 1:
                current_seq.append(patterns['cascade_sequences'][i])
            else:
                seqs.append(current_seq)
                current_seq = [patterns['cascade_sequences'][i]]
        seqs.append(current_seq)
        
        for seq in seqs:
            print(f"    Ayahs {seq[0]}-{seq[-1]}: {len(seq)} consecutive failures")
    
    print(f"\n  Isolated Failures: {len(patterns['isolated_failures'])} ayahs")
    if patterns['isolated_failures']:
        print(f"    Ayahs: {patterns['isolated_failures'][:10]}...")
    
    print(f"\n  Gap Issues: {len(patterns['gap_issues'])} ayahs")
    if patterns['gap_issues']:
        print(f"    Ayahs with large gaps before: {patterns['gap_issues'][:10]}...")
    
    print(f"\n  Duration Mismatches: {len(patterns['duration_mismatch'])} ayahs")
    
    print(f"\n  Critical Failures (<50%): {len(patterns['critical_failures'])} ayahs")
    if patterns['critical_failures']:
        print(f"    Ayahs: {patterns['critical_failures']}")
    
    # Detailed view of worst cases
    print("\n" + "-" * 80)
    print("  WORST CASES DETAIL (sorted by similarity)")
    print("-" * 80)
    
    worst = sorted(low_scoring, key=lambda x: x.similarity)[:20]
    
    for a in worst:
        print(f"\n  Ayah {a.ayah_number}: {a.similarity*100:.1f}%")
        print(f"    Duration: {a.duration:.1f}s ({a.start:.1f}s - {a.end:.1f}s)")
        print(f"    Words: {a.word_count} ({a.words_per_second:.2f} words/sec)")
        print(f"    Gap before: {a.gap_before:.2f}s, Gap after: {a.gap_after:.2f}s")
        prev_str = f"{a.prev_similarity:.2f}" if a.prev_similarity is not None else "N/A"
        next_str = f"{a.next_similarity:.2f}" if a.next_similarity is not None else "N/A"
        print(f"    Context: prev={prev_str}, next={next_str}")
        if a.issues:
            print(f"    Issues: {', '.join(a.issues)}")
        print(f"    Text: {a.text}")
    
    # Likely causes summary
    print("\n" + "-" * 80)
    print("  LIKELY CAUSES SUMMARY")
    print("-" * 80)
    
    cause_counts = {
        "Cascade Effect": len(patterns['cascade_sequences']),
        "Large Gaps (missed content)": len(patterns['gap_issues']),
        "Duration Mismatch": len(patterns['duration_mismatch']),
        "Critical Misalignment": len(patterns['critical_failures']),
        "Isolated Failures": len(patterns['isolated_failures']),
    }
    
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            bar = "█" * (count * 2)
            print(f"    {cause:<30}: {count:>3} ayahs  {bar}")
    
    # Recommendations
    print("\n" + "-" * 80)
    print("  RECOMMENDATIONS FOR ALGORITHM IMPROVEMENT")
    print("-" * 80)
    
    if len(patterns['cascade_sequences']) > 5:
        print("""
  1. CASCADE EFFECT DETECTED
     - Multiple consecutive ayahs failing suggests a single misalignment
       is cascading through subsequent ayahs
     - Consider: Adding recovery mechanism in DP to backtrack on low scores
     - Consider: Using silence detection more aggressively for resyncing""")
    
    if len(patterns['gap_issues']) > 3:
        print("""
  2. GAP ISSUES DETECTED  
     - Large gaps before low-scoring ayahs suggest content being missed
     - Consider: Lowering segment merge threshold during long silences
     - Consider: Adding gap penalty to DP cost function""")
    
    if len(patterns['critical_failures']) > 0:
        print("""
  3. CRITICAL MISALIGNMENTS DETECTED
     - Some ayahs below 50% are likely completely wrong alignments
     - Consider: Adding sanity check that rejects alignments below 40%
     - Consider: Forcing re-alignment when similarity is critically low""")
    
    print("\n" + "=" * 80 + "\n")
    
    return patterns, analyses


def main():
    parser = argparse.ArgumentParser(description="Investigate low-scoring ayahs in a surah")
    parser.add_argument("--surah", "-s", type=int, default=9, help="Surah number to investigate")
    parser.add_argument("--output-dir", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.8, help="Similarity threshold")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    try:
        surah_data = load_surah(output_dir, args.surah)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print_detailed_analysis(surah_data, args.threshold)
    
    return 0


if __name__ == "__main__":
    exit(main())
