#!/usr/bin/env python3
"""
Similarity Score Analysis for Munajjam Alignment Output

Analyzes alignment quality across all surahs by examining similarity scores
and identifying areas for algorithm improvement.

Usage:
    python analyze_similarity.py [--output-dir OUTPUT_DIR] [--threshold THRESHOLD]

Example:
    python analyze_similarity.py --output-dir output --threshold 0.9
"""

import json
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class AyahScore:
    """Represents a single ayah's similarity score with metadata."""
    surah_id: int
    surah_name: str
    ayah_number: int
    similarity: float
    text: str = ""


@dataclass
class SurahStats:
    """Statistics for a single surah."""
    surah_id: int
    surah_name: str
    total_ayahs: int
    aligned_ayahs: int
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    below_90_count: int
    below_80_count: int
    below_70_count: int


def load_output_files(output_dir: Path) -> list[dict]:
    """Load all surah JSON files from the output directory."""
    files = sorted(output_dir.glob("surah_*.json"))
    data = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data.append(json.load(fp))
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    return data


def extract_all_scores(surahs: list[dict]) -> list[AyahScore]:
    """Extract all ayah similarity scores from loaded surah data."""
    scores = []
    for surah in surahs:
        surah_id = surah.get("surah_id", 0)
        surah_name = surah.get("surah_name", "Unknown")
        for ayah in surah.get("ayahs", []):
            scores.append(AyahScore(
                surah_id=surah_id,
                surah_name=surah_name,
                ayah_number=ayah.get("ayah_number", 0),
                similarity=ayah.get("similarity", 0.0),
                text=ayah.get("text", "")[:50]  # Truncate for display
            ))
    return scores


def compute_surah_stats(surahs: list[dict]) -> list[SurahStats]:
    """Compute per-surah statistics."""
    stats = []
    for surah in surahs:
        similarities = [a.get("similarity", 0.0) for a in surah.get("ayahs", [])]
        if not similarities:
            continue
        stats.append(SurahStats(
            surah_id=surah.get("surah_id", 0),
            surah_name=surah.get("surah_name", "Unknown"),
            total_ayahs=surah.get("total_ayahs", len(similarities)),
            aligned_ayahs=surah.get("aligned_ayahs", len(similarities)),
            avg_similarity=statistics.mean(similarities),
            min_similarity=min(similarities),
            max_similarity=max(similarities),
            below_90_count=sum(1 for s in similarities if s < 0.9),
            below_80_count=sum(1 for s in similarities if s < 0.8),
            below_70_count=sum(1 for s in similarities if s < 0.7),
        ))
    return stats


def compute_percentiles(scores: list[float]) -> dict[str, float]:
    """Compute key percentiles for the score distribution."""
    if not scores:
        return {}
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    
    def percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_scores[idx]
    
    return {
        "1st": percentile(0.01),
        "5th": percentile(0.05),
        "10th": percentile(0.10),
        "25th": percentile(0.25),
        "50th (median)": percentile(0.50),
        "75th": percentile(0.75),
        "90th": percentile(0.90),
        "95th": percentile(0.95),
        "99th": percentile(0.99),
    }


def compute_threshold_breakdown(scores: list[float]) -> dict[str, tuple[int, float]]:
    """Count ayahs in each similarity threshold range."""
    n = len(scores)
    if n == 0:
        return {}
    
    ranges = {
        "Below 50% (critical)": (0.0, 0.5),
        "50-70% (poor)": (0.5, 0.7),
        "70-80% (needs improvement)": (0.7, 0.8),
        "80-90% (acceptable)": (0.8, 0.9),
        "90-95% (good)": (0.9, 0.95),
        "95-100% (excellent)": (0.95, 1.01),  # 1.01 to include 1.0
    }
    
    breakdown = {}
    for label, (low, high) in ranges.items():
        count = sum(1 for s in scores if low <= s < high)
        pct = (count / n) * 100
        breakdown[label] = (count, pct)
    
    return breakdown


def print_header(title: str):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_summary_statistics(scores: list[AyahScore]):
    """Print overall summary statistics."""
    print_header("SUMMARY STATISTICS")
    
    all_sims = [s.similarity for s in scores]
    
    print(f"\n  Total Ayahs Analyzed: {len(scores):,}")
    print(f"  Mean Similarity:      {statistics.mean(all_sims):.4f} ({statistics.mean(all_sims)*100:.2f}%)")
    print(f"  Median Similarity:    {statistics.median(all_sims):.4f} ({statistics.median(all_sims)*100:.2f}%)")
    print(f"  Std Deviation:        {statistics.stdev(all_sims):.4f}")
    
    min_score = min(scores, key=lambda x: x.similarity)
    max_score = max(scores, key=lambda x: x.similarity)
    
    print(f"\n  Minimum Score: {min_score.similarity:.4f}")
    print(f"    -> Surah {min_score.surah_id} ({min_score.surah_name}), Ayah {min_score.ayah_number}")
    print(f"  Maximum Score: {max_score.similarity:.4f}")
    print(f"    -> Surah {max_score.surah_id} ({max_score.surah_name}), Ayah {max_score.ayah_number}")


def print_threshold_breakdown(scores: list[float]):
    """Print the threshold breakdown table."""
    print_header("THRESHOLD BREAKDOWN")
    
    breakdown = compute_threshold_breakdown(scores)
    
    print(f"\n  {'Range':<30} {'Count':>10} {'Percentage':>12}")
    print("  " + "-" * 54)
    
    for label, (count, pct) in breakdown.items():
        bar = "█" * int(pct / 2)  # Scale bar to ~50 chars max
        print(f"  {label:<30} {count:>10,} {pct:>10.2f}%  {bar}")


def print_percentile_distribution(scores: list[float]):
    """Print the percentile distribution."""
    print_header("PERCENTILE DISTRIBUTION")
    
    percentiles = compute_percentiles(scores)
    
    print(f"\n  {'Percentile':<20} {'Score':>12} {'As Percentage':>15}")
    print("  " + "-" * 48)
    
    for label, value in percentiles.items():
        print(f"  {label:<20} {value:>12.4f} {value*100:>14.2f}%")


def print_worst_surahs(surah_stats: list[SurahStats], top_n: int = 10):
    """Print the surahs with lowest average similarity."""
    print_header(f"TOP {top_n} SURAHS WITH LOWEST AVERAGE SIMILARITY")
    
    sorted_stats = sorted(surah_stats, key=lambda x: x.avg_similarity)[:top_n]
    
    print(f"\n  {'#':<4} {'Surah':<25} {'Avg Sim':>10} {'<90%':>8} {'<80%':>8} {'<70%':>8}")
    print("  " + "-" * 66)
    
    for i, s in enumerate(sorted_stats, 1):
        name = f"{s.surah_id}. {s.surah_name}"[:24]
        print(f"  {i:<4} {name:<25} {s.avg_similarity:>10.4f} {s.below_90_count:>8} {s.below_80_count:>8} {s.below_70_count:>8}")


def print_surahs_with_most_low_scores(surah_stats: list[SurahStats], top_n: int = 10):
    """Print surahs with the most ayahs below 90%."""
    print_header(f"TOP {top_n} SURAHS WITH MOST AYAHS BELOW 90%")
    
    sorted_stats = sorted(surah_stats, key=lambda x: x.below_90_count, reverse=True)[:top_n]
    
    print(f"\n  {'#':<4} {'Surah':<25} {'<90% Count':>12} {'Total':>8} {'Ratio':>10}")
    print("  " + "-" * 62)
    
    for i, s in enumerate(sorted_stats, 1):
        name = f"{s.surah_id}. {s.surah_name}"[:24]
        ratio = (s.below_90_count / s.total_ayahs) * 100 if s.total_ayahs > 0 else 0
        print(f"  {i:<4} {name:<25} {s.below_90_count:>12} {s.total_ayahs:>8} {ratio:>9.1f}%")


def print_low_scoring_ayahs(scores: list[AyahScore], threshold: float = 0.9):
    """Print all ayahs below the threshold."""
    low_scores = [s for s in scores if s.similarity < threshold]
    low_scores.sort(key=lambda x: x.similarity)
    
    print_header(f"ALL AYAHS BELOW {threshold*100:.0f}% SIMILARITY ({len(low_scores):,} total)")
    
    if not low_scores:
        print("\n  No ayahs below threshold! Excellent alignment.")
        return
    
    print(f"\n  {'Surah':<20} {'Ayah':>6} {'Score':>10} {'Text Preview':<30}")
    print("  " + "-" * 70)
    
    for s in low_scores:
        name = s.surah_name[:18]
        text_preview = s.text[:28] + "..." if len(s.text) > 28 else s.text
        print(f"  {name:<20} {s.ayah_number:>6} {s.similarity:>10.4f} {text_preview:<30}")


def print_actionable_insights(scores: list[AyahScore], surah_stats: list[SurahStats]):
    """Print actionable insights for algorithm improvement."""
    print_header("ACTIONABLE INSIGHTS FOR ALGORITHM IMPROVEMENT")
    
    all_sims = [s.similarity for s in scores]
    below_90 = sum(1 for s in all_sims if s < 0.9)
    below_80 = sum(1 for s in all_sims if s < 0.8)
    below_70 = sum(1 for s in all_sims if s < 0.7)
    
    print(f"""
  Current State:
  - {below_90:,} ayahs ({below_90/len(all_sims)*100:.1f}%) below 90% similarity
  - {below_80:,} ayahs ({below_80/len(all_sims)*100:.1f}%) below 80% similarity
  - {below_70:,} ayahs ({below_70/len(all_sims)*100:.1f}%) below 70% similarity (critical)

  Priority Areas:
  1. Focus on surahs with highest count of low-scoring ayahs
  2. Investigate ayahs below 70% for potential transcription issues
  3. Review segment merging logic for ayahs in 70-80% range
""")
    
    # Find patterns in low-scoring ayahs
    low_scores = [s for s in scores if s.similarity < 0.8]
    if low_scores:
        # Group by surah
        surah_counts = {}
        for s in low_scores:
            key = (s.surah_id, s.surah_name)
            surah_counts[key] = surah_counts.get(key, 0) + 1
        
        top_problematic = sorted(surah_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("  Top 5 Problematic Surahs (ayahs < 80%):")
        for (sid, sname), count in top_problematic:
            print(f"    - Surah {sid} ({sname}): {count} ayahs")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze similarity scores from Munajjam alignment output"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Directory containing surah JSON files (default: output)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.9,
        help="Threshold for 'low' similarity scores (default: 0.9)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' not found.")
        return 1
    
    print("\n" + "=" * 70)
    print("  MUNAJJAM SIMILARITY SCORE ANALYSIS")
    print("=" * 70)
    print(f"\n  Analyzing: {output_dir.absolute()}")
    print(f"  Threshold: {args.threshold * 100:.0f}%")
    
    # Load data
    surahs = load_output_files(output_dir)
    print(f"  Loaded: {len(surahs)} surah files")
    
    if not surahs:
        print("Error: No surah files found.")
        return 1
    
    # Extract scores
    all_scores = extract_all_scores(surahs)
    surah_stats = compute_surah_stats(surahs)
    
    if not all_scores:
        print("Error: No ayah scores found.")
        return 1
    
    all_sims = [s.similarity for s in all_scores]
    
    # Print reports
    print_summary_statistics(all_scores)
    print_threshold_breakdown(all_sims)
    print_percentile_distribution(all_sims)
    print_worst_surahs(surah_stats)
    print_surahs_with_most_low_scores(surah_stats)
    print_low_scoring_ayahs(all_scores, threshold=args.threshold)
    print_actionable_insights(all_scores, surah_stats)
    
    print("\n" + "=" * 70)
    print("  Analysis Complete")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
