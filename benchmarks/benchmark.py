#!/usr/bin/env python3
"""
Munajjam Benchmark Harness

A comprehensive benchmarking tool for measuring Munajjam performance
across different alignment strategies and dataset sizes.

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --strategy hybrid  # Benchmark specific strategy
    python benchmark.py --output json      # Output format (json, markdown, both)
    python benchmark.py --compare-ground-truth  # Compare against ground truth data

Features:
- JSON output for programmatic access
- Markdown leaderboard generation
- Ground truth comparison
- Statistical analysis (mean, std, min, max, percentiles)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path for importing munajjam
sys.path.insert(0, str(Path(__file__).parent.parent / "munajjam"))

from munajjam.core import Aligner, AlignmentStrategy, align
from munajjam.data import load_surah_ayahs
from munajjam.models import AlignmentResult, Ayah, Segment, SegmentType


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    strategy: str
    dataset_size: int
    total_time_ms: float
    avg_time_per_ayah_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float
    avg_similarity: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class GroundTruthComparison:
    """Comparison between predicted and ground truth timings."""
    ayah_number: int
    predicted_start: float
    predicted_end: float
    ground_truth_start: float
    ground_truth_end: float
    start_error_ms: float
    end_error_ms: float
    duration_error_ms: float
    similarity_score: float


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    description: str
    timestamp: str
    version: str
    python_version: str
    results: list[BenchmarkResult]
    ground_truth_comparisons: list[GroundTruthComparison] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "version": self.version,
            "python_version": self.python_version,
            "results": [asdict(r) for r in self.results],
            "ground_truth_comparisons": [asdict(c) for c in self.ground_truth_comparisons],
        }


class BenchmarkHarness:
    """Main benchmark harness for Munajjam."""

    VERSION = "1.0.0"
    ITERATIONS = 10  # Number of iterations for statistical significance

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(__file__).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []
        self.ground_truth_comparisons: list[GroundTruthComparison] = []

    def _create_synthetic_segments(
        self, ayahs: list[Ayah], surah_id: int = 1
    ) -> list[Segment]:
        """Create synthetic segments that match the ayahs."""
        segments = []
        current_time = 0.0

        for i, ayah in enumerate(ayahs):
            # Estimate duration based on text length (rough approximation)
            duration = max(3.0, len(ayah.text) * 0.08)
            
            segment_type = SegmentType.AYAH
            if i == 0 and surah_id != 9:  # Add basmala for most surahs
                if "بسم" in ayah.text:
                    segment_type = SegmentType.BASMALA

            segment = Segment(
                id=i,
                surah_id=surah_id,
                start=current_time,
                end=current_time + duration,
                text=ayah.text[:50],  # Truncate for realistic transcription
                type=segment_type,
            )
            segments.append(segment)
            current_time += duration + 0.5  # Add gap between segments

        return segments

    def _create_ground_truth_timings(
        self, ayahs: list[Ayah]
    ) -> list[tuple[float, float]]:
        """Create synthetic ground truth timings for comparison."""
        timings = []
        current_time = 0.0

        for ayah in ayahs:
            duration = max(3.0, len(ayah.text) * 0.08)
            timings.append((current_time, current_time + duration))
            current_time += duration + 0.5

        return timings

    def run_alignment_benchmark(
        self,
        name: str,
        strategy: str,
        ayahs: list[Ayah],
        surah_id: int = 1,
        iterations: int | None = None,
    ) -> BenchmarkResult:
        """Run a single alignment benchmark."""
        iterations = iterations or self.ITERATIONS
        segments = self._create_synthetic_segments(ayahs, surah_id)
        
        times = []
        all_results = []
        
        for _ in range(iterations):
            aligner = Aligner(
                audio_path=f"test_surah_{surah_id}.wav",
                strategy=strategy,
                energy_snap=False,
                fix_drift=False,
                fix_overlaps=False,
            )
            
            start_time = time.perf_counter()
            try:
                results = aligner.align(segments, ayahs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                times.append(elapsed_ms)
                all_results.extend(results)
            except Exception as e:
                print(f"  Warning: Alignment failed: {e}")
                times.append(float('inf'))

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            raise RuntimeError(f"All {iterations} iterations failed for {name}")

        total_time = sum(valid_times)
        avg_time = statistics.mean(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        std_dev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0.0
        
        # Percentiles
        sorted_times = sorted(valid_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        p95 = sorted_times[min(p95_idx, len(sorted_times) - 1)]
        p99 = sorted_times[min(p99_idx, len(sorted_times) - 1)]

        # Success metrics
        success_rate = len(valid_times) / iterations
        avg_similarity = (
            statistics.mean([r.similarity_score for r in all_results])
            if all_results else 0.0
        )

        return BenchmarkResult(
            name=name,
            strategy=strategy,
            dataset_size=len(ayahs),
            total_time_ms=total_time,
            avg_time_per_ayah_ms=avg_time / len(ayahs) if ayahs else 0.0,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            success_rate=success_rate,
            avg_similarity=avg_similarity,
            metadata={
                "iterations": len(valid_times),
                "failed_iterations": iterations - len(valid_times),
            },
        )

    def run_ground_truth_comparison(
        self,
        ayahs: list[Ayah],
        ground_truth_timings: list[tuple[float, float]],
        strategy: str = "hybrid",
        surah_id: int = 1,
    ) -> list[GroundTruthComparison]:
        """Compare alignment results against ground truth timings."""
        segments = self._create_synthetic_segments(ayahs, surah_id)
        
        aligner = Aligner(
            audio_path=f"test_surah_{surah_id}.wav",
            strategy=strategy,
            energy_snap=False,
        )
        
        results = aligner.align(segments, ayahs)
        comparisons = []

        for i, (result, (gt_start, gt_end)) in enumerate(zip(results, ground_truth_timings)):
            start_error_ms = abs(result.start_time - gt_start) * 1000
            end_error_ms = abs(result.end_time - gt_end) * 1000
            pred_duration = result.end_time - result.start_time
            gt_duration = gt_end - gt_start
            duration_error_ms = abs(pred_duration - gt_duration) * 1000

            comparisons.append(GroundTruthComparison(
                ayah_number=result.ayah.ayah_number,
                predicted_start=result.start_time,
                predicted_end=result.end_time,
                ground_truth_start=gt_start,
                ground_truth_end=gt_end,
                start_error_ms=start_error_ms,
                end_error_ms=end_error_ms,
                duration_error_ms=duration_error_ms,
                similarity_score=result.similarity_score,
            ))

        return comparisons

    def run_all_benchmarks(
        self,
        strategies: list[str] | None = None,
        compare_ground_truth: bool = False,
    ) -> BenchmarkSuite:
        """Run the complete benchmark suite."""
        strategies = strategies or ["greedy", "dp", "hybrid"]
        
        print("=" * 70)
        print("Munajjam Benchmark Harness")
        print("=" * 70)
        print(f"Version: {self.VERSION}")
        print(f"Python: {sys.version}")
        print()

        # Test datasets of different sizes
        test_cases = [
            ("Surah Al-Fatiha (7 ayahs)", 1, 7),
            ("Surah Al-Ikhlas (4 ayahs)", 112, 4),
            ("Surah Al-Falaq (5 ayahs)", 113, 5),
            ("Surah An-Nas (6 ayahs)", 114, 6),
        ]

        for name, surah_id, expected_count in test_cases:
            print(f"\n📖 Loading {name}...")
            try:
                ayahs = load_surah_ayahs(surah_id)
                if len(ayahs) != expected_count:
                    print(f"  Warning: Expected {expected_count} ayahs, got {len(ayahs)}")
            except Exception as e:
                print(f"  Error loading surah {surah_id}: {e}")
                continue

            for strategy in strategies:
                print(f"  🔄 Benchmarking {strategy} strategy...")
                try:
                    result = self.run_alignment_benchmark(
                        name=name,
                        strategy=strategy,
                        ayahs=ayahs,
                        surah_id=surah_id,
                    )
                    self.results.append(result)
                    print(f"    ✓ Avg: {result.avg_time_per_ayah_ms:.2f}ms/ayah, "
                          f"Total: {result.avg_time_ms:.2f}ms, "
                          f"Similarity: {result.avg_similarity:.2%}")
                except Exception as e:
                    print(f"    ✗ Failed: {e}")

            # Ground truth comparison
            if compare_ground_truth:
                print(f"  📊 Running ground truth comparison...")
                try:
                    ground_truth = self._create_ground_truth_timings(ayahs)
                    comparisons = self.run_ground_truth_comparison(
                        ayahs=ayahs,
                        ground_truth_timings=ground_truth,
                        strategy="hybrid",
                        surah_id=surah_id,
                    )
                    self.ground_truth_comparisons.extend(comparisons)
                    avg_start_error = statistics.mean([c.start_error_ms for c in comparisons])
                    avg_end_error = statistics.mean([c.end_error_ms for c in comparisons])
                    print(f"    ✓ Avg start error: {avg_start_error:.1f}ms, "
                          f"Avg end error: {avg_end_error:.1f}ms")
                except Exception as e:
                    print(f"    ✗ Failed: {e}")

        suite = BenchmarkSuite(
            name="Munajjam Alignment Benchmarks",
            description="Performance benchmarks for Quran audio alignment strategies",
            timestamp=datetime.utcnow().isoformat(),
            version=self.VERSION,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            results=self.results,
            ground_truth_comparisons=self.ground_truth_comparisons,
        )

        return suite

    def save_json(self, suite: BenchmarkSuite, filename: str = "benchmark_results.json") -> Path:
        """Save benchmark results as JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(suite.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON results saved to: {output_path}")
        return output_path

    def generate_markdown_leaderboard(
        self, suite: BenchmarkSuite, filename: str = "LEADERBOARD.md"
    ) -> Path:
        """Generate Markdown leaderboard."""
        output_path = self.output_dir / filename

        md = []
        md.append("# Munajjam Benchmark Leaderboard\n")
        md.append(f"**Generated:** {suite.timestamp}\n")
        md.append(f"**Version:** {suite.version}\n")
        md.append(f"**Python:** {suite.python_version}\n")
        md.append("---\n")

        # Summary table
        md.append("\n## 📊 Performance Summary\n")
        md.append("| Benchmark | Strategy | Size | Avg/ayah | Total | P95 | P99 | Success | Similarity |")
        md.append("|-----------|----------|------|----------|-------|-----|-----|---------|------------|")

        for result in suite.results:
            md.append(
                f"| {result.name} | `{result.strategy}` | {result.dataset_size} | "
                f"{result.avg_time_per_ayah_ms:.2f}ms | {result.avg_time_ms:.2f}ms | "
                f"{result.p95_ms:.2f}ms | {result.p99_ms:.2f}ms | "
                f"{result.success_rate:.0%} | {result.avg_similarity:.1%} |"
            )

        # Strategy comparison
        md.append("\n## 🏆 Strategy Comparison\n")
        
        strategies = set(r.strategy for r in suite.results)
        for strategy in sorted(strategies):
            strategy_results = [r for r in suite.results if r.strategy == strategy]
            if strategy_results:
                avg_time = statistics.mean([r.avg_time_ms for r in strategy_results])
                avg_similarity = statistics.mean([r.avg_similarity for r in strategy_results])
                md.append(f"\n### `{strategy}` Strategy\n")
                md.append(f"- **Average Time:** {avg_time:.2f}ms\n")
                md.append(f"- **Average Similarity:** {avg_similarity:.1%}\n")
                md.append(f"- **Benchmarks:** {len(strategy_results)}\n")

        # Ground truth comparison
        if suite.ground_truth_comparisons:
            md.append("\n## 🎯 Ground Truth Accuracy\n")
            
            start_errors = [c.start_error_ms for c in suite.ground_truth_comparisons]
            end_errors = [c.end_error_ms for c in suite.ground_truth_comparisons]
            
            md.append(f"- **Mean Start Error:** {statistics.mean(start_errors):.1f}ms\n")
            md.append(f"- **Mean End Error:** {statistics.mean(end_errors):.1f}ms\n")
            md.append(f"- **Max Start Error:** {max(start_errors):.1f}ms\n")
            md.append(f"- **Max End Error:** {max(end_errors):.1f}ms\n")

            md.append("\n### Per-Ayah Details\n")
            md.append("| Ayah | Start Error | End Error | Duration Error | Similarity |")
            md.append("|------|-------------|-----------|----------------|------------|")
            
            for c in sorted(suite.ground_truth_comparisons, key=lambda x: x.ayah_number):
                md.append(
                    f"| {c.ayah_number} | {c.start_error_ms:.1f}ms | {c.end_error_ms:.1f}ms | "
                    f"{c.duration_error_ms:.1f}ms | {c.similarity_score:.1%} |"
                )

        # Methodology
        md.append("\n## 🧪 Methodology\n")
        md.append("\n### Benchmark Setup\n")
        md.append(f"- **Iterations per benchmark:** {self.ITERATIONS}\n")
        md.append("- **Timing method:** `time.perf_counter()`\n")
        md.append("- **Synthetic segments:** Created to match real ayah text\n")
        md.append("- **Strategies tested:** greedy, dp, hybrid\n")

        md.append("\n### Metrics\n")
        md.append("- **Avg/ayah:** Average processing time per ayah\n")
        md.append("- **P95/P99:** 95th and 99th percentile latencies\n")
        md.append("- **Success Rate:** Percentage of successful alignments\n")
        md.append("- **Similarity:** Text similarity between predicted and reference\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        print(f"💾 Markdown leaderboard saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Munajjam Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all benchmarks
  %(prog)s --strategy hybrid        # Benchmark only hybrid strategy
  %(prog)s --output both            # Generate both JSON and Markdown
  %(prog)s --compare-ground-truth   # Include ground truth comparison
  %(prog)s --iterations 20          # Run 20 iterations per benchmark
        """,
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "dp", "hybrid", "all"],
        default="all",
        help="Alignment strategy to benchmark (default: all)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--compare-ground-truth",
        action="store_true",
        help="Compare results against ground truth timings",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: benchmarks/)",
    )

    args = parser.parse_args()

    # Determine strategies to test
    if args.strategy == "all":
        strategies = ["greedy", "dp", "hybrid"]
    else:
        strategies = [args.strategy]

    # Run benchmarks
    harness = BenchmarkHarness(output_dir=args.output_dir)
    harness.ITERATIONS = args.iterations

    suite = harness.run_all_benchmarks(
        strategies=strategies,
        compare_ground_truth=args.compare_ground_truth,
    )

    # Save results
    if args.output in ("json", "both"):
        harness.save_json(suite)

    if args.output in ("markdown", "both"):
        harness.generate_markdown_leaderboard(suite)

    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
