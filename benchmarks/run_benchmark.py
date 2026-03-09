"""
Benchmark harness for comparing alignment strategies against ground-truth data.

Runs all strategies (GREEDY, DP, HYBRID) against verified ground-truth
timestamps and produces JSON results + Markdown leaderboard.

Usage:
    python -m benchmarks.run_benchmark
    python benchmarks/run_benchmark.py
"""

import json
import time
from pathlib import Path

from munajjam.core import Aligner, AlignmentStrategy
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
RESULTS_DIR = Path(__file__).parent / "results"
STRATEGIES = [AlignmentStrategy.GREEDY, AlignmentStrategy.DP, AlignmentStrategy.HYBRID]


def load_ground_truth(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def create_mock_segments(ground_truth: dict) -> list[Segment]:
    """Create mock segments from ground-truth data (simulates transcriber output)."""
    segments = []
    for i, ayah in enumerate(ground_truth["ayahs"]):
        segments.append(
            Segment(
                id=i,
                surah_id=ground_truth["surah_id"],
                start=ayah["start_time"],
                end=ayah["end_time"],
                text=ayah["text"],
                type=SegmentType.AYAH,
            )
        )
    return segments


def compute_mae(results, ground_truth_ayahs) -> float:
    """Compute Mean Absolute Error of timestamps vs ground truth."""
    if not results:
        return float("inf")

    errors = []
    gt_by_num = {a["ayah_number"]: a for a in ground_truth_ayahs}

    for result in results:
        gt = gt_by_num.get(result.ayah.ayah_number)
        if gt is None:
            continue
        errors.append(abs(result.start_time - gt["start_time"]))
        errors.append(abs(result.end_time - gt["end_time"]))

    return sum(errors) / len(errors) if errors else float("inf")


def compute_metrics(results, ground_truth_ayahs) -> dict:
    """Compute all benchmark metrics for a set of alignment results."""
    if not results:
        return {
            "mae_seconds": float("inf"),
            "avg_similarity": 0.0,
            "high_confidence_pct": 0.0,
            "aligned_count": 0,
            "expected_count": len(ground_truth_ayahs),
        }

    mae = compute_mae(results, ground_truth_ayahs)
    avg_sim = sum(r.similarity_score for r in results) / len(results)
    high_conf = sum(1 for r in results if r.is_high_confidence) / len(results) * 100

    return {
        "mae_seconds": round(mae, 4),
        "avg_similarity": round(avg_sim, 4),
        "high_confidence_pct": round(high_conf, 2),
        "aligned_count": len(results),
        "expected_count": len(ground_truth_ayahs),
    }


def run_single_benchmark(
    strategy: AlignmentStrategy, segments: list[Segment], ayahs, ground_truth: dict
) -> dict:
    """Run a single strategy benchmark and return metrics."""
    aligner = Aligner(
        audio_path="benchmark_mock.wav",
        strategy=strategy,
        fix_drift=False,
        fix_overlaps=True,
        energy_snap=False,
    )

    start = time.perf_counter()
    results = aligner.align(segments, ayahs)
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(results, ground_truth["ayahs"])
    metrics["runtime_seconds"] = round(elapsed, 4)
    metrics["strategy"] = strategy.value

    return metrics


def run_benchmarks() -> dict:
    """Run all benchmarks across all ground-truth files and strategies."""
    gt_files = sorted(GROUND_TRUTH_DIR.glob("surah_*.json"))
    if not gt_files:
        print("No ground-truth files found in", GROUND_TRUTH_DIR)
        return {}

    all_results = {"surahs": {}, "summary": {}}

    for gt_file in gt_files:
        ground_truth = load_ground_truth(gt_file)
        surah_id = ground_truth["surah_id"]
        surah_name = ground_truth["surah_name"]
        print(f"\nBenchmarking Surah {surah_id} ({surah_name})...")

        segments = create_mock_segments(ground_truth)
        ayahs = load_surah_ayahs(surah_id)

        surah_results = {}
        for strategy in STRATEGIES:
            print(f"  Strategy: {strategy.value}...", end=" ")
            metrics = run_single_benchmark(strategy, segments, ayahs, ground_truth)
            surah_results[strategy.value] = metrics
            print(
                f"MAE={metrics['mae_seconds']:.4f}s, "
                f"sim={metrics['avg_similarity']:.4f}, "
                f"high_conf={metrics['high_confidence_pct']:.1f}%, "
                f"time={metrics['runtime_seconds']:.4f}s"
            )

        all_results["surahs"][str(surah_id)] = {
            "name": surah_name,
            "strategies": surah_results,
        }

    # Compute summary (average across surahs)
    for strategy in STRATEGIES:
        strategy_metrics = []
        for surah_data in all_results["surahs"].values():
            m = surah_data["strategies"].get(strategy.value)
            if m:
                strategy_metrics.append(m)

        if strategy_metrics:
            all_results["summary"][strategy.value] = {
                "avg_mae_seconds": round(
                    sum(m["mae_seconds"] for m in strategy_metrics)
                    / len(strategy_metrics),
                    4,
                ),
                "avg_similarity": round(
                    sum(m["avg_similarity"] for m in strategy_metrics)
                    / len(strategy_metrics),
                    4,
                ),
                "avg_high_confidence_pct": round(
                    sum(m["high_confidence_pct"] for m in strategy_metrics)
                    / len(strategy_metrics),
                    2,
                ),
                "avg_runtime_seconds": round(
                    sum(m["runtime_seconds"] for m in strategy_metrics)
                    / len(strategy_metrics),
                    4,
                ),
                "surahs_tested": len(strategy_metrics),
            }

    return all_results


def generate_leaderboard(results: dict) -> str:
    """Generate Markdown leaderboard from benchmark results."""
    lines = [
        "# Munajjam Alignment Benchmark Leaderboard",
        "",
        "Auto-generated by `benchmarks/run_benchmark.py`.",
        "",
        "## Summary (averaged across all surahs)",
        "",
        "| Rank | Strategy | Avg MAE (s) | Avg Similarity | High Confidence % | Avg Runtime (s) |",
        "|------|----------|-------------|----------------|-------------------|-----------------|",
    ]

    summary = results.get("summary", {})
    ranked = sorted(summary.items(), key=lambda x: x[1]["avg_mae_seconds"])

    for rank, (strategy, metrics) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | **{strategy.upper()}** | "
            f"{metrics['avg_mae_seconds']:.4f} | "
            f"{metrics['avg_similarity']:.4f} | "
            f"{metrics['avg_high_confidence_pct']:.1f}% | "
            f"{metrics['avg_runtime_seconds']:.4f} |"
        )

    lines.append("")
    lines.append("## Per-Surah Results")

    for surah_id, surah_data in results.get("surahs", {}).items():
        lines.append("")
        lines.append(f"### Surah {surah_id} — {surah_data['name']}")
        lines.append("")
        lines.append(
            "| Strategy | MAE (s) | Similarity | High Confidence % | Runtime (s) | Aligned/Expected |"
        )
        lines.append(
            "|----------|---------|------------|-------------------|-------------|------------------|"
        )

        strategies = surah_data.get("strategies", {})
        ranked_s = sorted(strategies.items(), key=lambda x: x[1]["mae_seconds"])

        for strategy, metrics in ranked_s:
            lines.append(
                f"| {strategy.upper()} | "
                f"{metrics['mae_seconds']:.4f} | "
                f"{metrics['avg_similarity']:.4f} | "
                f"{metrics['high_confidence_pct']:.1f}% | "
                f"{metrics['runtime_seconds']:.4f} | "
                f"{metrics['aligned_count']}/{metrics['expected_count']} |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Munajjam Alignment Benchmark")
    print("=" * 60)

    results = run_benchmarks()
    if not results:
        return

    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    leaderboard = generate_leaderboard(results)
    leaderboard_path = RESULTS_DIR / "LEADERBOARD.md"
    with open(leaderboard_path, "w") as f:
        f.write(leaderboard)
    print(f"Leaderboard saved to: {leaderboard_path}")

    print("\n" + leaderboard)


if __name__ == "__main__":
    main()
