"""
Benchmark harness for Munajjam alignment strategies.

This module provides tools to objectively compare alignment quality
and speed across GREEDY, DP, HYBRID, and AUTO strategies using
ground-truth timestamp data.

Usage:
    python -m benchmarks.benchmark

Output:
    - benchmarks/results/benchmark_results.json  (raw results)
    - benchmarks/LEADERBOARD.md                  (auto-generated leaderboard)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from munajjam.core import Aligner, AlignmentStrategy
from munajjam.models import Segment, SegmentType, Ayah, AlignmentResult
from munajjam.core.arabic import normalize_arabic
from munajjam.core.matcher import similarity


# ---------------------------------------------------------------------------
# Ground-truth data (manually verified timestamps for 2 surahs)
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, list[dict]] = {
    "surah_1_fatiha": [
        {
            "ayah_number": 1,
            "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "start_time": 0.50,
            "end_time": 4.80,
        },
        {
            "ayah_number": 2,
            "text": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            "start_time": 5.20,
            "end_time": 10.10,
        },
        {
            "ayah_number": 3,
            "text": "الرَّحْمَٰنِ الرَّحِيمِ",
            "start_time": 10.60,
            "end_time": 13.90,
        },
        {
            "ayah_number": 4,
            "text": "مَالِكِ يَوْمِ الدِّينِ",
            "start_time": 14.30,
            "end_time": 17.50,
        },
        {
            "ayah_number": 5,
            "text": "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
            "start_time": 18.00,
            "end_time": 22.80,
        },
        {
            "ayah_number": 6,
            "text": "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
            "start_time": 23.30,
            "end_time": 27.50,
        },
        {
            "ayah_number": 7,
            "text": "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
            "start_time": 28.00,
            "end_time": 38.50,
        },
    ],
    "surah_112_ikhlas": [
        {
            "ayah_number": 1,
            "text": "قُلْ هُوَ اللَّهُ أَحَدٌ",
            "start_time": 0.30,
            "end_time": 3.20,
        },
        {
            "ayah_number": 2,
            "text": "اللَّهُ الصَّمَدُ",
            "start_time": 3.70,
            "end_time": 6.10,
        },
        {
            "ayah_number": 3,
            "text": "لَمْ يَلِدْ وَلَمْ يُولَدْ",
            "start_time": 6.60,
            "end_time": 10.20,
        },
        {
            "ayah_number": 4,
            "text": "وَلَمْ يَكُن لَّهُ كُفُوًا أَحَدٌ",
            "start_time": 10.70,
            "end_time": 14.50,
        },
    ],
}


# ---------------------------------------------------------------------------
# Synthetic segment generator (no GPU / audio file needed)
# ---------------------------------------------------------------------------

def _make_segments_from_ground_truth(
    gt_ayahs: list[dict],
    surah_id: int,
    noise_factor: float = 0.05,
) -> list[Segment]:
    """
    Generate synthetic transcribed segments from ground-truth data.

    Adds slight timing noise to simulate real transcription output,
    making the benchmark realistic without requiring actual audio files.
    """
    segments: list[Segment] = []
    for i, gt in enumerate(gt_ayahs):
        # Add small timing noise (±noise_factor * duration)
        duration = gt["end_time"] - gt["start_time"]
        start = gt["start_time"] + noise_factor * duration * (-1 if i % 2 == 0 else 1)
        end = gt["end_time"] + noise_factor * duration * (1 if i % 2 == 0 else -1)
        segments.append(
            Segment(
                id=i,
                surah_id=surah_id,
                start=max(0.0, start),
                end=max(start + 0.1, end),
                text=gt["text"],
                type=SegmentType.AYAH,
            )
        )
    return segments


def _make_ayahs_from_ground_truth(
    gt_ayahs: list[dict],
    surah_id: int,
) -> list[Ayah]:
    """Build Ayah objects from ground-truth data."""
    return [
        Ayah(
            id=i + 1,
            surah_id=surah_id,
            ayah_number=gt["ayah_number"],
            text=gt["text"],
        )
        for i, gt in enumerate(gt_ayahs)
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class StrategyMetrics:
    """Metrics for a single strategy on a single surah."""

    strategy: str
    surah_name: str
    num_ayahs: int
    mae_start: float = 0.0          # Mean Absolute Error of start timestamps (s)
    mae_end: float = 0.0            # Mean Absolute Error of end timestamps (s)
    mae_combined: float = 0.0       # Average of mae_start and mae_end
    avg_similarity: float = 0.0     # Average similarity score (0–1)
    pct_high_confidence: float = 0.0  # % of ayahs with similarity >= 0.8
    runtime_seconds: float = 0.0    # Wall-clock time for alignment
    aligned_count: int = 0          # Number of ayahs successfully aligned


def _compute_metrics(
    results: list[AlignmentResult],
    ground_truth: list[dict],
    strategy: str,
    surah_name: str,
    runtime: float,
) -> StrategyMetrics:
    """Compute benchmark metrics by comparing results to ground truth."""
    n = min(len(results), len(ground_truth))
    if n == 0:
        return StrategyMetrics(
            strategy=strategy,
            surah_name=surah_name,
            num_ayahs=0,
            runtime_seconds=runtime,
        )

    mae_starts, mae_ends, similarities = [], [], []
    high_confidence = 0

    for result, gt in zip(results[:n], ground_truth[:n]):
        mae_starts.append(abs(result.start_time - gt["start_time"]))
        mae_ends.append(abs(result.end_time - gt["end_time"]))
        similarities.append(result.similarity_score)
        if result.is_high_confidence:
            high_confidence += 1

    mae_start = sum(mae_starts) / len(mae_starts)
    mae_end = sum(mae_ends) / len(mae_ends)

    return StrategyMetrics(
        strategy=strategy,
        surah_name=surah_name,
        num_ayahs=len(ground_truth),
        mae_start=round(mae_start, 4),
        mae_end=round(mae_end, 4),
        mae_combined=round((mae_start + mae_end) / 2, 4),
        avg_similarity=round(sum(similarities) / len(similarities), 4),
        pct_high_confidence=round(high_confidence / n * 100, 1),
        runtime_seconds=round(runtime, 4),
        aligned_count=n,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

STRATEGIES = [
    AlignmentStrategy.GREEDY,
    AlignmentStrategy.DP,
    AlignmentStrategy.HYBRID,
    AlignmentStrategy.AUTO,
]

SURAH_IDS = {
    "surah_1_fatiha": 1,
    "surah_112_ikhlas": 112,
}

DUMMY_AUDIO = "benchmark_dummy.wav"


def run_benchmark() -> list[StrategyMetrics]:
    """
    Run the full benchmark across all strategies and surahs.

    Returns a list of StrategyMetrics, one per (strategy, surah) combination.
    """
    all_metrics: list[StrategyMetrics] = []

    for surah_name, gt_ayahs in GROUND_TRUTH.items():
        surah_id = SURAH_IDS[surah_name]
        segments = _make_segments_from_ground_truth(gt_ayahs, surah_id)
        ayahs = _make_ayahs_from_ground_truth(gt_ayahs, surah_id)

        for strategy in STRATEGIES:
            aligner = Aligner(
                audio_path=DUMMY_AUDIO,
                strategy=strategy,
                energy_snap=False,   # no real audio → skip acoustic features
                fix_drift=True,
                fix_overlaps=True,
            )

            t0 = time.perf_counter()
            results = aligner.align(segments, ayahs)
            runtime = time.perf_counter() - t0

            metrics = _compute_metrics(
                results, gt_ayahs, strategy.value, surah_name, runtime
            )
            all_metrics.append(metrics)
            print(
                f"  [{strategy.value:6s}] {surah_name:25s} "
                f"MAE={metrics.mae_combined:.3f}s  "
                f"sim={metrics.avg_similarity:.3f}  "
                f"hc={metrics.pct_high_confidence:.0f}%  "
                f"t={metrics.runtime_seconds:.3f}s"
            )

    return all_metrics


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_results_json(metrics: list[StrategyMetrics], output_path: Path) -> None:
    """Save raw benchmark results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [asdict(m) for m in metrics],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Markdown leaderboard generator
# ---------------------------------------------------------------------------

def _rank_strategies(metrics: list[StrategyMetrics]) -> list[tuple[str, float]]:
    """
    Rank strategies by a composite score across all surahs.

    Score = avg_similarity * 0.5 + (1 - mae_combined_norm) * 0.3 + pct_hc_norm * 0.2
    Lower MAE and higher similarity = better rank.
    """
    # Aggregate per strategy
    strategy_scores: dict[str, list[float]] = {}
    for m in metrics:
        if m.strategy not in strategy_scores:
            strategy_scores[m.strategy] = []
        # Composite score (higher is better)
        # Normalize MAE: assume max acceptable MAE = 2.0 seconds
        mae_norm = min(m.mae_combined / 2.0, 1.0)
        score = (
            m.avg_similarity * 0.5
            + (1.0 - mae_norm) * 0.3
            + (m.pct_high_confidence / 100.0) * 0.2
        )
        strategy_scores[m.strategy].append(score)

    ranked = [
        (strategy, sum(scores) / len(scores))
        for strategy, scores in strategy_scores.items()
    ]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def generate_leaderboard(metrics: list[StrategyMetrics], output_path: Path) -> None:
    """Auto-generate a Markdown leaderboard from benchmark results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = _rank_strategies(metrics)
    medal = ["🥇", "🥈", "🥉", "4️⃣"]

    lines: list[str] = [
        "# Munajjam Alignment Strategy Leaderboard",
        "",
        "> Auto-generated by `benchmarks/benchmark.py` — do not edit manually.",
        f"> Last updated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Overall Ranking",
        "",
        "Strategies are ranked by a composite score:",
        "- **50%** average similarity score",
        "- **30%** timestamp accuracy (lower MAE = better)",
        "- **20%** percentage of high-confidence ayahs (similarity ≥ 0.8)",
        "",
        "| Rank | Strategy | Composite Score |",
        "|------|----------|----------------|",
    ]
    for i, (strategy, score) in enumerate(ranked):
        m_str = medal[i] if i < len(medal) else f"{i+1}."
        lines.append(f"| {m_str} | `{strategy}` | {score:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Detailed Results by Surah",
        "",
    ]

    # Group by surah
    surahs = sorted({m.surah_name for m in metrics})
    for surah in surahs:
        surah_metrics = [m for m in metrics if m.surah_name == surah]
        lines += [
            f"### {surah.replace('_', ' ').title()}",
            "",
            "| Strategy | MAE Start (s) | MAE End (s) | MAE Combined (s) | Avg Similarity | High-Confidence % | Runtime (s) |",
            "|----------|--------------|------------|-----------------|---------------|------------------|-------------|",
        ]
        for m in sorted(surah_metrics, key=lambda x: x.mae_combined):
            lines.append(
                f"| `{m.strategy}` "
                f"| {m.mae_start:.4f} "
                f"| {m.mae_end:.4f} "
                f"| {m.mae_combined:.4f} "
                f"| {m.avg_similarity:.4f} "
                f"| {m.pct_high_confidence:.1f}% "
                f"| {m.runtime_seconds:.4f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| **MAE Start** | Mean Absolute Error of predicted start timestamps vs ground truth (seconds) |",
        "| **MAE End** | Mean Absolute Error of predicted end timestamps vs ground truth (seconds) |",
        "| **MAE Combined** | Average of MAE Start and MAE End |",
        "| **Avg Similarity** | Average text similarity score between transcribed and reference ayah text (0–1) |",
        "| **High-Confidence %** | Percentage of ayahs with similarity score ≥ 0.8 |",
        "| **Runtime** | Wall-clock time for the alignment call (seconds) |",
        "",
        "## How to Run",
        "",
        "```bash",
        "# From the repository root:",
        "python -m benchmarks.benchmark",
        "```",
        "",
        "Results are saved to `benchmarks/results/benchmark_results.json`.",
        "This leaderboard is regenerated automatically on each run.",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Leaderboard saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the benchmark and generate outputs."""
    root = Path(__file__).parent

    print("=" * 60)
    print("Munajjam Benchmark Harness")
    print("=" * 60)
    print(f"Strategies : {[s.value for s in STRATEGIES]}")
    print(f"Surahs     : {list(GROUND_TRUTH.keys())}")
    print("-" * 60)

    metrics = run_benchmark()

    print("-" * 60)
    save_results_json(metrics, root / "results" / "benchmark_results.json")
    generate_leaderboard(metrics, root / "LEADERBOARD.md")
    print("=" * 60)
    print("Done! ✅")


if __name__ == "__main__":
    main()
