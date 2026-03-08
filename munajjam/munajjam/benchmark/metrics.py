"""
Pure metric computation functions for the benchmark harness.

All functions are side-effect-free and operate on in-memory data only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from munajjam.benchmark.models import GroundTruthSurah, StrategyMetrics
from munajjam.models import AlignmentResult


def compute_mae(predicted: list[float], actual: list[float]) -> float:
    """Compute mean absolute error between two lists of values.

    Uses min(len(predicted), len(actual)) pairs for comparison,
    so mismatched lengths degrade gracefully.

    Returns 0.0 if either list is empty.
    """
    if not predicted or not actual:
        return 0.0
    n = min(len(predicted), len(actual))
    return sum(abs(p - a) for p, a in zip(predicted[:n], actual[:n])) / n  # noqa: B905


def compute_avg_similarity(results: list[AlignmentResult]) -> float:
    """Compute average similarity score across alignment results.

    Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(r.similarity_score for r in results) / len(results)


def compute_pct_high_confidence(results: list[AlignmentResult]) -> float:
    """Compute percentage of results with high confidence (similarity >= 0.8).

    Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return (sum(1 for r in results if r.is_high_confidence) / len(results)) * 100.0


def compute_strategy_metrics(
    strategy: str,
    surah_id: int,
    results: list[AlignmentResult],
    ground_truth: GroundTruthSurah,
    runtime_seconds: float,
) -> StrategyMetrics:
    """Assemble all metrics for one strategy + surah combination.

    Matches predicted results to ground-truth ayahs by ayah_number order.
    """
    gt_sorted = sorted(ground_truth.ayahs, key=lambda a: a.ayah_number)
    res_sorted = sorted(results, key=lambda r: r.ayah.ayah_number)

    gt_starts = [a.start_time for a in gt_sorted]
    gt_ends = [a.end_time for a in gt_sorted]
    pred_starts = [r.start_time for r in res_sorted]
    pred_ends = [r.end_time for r in res_sorted]

    return StrategyMetrics(
        strategy=strategy,
        surah_id=surah_id,
        mae_start=compute_mae(pred_starts, gt_starts),
        mae_end=compute_mae(pred_ends, gt_ends),
        avg_similarity=compute_avg_similarity(results),
        pct_high_confidence=compute_pct_high_confidence(results),
        runtime_seconds=runtime_seconds,
        ayah_count=len(results),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
