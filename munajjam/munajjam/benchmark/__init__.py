"""
Benchmark harness for Munajjam alignment quality evaluation.

Provides ground-truth loading, metric computation, benchmark running,
and Markdown leaderboard generation.
"""

from munajjam.benchmark.leaderboard import generate_leaderboard, save_leaderboard
from munajjam.benchmark.loader import list_ground_truth_files, load_ground_truth
from munajjam.benchmark.metrics import (
    compute_avg_similarity,
    compute_mae,
    compute_pct_high_confidence,
    compute_strategy_metrics,
)
from munajjam.benchmark.models import (
    BenchmarkReport,
    GroundTruthAyah,
    GroundTruthSurah,
    StrategyMetrics,
)
from munajjam.benchmark.runner import BenchmarkRunner

__all__ = [
    "BenchmarkReport",
    "BenchmarkRunner",
    "GroundTruthAyah",
    "GroundTruthSurah",
    "StrategyMetrics",
    "compute_avg_similarity",
    "compute_mae",
    "compute_pct_high_confidence",
    "compute_strategy_metrics",
    "generate_leaderboard",
    "list_ground_truth_files",
    "load_ground_truth",
    "save_leaderboard",
]
