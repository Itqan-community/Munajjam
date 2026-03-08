"""
Benchmark runner for evaluating alignment strategies.

Converts ground-truth data into Segments (no real transcriber needed),
runs the Aligner with each strategy, and collects metrics.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import munajjam
from munajjam.benchmark.loader import list_ground_truth_files, load_ground_truth
from munajjam.benchmark.metrics import compute_strategy_metrics
from munajjam.benchmark.models import (
    BenchmarkReport,
    GroundTruthSurah,
    StrategyMetrics,
)
from munajjam.core.aligner import Aligner, AlignmentStrategy
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType

_DEFAULT_STRATEGIES = [
    AlignmentStrategy.GREEDY,
    AlignmentStrategy.DP,
    AlignmentStrategy.HYBRID,
]


class BenchmarkRunner:
    """Run alignment benchmarks against ground-truth data.

    Converts ground-truth ayahs into Segment objects (using the
    ground-truth text as perfectly-transcribed text), then runs
    each configured alignment strategy and collects metrics.

    No GPU, real audio, or transcriber is required.
    """

    def __init__(
        self,
        ground_truth_dir: Path,
        results_dir: Path | None = None,
        strategies: list[AlignmentStrategy] | None = None,
    ) -> None:
        self.ground_truth_dir = ground_truth_dir
        self.results_dir = results_dir
        self.strategies = strategies if strategies is not None else list(_DEFAULT_STRATEGIES)

    def _ground_truth_to_segments(
        self,
        ground_truth: GroundTruthSurah,
    ) -> list[Segment]:
        """Convert ground-truth ayahs into Segment objects for alignment."""
        return [
            Segment(
                id=i + 1,
                surah_id=ground_truth.surah_id,
                start=gt_ayah.start_time,
                end=gt_ayah.end_time,
                text=gt_ayah.text,
                type=SegmentType.AYAH,
            )
            for i, gt_ayah in enumerate(
                sorted(ground_truth.ayahs, key=lambda a: a.ayah_number),
            )
        ]

    def run_surah(
        self,
        ground_truth: GroundTruthSurah,
    ) -> list[StrategyMetrics]:
        """Run all strategies on one surah and return metrics.

        Args:
            ground_truth: Ground-truth data for the surah.

        Returns:
            List of StrategyMetrics, one per strategy.
        """
        ayahs = load_surah_ayahs(ground_truth.surah_id)
        segments = self._ground_truth_to_segments(ground_truth)
        metrics_list: list[StrategyMetrics] = []

        for strategy in self.strategies:
            aligner = Aligner(
                audio_path=ground_truth.audio_filename,
                strategy=strategy,
                energy_snap=False,
            )
            t0 = time.perf_counter()
            results = aligner.align(segments, ayahs)
            runtime = time.perf_counter() - t0

            metrics = compute_strategy_metrics(
                strategy=strategy.value,
                surah_id=ground_truth.surah_id,
                results=results,
                ground_truth=ground_truth,
                runtime_seconds=round(runtime, 6),
            )
            metrics_list.append(metrics)

        return metrics_list

    def run_all(self) -> BenchmarkReport:
        """Load all ground-truth files and benchmark every strategy.

        Returns:
            BenchmarkReport containing all results.
        """
        gt_files = list_ground_truth_files(self.ground_truth_dir)
        all_metrics: list[StrategyMetrics] = []
        for gt_file in gt_files:
            gt = load_ground_truth(gt_file)
            all_metrics.extend(self.run_surah(gt))

        return BenchmarkReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            munajjam_version=munajjam.__version__,
            results=all_metrics,
        )

    def save_json(
        self,
        report: BenchmarkReport,
        output_path: Path,
    ) -> None:
        """Write the benchmark report as JSON.

        Creates parent directories if they do not exist.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
