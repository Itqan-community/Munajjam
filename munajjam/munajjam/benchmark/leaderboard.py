"""
Markdown leaderboard generator for benchmark results.

Produces a formatted Markdown string with per-surah tables
and an overall summary table.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from munajjam.benchmark.models import BenchmarkReport, StrategyMetrics

_TABLE_HEADER = (
    "| Strategy | MAE Start (s) | MAE End (s) | Avg Similarity "
    "| % High Confidence | Runtime (s) | Ayahs |\n"
    "|----------|---------------|-------------|----------------"
    "|-------------------|-------------|-------|"
)


@dataclass
class _SummaryRow:
    """Internal row for the overall summary table (not a Pydantic model)."""

    strategy: str
    mae_start: float
    mae_end: float
    avg_similarity: float
    pct_high_confidence: float
    runtime_seconds: float
    ayah_count: int


def _format_row(
    strategy: str,
    mae_start: float,
    mae_end: float,
    avg_similarity: float,
    pct_high_confidence: float,
    runtime_seconds: float,
    ayah_count: int,
) -> str:
    return (
        f"| {strategy:<8} "
        f"| {mae_start:>13.3f} "
        f"| {mae_end:>11.3f} "
        f"| {avg_similarity:>14.3f} "
        f"| {pct_high_confidence:>17.1f} "
        f"| {runtime_seconds:>11.3f} "
        f"| {ayah_count:>5} |"
    )


def _format_metrics_row(m: StrategyMetrics) -> str:
    return _format_row(
        m.strategy,
        m.mae_start,
        m.mae_end,
        m.avg_similarity,
        m.pct_high_confidence,
        m.runtime_seconds,
        m.ayah_count,
    )


def generate_leaderboard(report: BenchmarkReport) -> str:
    """Generate a Markdown leaderboard from benchmark results.

    Produces:
    - A header with generation timestamp and library version
    - One section per surah, rows sorted by MAE start ascending
    - An overall summary with per-strategy means across all surahs

    Args:
        report: BenchmarkReport to render.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = [
        "# Munajjam Alignment Benchmark Leaderboard\n",
        f"\n> Generated: {report.generated_at}  \n",
        f"> Library version: `{report.munajjam_version}`\n",
        "\n---\n",
    ]

    # Per-surah sections
    sorted_results = sorted(
        report.results,
        key=lambda r: (r.surah_id, r.mae_start),
    )
    for surah_id, group in groupby(sorted_results, key=lambda r: r.surah_id):
        group_list = list(group)
        lines.append(f"\n## Surah {surah_id}\n\n")
        lines.append(_TABLE_HEADER + "\n")
        for m in group_list:
            lines.append(_format_metrics_row(m) + "\n")

    # Overall summary
    lines.append("\n---\n\n## Overall Summary (mean across all surahs)\n\n")
    lines.append(_TABLE_HEADER + "\n")

    strategies = sorted({m.strategy for m in report.results})
    for strat in strategies:
        strat_results = [m for m in report.results if m.strategy == strat]
        n = len(strat_results)
        if n == 0:
            continue
        summary = _SummaryRow(
            strategy=strat,
            mae_start=sum(m.mae_start for m in strat_results) / n,
            mae_end=sum(m.mae_end for m in strat_results) / n,
            avg_similarity=sum(m.avg_similarity for m in strat_results) / n,
            pct_high_confidence=sum(m.pct_high_confidence for m in strat_results) / n,
            runtime_seconds=sum(m.runtime_seconds for m in strat_results) / n,
            ayah_count=sum(m.ayah_count for m in strat_results),
        )
        lines.append(
            _format_row(
                summary.strategy,
                summary.mae_start,
                summary.mae_end,
                summary.avg_similarity,
                summary.pct_high_confidence,
                summary.runtime_seconds,
                summary.ayah_count,
            )
            + "\n",
        )

    return "".join(lines)


def save_leaderboard(report: BenchmarkReport, output_path: Path) -> None:
    """Write the leaderboard to a Markdown file.

    Creates parent directories if they do not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_leaderboard(report), encoding="utf-8")
