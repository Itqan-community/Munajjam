"""
Unit tests for benchmark leaderboard generation (Issue #45).

Tests cover:
- generate_leaderboard produces valid Markdown
- Sections for each surah and overall summary
- Rows sorted by MAE ascending within each surah
- save_leaderboard writes file
"""

import pytest

from munajjam.benchmark.leaderboard import generate_leaderboard, save_leaderboard
from munajjam.benchmark.models import BenchmarkReport, StrategyMetrics


def _make_report() -> BenchmarkReport:
    """Create a fixture BenchmarkReport with 2 surahs x 3 strategies."""
    results = []
    for surah_id in (1, 112):
        for strategy, mae in [("hybrid", 0.05), ("greedy", 0.15), ("dp", 0.10)]:
            results.append(
                StrategyMetrics(
                    strategy=strategy,
                    surah_id=surah_id,
                    mae_start=mae,
                    mae_end=mae + 0.05,
                    avg_similarity=0.95 if strategy == "hybrid" else 0.90,
                    pct_high_confidence=90.0 if strategy == "hybrid" else 80.0,
                    runtime_seconds=0.01 * (surah_id + 1),
                    ayah_count=7 if surah_id == 1 else 4,
                    timestamp="2026-02-26T12:00:00+00:00",
                )
            )
    return BenchmarkReport(
        generated_at="2026-02-26T12:00:00+00:00",
        munajjam_version="2.0.0a1",
        results=results,
    )


class TestGenerateLeaderboard:
    """Test Markdown leaderboard generation."""

    def test_contains_header(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "# Munajjam Alignment Benchmark Leaderboard" in md

    def test_contains_version(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "2.0.0a1" in md

    def test_contains_surah_sections(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "## Surah 1" in md
        assert "## Surah 112" in md

    def test_contains_overall_summary(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "Overall Summary" in md

    def test_contains_all_strategies(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "greedy" in md
        assert "dp" in md
        assert "hybrid" in md

    def test_contains_table_header(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert "MAE Start" in md
        assert "MAE End" in md
        assert "Avg Similarity" in md
        assert "High Confidence" in md
        assert "Runtime" in md

    def test_rows_sorted_by_mae_start_ascending(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        # Within Surah 1 section, hybrid (0.05) should come before dp (0.10) before greedy (0.15)
        lines = md.split("\n")
        surah_1_lines = []
        in_surah_1 = False
        for line in lines:
            if line.strip() == "## Surah 1":
                in_surah_1 = True
                continue
            if in_surah_1 and line.startswith("## "):
                break
            if in_surah_1 and line.startswith("|") and ("hybrid" in line or "dp" in line or "greedy" in line):
                surah_1_lines.append(line)
        # hybrid should appear before dp, dp before greedy
        strategy_order = []
        for line in surah_1_lines:
            if "hybrid" in line:
                strategy_order.append("hybrid")
            elif "dp" in line:
                strategy_order.append("dp")
            elif "greedy" in line:
                strategy_order.append("greedy")
        assert strategy_order == ["hybrid", "dp", "greedy"]

    def test_empty_report(self) -> None:
        report = BenchmarkReport(
            generated_at="2026-02-26T12:00:00+00:00",
            munajjam_version="2.0.0a1",
            results=[],
        )
        md = generate_leaderboard(report)
        assert "# Munajjam Alignment Benchmark Leaderboard" in md
        # No surah sections, but summary section should still be present
        assert "Overall Summary" in md

    def test_returns_string(self) -> None:
        report = _make_report()
        md = generate_leaderboard(report)
        assert isinstance(md, str)


class TestSaveLeaderboard:
    """Test save_leaderboard writes file."""

    def test_writes_markdown_file(self, tmp_path) -> None:
        report = _make_report()
        output_path = tmp_path / "LEADERBOARD.md"
        save_leaderboard(report, output_path)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "# Munajjam Alignment Benchmark Leaderboard" in content

    def test_creates_parent_directories(self, tmp_path) -> None:
        report = _make_report()
        output_path = tmp_path / "nested" / "dir" / "LEADERBOARD.md"
        save_leaderboard(report, output_path)
        assert output_path.exists()
