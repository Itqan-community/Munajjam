"""
Unit tests for zone-level realignment helpers.
"""

from munajjam.core.zone_realigner import _find_problem_runs
from munajjam.models import AlignmentResult, Ayah


def _make_result(
    ayah_number: int,
    start: float,
    end: float,
    similarity: float,
    text: str = "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
) -> AlignmentResult:
    ayah = Ayah(
        id=ayah_number,
        surah_id=112,
        ayah_number=ayah_number,
        text=text,
    )
    return AlignmentResult(
        ayah=ayah,
        start_time=start,
        end_time=end,
        transcribed_text=text,
        similarity_score=similarity,
        overlap_detected=False,
    )


def test_find_problem_runs_detects_low_similarity_sequence():
    results = [
        _make_result(1, 0.0, 3.0, 0.95),
        _make_result(2, 3.2, 6.0, 0.62),
        _make_result(3, 6.2, 9.0, 0.58),
        _make_result(4, 9.2, 12.0, 0.92),
    ]

    runs = _find_problem_runs(
        results=results,
        similarity_threshold=0.75,
        min_consecutive=2,
        max_pace_ratio=2.5,
    )

    assert runs == [(1, 3)]


def test_find_problem_runs_no_problems():
    """All high similarity scores should yield no problem runs."""
    results = [
        _make_result(1, 0.0, 3.0, 0.95),
        _make_result(2, 3.2, 6.0, 0.92),
        _make_result(3, 6.2, 9.0, 0.88),
        _make_result(4, 9.2, 12.0, 0.91),
    ]

    runs = _find_problem_runs(
        results=results,
        similarity_threshold=0.75,
        min_consecutive=2,
        max_pace_ratio=2.5,
    )

    assert runs == []


def test_find_problem_runs_single_low():
    """A single low-similarity result (below min_consecutive) should yield no runs."""
    results = [
        _make_result(1, 0.0, 3.0, 0.95),
        _make_result(2, 3.2, 6.0, 0.50),  # Single low score
        _make_result(3, 6.2, 9.0, 0.92),
        _make_result(4, 9.2, 12.0, 0.91),
    ]

    runs = _find_problem_runs(
        results=results,
        similarity_threshold=0.75,
        min_consecutive=2,
        max_pace_ratio=2.5,
    )

    assert runs == []
