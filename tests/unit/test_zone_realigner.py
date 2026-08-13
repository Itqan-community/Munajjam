"""
Unit tests for zone-level realignment helpers.
"""

from munajjam.core.cascade_recovery import (
    detect_unaligned_word_gaps,
    recover_unaligned_word_gaps,
)
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


def test_detect_unaligned_word_gaps():
    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحْمَـٰنِ", "start": 0.9, "end": 1.0, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 3.0, "end": 4.5, "confidence": 0.92},
    ]

    gaps = detect_unaligned_word_gaps(words)
    assert len(gaps) == 1
    assert gaps[0].start_word_idx == 1
    assert gaps[0].end_word_idx == 3
    assert gaps[0].words == ["ٱللَّهِ", "ٱلرَّحْمَـٰنِ"]
    assert gaps[0].gap_start_time == 0.8
    assert gaps[0].gap_end_time == 3.0


def test_recover_unaligned_word_gaps():
    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحْمَـٰنِ", "start": 0.9, "end": 1.0, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 3.0, "end": 4.5, "confidence": 0.92},
    ]

    recovered = recover_unaligned_word_gaps(words)
    assert recovered[1]["confidence"] > 0.5
    assert recovered[2]["confidence"] > 0.5
    assert recovered[1]["start"] == 0.8
    assert recovered[2]["end"] == 3.0
    assert recovered[1]["end"] == recovered[2]["start"]


def test_detect_unaligned_word_gaps_preserves_high_confidence_short_words():
    """Verify that a valid short word (duration <= 0.15s) with high confidence is NOT detected as unaligned."""
    words = [
        {"word": "قُلْ", "start": 0.0, "end": 0.5, "confidence": 0.95},
        {"word": "فِي", "start": 0.5, "end": 0.62, "confidence": 0.95},
        {"word": "ٱلْأَرْضِ", "start": 0.62, "end": 1.5, "confidence": 0.92},
    ]

    gaps = detect_unaligned_word_gaps(words)
    assert len(gaps) == 0
