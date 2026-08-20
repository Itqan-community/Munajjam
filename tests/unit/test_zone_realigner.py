"""
Unit tests for zone-level realignment helpers.
"""

import numpy as np

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


def test_detect_unaligned_word_gaps_with_placeholder_flags():
    """Verify that an explicit placeholder/fallback record (even with high confidence) is detected."""
    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {
            "word": "ٱللَّهِ",
            "start": 0.8,
            "end": 0.9,
            "confidence": 0.95,
            "is_placeholder": True,
        },
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


def test_slice_audio_array():
    """Verify audio slicing returns correct duration and sample bounds."""
    from munajjam.core.cascade_recovery import slice_audio_array

    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds
    slice_arr, start_sec, end_sec = slice_audio_array(
        dummy_audio, 1.0, 2.5, sample_rate=16000
    )
    assert len(slice_arr) == 16000 * 1.5
    assert start_sec == 1.0
    assert end_sec == 2.5


def test_recover_unaligned_word_gaps_acoustic():
    """Verify recover_unaligned_word_gaps uses acoustic realignment when provided."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 2.0, "end": 3.0, "confidence": 0.92},
    ]
    dummy_audio = np.zeros(16000 * 4, dtype=np.float32)
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    mock_align_result = {
        "segments": [
            {
                "text": "ٱللَّهِ",
                "words": [{"word": "ٱللَّهِ", "start": 0.25, "end": 1.15, "score": 0.88}],
            }
        ]
    }
    mock_whisperx = MagicMock()
    mock_whisperx.align.return_value = mock_align_result

    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        recovered = recover_unaligned_word_gaps(
            words,
            audio=dummy_audio,
            align_model=mock_align_model,
            align_metadata=mock_align_metadata,
        )
        assert recovered[1]["confidence"] == 0.88
        assert recovered[1]["start"] > 0.5
        assert recovered[1]["end"] <= 2.0


def test_realign_trailing_gap_acoustic():
    """Verify trailing unaligned gap resolves recovery interval using total audio duration."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},  # Trailing gap
    ]
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3.0s total audio
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    mock_align_result = {
        "segments": [
            {
                "text": "ٱللَّهِ",
                "words": [{"word": "ٱللَّهِ", "start": 0.35, "end": 1.5, "score": 0.90}],
            }
        ]
    }
    mock_whisperx = MagicMock()
    mock_whisperx.align.return_value = mock_align_result

    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        recovered = recover_unaligned_word_gaps(
            words,
            audio=dummy_audio,
            align_model=mock_align_model,
            align_metadata=mock_align_metadata,
        )
        assert recovered[1]["confidence"] == 0.90
        assert recovered[1]["end"] <= 3.0


def test_realign_rejects_out_of_bounds_acoustic():
    """Verify acoustic realignment is rejected and falls back if timestamps exceed anchor bounds."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 2.0, "end": 3.0, "confidence": 0.92},
    ]
    dummy_audio = np.zeros(16000 * 4, dtype=np.float32)
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    # Mock returns timestamp placed in padded context past anchor end (2.5s > 2.0s)
    mock_align_result = {
        "segments": [
            {
                "text": "ٱللَّهِ",
                "words": [{"word": "ٱللَّهِ", "start": 1.0, "end": 2.5, "score": 0.90}],
            }
        ]
    }
    mock_whisperx = MagicMock()
    mock_whisperx.align.return_value = mock_align_result

    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        recovered = recover_unaligned_word_gaps(
            words,
            audio=dummy_audio,
            align_model=mock_align_model,
            align_metadata=mock_align_metadata,
        )
        # Should fall back to bounded interpolation (confidence 0.60, bounded by 2.0)
        assert recovered[1]["confidence"] == 0.60
        assert recovered[1]["end"] <= 2.0


def test_realign_preserves_canonical_reference_word():
    """Verify acoustic recovery preserves the canonical Quranic reference word text."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 2.0, "end": 3.0, "confidence": 0.92},
    ]
    dummy_audio = np.zeros(16000 * 4, dtype=np.float32)
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    # Mock returns a normalized or altered word string
    mock_align_result = {
        "segments": [
            {
                "text": "الله",
                "words": [
                    {
                        "word": "الله_whisper_normalized",
                        "start": 0.25,
                        "end": 1.15,
                        "score": 0.88,
                    }
                ],
            }
        ]
    }
    mock_whisperx = MagicMock()
    mock_whisperx.align.return_value = mock_align_result

    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        recovered = recover_unaligned_word_gaps(
            words,
            audio=dummy_audio,
            align_model=mock_align_model,
            align_metadata=mock_align_metadata,
        )
        # Canonical reference word must be preserved intact
        assert recovered[1]["word"] == "ٱللَّهِ"
        assert recovered[1]["confidence"] == 0.88
        assert recovered[1]["start"] == 0.8
        assert recovered[1]["end"] == 1.7


def test_realign_rejects_pre_start_timestamp():
    """Verify acoustic realignment is rejected if timestamp starts before recovery_start."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 2.0, "end": 3.0, "confidence": 0.92},
    ]
    dummy_audio = np.zeros(16000 * 4, dtype=np.float32)
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    # slice_start is 0.55s. start: 0.10s gives actual start 0.65s (< 0.80s anchor end)
    mock_align_result = {
        "segments": [
            {
                "text": "ٱللَّهِ",
                "words": [{"word": "ٱللَّهِ", "start": 0.10, "end": 1.15, "score": 0.88}],
            }
        ]
    }
    mock_whisperx = MagicMock()
    mock_whisperx.align.return_value = mock_align_result

    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        recovered = recover_unaligned_word_gaps(
            words,
            audio=dummy_audio,
            align_model=mock_align_model,
            align_metadata=mock_align_metadata,
        )
        # Must be rejected and fall back to bounded interpolation
        assert recovered[1]["confidence"] == 0.60
        assert recovered[1]["start"] >= 0.80
        assert recovered[1]["end"] <= 2.00
