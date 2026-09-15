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


def test_realign_normalizes_quranic_text_for_whisperx():
    """Verify that un-normalized Quranic text with full diacritics is normalized before passing to whisperx.align."""
    import sys
    from unittest.mock import MagicMock, patch

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحْمَـٰنِ", "start": 0.9, "end": 1.0, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 2.0, "end": 3.0, "confidence": 0.92},
    ]
    dummy_audio = np.zeros(16000 * 4, dtype=np.float32)
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "ar"}

    mock_align_result = {
        "segments": [
            {
                "text": "الله الرحمن",
                "words": [
                    {"word": "الله", "start": 0.25, "end": 0.70, "score": 0.90},
                    {"word": "الرحمن", "start": 0.70, "end": 1.15, "score": 0.88},
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
        # Check that whisperx.align received normalized text without diacritics / Alif Wasla
        args, _ = mock_whisperx.align.call_args
        segments_passed = args[0]
        assert segments_passed[0]["text"] == "الله الرحمن"

        # Check canonical words with diacritics were preserved in output
        assert recovered[1]["word"] == "ٱللَّهِ"
        assert recovered[2]["word"] == "ٱلرَّحْمَـٰنِ"
        assert recovered[1]["confidence"] == 0.90
        assert recovered[2]["confidence"] == 0.88


def test_local_to_global_timestamp():
    """local_to_global_timestamp must apply the slice offset exactly once."""
    from munajjam.core.cascade_recovery import local_to_global_timestamp

    gs, ge = local_to_global_timestamp(0.70, 1.40, slice_start=132.40)
    assert gs == 132.40 + 0.70
    assert ge == 132.40 + 1.40


def test_is_placeholder_word_ignores_valid_short_word():
    """A short word with positive confidence and positive duration is not a placeholder."""
    from munajjam.core.cascade_recovery import is_placeholder_word

    assert not is_placeholder_word(
        {"word": "قُلْ", "start": 0.0, "end": 0.12, "confidence": 0.9}
    )
    assert is_placeholder_word(
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0}
    )
    assert is_placeholder_word(
        {"word": "x", "start": 1.0, "end": 1.0, "confidence": 0.5}
    )
    assert is_placeholder_word(
        {"word": "x", "start": 1.0, "end": 1.2, "confidence": 0.5, "fallback": True}
    )


def test_validate_recovered_gap_rejects_non_monotonic():
    from munajjam.core.cascade_recovery import RecoveredWord, validate_recovered_gap

    words = [
        RecoveredWord("a", 1.0, 2.0, 0.9, "acoustic_recovery"),
        RecoveredWord("b", 1.5, 1.8, 0.9, "acoustic_recovery"),
    ]
    valid, reason = validate_recovered_gap(words, 1.0, 3.0, None, None)
    assert not valid
    assert reason is not None


def test_validate_recovered_gap_rejects_out_of_bounds():
    from munajjam.core.cascade_recovery import RecoveredWord, validate_recovered_gap

    words = [RecoveredWord("a", 99.0, 100.5, 0.9, "acoustic_recovery")]
    valid, reason = validate_recovered_gap(words, 100.0, 102.0, None, None)
    assert not valid
    assert reason is not None


def test_validate_recovered_gap_accepts_in_bounds():
    from munajjam.core.cascade_recovery import RecoveredWord, validate_recovered_gap

    words = [
        RecoveredWord("a", 100.1, 100.5, 0.9, "acoustic_recovery"),
        RecoveredWord("b", 100.6, 101.2, 0.9, "acoustic_recovery"),
    ]
    valid, reason = validate_recovered_gap(words, 100.0, 102.0, 100.0, 102.0)
    assert valid
    assert reason is None


def test_validate_recovered_gap_rejects_anchor_overlap():
    from munajjam.core.cascade_recovery import RecoveredWord, validate_recovered_gap

    words = [RecoveredWord("a", 99.5, 100.5, 0.9, "acoustic_recovery")]
    valid, reason = validate_recovered_gap(words, 100.0, 102.0, 100.0, None)
    assert not valid
    assert reason is not None


def test_recover_leading_gap_starts_at_zero():
    """Leading gap recovery must not produce a negative start time."""
    words = [
        {"word": "ٱللَّهِ", "start": 0.0, "end": 0.1, "confidence": 0.0},
        {"word": "ٱلرَّحْمَـٰنِ", "start": 0.1, "end": 0.2, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 3.0, "end": 4.5, "confidence": 0.92},
    ]
    recovered = recover_unaligned_word_gaps(words)
    assert recovered[0]["start"] >= 0.0
    assert recovered[0]["end"] <= recovered[1]["end"]
    assert recovered[1]["end"] <= recovered[2]["start"] + 1e-3


def test_recover_trailing_gap_ends_at_audio_duration():
    """Trailing gap recovery must not extend beyond the audio duration."""
    import numpy as np

    words = [
        {"word": "بِسْمِ", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "ٱللَّهِ", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "ٱلرَّحِيمِ", "start": 0.9, "end": 1.0, "confidence": 0.0},
    ]
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2.0s
    recovered = recover_unaligned_word_gaps(
        words, audio=dummy_audio, audio_duration=2.0
    )
    for w in recovered:
        assert w["end"] <= 2.0 + 1e-3
    # Without alignment model, falls back to interpolation (confidence 0.60).
    assert recovered[1]["confidence"] == 0.60


def test_recover_multiple_gaps_independent():
    """Two separated gaps must be recovered independently without overlap."""
    words = [
        {"word": "a", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "b", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "c", "start": 2.0, "end": 2.8, "confidence": 0.95},
        {"word": "d", "start": 2.8, "end": 2.9, "confidence": 0.0},
        {"word": "e", "start": 4.0, "end": 4.8, "confidence": 0.95},
    ]
    recovered = recover_unaligned_word_gaps(words)
    # b recovered within [0.8, 2.0]; d recovered within [2.8, 4.0]; anchors intact.
    assert recovered[1]["start"] >= 0.8
    assert recovered[1]["end"] <= 2.0
    assert recovered[3]["start"] >= 2.8
    assert recovered[3]["end"] <= 4.0
    assert recovered[0]["end"] == 0.8
    assert recovered[2]["start"] == 2.0
    assert recovered[4]["start"] == 4.0


def test_recover_preserves_anchor_timestamps():
    """Recovery must never modify surrounding valid anchor words."""
    words = [
        {"word": "a", "start": 0.0, "end": 0.8, "confidence": 0.95},
        {"word": "b", "start": 0.8, "end": 0.9, "confidence": 0.0},
        {"word": "c", "start": 0.9, "end": 1.0, "confidence": 0.0},
        {"word": "d", "start": 3.0, "end": 4.5, "confidence": 0.92},
    ]
    recovered = recover_unaligned_word_gaps(words)
    assert recovered[0]["start"] == 0.0
    assert recovered[0]["end"] == 0.8
    assert recovered[3]["start"] == 3.0
    assert recovered[3]["end"] == 4.5
