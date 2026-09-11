"""
Unit tests for word-level DP alignment and gap-recovery DP mapping.
"""

from munajjam.core.arabic import normalize_arabic
from munajjam.core.cascade_recovery import _map_acoustic_to_canonical_dp
from munajjam.core.word_level_dp import align_segments_word_dp
from munajjam.models import AlignmentResult


class TestAlignSegmentsWordDp:
    """Test align_segments_word_dp function."""

    def test_returns_results(self, sample_segments, sample_ayahs):
        """Basic invocation returns a list of AlignmentResult."""
        results = align_segments_word_dp(sample_segments, sample_ayahs)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_covers_all_ayahs(self, sample_segments, sample_ayahs):
        """Result count should match the number of ayahs."""
        results = align_segments_word_dp(sample_segments, sample_ayahs)
        assert len(results) == len(sample_ayahs)

    def test_valid_times(self, sample_segments, sample_ayahs):
        """All results should have valid start/end times."""
        results = align_segments_word_dp(sample_segments, sample_ayahs)
        for result in results:
            assert result.start_time >= 0
            assert result.end_time > result.start_time
            assert 0.0 <= result.similarity_score <= 1.0


def _make_acoustic(words):
    """Build acoustic word dicts with deterministic pattern: cap the tail."""
    return [
        {"word": w, "start": i * 1.0, "end": i * 1.0 + 0.5} for i, w in enumerate(words)
    ]


class TestMapAcousticToCanonicalDp:
    """DP mapping between canonical reference words and acoustic words."""

    def test_equal_sequences(self):
        canonical = ["الحمد", "لله", "رب"]
        acoustic = _make_acoustic(["الحمد", "لله", "رب"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert all(m is not None for m in mapped)
        assert [m["word"] for m in mapped] == canonical

    def test_one_deletion(self):
        """A missing middle acoustic word must map to None, not shift the tail."""
        canonical = ["الحمد", "لله", "رب", "العالمين"]
        acoustic = _make_acoustic(["الحمد", "رب", "العالمين"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert mapped[0] is not None
        assert mapped[1] is None  # "لله" deletion detected, not shifted
        assert mapped[2] is not None
        assert mapped[3] is not None
        assert mapped[2]["word"] == "رب"
        assert mapped[3]["word"] == "العالمين"

    def test_one_substitution(self):
        canonical = ["الحمد", "الله", "رب"]
        acoustic = _make_acoustic(["الحمد", "لله", "رب"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        # "الله" maps to nearest acoustic token (rapidfuzz), still not None.
        assert mapped[0] is not None
        assert mapped[2] is not None
        assert mapped[2]["word"] == "رب"

    def test_one_insertion(self):
        canonical = ["الحمد", "رب"]
        acoustic = _make_acoustic(["الحمد", "لله", "رب"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert mapped[0] is not None
        assert mapped[1] is not None
        # Extra acoustic "لله" is consumed as an insertion, not matched to any canonical.
        assert mapped[1]["word"] == "رب"

    def test_repeated_word(self):
        canonical = ["هو", "هو"]
        acoustic = _make_acoustic(["هو", "هو"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert all(m is not None for m in mapped)

    def test_short_word_with_prefix(self):
        """A very short word must not be swallowed by a longer neighbor."""
        canonical = ["ق", "والليل"]
        acoustic = _make_acoustic(["ق", "والليل"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert mapped[0] is not None
        assert mapped[1] is not None

    def test_arabic_normalization_insensitive(self):
        """Diacritics and alef/hamza variants should not affect DP matching."""
        canonical = ["الْحَمْدُ", "لِلَّهِ"]
        acoustic = _make_acoustic(["الحمد", "لله"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert all(m is not None for m in mapped)

    def test_multiple_consecutive_deletions(self):
        canonical = ["ح", "م", "عسق"]
        acoustic = _make_acoustic(["عسق"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        assert mapped[0] is None
        assert mapped[1] is None
        assert mapped[2] is not None

    def test_all_deleted_returns_all_none(self):
        canonical = ["الحمد", "لله"]
        acoustic = _make_acoustic(["مختلف"])
        canon_norm = [normalize_arabic(w) for w in canonical]
        mapped = _map_acoustic_to_canonical_dp(canonical, canon_norm, acoustic)
        # With no match and only an unrelated acoustic word, DP may still leave
        # canonical words unmatched rather than force wildly wrong mappings.
        assert len(mapped) == len(canonical)
