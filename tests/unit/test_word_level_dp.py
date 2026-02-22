"""
Unit tests for word-level DP alignment.
"""

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
