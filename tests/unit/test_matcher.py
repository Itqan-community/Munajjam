"""
Unit tests for text similarity matching.
"""

import pytest
from munajjam.core.matcher import similarity


class TestSimilarity:
    """Test similarity calculation functions."""

    @pytest.mark.parametrize(
        "text1,text2,min_score,max_score",
        [
            # Identical texts
            ("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم", 1.0, 1.0),
            # Substring match
            ("بسم الله", "بسم الله الرحمن الرحيم", 0.4, 1.0),
            # Single character difference
            ("الحمد لله رب العالمين", "الحمد لله رب العلمين", 0.9, 1.0),
            # Completely different texts
            ("الحمد لله", "قل أعوذ برب الناس", 0.0, 0.5),
            # With diacritics
            ("بِسْمِ اللَّهِ", "بسم الله", 0.9, 1.0),
        ],
    )
    def test_similarity_scores(self, text1, text2, min_score, max_score):
        """Test similarity produces expected score ranges."""
        score = similarity(text1, text2)
        assert min_score <= score <= max_score

    def test_empty_strings(self):
        """Test similarity with empty strings."""
        assert similarity("", "") == 1.0
        assert similarity("الحمد لله", "") == 0.0
        assert similarity("", "الحمد لله") == 0.0

    def test_symmetric_property(self):
        """Test that similarity is symmetric: sim(a,b) == sim(b,a)."""
        text1 = "الحمد لله رب العالمين"
        text2 = "الحمد لله"

        score1 = similarity(text1, text2)
        score2 = similarity(text2, text1)

        assert abs(score1 - score2) < 0.001

    def test_similarity_always_in_valid_range(self, short_arabic_text_pairs):
        """Test that similarity is always in [0, 1]."""
        for text1, text2, _ in short_arabic_text_pairs:
            score = similarity(text1, text2)
            assert 0.0 <= score <= 1.0
