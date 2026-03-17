"""
Unit tests for phonetic similarity scoring.
"""

from munajjam.core.phonetic import phonetic_word_similarity, phonetic_similarity


class TestPhoneticWordSimilarity:
    """Test phonetic_word_similarity function."""

    def test_identical_words(self):
        """Identical words should return 1.0."""
        assert phonetic_word_similarity("بسم", "بسم") == 1.0

    def test_completely_different(self):
        """Completely different words should return a low score."""
        score = phonetic_word_similarity("الله", "كتب")
        assert score < 0.5

    def test_confusion_pairs_score_higher(self):
        """Known confusion pairs (ت/ط, د/ض) should score higher than unrelated chars."""
        # ت and ط are a common ASR confusion pair
        score_confused = phonetic_word_similarity("تب", "طب")
        # ت and ك are unrelated
        score_unrelated = phonetic_word_similarity("تب", "كب")
        assert score_confused > score_unrelated

    def test_empty_strings(self):
        """Empty strings should be handled gracefully."""
        score = phonetic_word_similarity("", "")
        assert isinstance(score, float)

    def test_one_empty_string(self):
        """One empty string should return a low score."""
        score = phonetic_word_similarity("بسم", "")
        assert score < 0.5


class TestPhoneticSimilarity:
    """Test phonetic_similarity function (full text)."""

    def test_identical_text(self):
        """Identical texts should return 1.0."""
        text = "بسم الله الرحمن الرحيم"
        assert phonetic_similarity(text, text) == 1.0

    def test_different_text(self):
        """Different texts should return a score between 0 and 1."""
        score = phonetic_similarity("بسم الله الرحمن الرحيم", "الحمد لله رب العالمين")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        """Result should always be a float."""
        score = phonetic_similarity("بسم", "الله")
        assert isinstance(score, float)

    def test_empty_strings(self):
        """Empty strings should be handled gracefully."""
        score = phonetic_similarity("", "")
        assert isinstance(score, float)
